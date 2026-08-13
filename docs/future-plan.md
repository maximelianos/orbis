# Future plan: probing Orbis representations for ego-motion and planning

Written 2026-07-30. Two parts:

1. [nuReasoning dataset facts](#1-nureasoning-dataset-facts) — answers to the three
   questions in [idea.md](idea.md) (duration, size/structure, 2D trajectories).
2. [Three experiments not done yet](#3-three-experiments-not-done-yet) — grounded in
   what [report_synthetic.md](report_synthetic.md) and
   [exp_navsim/readme.md](../exp_navsim/readme.md) already cover.

---

## 1. nuReasoning dataset facts

Source: the [HF dataset card](https://huggingface.co/datasets/qixuewei/nuReasoning),
its [`data_schema.py`](https://huggingface.co/datasets/qixuewei/nuReasoning/raw/main/data_schema.py),
and the [nuReasoning paper](https://arxiv.org/html/2605.31572). Owned by Motional AD
Inc., collected with their internal AV fleet — **not** derived from nuScenes or nuPlan,
so it is a genuinely held-out domain relative to the NAVSIM/nuPlan data used so far.
Dual licence: non-commercial by default, commercial terms on request.

### 1.1 Clip duration

- **~20 seconds per clip**, sampled at **10 Hz** → **~200 frames per clip**.
- That is 5× the temporal density of the NAVSIM logs currently in use
  (`stored_data_frame_rate: 2` in [exp_navsim/config.yaml](../exp_navsim/config.yaml)),
  and ~5× longer than a navtrain scene (~40 frames @ 2 Hz = 20 s — same wall-clock
  span, but 200 usable frames instead of 40).

### 1.2 How many clips, and how the dataset is structured

**20K clips ≈ 105 hours** of curated driving, split:

| split | clips |
|-------|-------|
| train | 17K |
| validation | 2K |
| test (private) | 1K |

Layout — one self-contained directory per clip, named `<log_name>_<keyframe_token>`:

```
data/
├── train/part_1, part_2, .../<log>_<keyframe_token>/
├── validation/part_1/...
└── test/part_1/...

<log>_<keyframe_token>/
├── metadata.json          # clip-level: location, scenario type, frame rate, temporal bounds
├── map.pkl                # static vectorised map: lanes, baseline paths
├── ego_state/<ts_us>.pkl  # pose, velocity, acceleration, dimensions, history + future trajectory
├── annotations/<ts_us>.pkl# 3D detections, traffic-light states
├── reasoning/             # spatial / decision / counterfactual annotations
└── sensor assets          # 8 cameras + LiDAR
```

Repo-root files worth reading before writing a loader: `data_schema.py` (the
dataclasses), `view_data.ipynb`, `view_reasoning.ipynb`.

- **Cameras (8):** `front, front_left, front_right, left, right, back, back_left,
  back_right`, each with intrinsics + sensor→lidar extrinsics and W/H.
- **Reasoning annotations** are sparse and of three kinds — *spatial* (2D/3D
  detections, object relations, projected map features), *decision* (scene
  interpretation, driving action), *counterfactual* (alternative action, safety
  outcome). Per the paper: spatial from 3 s into the clip at 1 Hz, decision/
  counterfactual from 5 s before the keyframe at 0.2 Hz; 247K spatial frames and
  57K decision/counterfactual frames over 19K annotated clips.
- Each frame also carries the **ego route path** and a **high-level navigation
  command** — directly useful for experiment 3 below.
- **Size: ~4.94 TB total.** Plan for a subset download; 17K clips × 200 frames × 8
  cameras is far more than the probe needs.

### 1.3 Do clips have 2D trajectories?

**Yes.** `EgoState` in `data_schema.py` has exactly the fields needed:

```python
pose         = {"x","y","z","yaw","qw","qx","qy","qz"}
velocity     = {"vx","vy","vz"}          # ego frame
acceleration = {...}
dimensions   = {"l","w","h"}
trajectory_history: Optional[List[List[float]]]
trajectory_future: Optional[List[List[float]]]
```

So there are two independent routes to a 2D BEV trajectory:

1. **Precomputed** — read `trajectory_future` directly. The paper's planning
   benchmark uses a **5-second ego-frame trajectory**, so this is the intended
   supervision target.
2. **Derived** — integrate `pose` (x, y, yaw) across `ego_state/<ts_us>.pkl` and
   transform into the ego frame at the anchor frame. This is the same computation
   `NavsimLongBase` already does from `ego2global_translation` /
   `ego2global_rotation`, so the existing pose/velocity helpers port over almost
   unchanged.

⚠️ **To verify from `view_data.ipynb` before writing the loader:** the column layout
of `trajectory_future` (whether it is `(x, y)`, `(x, y, yaw)`, or includes `z`) and
its sampling rate — 5 s could be stored as 50 points @ 10 Hz or subsampled to the
8-point @ 2 Hz NAVSIM convention. The card does not state it. Route 2 sidesteps this
entirely and is the safer default.

**Verdict for the project:** nuReasoning is a good fit. 10 Hz gives dense
supervision, 20 s clips are long enough for a real context/future split, ego-frame
future trajectories come for free, and the map + navigation command + counterfactual
annotations enable stratified and multimodality analyses that NAVSIM cannot support.

---

## 2. What is already done (baseline for "not done yet")

- **[report_synthetic.md](report_synthetic.md)** — flow-matching + temporal
  DiT-style denoiser validated on synthetic Bézier trajectories. Conditioning was
  *deliberately zeroed* (§4.5), so it measured distribution coverage only. Ablations:
  hidden width (8 vs 16), per-channel velocity normalisation (on vs off).
- **[exp_navsim](../exp_navsim/)** — real NAVSIM (navhard + navtrain) pipeline:
  cached **stage-1 tokenizer latents** (`latent_key: encoded_q_sem`, 16 channels),
  flow-matching model conditioned on those latents, `context_images: 2`,
  `num_diffusion_steps: 50`, val metrics `val/mse` and `val/std` (STD of total turn).

The gap that matters: **everything so far conditions on VAE latents only.** The
world model — `models.second_stage.fm_model.Model` wrapping
`networks.DiT.dit.STDiT` (depth 24, hidden 768, 16×16 latent grid, 16 channels) —
has never been touched by a probe, even though "the WM has learned ego-motion
representations" is the project's actual claim. There are also no control baselines
and no proper distributional metric.

---

## 3. Three experiments not done yet

### Experiment 1 — Where does the ego-motion information live? Layer-wise probe of the STDiT world model

**Why.** The stated aim is that *the VAE **+ world model*** encode ego-motion. So far
only the VAE has been probed. Without reading inside STDiT there is no evidence about
the world model at all, and no way to say whether next-frame prediction training adds
planning-relevant structure on top of what the tokenizer already provides.

**Setup.** Freeze the stage-2 model. Add a forward hook that dumps STDiT block
activations — a sparse ladder (blocks 0, 4, 8, 12, 16, 20, 23) plus the stage-1
`encoded_q_sem` as layer "−1". Extend `cache_latents.py` to write one HDF group per
tap so the cache is built in a single pass. Train **one identical small attentive
probe per tap** (the existing `TrajectoryDenoiser` at `hidden_dim: 64`, cross-attending
over the 16×16 tokens of the context frames), same steps, same seed, same split.

Two things to decide and report explicitly, since they change what the result means:

- **Which diffusion timestep** the activations are read at. Representation quality in
  diffusion transformers is strongly t-dependent, so sweep at least `t ∈ {0.2, 0.5,
  0.8}` at one mid-depth block before fixing t for the full ladder.
- **Teacher-forced vs rolled-out** latents. Teacher-forced answers "is the info
  present"; rollout answers "does it survive generation".

**Metrics.** ADE / FDE at 1 s, 3 s, 5 s + `val/std` of total turn, plotted as a curve
over tap depth.

**Success criteria.** A non-flat depth curve is the result either way. If mid-depth
STDiT blocks beat `encoded_q_sem` by a clear margin, the world model demonstrably adds
ego-motion structure. If the curve is flat, the honest conclusion is that the
tokenizer already carries everything and the WM adds nothing decodable — worth
knowing, and worth reporting rather than burying.

**Cost.** The dominant cost is cache size: 7 taps × 768 dims is much larger than the
16-channel stage-1 cache. Mitigate by caching a subset of clips (~2K), storing fp16,
and mean-pooling over the token grid for a first pass before committing to full
per-token caches.

---

### Experiment 2 — Probe capacity sweep and control baselines: is the information *in the latents*, or trivially available?

**Why.** A probe that decodes trajectories well proves nothing on its own. Future
motion is largely predictable from current ego velocity alone, and a large enough
probe can learn the marginal trajectory distribution while ignoring its conditioning
entirely — exactly the failure mode that made the synthetic experiment
unconditional-by-accident (§4.5). None of these controls exist yet, and without them
the project's central claim is not falsifiable.

**Setup.** Fix one tap (the winner from experiment 1) and run a 2D grid:

*Probe capacity* — `hidden_dim ∈ {16, 32, 64, 128}` × depth `∈ {1, 2, 4}` blocks,
reported against parameter count. The "small probe" claim in [idea.md](idea.md) needs
a number attached to it.

*Controls, all with the identical probe:*

| control | what it rules out |
|---------|-------------------|
| **Constant-velocity extrapolation** (no learning) | how much is just kinematics |
| **Ego-state-only probe** (velocity + accel, no latents) | how much needs vision at all |
| **Randomly-initialised frozen tokenizer/STDiT** | credit to *training*, not architecture |
| **Shuffled latents** (latents paired with another clip's trajectory) | probe memorising the marginal |
| **Latents only, ego state withheld** | the actual quantity of interest |

**Metrics.** ADE/FDE for every cell, plus the **gap** to constant-velocity as the
headline number. Report ADE against parameter count on a log-x axis and stratify by
manoeuvre (straight / turning / stopping, from the nuReasoning navigation command) —
the constant-velocity baseline is near-perfect on straights, so an aggregate mean
hides the entire effect.

**Success criteria.** The claim is supported only if the latent probe beats
constant-velocity *and* the ego-state-only probe on the turning/stopping strata, the
random-init control is clearly worse than the trained one, and the shuffled control
collapses to the marginal. A small probe (≤ a few M params) reaching most of the
large probe's accuracy is what turns "decodable" into "linearly-ish accessible".

**Cost.** Cheap — reuses one cache, ~15 short training runs.

---

### Experiment 3 — Does the diffusion probe recover a *calibrated* multimodal distribution?

**Why.** "Using diffusion, recover a distribution of future trajectories"
([idea.md](idea.md)) has never been evaluated as a distribution. The synthetic
experiment measured coverage of a fan by eye with conditioning disabled; exp_navsim
logs `val/std` of total turn, which detects *some* spread but cannot distinguish a
well-calibrated bimodal distribution at a T-junction from an over-dispersed unimodal
blob. nuReasoning is the right dataset for fixing this: it supplies a high-level
navigation command, a vectorised map, and explicit **counterfactual** annotations
("alternative action, safety outcome").

**Setup.** Take the best probe from experiments 1–2 and sample `k ∈ {1, 5, 10, 20}`
trajectories per context. Evaluate on three strata drawn from nuReasoning metadata:
(a) straight/free-flow, (b) intersections with a branching route, (c) clips with
counterfactual annotations. Two conditioning variants:

- **unconditioned on command** — does the model spontaneously produce the modes?
- **conditioned on the navigation command** — does the correct mode get selected?
  This is also the first real test that conditioning is wired up, which the synthetic
  experiment explicitly never verified.

**Metrics.**

- `minADE_k` / `minFDE_k` and the `minADE_k` vs `k` curve (mode coverage).
- **Mode recall on map branches** — project samples onto map baseline paths and count
  which reachable branches get ≥1 sample. Directly answers "did it find the turn?".
- **Calibration** — the fraction of contexts where the ground truth lies inside the
  sample envelope, target ≈ the nominal level; plus a miss-rate at a 2 m threshold.
- **Sharpness on straights** — mean pairwise sample distance, which *should* be small.
  Reporting it alongside coverage is what prevents "sample everything" from scoring well.

**Success criteria.** `minADE_k` improving with k while sharpness on straights stays
tight, and mode recall > 1 branch at branching intersections. Command conditioning
should reduce ADE on stratum (b) specifically — if it does not, conditioning is
either broken or being ignored, which is itself the finding.

**Cost.** Moderate — `num_diffusion_steps: 50` × 20 samples × the eval set. Sampling
dominates; the metrics are cheap. Map-projection code for branch recall is the only
new component.

---

## 4. Prerequisite: nuReasoning ingest

All three experiments need the same loader, so build it once:

1. Download a **subset** — a few hundred train clips + a few hundred validation
   clips is enough for probing, against 4.94 TB total. Front camera only for the
   probe; surround only for BEV visualisation.
2. `exp_navsim/data/nureasoning_long.py` — a third `NavsimLongBase` subclass. Only
   `_episode_frames` / `_build_index` should need writing, exactly as the readme's
   "Data-format assumption" section predicts; the pose/velocity/camera helpers are
   reusable. Map the schema's `pose` dict onto the `ego2global_*` interface rather
   than changing the base class.
3. Add `nureasoning` to the `dataset:` switch, `data_long.nureasoning`, and
   `cache.dir.nureasoning` in [exp_navsim/config.yaml](../exp_navsim/config.yaml).
4. Set `stored_data_frame_rate: 10`; keep `frame_rate` configurable so the 2 Hz
   NAVSIM-comparable setting stays available (useful as a like-for-like control).
5. Verify with `visualize_dataset.py` — the stats page plus a few episode pages will
   immediately show whether poses and trajectories are being read correctly.

Do this before experiment 1; experiments 2 and 3 then need no further data work.

---

## Sources

- [qixuewei/nuReasoning · Hugging Face](https://huggingface.co/datasets/qixuewei/nuReasoning)
- [nuReasoning `data_schema.py`](https://huggingface.co/datasets/qixuewei/nuReasoning/raw/main/data_schema.py)
- [nuReasoning: A Reasoning-Centric Dataset and Benchmark for Long-Tail Autonomous Driving (arXiv 2605.31572)](https://arxiv.org/html/2605.31572)
- [nuReasoning project page](https://nureasoning.github.io/)
- [Motional announcement](https://motional.com/news/cracking-long-tail-code-autonomous-driving-nureasoning)
