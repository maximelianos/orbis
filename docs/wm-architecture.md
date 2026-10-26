# Which latents does the world model produce, and which ones can I use?

Written 2026-09-17. About the world-model-conditioned trajectory model in
[exp_navsim/wm/](../exp_navsim/wm/). Companion to
[future-plan.md](future-plan.md), whose Experiment 1 asks the same question as a
measurement; this document answers the *inventory* half — what tensors exist,
where they are in the code, and what each one can and cannot carry.

## The prompt

> This code uses the world model as pretrained latent representation. It uses the
> final denoised latent though, which does not contain egomotion information.
> Which latents does the world model produce and which ones can I use? Write this
> prompt and answer in `docs/world-model.md`.

---

## 1. One correction to the premise

The conditioning tensor is **not** only the denoised latent. `imagine()` returns
`cat([context, imagined])` ([wm_runner.py:191-200](../exp_navsim/wm/wm_runner.py#L191-L200)),
so the aggregator sees `context_images` observed latents *plus* the imagined one —
3 frames at the current config. The two observed frames are there, and ego-motion
between them is in principle recoverable from that pair.

The rest of the premise holds, and for a sharper reason than "the latent has no
ego-motion":

1. **The imagined latent is an appearance latent, not a motion latent.** It lives
   in the tokenizer's per-frame space. Nothing in it is ego-motion; ego-motion is
   only the *difference* between it and the previous frame, which some downstream
   module has to compute.
2. **It is a random sample, not an estimate.** `_denoise_next`
   ([wm_runner.py:202-223](../exp_navsim/wm/wm_runner.py#L202-L223)) starts from
   `torch.randn` and solves 4 Euler steps. Two calls on the same context give two
   different futures. As a conditioning signal that draw is variance, not
   information. (§3.1 shows this is fixable for free.)
3. **The world model is not action-conditioned.** It is trained purely as
   `p(x_{t+1} | x_{≤t}, frame_rate)`
   ([fm_model.py:370-398](../models/second_stage/fm_model.py#L370-L398)). Whatever
   ego-motion it implies is its *prior* over what the car will do, not what the car
   did. That prior is genuinely useful to a planner, but it is not a measurement.
4. **The aggregator destroys cross-frame correspondence before anything can
   difference the frames.** `LatentAggregator.forward`
   ([denoiser.py:128-147](../exp_navsim/wm/denoiser.py#L128-L147)) reshapes to
   `(B*F, D, H, W)` and pools **each frame independently** into 8 tokens of width
   64. Frames never interact inside the aggregator; they only meet later, in the
   `TemporalBlock`s. Ego-motion is a sub-token spatial correspondence between two
   18×32 grids, and 8 pooled tokens per frame is a hard place to read it from.

So the honest statement of the problem is: *the pipeline throws away the world
model's temporal reasoning and keeps only its appearance output, then compresses
each frame past the point where motion is recoverable.*

---

## 2. The full inventory

Resolution throughout: images 288×512, tokenizer patch 16 → an **18×32 latent
grid**; the backbone uses `patch_size: 1`, so one frame is **576 tokens** and the
grid never changes inside the network
([logs_wm/orbis_288x512/config.yaml](../logs_wm/orbis_288x512/config.yaml)).

### Tier 0 — stage-1 tokenizer latents (cached on disk)

`VQModelIF.encode` ([vqgan.py:391-415](../models/first_stage/vqgan.py#L391-L415))
runs **two** independent MAE-initialised encoders and returns
`continuous = (h, h2)`, each 16 channels after its `quant_conv`, L2-normalised
over channels. [cache_latents.py](../exp_navsim/data/cache_latents.py) stores them
as `encoded_q_rec` and `encoded_q_sem`, `(T, 16, 18, 32)` per episode. The
`VIT_DINOv2` distillation head is what makes one branch semantic.

| name | shape | where |
| --- | --- | --- |
| `encoded_q_rec` | `(T, 16, 18, 32)` | HDF5 cache, `data/navtrain_latents_288x512` |
| `encoded_q_sem` | `(T, 16, 18, 32)` | same |

Strictly per-frame. **Zero motion content by construction.** The basic model
([exp_navsim/model.py:104](../exp_navsim/model.py#L104)) uses `encoded_q_sem` alone.

### Tier 1 — world-model input space

`pack()` ([wm_runner.py:151-166](../exp_navsim/wm/wm_runner.py#L151-L166)) mirrors
`ModelIF.encode_frames` ([fm_model.py:431-445](../models/second_stage/fm_model.py#L431-L445)):
`cat([q_rec * 4, q_sem * 4])` → `(B, F, 32, 18, 32)`. This is a rescaled
concatenation, no new information. It is what `image_encoder_channels: 32` refers
to.

### Tier 2 — what the backbone computes internally

`SwinSTDiTNoExtraMLP` is `STDiT` with the spatial attention swapped for windowed
Swin attention ([dit.py:616-629](../networks/DiT/dit.py#L616-L629)). Depth 24,
hidden 768, 16 heads, `causal_time_attn: true`, `modulate_time_attn: true`,
`max_num_frames: 6`.

`preprocess_inputs` ([dit.py:519-542](../networks/DiT/dit.py#L519-L542)) patchifies
`cat([context, target])` together, adds the 2D sin-cos position embedding, a
learned per-slot frame embedding and a frame-rate embedding. Then every block
([dit.py:271-303](../networks/DiT/dit.py#L271-L303)) does *spatial attention →
spatial MLP → **temporal attention across frames** → temporal MLP*, with adaLN
conditioned on the diffusion time.

That temporal attention is the only place in the whole system where information
from different frames is mixed at full spatial resolution. **This is where
ego-motion lives.** The tensor is:

```
x : (B, F, 576, 768)      after every one of the 24 blocks
      ^  ^    ^     ^
      |  |    |     hidden_size
      |  |    576 = 18*32 grid positions, patch_size=1
      |  F = context_images + 1 (target)
      batch
```

Because time attention is **causal**, frame *i* attends to frames ≤ *i*: the last
context frame's tokens are a full contextualised encoding of everything before it,
and the target frame's tokens see all context.

This tensor is currently unreachable. `STDiT.forward` already has the hook —
`return_features`, a `features = []` list, a commented-out `# features[i] = x`, and
a `raise NotImplementedError`
([dit.py:583-605](../networks/DiT/dit.py#L583-L605)). Enabling it is three lines.

### Tier 3 — what the backbone returns

`final_layer` is applied **only to the target frame's tokens**
([dit.py:600](../networks/DiT/dit.py#L600)) and unpatchified, giving

```
v : (B, 1, 32, 18, 32)     the flow-matching velocity field, -dx_t/dt
```

The training target is `A(t)·x_0 + B(t)·ε` with `A = 1`, `B = -(1-σ_min)`
([fm_model.py:76-86](../models/second_stage/fm_model.py#L76-L86)), i.e. the network
regresses `x_0 - (1-σ_min)·ε`. That identity is what §3.1 exploits.

### Tier 4 — the sampled latent (what is used today)

`(B, 1, 32, 18, 32)` after 4 Euler steps, concatenated onto the context and handed
to the aggregator ([model.py:174-205](../exp_navsim/wm/model.py#L174-L205)).

### What does not exist anywhere

No action head, no pose head, no explicit motion or flow output, no
ego-motion-conditioned variant. Nothing in the released checkpoint predicts a
velocity in metres. Any ego-motion signal must be *read out* of one of the tensors
above by a trained probe.

---

## 3. What to use, in order of effort

### 3.1 Free fix: `wm_steps: 1` turns the sample into a conditional mean

At `t = 1` the noised target is `x_1 = α(1)·x_0 + σ(1)·ε = ε`, so the network's
input carries no information about `x_0` and its output is
`v = E[x_0 | context] - (1-σ_min)·ε`. One Euler step over the whole interval
([wm_runner.py:219-222](../exp_navsim/wm/wm_runner.py#L219-L222)) gives

```
x ← ε + v·(1 - 0) = E[x_0 | context] + σ_min·ε        (σ_min = 1e-5)
```

**One Euler step is exactly the posterior mean of the next latent** — deterministic,
seed-independent, and one backbone forward instead of four. More steps buy a
sharper, more photographic sample at the cost of injecting the noise seed into the
conditioning. For a *picture* that trade is right; for a conditioning signal it is
backwards. The config comment at
[config.yaml:110-112](../exp_navsim/wm/config.yaml#L110-L112) already says "a blurry
latent still carries the ego-motion" — `wm_steps: 1` is the blurriest and the
cheapest, and it is the only setting with no sampling variance at all.

Try `wm_steps: 1` before anything else. It is a one-character change, it makes the
run 4× cheaper, and if the metrics do not move you have learned that the sampling
detail was never the bottleneck.

### 3.2 Cheap: give the aggregator the difference explicitly

The model has to derive motion from a pair of independently-pooled frames. Hand it
the pair difference instead:

```python
# in get_encoded_q, after imagine():
deltas = latents[:, 1:] - latents[:, :-1]          # (B, F-1, 32, 18, 32)
latents = torch.cat([latents[:, 1:], deltas], dim=2)   # 64 channels
```

Costs one `image_encoder_channels: 64` edit and nothing at run time. This is the
control baseline for everything below: if explicit differencing of *tokenizer*
latents already gets the gain, the world model's internals are not what is missing.

### 3.3 The real answer: mid-depth block features (Tier 2)

The representation-learning use of a diffusion transformer is its intermediate
activations, not its output — that is the finding behind DIFT and REPA, and it is
what [future-plan.md](future-plan.md) Experiment 1 is designed to measure. Here the
argument is structural rather than empirical: blocks are the only components that
mix frames, and `final_layer` reads out the target frame alone, so *every* purely
temporal computation the model performs is discarded before the latent the pipeline
currently consumes.

**Concretely.** Run the backbone once with `target = randn` at `t = 1` — no
sampling loop at all — and read `x` after a middle block:

```
(B, F, 576, 768)  ->  reshape to (B, F, 768, 18, 32)  ->  LatentAggregator
```

The grid is unchanged at 18×32, so it drops into `LatentAggregator` with only
`latent_channels: 768`; its `nn.Linear(768, 64)`
([denoiser.py:113](../exp_navsim/wm/denoiser.py#L113)) does the reduction. Memory at
`batch_size: 16`, F=3 is ~40 MB in fp16 — not a constraint. Cost is one forward
pass, i.e. cheaper than today.

Three choices to make, each worth reporting:

- **Which block.** Mid-depth first (block 12 of 24), then a ladder {0, 6, 12, 18, 23}.
  The depth curve is the result either way.
- **Which frames.** The last context frame's tokens are clean and see all history;
  the target frame's tokens at `t = 1` are noise-driven but have pulled in context
  through 24 rounds of temporal attention. Start with all F, ablate later.
- **Which `t`.** Feature quality in diffusion transformers is strongly `t`-dependent.
  `t = 1` needs no sampling; `t ≈ 0.5` with a teacher-forced noised target is the
  standard choice and is what the literature reports best features at, but it
  requires the true next latent, so it only works at training time.

**Implementation sketch** (four touch points):

| file | change |
| --- | --- |
| [networks/DiT/dit.py:583-605](../networks/DiT/dit.py#L583-L605) | collect `x` into `features` per block, return `(out, features)` when `return_features` |
| [exp_navsim/wm/wm_runner.py](../exp_navsim/wm/wm_runner.py) | add `features(context, frame_rate, t, block)` alongside `imagine()`; returns `(B, F, 768, 18, 32)` |
| [exp_navsim/wm/model.py:174](../exp_navsim/wm/model.py#L174) | call it instead of `self.world_model(...)`; memoise on the batch as today |
| [exp_navsim/wm/model.py:154](../exp_navsim/wm/model.py#L154) | the `latent_channels == image_encoder_channels` assert compares against `net.in_channels` (32) and must compare against `hidden_size` (768) in feature mode |

### 3.4 Also available, smaller payoff

- **Longer rollout.** `wm_imagined_frames: 2-3`. The imagined *sequence* carries
  implied speed more legibly than a single frame, at `wm_steps` forwards each. Note
  `max_num_frames: 6` and the sliding window at
  [wm_runner.py:199](../exp_navsim/wm/wm_runner.py#L199).
- **Seed ensemble.** Several `_denoise_next` draws on one context; their per-cell
  variance is a map of where the future is uncertain. Diagnostic more than feature —
  and if §3.1 is adopted the mean is already free, so this only adds the spread.
- **The velocity field itself** as extra channels, rather than the integrated
  latent. Same tensor as §3.1 before integration; probably redundant with it.

### 3.5 Deciding between them without a full training run

Every option above is a different conditioning tensor on the same task, so compare
them with the cheapest possible readout first: a **linear probe on the known ego
velocity**, one probe per candidate tensor, same split and seed. That isolates "is
the information present" from "can the trajectory head use it", which is exactly
the confound [future-plan.md](future-plan.md) Experiment 2 is about. Only promote
the winners to a full `train_nuplan.py` run.

---

## 4. Summary

| latent | shape | motion content | cost | verdict |
| --- | --- | --- | --- | --- |
| `encoded_q_rec` / `q_sem` | `(T, 16, 18, 32)` | none (per-frame) | cached | baseline |
| packed WM input | `(B, F, 32, 18, 32)` | none (rescaled Tier 0) | free | — |
| **block activations** | `(B, F, 576, 768)` ×24 | **temporal attention across frames** | 1 forward | **best candidate** |
| velocity field `v` | `(B, 1, 32, 18, 32)` | implied, target frame only | 1 forward | ≈ posterior mean |
| **1-step denoised latent** | `(B, 1, 32, 18, 32)` | implied, deterministic | 1 forward | **free improvement** |
| 4-step sampled latent | `(B, 1, 32, 18, 32)` | implied, + seed noise | 4 forwards | used today |
| frame differences | `(B, F-1, ·, 18, 32)` | explicit | free | control baseline |

Start with `wm_steps: 1` (§3.1) and the difference channels (§3.2) — both are
config-scale changes that make the run cheaper. Then open up the block activations
(§3.3), which is the only option that reaches computation the current pipeline
throws away.
