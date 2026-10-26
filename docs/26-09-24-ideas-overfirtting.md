# Ideas how to go from memorization to a distribution of trajectories

Written 2026-09-24. Based on the two TensorBoard exports in the repo root,
[log_260923_navtrain.pdf](../log_260923_navtrain.pdf) and
[log_260924_navtrain.pdf](../log_260924_navtrain.pdf), and the code in
[exp_navsim/](../exp_navsim/). Nothing in the code was changed for this note.

## The prompt

> Look at docs/ and .pdf logs - the model overfits. I have options to use
> semantic encoder, detail encoder, or ORBIS world model. How to prevent
> overfitting? Suggest 10 ideas, think carefully about the reasons for
> overfitting, and what could be the fundamental problems with the current
> implementation.

Follow-up:

> I want to not only prevent memorizing, but encourage to predict a
> distribution of trajectories instead of conditional mean.

---

## The goal, stated as a data problem

Every training scene comes with exactly one future. The empirical conditional
distribution is therefore a single point per scene. A model that fits it
perfectly is a lookup table: one context, one trajectory, no spread. That is
the memorization we see in the logs.

So a distribution can only come from pooling. The model has to treat scenes
that look alike as the same context, and then the futures of all those scenes
become the samples of one distribution. How much it pools is decided by what
the conditioning lets it tell apart:

- the condition identifies the scene → one future per scene, a delta (memorization);
- the condition says nothing → the marginal over all driving, every sample
  looks like the dataset average (the other failure, close to the
  unconditional mean);
- the condition carries only what can be predicted (speed, road layout,
  maybe the route) → the right answer, a spread over what drivers actually do
  in that kind of situation.

Most of the ideas below come down to moving the condition into that last
regime, and then making sure the sampler and the metrics do not squash the
spread back to a mean.

## What the logs show

| run | train loss | val/loss | val/mse |
|---|---|---|---|
| 09-22 (D=64, 64 tokens, no regularization) | 0.59 → **0.02** | lowest before ~60k steps, then up to **1.09** | 26 → 21.7 |
| 09-24 (D=32, 16 tokens, noise/mask/dropout, random start) | → 0.17 | lowest at ~50–90k steps, then up to 0.36 | 34 → 26.4 |

The regularization in [regularization.py](../exp_navsim/regularization.py)
shrank the gap a lot, but it did not change the trend. Both runs reach their
best val/loss early and get worse from there on.

**A train loss of 0.02 is a delta per scene.** The flow-matching loss can only
get near zero when the model knows the clean trajectory from the condition
alone. At high t the input is almost pure noise, so a low loss there means the
image alone pins down up to 20 s of driving. That information is not in two
frames 0.5 s apart. The only way there is to recognise the scene.

**val/loss goes up while val/mse keeps going down.** At high t the loss-optimal
answer is the mean of the futures the model considers possible. A memorizing
model answers with one confident trajectory and is punished for it on unseen
scenes. So the rising val/loss is literally the model losing its spread. val/mse
keeps improving because it does not measure spread at all (see problem 5
below), which is why it cannot be the selection criterion.

## Fundamental problems

### 1. Thousands of real epochs

There are 9577 cached episodes in `data/navtrain_latents`, about 8.6k in the
train split. `dataset_length: 100000` makes an "epoch" virtual. 1M steps at
batch 100 is about 100M draws, roughly 11,600 passes over each scene. The 500k
run is about 5,800. The random start adds at most ~28 windows per scene and
they overlap heavily. With that many passes any condition that separates the
scenes will eventually be used to separate them.

### 2. The task is ill-posed at 20 s

Training asks for at least 6 s (`random_start_min_frames: 12`), validation for
the whole episode from frame 0, up to about 20 s. The model gets no driving
command and no ego speed. At that horizon the true conditional distribution is
extremely wide, and 8.6k single samples cannot describe it. The loss is
dominated by the far future, where the only cheap way to reduce it is scene
identity.

### 3. The honest signal is hard to reach, the shortcut is easy

The most useful predictable cue is ego speed, which lives in the motion between
frame 0 and frame 1. The aggregator pools every frame on its own
([denoiser.py](../exp_navsim/denoiser.py), `ImageAggregator`), so that motion
is nearly unreadable (see [world-model.md](world-model.md), section 1, point
4). Meanwhile 2 × 16 × 18 × 32 latents per frame, `q_rec` especially, are close
to a unique fingerprint of the scene. The network takes the easy path.

### 4. The output space is unconstrained

Every frame gets a free (dx, dy) with independent Gaussian noise. Nothing ties
the samples to shapes that cars drive, so a distribution learned from few
examples has no structure to lean on, and it is easy to fit a separate curve
per scene.

### 5. The metrics reward the mean and mismeasure the spread

`sampled_mse` in [model.py](../exp_navsim/model.py) averages the squared error
of every sample against the one ground truth. That average equals the squared
error of the sample mean plus the variance of the samples. So every bit of
spread is counted as error, and a model that returns the same mean trajectory
50 times gets the best val/mse. Selecting checkpoints on it pushes towards the
conditional mean, the thing we want to avoid.

`val/std` is meant to measure spread, but `total_turn` in
[metrics.py](../exp_navsim/metrics.py) takes `atan2` of every segment. When the
car is slow or stopped the segments are tiny and the heading is noise, so a
stopped car produces a large "turn STD" that has nothing to do with the
modes. The 0.2–0.7 swings between validations are probably mostly this.

### 6. The sampler leans towards the mean

At t = 1 the model's velocity points at the conditional mean
([world-model.md](world-model.md), section 3.1): one Euler step from noise
lands exactly on it. With `num_diffusion_steps: 10`, the first steps each take a
big stride towards the mean before the model has had a chance to commit to a
mode, and samples end up between modes. The fewer the steps, the more the
samples look like the mean.

## 10 ideas, roughly in order of impact

### 1. Measure the distribution, not the error of each sample

Replace val/mse as the selection metric with a proper scoring rule, which is
minimized only by the true distribution. The energy score over K samples
x_1..x_K of the trajectory and the ground truth y:

```
ES = mean_k ||x_k - y||  -  0.5 * mean_{k,l} ||x_k - x_l||
```

The first term wants samples near the truth, the second rewards spread, and
the balance between them is exactly right for a calibrated model. Log it next to:

- minADE@K and minFDE@K (does some sample cover the truth);
- mean pairwise distance between samples, as the spread measure, instead of
  total-turn STD (or fix `total_turn` to skip segments shorter than ~0.5 m);
- a coverage check: the fraction of val scenes whose true endpoint lies inside
  the region spanned by the samples, per horizon. A calibrated model covers
  about as often as the region's nominal level.

Also log val/loss per t-bin, with fixed t and noise, to see where the
generalization gap sits. And the working-copy
[config.yaml](../exp_navsim/config.yaml) has `split: train` under
`validation`, while both logged runs used `val`. If that is a memorization
check, keep it as a second val loader next to the real one.

### 2. Shorten the horizon to 4 s

`num_frames: 8`, as in NAVSIM. Random windows then give about 30 samples per
scene instead of one long target. At 4 s the distribution is narrow enough to
be learned from this much data and still has real modes (brake or go, lane
keep or change, turn or not). The far future, which today dominates the loss
and drives the memorization, is gone.

### 3. Condition on what is predictable, and decide about the route

Give the model ego speed and acceleration. The cheap version needing no new
caching is `context_traj: 2`: frames 0 and 1 are observed anyway, so their
velocity is legitimate input. Speed is the part of the future that is not
random, and once the model has it, the remaining spread is the real
uncertainty.

The driving command is a design choice. With it, the model predicts the
distribution within a route. Without it, the route itself is one of the modes
(left, straight, right at a junction). The best of both: train with the
command and drop it with some probability (like `cond_dropout`), so at test
time you can sample either way.

### 4. Stop early, select on the energy score, keep an EMA

Train 50–100k steps, set `val_check_interval` in steps, and change
`monitor="train/loss"` in [train_nuplan.py](../train_nuplan.py) (line 171) to
the energy score from idea 1. Evaluate with an EMA of the weights. The val/loss
minimum is where the spread is still intact, and the logs say that is early.

### 5. Make the image a worse fingerprint: `q_sem`, bottleneck, masking

This is what makes the model pool similar scenes, so it creates the
distribution directly. Drop `q_rec`: it carries texture and exact appearance,
which identifies the scene and not much else useful for planning. Keep the
DINO-distilled `q_sem`, which encodes layout and is more invariant. Then
tighten the aggregator: 2–4 image tokens instead of 16, and harder training
perturbations (`mask_ratio` 0.5–0.75 over spatial blocks instead of single
cells, `noise_std` around 0.3, `cond_dropout` 0.2–0.3). Run the ablations
sem only, rec only, both, and compare them on the energy score, not on MSE.

### 6. Hand over motion explicitly, or take it from ORBIS

Feed the difference latent of frame 1 minus frame 0
([world-model.md](world-model.md), section 3.2), or let the two frames attend
to each other before pooling. That makes the predictable cue as easy to use as
the shortcut.

ORBIS mid-depth features do the same with much more prior knowledge: its
temporal attention computes ego motion across frames, and it was trained on far
more driving than 8.6k scenes, so the features are generic rather than
scene-specific. Use `wm_steps: 1` or block ~12 features at t = 1
([world-model.md](world-model.md), sections 3.1 and 3.3), cache them offline,
and bottleneck them hard. At 768 channels they are an even better fingerprint
than the tokenizer latents if passed through as they are.

### 7. Sample so that modes survive

- More steps (30–50) or a second-order solver (Heun), so early steps do not
  jump straight to the mean. Compare the spread at 1, 5, 10, 50 steps: if it
  grows with steps, the sampler was part of the collapse.
- A stochastic sampler (inject a little noise at each step, as in SDE-style
  samplers) spreads samples within a mode.
- `cond_dropout` already trains an unconditional model inside the same network,
  so classifier-free guidance comes for free: `v = v_uncond + w (v_cond - v_uncond)`.
  w > 1 sharpens towards the conditional mode, w < 1 widens towards the
  marginal. That gives a direct diversity knob to tune against the energy score.
- The logit-normal t draw puts little training at t near 1, which is where a
  sample picks its mode. If the per-t-bin loss from idea 1 shows the high-t
  bins lagging, shift the draw's centre upwards or mix in some uniform t.

### 8. Pool futures from neighbouring scenes into the target

Since one scene gives one future, build more explicitly. Embed every scene by
what should matter (speed, curvature of the first second, command, a pooled
`q_sem` vector), find its K nearest neighbours, and with some probability
train on a neighbour's future instead of its own. The target becomes a sample
from "what drivers did in situations like this", which is the distribution we
want. K and the probability set how much spread is taught. Keep validation on
the scene's own future.

### 9. Anchored multimodal output

Cluster the training trajectories (at 4 s) into 20–256 anchors. Then either
diffuse a residual from an anchor, as DiffusionDrive does, or add a small head
that predicts a categorical distribution over anchors and diffuse the residual
conditioned on the chosen one. The categorical part represents the modes
explicitly and cannot average them, and the residual keeps the fine
distribution. This also fixes problem 4: samples start from shapes that cars
really drive.

### 10. Mirror and add data

Encode the flipped images once, cache them, and negate y in the trajectory.
Flipping the latent grid is not the same as encoding a flipped image, so this
needs a re-encode. Beyond doubling the data, it makes left and right turns
equally represented, which helps the turn modes stay balanced. Separately:
check if more navtrain sensor blobs can be downloaded, and split by log rather
than by scene token. Neighbouring NAVSIM scenes from one log can overlap in
time and leak across the split, which makes memorization look like
generalization.

## Which encoder

Encoder choice sets how much the model pools, so it decides the spread.

- **Detail encoder (`q_rec`).** The strongest fingerprint. Expect a delta per
  scene and the least spread. Only useful as a control.
- **Semantic encoder (`q_sem`).** More invariant, so similar scenes collapse
  onto similar conditions and their futures pool into one distribution. The
  main candidate, together with ego speed from idea 3.
- **ORBIS.** The best source of motion, used as mid-depth features and
  bottlenecked. It also opens a second route to diversity that neither
  tokenizer latent has: ORBIS is itself a generative model of the future. With
  `wm_steps > 1` the imagined frame is a sample, and conditioning the trajectory
  on different imagined futures gives different trajectories. For that to mean
  something the trajectory model has to be trained on the true next latent (so
  it learns to read the future frame), and then fed ORBIS samples at test time.
  Worth trying after the basics work, as it hands the multimodality to a model
  trained on far more data.

Start with ideas 1–4. Without a distributional metric none of the other
comparisons can be judged, and with a 20 s horizon and no speed input every
encoder will memorize.
