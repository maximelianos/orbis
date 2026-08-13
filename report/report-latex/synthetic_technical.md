# Synthetic Trajectory Diffusion: Data, Model, and Ablations

This report documents the synthetic-trajectory experiment used to validate the
flow-matching trajectory generator before moving to real nuPlan data. It covers
(1) how the synthetic data is generated, (2) the coordinate
normalization/denormalization scheme with exact formulas, (3) the transformer
denoiser architecture, (4) the flow-matching diffusion formulation, and (5) two
ablations — hidden-unit width and the effect of velocity normalization —
illustrated with the generated-vs-ground-truth comparison plots.

**Primary sources**

- Experiment config: [exp/whole_trajectory.yaml](../exp/whole_trajectory.yaml)
- Model: [exp/whole_t_att.py](../exp/whole_t_att.py)
- Data loader: [synthetic_trajectory_dataloader_IMPORTS.py](../loaders_for_projects/exp_loaders/synthetic_trajectory_dataloader_IMPORTS.py)
- Building blocks: [networks/blocks.py](../networks/blocks.py), [networks/pos_emb.py](../networks/pos_emb.py)
- Evaluation harness: [exp/test_trajectory_diffusion.py](../exp/test_trajectory_diffusion.py)
- Result figures: [evaluate/trajectory_comparison_*](../evaluate/)

---

## 1. Overview and motivation

The end goal of the project is a diffusion model that predicts a vehicle's
future motion. Real driving data (nuPlan) is expensive to debug against: the
trajectories are noisy, the conditioning is high-dimensional (camera latents),
and a failure could come from the data pipeline, the tokenizer, the
normalization, or the diffusion model itself. To isolate the *diffusion + sequence
model* component, we first train on a fully controlled synthetic distribution of
2-D trajectories whose ground-truth shape we know in closed form.

The synthetic task is deliberately simple: trajectories are smooth cubic Bézier
curves that start at the origin, head in the +x direction, and fan out left or
right depending on a scalar *turn* parameter. The model must learn to denoise a
whole trajectory at once (not autoregressively) using a flow-matching objective,
and to produce samples that, as a *set*, reproduce the fan-shaped distribution of
the ground truth. Because the conditioning is intentionally disabled in this
experiment (see §4.5), the model is evaluated on *distribution coverage* rather
than per-sample turn accuracy.

The same model code (`DiffusionModel` in [exp/whole_t_att.py](../exp/whole_t_att.py))
is later reused, with image conditioning added, as the production
`networks.whole_context.DiffusionModel`. So everything validated here transfers
directly.

---

## 2. Data synthesis

All data generation lives in
[synthetic_trajectory_dataloader_IMPORTS.py](../loaders_for_projects/exp_loaders/synthetic_trajectory_dataloader_IMPORTS.py).
The dataset is procedural: there is no file on disk, every sample is computed on
the fly from a turn parameter.

### 2.1 Turn parameter to endpoint

Each trajectory is parameterized by a single scalar `turn` ∈ [0, 1] drawn
uniformly:

```python
self.turns = np.random.uniform(turn_min, turn_max, size=dataset_size)
```

The turn maps to a heading angle and an endpoint on a circle of radius `r`:

$$\alpha(\text{turn}) = 30^\circ - 60^\circ \cdot \text{turn}, \qquad
(X, Y) = \big(r\cos\alpha,\; r\sin\alpha\big).$$

So `turn = 0` gives a **left** turn at `+30°`, `turn = 0.5` goes **straight
ahead** (`0°`), and `turn = 1` gives a **right** turn at `−30°`. With the config
value `r = 1`, all endpoints lie on the unit circle between `+30°` and `−30°`,
i.e. an arc spanning a 60° wedge centered on the +x axis. This is exactly the
fan visible in the "Ground Truth" panels of every figure below.

### 2.2 Cubic Bézier shape

Given the start `p0` and the endpoint `p3 = (X, Y)`, the curve is a cubic Bézier:

$$B(s) = (1-s)^3\,p_0 + 3(1-s)^2 s\,p_1 + 3(1-s)\,s^2\,p_2 + s^3\,p_3,
\qquad s \in [0, 1].$$

The two interior control points are placed to force a **straight initial
segment**:

```python
p0 = (0, 0)
p3 = (X, Y)
p1 = p0 + (r, 0) * 0.5      # = (0.5 r, 0)
p2 = p1                     # identical to p1
```

Because `p1 = p2 = (0.5r, 0)` sit on the +x axis, the tangent at the start
points purely along +x, and the curve only begins to bend toward its endpoint in
its second half. This is why every ground-truth trajectory leaves the origin
along a common straight stem before fanning out — clearly visible as the shared
horizontal segment near `x ∈ [0, 0.2]` in the plots. (The commented-out
alternative control points would have produced S-shaped curves; the active code
uses the simpler straight-stem variant.)

The curve is sampled at `T` equally spaced parameter values:

```python
t_values = np.linspace(0, 1, T)
trajectory = np.array([bezier_curve(p0, p1, p2, p3, t) for t in t_values])
```

producing a `(T, 2)` array of positions. The config uses `T = 20`.

### 2.3 Positions to velocities

The model operates on **velocities** (position deltas), not absolute positions.
Velocities are first differences:

$$v_t = p_{t+1} - p_t, \qquad t = 0, \dots, T-2,$$

with the last entry padded to zero (there is no frame after the last):

```python
velocity = np.zeros_like(trajectory)
velocity[:-1] = trajectory[1:] - trajectory[:-1]   # (T, 2)
```

Each sample is therefore a dict:

| key        | shape   | meaning                                   |
|------------|---------|-------------------------------------------|
| `turn`     | scalar  | turn parameter in [0, 1]                  |
| `position` | (T, 2)  | absolute (x, y) along the Bézier curve    |
| `velocity` | (T, 2)  | first-difference deltas, last row = 0     |

Working in velocity space has two advantages: (a) the signal is roughly
zero-mean and stationary along the trajectory, which is friendlier to a
diffusion model than absolute coordinates that grow monotonically in x; and (b)
positions are trivially recovered by a cumulative sum at sampling time
(see §4.4). The trade-off is that the predicted velocities must be denormalized
before integration, and any per-step error accumulates along the cumsum.

### 2.4 Dataset configuration

From [exp/whole_trajectory.yaml](../exp/whole_trajectory.yaml):

| param               | value   | role                                         |
|---------------------|---------|----------------------------------------------|
| `T`                 | 20      | timesteps per trajectory                     |
| `r`                 | 1       | endpoint radius                              |
| `turn_min`/`turn_max` | 0 / 1 | full left-to-right fan                       |
| `dataset_size`      | 10000   | number of trajectories per epoch             |
| `output_normalization` | true | enable velocity normalization (§3)           |
| `batch_size`        | 30      | per-GPU batch                                |

The unused image-related fields (`hdf5_paths_file`, `aug`, `scale_min/max`,
frame rates) are inherited boilerplate from the nuPlan config and have no effect
on synthetic generation — the loader ignores them via `*args, **kwargs`.

---

## 3. Coordinate normalization and denormalization

Velocities have very different statistics along x and y. The x-deltas are always
positive and fairly large (the trajectory marches steadily in +x), while the
y-deltas are near zero on average and symmetric (left turns cancel right turns).
A diffusion model adds isotropic Gaussian noise, so unequal per-channel scales
make the noise level effectively channel-dependent. Standardizing each channel
to zero mean and unit variance fixes this.

### 3.1 Computing the statistics

Statistics are computed once over the whole dataset and cached to disk
(`precompute_normalization`). Every trajectory's velocity array is stacked and
reduced over all points and all samples, **per channel**:

```python
velocities = np.concatenate(velocities, axis=0)   # (dataset_size * T, 2)
self.velocity_mean = velocities.mean(axis=0)       # (2,)  -> (mean_dx, mean_dy)
self.velocity_std  = velocities.std(axis=0)        # (2,)  -> (std_dx,  std_dy)
```

The result is two 2-vectors `(mean_dx, mean_dy)` and `(std_dx, std_dy)`. They are
saved to a parameter-keyed cache file so repeated runs with the same
`(dataset_size, T, r)` skip recomputation:

```python
cache_filename = f"norm_stats_size{dataset_size}_T{T}_r{r}.npz"
np.savez(cache_path, mean=self.velocity_mean, std=self.velocity_std)
```

Note that because the padded last velocity row is `0`, it is included in the
statistics; with `T = 20` this is a 1/20 contribution that slightly shrinks the
mean and std but is harmless and consistent between normalize and denormalize.

### 3.2 Normalization formula

With a small epsilon for numerical safety, the forward transform applied per
channel is:

$$\hat{v} = \frac{v - \mu}{\sigma + \varepsilon}, \qquad \varepsilon = 10^{-8},$$

```python
mu    = torch.tensor(self.velocity_mean)
sigma = torch.tensor(self.velocity_std) + 1e-8
batch['velocity'] = (velocity - mu) / sigma
```

This is applied in `__getitem__`, so the model always sees standardized
velocities `v̂` during training. The diffusion process therefore operates in a
space where both channels are ~N(0, 1), matching the isotropic Gaussian prior
the sampler starts from.

### 3.3 Denormalization formula

The exact inverse is applied to the model's sampled velocities before
integrating them back into positions:

$$v = \hat{v}\,(\sigma + \varepsilon) + \mu,$$

```python
mu    = torch.tensor(self.velocity_mean)
sigma = torch.tensor(self.velocity_std) + 1e-8
batch['velocity'] = velocity * sigma + mu
```

The same `(σ + ε)` factor is used in both directions, so the round trip is exact
up to floating point. The `output_normalization` flag short-circuits both
methods to identity when disabled, which is precisely the toggle exercised in
the normalization ablation (§5.2).

### 3.4 Relationship to the online normalizer

In the production code path, this precomputed cache is replaced by an
EMA-tracked `TrajectoryNorm` module (see
[networks/norm.py](../networks/norm.py)) that maintains the same
`(mean, var)` per channel and applies the identical `(v − μ)/σ` /
`v·σ + μ` formulas, but updates the statistics online during training instead of
in a one-shot pass. The synthetic experiments here use the dataset-side cached
version; the formulas are mathematically the same.

---

## 4. Model and diffusion formulation

The model is `DiffusionModel`, a `pl.LightningModule` in
[exp/whole_t_att.py](../exp/whole_t_att.py). It denoises an entire trajectory of
`T` velocity vectors simultaneously using conditional flow matching, with a
transformer (cross-attention) backbone over the time axis.

### 4.1 Inputs and per-point featurization

The denoiser consumes three things and produces a velocity field:

- `noisy_deltas`: `(B, T, 2)` — the noised velocity trajectory at diffusion time `t`
- `t`: `(B,)` — the diffusion timestep (one per trajectory, shared across points)
- `turn`: `(B,)` — turn conditioning (disabled, see §4.5)

Two scalars are lifted into embeddings of width `D = hidden_dim`:

- **Diffusion-time embedding** `fm_time_mlp`: sinusoidal `PositionEmbedding1d`
  followed by `Linear → Tanh → Linear`. Produces a `(B, D)` vector, broadcast to
  all `T` points.
- **Frame-index embedding** `frame_mlp`: the normalized frame index
  `arange(T)/T` passed through an identical sinusoidal + MLP stack. Produces a
  `(T, D)` vector, broadcast across the batch. This is what tells the model
  *where* along the trajectory each point sits.

The per-point input is the concatenation

$$x_i = \big[\, \underbrace{\hat{v}_i}_{2},\; \underbrace{e_{\text{time}}}_{D},\;
\underbrace{e_{\text{frame},i}}_{D},\; \underbrace{\text{turn}}_{1} \,\big]
\in \mathbb{R}^{2 + 2D + 1},$$

```python
x = torch.cat([noisy_deltas, fm_emb_expanded, frames_emb_expanded, turn_expanded], dim=-1)
```

This is reshaped to `(B·T, 2 + 2D + 1)` and pushed through a **pointwise MLP**
(`denoiser`): four `Linear` layers with `Tanh` nonlinearities that map each point
independently into a `D`-dimensional feature. At this stage there is no
interaction between timesteps — it is a shared per-frame encoder.

### 4.2 Transformer architecture (temporal cross-attention)

Temporal mixing is done by a stack of four `TemporalBlock`s (`cross_denoiser`),
each built on the shared `CrossAttentionBlock` in
[networks/blocks.py](../networks/blocks.py). In this experiment the blocks are
used in a **self-attention** configuration — the per-point features serve as both
the key/value context and the query:

```python
pred_flow = features                       # (B, T, D)
for layer in self.cross_denoiser:
    pred_flow = layer(pred_flow, pred_flow, fm_emb)   # context=query=pred_flow
```

Each `CrossAttentionBlock` is a DiT-style transformer block with **AdaLN-Zero**
conditioning:

1. **LayerNorm (no affine)** on the (shared) context and query streams.
2. **AdaLN modulation**: the conditioning vector — here the diffusion-time
   embedding `fm_emb` `(B, D)` — is passed through `SiLU → Linear(D, 8D)` to
   produce eight `(B, D)` vectors: shift/scale/gate for the query branch,
   shift/scale for the context branch, and shift/scale/gate for the output
   branch. The final linear is **zero-initialized** (AdaLN-Zero), so each block
   starts as an identity map and learns to deviate — this is what makes deep
   diffusion transformers train stably.
3. **Multi-head attention**: 8-head `nn.MultiheadAttention` (batch-first). The
   queries attend over all `T` timesteps, letting each point gather information
   from the whole trajectory. The result is gated and added back to the query
   (residual).
4. **Feed-forward**: `Linear(D, 2D) → GELU → Dropout → Linear(2D, D)`, modulated
   by the output shift/scale, gated, and added residually.

Stacking four such blocks gives the model enough capacity to enforce trajectory
smoothness and the global fan structure. The `modulate(x, shift, scale)` helper
applies `x * (1 + scale) + shift`. Frame ordering enters only through the frame
embedding added in §4.1; attention itself is permutation-equivariant.

### 4.3 Output head

After the four temporal blocks, a final `output_mlp = Linear(D, 2)` projects each
point's feature back to a 2-D velocity:

$$\text{pred\_flow} \in \mathbb{R}^{B \times T \times 2}.$$

This is the model's estimate of the flow-matching velocity field (§4.4), one
2-vector per timestep.

### 4.4 Flow-matching diffusion

Training uses **conditional flow matching** with a linear interpolation
("rectified-flow"-style) schedule. The schedule coefficients are:

$$\alpha(t) = 1 - t, \qquad \sigma(t) = \sigma_{\min} + t\,(1 - \sigma_{\min}),
\qquad \sigma_{\min} = 10^{-5}.$$

**Forward (noising).** Given clean (normalized) velocities `x = v̂` and noise
`ε ∼ N(0, I)`, the noisy sample at diffusion time `t ∈ [0, 1)` is

$$x_t = \alpha(t)\,x + \sigma(t)\,\varepsilon
      = (1-t)\,x + \big(\sigma_{\min} + t(1-\sigma_{\min})\big)\,\varepsilon.$$

At `t = 0`, `x_t = x` (clean data); at `t = 1`, `x_t ≈ ε` (pure noise). `t` is
sampled uniformly per trajectory:

```python
t = torch.rand(B, device=deltas.device)
noisy_deltas, noise = self.add_noise(deltas, t)
```

**Target velocity field.** The regression target is the time-derivative of the
interpolation path, expressed via the constant coefficients `A(t) = 1` and
`B(t) = −(1 − σ_min)`:

$$u_t = A(t)\,x + B(t)\,\varepsilon = x - (1 - \sigma_{\min})\,\varepsilon.$$

**Loss.** Plain MSE between the predicted field and the target, averaged over all
batch elements, timesteps, and channels:

$$\mathcal{L} = \big\lVert\, f_\theta(x_t, t) - u_t \,\big\rVert_2^2.$$

```python
target = self.A(t_expanded) * deltas + self.B(t_expanded) * noise
loss = ((pred_v - target) ** 2).mean()
```

Note the loss is taken over **all** `T` points (this synthetic experiment does
not split out a context prefix; the later image-conditioned model does).

**Sampling.** Generation integrates the learned ODE `dx/dt = f_θ(x_t, t)`
backward from `t = 1` (noise) to `t = 0` (data) with a first-order Euler scheme
over `num_diffusion_steps = 20` uniform steps:

```python
sampled_deltas = torch.randn(B, T, 2, device=device)     # x at t = 1
t_steps = torch.linspace(1, 0, num_diffusion_steps + 1)
for i in range(num_diffusion_steps):
    pred_v = self.forward(turn, sampled_deltas, t_steps[i].repeat(B))
    dt = t_steps[i] - t_steps[i + 1]                      # > 0
    sampled_deltas = sampled_deltas + pred_v * dt
```

After integration the velocities are **denormalized** (§3.3) and integrated into
positions by cumulative summation, seeded with `context_length` ground-truth
points (`context_length = 0` ⇒ start at the origin):

```python
trajectory[:, :context_length] = position[:, :context_length]
for f in range(1, T):
    trajectory[:, f] = trajectory[:, f-1] + sampled_deltas[:, f-1]
```

This last step is where any per-velocity error compounds: a small bias in
`dx`/`dy` accumulates over the 20-step cumsum, which is why endpoint scatter
(rather than path wiggle) is the most sensitive quality indicator in the plots.

### 4.5 Conditioning is disabled

A crucial detail for interpreting the results: the turn parameter is **zeroed
out** inside `forward`:

```python
turn = torch.zeros_like(turn)  # TODO remove conditioning
```

So although `turn` is plumbed through the input, the model receives no
information about which way to turn. It is effectively an **unconditional**
generator of trajectories. Consequently:

- The model cannot and should not map a *specific* turn value to a *specific*
  endpoint.
- The right thing to ask is whether the *set* of generated trajectories covers
  the same fan-shaped distribution as the ground truth.
- In the figures, the color of a generated curve (assigned by sample index) is
  **not** expected to match the same-colored ground-truth curve. Only the
  overall envelope should match.

### 4.6 Optimization

`configure_optimizers` uses `AdamW` (`weight_decay = 0.01`) with a linear warmup
(`warmup_steps = 0` here, so warmup is skipped) followed by cosine decay down to
`min_lr_multiplier = 0.1` of the base LR over `max_steps`. The base learning rate
is `1e-3` (config `base_learning_rate`). Training in the ablations runs for
**5000 steps** at `batch_size = 30`, `precision: 16-mixed`, with gradient
clipping by norm (`grad_clip: 1.0`).

| hyperparameter        | value      |
|-----------------------|------------|
| base LR               | 1e-3       |
| optimizer             | AdamW, wd 0.01 |
| LR schedule           | cosine to 0.1× |
| warmup steps          | 0          |
| precision             | 16-mixed   |
| grad clip (norm)      | 1.0        |
| diffusion steps (sampling) | 20    |
| `sigma_min`           | 1e-5       |
| temporal blocks       | 4          |
| attention heads       | 8          |

---

## 5. Ablations

All ablations are produced by
[exp/test_trajectory_diffusion.py](../exp/test_trajectory_diffusion.py), which
loads a trained checkpoint, samples `num_samples` trajectories at turn values
swept uniformly with `linspace(0, 1, N)`, and draws ground-truth vs. generated
side by side. Start points are circles, end points are squares, and the origin
is the green star. Color encodes the swept turn index (blue → red), but recall
from §4.5 that the model is unconditional, so color correspondence across the two
panels is not expected — coverage of the fan is what matters.

### 5.1 Hidden-unit size

We compare `hidden_dim = 8` against `hidden_dim = 16`, both with 4 temporal
blocks and 5000 training steps. (Note: `hidden_dim` sets `D` everywhere — the
time/frame embedding width, the per-point MLP width, the attention embedding
dimension, and the feed-forward width `2D` — so it is the single knob governing
model capacity.)

**`hidden_dim = 8`** — [evaluate/trajectory_comparison_8h_4layers_5000_steps.png](../evaluate/trajectory_comparison_8h_4layers_5000_steps.png):

![hidden_dim = 8](../evaluate/trajectory_comparison_8h_4layers_5000_steps.png)

The 8-wide model already recovers the qualitative structure: a shared straight
stem out of the origin, a left/right fan, and endpoints clustered near the unit
arc around `x ≈ 0.9–1.0`. The fan is a little ragged — some endpoints fall short
of the arc and the spacing of generated curves is uneven — but the envelope is
clearly correct.

**`hidden_dim = 16`** — [evaluate/trajectory_comparison_16h_4layers_5000_steps.png](../evaluate/trajectory_comparison_16h_4layers_5000_steps.png):

![hidden_dim = 16](../evaluate/trajectory_comparison_16h_4layers_5000_steps.png)

Doubling the width produces a visibly **cleaner and more evenly spread** fan. The
straight initial stem is reproduced more faithfully, individual curves are
smoother, and the endpoint cloud hugs the `x ≈ 0.9–1.0` arc more tightly with
fewer stragglers. The extra capacity mostly buys smoother per-step velocities
(less cumsum drift) and better coverage of the extreme turns.

| width `D` | fan shape | endpoint clustering | path smoothness |
|-----------|-----------|---------------------|-----------------|
| 8         | correct, slightly ragged | moderate spread around arc | minor wobble |
| 16        | correct, evenly spread   | tighter on the arc | smoother |

**Takeaway.** Capacity helps, but with diminishing returns — even `D = 8` learns
the task. `D = 16` is a good default for this synthetic problem; the config's
nominal `16` (with `64` noted as an alternative) is consistent with this.

### 5.2 With vs. without velocity normalization

This ablation toggles `output_normalization`. With it on, the model trains and
samples in standardized velocity space (§3); with it off, the model sees raw
velocities directly and the denormalize step is the identity.

**With normalization** — [evaluate/trajectory_comparison_with_norm.png](../evaluate/trajectory_comparison_with_norm.png):

![with normalization](../evaluate/trajectory_comparison_with_norm.png)

Generated trajectories are smooth, stay within a sensible bounding box
(`x ≲ 1.0`), and the endpoint cloud forms a recognizable arc covering the full
left-to-right fan. There is scatter (expected from an unconditional model), but
no gross scale errors and no runaway trajectories.

**Without normalization** — [evaluate/trajectory_comparison_without_norm.png](../evaluate/trajectory_comparison_without_norm.png):

![without normalization](../evaluate/trajectory_comparison_without_norm.png)

Without normalization the samples are noticeably **wavier and less controlled**.
Several trajectories overshoot to `x ≈ 1.2` (beyond the unit-radius ground
truth), individual paths show kinks rather than smooth arcs, and the endpoint
cloud is more diffuse. The root cause is the channel-scale mismatch described in
§3: the diffusion prior is isotropic `N(0, I)`, but the raw x-velocity has a much
larger scale than the raw y-velocity, so a single shared noise level is too large
for one channel and too small for the other. The model wastes capacity modeling
the scale instead of the shape, and the Euler-integrated cumsum amplifies the
residual per-step errors into endpoint overshoot.

| setting       | path smoothness | endpoint spread | scale errors |
|---------------|-----------------|-----------------|--------------|
| with norm     | smooth          | arc-like, bounded `x ≲ 1.0` | none |
| without norm  | wavy, kinked    | diffuse, `x` up to ~1.2     | overshoot |

**Takeaway.** Per-channel velocity normalization is clearly beneficial.
It is cheap (two 2-vectors), exactly invertible, and it removes the channel-scale
mismatch that otherwise degrades both smoothness and endpoint accuracy. This
result is the empirical justification for keeping normalization in the
production pipeline (and for the online `TrajectoryNorm` variant described in
§3.4).

---

## 6. How to reproduce

Training and evaluation commands are documented in the header of
[exp/test_trajectory_diffusion.py](../exp/test_trajectory_diffusion.py):

```bash
# Train (5000 steps)
python train_nuplan.py --config exp/whole_trajectory.yaml --logdir log_exp --max_steps 5000

# Evaluate: sample and plot generated vs. ground truth
python exp/test_trajectory_diffusion.py --config exp/whole_trajectory.yaml \
    --last_ckpt --logdir log_exp --num_samples 100 --num_steps 20 --T 20
```

To reproduce a specific ablation, edit `model.params.hidden_dim` (8 vs. 16) or
`data.params.train.params.output_normalization` (true vs. false) in
[exp/whole_trajectory.yaml](../exp/whole_trajectory.yaml), retrain, and re-run
the evaluation. The output figure is written to
`evaluate/trajectory_comparison.png` by default.

---

## 7. Summary

- **Data.** Synthetic 2-D trajectories are cubic Bézier curves from the origin to
  a point on a 60°-wide unit arc, parameterized by a turn scalar; the model
  consumes first-difference velocities with a zero-padded last step.
- **Normalization.** Velocities are standardized per channel with cached
  `(mean, std)` 2-vectors using `v̂ = (v − μ)/(σ + ε)` and inverted exactly with
  `v = v̂(σ + ε) + μ`.
- **Model.** A pointwise MLP encoder feeds a 4-layer DiT-style temporal
  self-attention stack with AdaLN-Zero conditioning on the diffusion time, and a
  linear head outputs a per-step flow-matching velocity field.
- **Diffusion.** Linear-schedule conditional flow matching
  (`α = 1−t`, `σ = σ_min + t(1−σ_min)`) trained with MSE against the field
  `x − (1−σ_min)ε`, sampled by 20-step Euler integration from noise to data.
- **Conditioning.** The turn input is intentionally zeroed, so evaluation
  measures distribution coverage of the fan, not per-turn accuracy.
- **Ablations.** Wider hidden size (16 > 8) yields a cleaner, better-spread fan
  and tighter endpoints; per-channel normalization is essential — disabling it
  produces wavy, overshooting trajectories with diffuse endpoints.

Together these results validate that the flow-matching + temporal-transformer
recipe learns smooth, well-distributed trajectories on a controlled problem,
clearing the way for adding image conditioning on real nuPlan data with the same
architecture.

