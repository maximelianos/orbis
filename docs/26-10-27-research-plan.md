# Research plan: trajectory distributions from frozen features on NAVSIM

Written 2026-10-27 from imbi.md. Results go to outputs/<nn_name>/README.md, major
steps to docs/log.md.

## Protocol (changed from the earlier episode probe)

- NAVSIM protocol: 4 history front-camera frames at 2 Hz in, 8 poses over 4 s out.
  The earlier probe saw frames 0 and 1 of an episode and predicted the trajectory
  from frame 0 on, so the image of t = 0.5 s was an input while the motion up to
  0.5 s was a target. It also predicted up to 20 s.
- Data: every navtrain scene with its 4 history images on disk, 102k, instead of
  9.6k longest-contiguous-run episodes.
- Validation holds out whole logs. navtrain scenes overlap by 13 of 14 frames, so
  splitting by scene leaks.
- Test: navtest (EPDMS one-stage, the most published comparisons) and
  navhard_two_stage (NAVSIM v2 two-stage EPDMS), both from the OpenScene test logs.
- Encoders reduced to one interface: 9 x 16 grid, 64 PCA channels (16 for the
  tokenizer latents), PCA fitted on training logs only.

## Metrics

Per scene, K = 16 (validation) or 32 (offline) samples:
- ADE / FDE averaged over samples and best-of-K, error of the sample mean;
- energy score: proper scoring rule, minimized only by the true distribution;
- spread E||X - X'|| and dispersion = spread / E||X - y||: 1 for a calibrated
  sampler, < 1 when samples collapse;
- NAVSIM-style L2 at 1-4 s; EPDMS and its components on the test splits, for one
  random sample, the sample mean and the best of K (oracle).
Memorization shows as the train_sub and val curves separating: on seen scenes the
spread and error go to zero, on unseen ones the error stays and the spread still
shrinks.

## Experiments

| nn | question |
|---|---|
| 02 collapse | does the distribution collapse with training? q_sem, full data and 10k scenes |
| 03 baselines | constant velocity, train mean, nearest ego status |
| 04 data scaling | collapse onset vs training-set size (Bonnaire et al. 2025, Favero et al. 2025: memorization time grows linearly with n) |
| 05 codec | raw poses vs polynomial orders as the generated representation |
| 06 capacity | probe width / depth vs collapse |
| 07 encoders | q_sem, q_rec, both, DINOv2, V-JEPA 2, world model; ego-status and no-input controls |
| 08 wm blocks | world-model depth curve, next-frame slot vs last observed frame |
| 09 generalization | held-out logs vs navtest vs navhard; held-out city |
| 10 compositional | (city, command) pairs held out of training |
| 11 epdms | EPDMS and components vs published planners |
| 12 efficiency | Euler steps, PCA width, history length |
| 13 linear probes | what each feature space holds linearly (speed, progress, lateral offset, command) |

## Sources

- NAVSIM docs (navsim/docs/metrics.md, splits.md, agents.md): EPDMS definition, 4 s
  horizon at 10 Hz scoring, 4 history frames, navtrain from trainval, navtest and
  navhard from test.
- Bonnaire, Urfin, Biroli, Mezard, "Why diffusion models don't memorize: the role of
  implicit dynamical regularization in training", NeurIPS 2025: generalization
  time constant, memorization time proportional to n.
- Favero, Sclocchi, Wyart, arXiv 2505.16959 (2025): memorization onset proportional
  to dataset size across image and language diffusion models.
