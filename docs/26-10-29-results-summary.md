# Results summary, NAVSIM-protocol study (2026-10-29)

All experiments of docs/26-10-27-research-plan.md have run. Details and tables per
experiment are in outputs/<nn>/README.md. The LaTeX report (report/report-latex,
`report.pdf`) is rewritten around these results.

## Setup in one paragraph

NAVSIM protocol: 4 front-camera frames at 2 Hz in, 8 poses over 4 s out. 92,381
training scenes, 9,675 validation scenes from 98 held-out logs (scenes overlap,
so the split is by log), navtest (12,146) and navhard two-stage (5,915) for test.
Every encoder goes to the same 9 x 16 grid with 64 PCA channels (tokenizer
latents: 16). The probe is a 19k-parameter flow-matching model on a 9-number
cubic-polynomial code of the trajectory. Main metric: energy score; collapse
measured by dispersion = spread / error (1 = calibrated).

## Question 1: why does the distribution collapse?

| finding | evidence | where |
|---|---|---|
| Few Euler steps pull samples to the conditional mean | 1 step: dispersion 0.37; 10 steps 0.89; 50 steps 0.97 (WM11-ctx) | 12_efficiency |
| For the small probe most of the earlier narrowing was this artefact | 10k scenes, 50 steps: dispersion 1.03 -> 0.95, minADE@32 stays 0.67 | 02_collapse |
| Real collapse needs capacity | width-64 probe on 10k scenes: dispersion 0.91 -> 0.16, minADE@32 0.65 -> 2.06 even with 50 steps; mean ADE stays best | 02_collapse, 06_capacity |
| Memorization onset grows linearly with data | val-loss minimum at 8k / 12k / 40k steps for 2.5k / 5k / 10k scenes, log-log slope 1.16; not reached for >= 20k | 04_data_scaling |
| Polynomial code helps accuracy, not collapse | poly2/3 energy 6.0 vs raw 7.1, but dispersion 0.82-0.87 | 05_codec |
| Smaller probe only avoids collapse by underfitting | width 8: dispersion 1.02, energy 7.63 | 06_capacity |

Remedy: all 92k scenes, >= 10-20 Euler steps, select on validation energy.

## Question 2: which representation knows most?

| finding | evidence | where |
|---|---|---|
| Orbis world model is the best encoder | val energy WM11-ctx 2.91, V-JEPA 2 4.66, DINOv2 5.37, q_sem 5.55, q_rec 7.34 | 07_encoders |
| Middle block, last observed frame | block 11 best of 3..23; last observed frame beats next-frame slot at every depth | 08_wm_blocks |
| Ego motion is linearly readable | speed R^2 0.95 (WM11-ctx) vs 0.87 V-JEPA 2, 0.59 q_sem; acceleration peaks at block 11 | 13_linear_probes |
| DINOv2 > q_sem is only width | at 16 channels DINOv2 5.81 vs q_sem 5.55 | 07, 12 |
| Ego status is a shortcut for displacement metrics | status + command energy 2.39 beats every image probe; nearest-status lookup ADE 1.26 vs constant velocity 2.76 | 03, 07 |
| Held-out logs predict navtest | navtest 3-10% worse, same ranking | 09_generalization |
| No compositional generalization, overconfident | unseen (Singapore, left): energy 9.11 vs 4.51, spread unchanged | 10_compositional |
| Unseen city | without Singapore: energy there 9.43 vs 6.31, dispersion 0.66 | 09_generalization |

## Question 3: useful for planning?

| finding | evidence | where |
|---|---|---|
| Diffusion probe, image only | navtest EPDMS 0.788 (sample mean), best of 4 0.882 | 11_epdms |
| Ego status alone drives badly | EPDMS 0.645 despite best energy (DAC 0.77) | 11_epdms |
| Adding ego status + command | 0.810, best of 4 0.900; gain is mostly comfort (EC) | 15_planner |
| Fast one-pass planner on frozen features | full navtest 0.881 (regression, 1.8 min training), 0.857 (vocabulary) | 16_fast_planner |
| 500-scene EPDMS proxy | matches official per-scene scores exactly; Spearman 0.98 vs full navtest | 16_fast_planner |
| Proposals are good, selection is the bottleneck | best of 8 anchors 0.95-0.96 | 16_fast_planner |
| Sub-score distillation with 5% of targets does not help | 0.872-0.877 vs seeds 0.863-0.885 | 16_fast_planner |
| navhard | all probes 0.33-0.41; stage-two lane keeping ~0.45 for every planner; not comparable with leaderboard | 11, 15 |

## Open items

- Sub-score targets for all navtrain (about 15 h of PDM caching and scoring) to
  test Hydra-style selection properly.
- More seeds for the encoder and block comparisons (single seeds now).
- Samples of the compositional held-out model on Singapore left turns (what does
  it predict instead?).
- Commit the code restructure and the report.
