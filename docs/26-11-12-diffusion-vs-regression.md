# Diffusion probe vs regression for trajectory prediction

## The question

We take a pretrained driving world model (ORBIS), freeze it, and ask what its internal
features know about where the car will drive next. To answer it we train a small network on
top of the frozen features, a *probe*, that reads the camera features of the last few frames
and predicts the future path of the ego car.

There are two ways to train such a probe.

- **Regression.** The probe outputs one trajectory and is trained to be as close as possible
  to the trajectory the human driver actually drove (L1 loss on the positions). At test time
  it gives one answer, which is roughly the average of all futures that are plausible for
  that scene.
- **Diffusion (flow matching).** The probe learns a whole distribution of futures. In
  training, the true future is mixed with random noise and the network learns to undo the
  noise. At test time it starts from pure noise and removes it in 10 small steps. Each
  start gives a different sample, so we can draw 32 plausible futures for the same scene.

The diffusion probe is the more interesting one for analysis, because it can show if the
features support several options (turn left or go straight) instead of one average. The
question here is what it costs: does a diffusion probe drive as well as a regression probe,
how do the two behave while training, and which variants help.

## Setup

**Data.** NAVSIM, a benchmark built on real driving logs from four cities. The training
split has 92k scenes (navtrain). The test split navtest has 12k scenes, and navhard is a
harder split of 5.9k scenes where the start is a slightly displaced, synthetic version of
the real scene. A validation set of 2000 scenes comes from training logs that were held out.

**Input.** Features of the ORBIS world model (block 11 of its transformer) for the front
camera, 4 history frames at 2 Hz. Some runs also get the *ego status* (current velocity
and acceleration) and the driving command (left, straight, right), which every NAVSIM
planner is allowed to use.

**Output.** 8 future poses (x, y, heading) at 0.5 s intervals, so 4 s ahead.

**Metrics.**

| metric | what it measures | better |
|---|---|---|
| EPDMS | the official NAVSIM driving score, 0 to 1. A simulator checks the plan for collisions, leaving the road, progress, comfort and so on, and multiplies the sub-scores | higher |
| L2 at 4 s | distance between the plan and the human position 4 s ahead, in metres | lower |
| minADE | for diffusion: error of the best of the samples. Shows if *any* sample matches the human | lower |
| spread | average distance between two samples of the same scene, in metres. 0 means all samples are the same | (no better/worse, it describes the width) |
| energy score | a single number for how good the whole distribution is. It rewards samples close to the truth and penalizes samples that are all identical. Lowest only for the true distribution | lower |

For a regression probe the "distribution" is one point, so its spread is 0 and its energy
score equals its average error.

EPDMS needs one plan per scene. For the diffusion probe we use the mean of the 32 samples
unless written otherwise.

## Two comparisons

The comparison was done twice, and the two give different answers, which is the main point
of this note.

1. **The original diffusion probe against a regression planner.** The diffusion probe is
   deliberately tiny (hidden width 16, about 20k parameters) so that it measures what is
   in the features and not what a big network can compute. The regression planner is a
   normal-size transformer head (width 256, 4 layers, about 5M parameters) built for
   driving.
2. **Same network, same inputs, only the training objective changes.** The diffusion head is
   put on exactly the network of the regression planner, with the same inputs, the same
   data and the same optimizer. (The flow version has 7.5M parameters instead of 4.9M, because
the denoising head is a bit larger.) Then several regression and diffusion variants are compared.

## Comparison 1: the small diffusion probe against regression

### Training dynamics

![probe dynamics](26-11-12-diffusion-vs-regression/probe_dynamics.png)

*Training of the original diffusion probe for 100k steps on all 92k scenes, three input
variants. Dotted blue line: the final regression planner on the same validation scenes.
Panel (b) solid: error of the mean of the 16 samples; dashed: error of the best of the 16.*

The probe learns fast in the first 10-20k steps and then improves slowly until the end.
On all 92k scenes it does not collapse within 100k steps: the spread of the samples falls
from 1.6 m to 0.8 m, but the error falls with it, and the energy score is still at its lowest
at the end (1.80 for the best input variant). The samples become narrower because the model
gets more certain, not because they all turn into one trajectory. (A 4 times wider probe trained on only 10k scenes
does collapse to one trajectory per scene.)

Against regression the picture is split. The error of the mean of the samples stays well
above the regression planner (0.68 m against 0.54 m average displacement). The best of the
16 samples is below it (0.31 m), and the energy score drops below regression's after about
20k steps. So as a *distribution* the probe is better than one regression answer, but as a
*single plan* it is worse.

The inputs matter a lot. Image only and ego status only are both clearly worse than the
two together, so the image features and the current speed carry different information.

### End result

![end results](26-11-12-diffusion-vs-regression/end_results.png)

*Left: official EPDMS on the full navtest split. Right: comparison 2, see below.*

| model | navtest EPDMS | navhard EPDMS (two-stage) | NAVSIM v1 PDMS | training time |
|---|---|---|---|---|
| constant velocity (no learning) | 0.259 | 0.115 | 20.7 | |
| diffusion probe, image only | 0.788 | 0.177 | 78.7 | 1.9 h |
| diffusion probe, image + ego status + command | 0.810 | 0.197 | 80.2 | 1.9 h |
| diffusion probe, image + ego status, best of 4 samples (oracle) | 0.900 | | | |
| vocabulary planner (picks one of 256 template trajectories) | 0.857 | | | about 2 min |
| regression planner | **0.881** | **0.265** | **87.6** | about 2 min |

The regression planner wins clearly: +0.07 EPDMS on navtest, +0.07 on navhard and +7 points
on the older NAVSIM v1 score. It also trains in minutes instead of hours.

But this is not a fair test of diffusion. The two models differ in size and design, not
only in the loss. If we pick the best of 4 diffusion samples with hindsight, the score is
0.900, above regression. So the samples contain good plans, and the problem is partly that
the average of the samples is a worse plan than one sharp regression output, and partly that
the tiny probe cannot use the inputs as well. Comparison 2 separates these effects.

## Comparison 2: same network, only the objective changes

All runs here use the regression planner's network and inputs (world-model features, ego
status, past poses, driving command), all 92k training scenes, and are scored on a fixed
random subset of 500 navtest scenes. On this subset the score of a planner is within about
0.02 of its full-navtest score, so differences below 0.02 are noise. The intervals below
are 95% bootstrap intervals over the 500 scenes.

The variants:

| name | what it is |
|---|---|
| regression | one trajectory, L1 loss, 10k steps |
| flow (predict v) | plain flow matching, 30k steps. Each coordinate is standardized to unit variance before adding noise. The network predicts the direction from noise to data |
| flow (predict x0) | same, but the network predicts the clean trajectory directly |
| flow (predict noise) | same, but the network predicts the noise that was added (classic DDPM choice) |
| flow, no normalization | positions only divided by 10 m instead of standardized |
| residual flow | regression gives a base trajectory, the flow only learns the correction on top of it |
| residual + reference perturbation | the base trajectory is randomly disturbed in training so the flow learns to fix bigger errors |
| multi-modal targets | for 10k scenes a simulator scored many alternative trajectories. Part of the time the model learns a random *good* alternative instead of the human one, so it sees more than one correct answer |
| reward fine-tuning | after training, the model samples 16 plans per training scene, the simulator scores them, and the model is trained further on its own good samples (weighted by how much better than average they were) |

### Training dynamics

![same backbone dynamics](26-11-12-diffusion-vs-regression/same_backbone_dynamics.png)

*Validation metrics during training, evaluated every 5k steps on the same 2000 validation
scenes as above. Regression trains for 10k steps, the flows for 30k. For the flows the plan is
the mean of 16 samples.*

- **Plan quality (a, d) is flat.** Regression and the v, x0 and residual flows all sit at
  about 1.35-1.4 m error at 4 s and 0.86-0.88 EPDMS from the first evaluation on. Regression
  is at its final quality after 5k steps; the flows need about 10k. Longer training does not
  improve the plan of any of them.
- **The distribution gets worse after 10k steps (b, c).** For the v, x0 and residual flows
  the energy score is lowest at 5-10k steps (1.48 for v) and then rises steadily to 1.69-1.73
  at 30k, while the spread falls from 0.67 m to 0.30 m and keeps falling. The plan does not
  change, so the samples are just pulled together around it. This is the collapse: with
  enough capacity and steps, the flow becomes a regression model with a little noise on top.
  The residual flow starts narrowest and stays narrowest.
- **Predicting the noise behaves differently.** Its samples start extremely wide (5.7 m
  spread at 5k steps) and the mean of them is a poor plan (4 m error, EPDMS 0.45). Both
  improve slowly and are still far from the others at 30k. Its energy is the best at the end
  only because it is still wide, not because the samples are good.
- Compared with comparison 1, the plain flow on the big network reaches a better energy
  (1.48 after 10k steps, 24 min) than the small probe after 100k steps (1.80, 1.9 h).

### End result

| model | EPDMS of the plan | difference to regression [95% interval] | samples that are bad (EPDMS < 0.5) | spread [m] | energy | ms per scene |
|---|---|---|---|---|---|---|
| regression | 0.893 | | 4.8% | 0 | 2.01 | 0.3 |
| flow (predict v) | 0.897 | +0.004 [-0.014, +0.022] | 5.5% | 0.30 | 1.69 | 2.6 |
| flow (predict x0) | 0.896 | +0.003 [-0.016, +0.021] | 5.5% | 0.25 | 1.72 | 2.6 |
| flow (predict noise) | 0.801 | **-0.092** [-0.113, -0.072] | 10.5% | 1.25 | **1.53** | 2.6 |
| flow, no normalization | 0.875 | -0.017 [-0.037, +0.001] | 7.9% | 0.25 | 1.75 | 2.6 |
| flow, multi-modal targets | 0.894 | +0.001 [-0.016, +0.018] | 7.7% | 0.59 | 1.58 | 2.6 |
| flow + reward fine-tuning | **0.909** | **+0.017** [+0.001, +0.034] | **4.6%** | 0.25 | 1.72 | 2.6 |
| residual flow | 0.891 | -0.002 [-0.018, +0.014] | 5.8% | 0.22 | 1.73 | 2.6 |
| residual flow (predict x0) | 0.896 | +0.004 [-0.011, +0.018] | 5.5% | 0.21 | 1.75 | 2.6 |
| residual + reference perturbation | 0.878 | -0.014 [-0.033, +0.003] | 7.5% | 0.26 | 1.72 | 2.6 |
| residual, multi-modal targets | 0.891 | -0.001 [-0.017, +0.015] | 6.9% | 0.38 | 1.65 | 2.6 |
| residual + 2 rounds reward fine-tuning | 0.892 | -0.001 [-0.016, +0.015] | 6.2% | 0.25 | 1.74 | 2.6 |
| flow, best of 32 samples (oracle) | 0.941 | | | | | |

The right panel of the end-result figure above shows the same differences with their intervals.

On the same navtest subset the human driver's own trajectory scores 0.949 and a rule-based
planner with access to the map and all other cars (PDM-Closed) 0.871.

Some example scenes, seen from above (bird's-eye view), for regression (left) and five diffusion
variants (plain flow, residual flow, residual + reference perturbation, residual with
multi-modal targets, residual + reward fine-tuning). Black: what the human drove. Green to red lines: diffusion samples, coloured by
the predicted driving score (green good, red bad).

![bev examples](26-11-12-diffusion-vs-regression/bev_examples.png)

In most scenes the samples lie on top of each other and on top of the regression line. In
the second row (a right turn among other cars) the plain flow puts some samples off to the
side, and those get a low score.

## What this means

**With the same network and inputs, diffusion is as good as regression.** The plain flow
scores 0.897 against 0.893, a difference far inside the noise. Single samples are almost as
good as the mean (0.885). So the large gap in comparison 1 came from the tiny probe and its
missing inputs, not from diffusion itself.

**Diffusion costs more.** It needs 3 times the training steps (30k vs 10k), about 4 times
the training time, and 9 times the inference time (32 samples with 10 steps each).

**Diffusion narrows as it trains.** On the big network the distribution is best (lowest
energy) after 5 to 10k steps. After that the spread keeps falling and the energy rises
again, even though the plan quality stays the same. The model slowly turns into a regression
model that returns one trajectory per scene. The tiny probe does not do this on all 92k
scenes, only when it is made wider and trained on 10k scenes, so the collapse depends on how much capacity
the probe has per training scene.

**How the target is written matters more than anything else.** Predicting the noise is
clearly worse (-0.09): the first denoising steps are inaccurate, the samples scatter, and
their mean is a bad plan. This also explains why its energy score is the best: the samples
are wide, so they cover the truth more often, but many of them are bad. Not standardizing the
positions costs about 0.02, because the noise is then too large for the near poses and too
small for the far ones. Predicting v or x0 makes no difference.

**Building on regression (residual flow) does not help.** With ego status, the regression
answer is already close to the truth, the residual is small, and the flow on it collapses
even faster (lowest spread of all).

**Multi-modal targets widen the distribution but do not improve the plan.** Spread doubles
and the best-of-32 oracle rises to 0.96, but there are more bad samples and the mean plan
is unchanged.

**The only clear gain is reward fine-tuning**, +0.017 with an interval above zero, and the
fewest bad samples. It is the only variant that gets information about *driving quality*
and not just about what the human did. Regression cannot use this kind of signal directly,
because it only has one output and nothing to compare.

**The samples contain much better plans than any rule picks.** The best of 32 samples
reaches 0.94, near the human driver (0.949). A learned scorer that picks one sample gains at
most 0.016 over the mean. Choosing among the samples is the weak step, not generating them.

**Summary.** For the driving score alone, regression is the better choice: same quality,
much cheaper, and nothing to tune. Diffusion is worth it if one needs the distribution
(for analysis of the features, for a scorer that picks among options, or for reward
fine-tuning), and then it should predict x0 or v on standardized coordinates and be stopped
early or kept wide, because longer training collapses it.
