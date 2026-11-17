# Trajectory diversity: plan (2026-11-02)

## What we know

- The flow-matching probe collapses when it can memorize: width 64 on 10k scenes goes from
  dispersion 0.91 to 0.16 and minADE@32 from 0.65 to 2.06 m, with 50 Euler steps
  (outputs/02_collapse, 06_capacity). The mean error stays the best of all runs, so the
  model keeps the conditional mean and loses the spread around it.
- Memorization onset grows linearly with the number of scenes (04_data_scaling). The same
  law is reported for image and language diffusion models ("memorization time is
  proportional to dataset size", arXiv 2505.16959), so it is not specific to us.
- Ego speed is linearly readable from WM block 11 with R^2 0.95 (13_linear_probes). Most of
  a 4 s trajectory is "how far", and that is set by the current speed. A diffusion model
  spends most of its capacity on this easy, nearly deterministic part.

## Idea

Split the trajectory into the parts that are easy and the parts that are multimodal, and
give each part the model that fits it.

    future = CV(v_hat)            constant-velocity rollout of the regressed speed   (deterministic)
           + a_k                  anchor k: a k-means centre of residual manoeuvres   (discrete, classifier)
           + e                    residual inside the cell of anchor k               (flow matching)

1. **Kinematic head.** A small head regresses ego velocity and acceleration from the ORBIS
   block-11 tokens (supervised with the logged ego status, which every NAVSIM scene has).
   The prediction, detached, is a condition token of the flow model and defines the
   constant-velocity prior. Inputs stay image-only: the logged status is a target, never an
   input. This is the condition-dependent source idea of CSFM / CPD (arXiv 2602.05951):
   the flow starts near the right place, so the velocity field is easier to learn and the
   intrinsic ambiguity of the supervision drops.
2. **Intent anchors.** k-means on the training residuals future - CV(v) gives M manoeuvre
   prototypes (brake, accelerate, keep, turn left / right at a few curvatures, lane
   changes). They are learned from the data, not written by hand. The partition of the
   output space into anchors is what DiffusionDrive (arXiv 2411.15139) and GoalFlow
   (arXiv 2503.05689) use against mode collapse in driving: each anchor is one intention,
   and denoising starts from that intention.
3. **Intent classifier.** p(k | scene) with a soft cross-entropy (softmax of the negative
   distance to each anchor), as in the vocabulary planner of 16_fast_planner, which
   reached 0.95 EPDMS best-of-8.
4. **Anchor-conditioned flow.** Flow matching on the residual e = future - CV(v_hat) - a_k,
   conditioned on the image tokens, the kinematics token and an embedding of k, trained
   with the hard nearest anchor k*. The residual of a training sample lies in the Voronoi
   cell of its anchor, so a sample drawn for anchor k lands near anchor k, whatever the
   flow memorizes.

## Why this guarantees diversity

Collapse inside the flow can only shrink the spread within one anchor cell. The spread
between anchors is set at sampling time:

- **proportional**: k ~ p(k | scene) per sample. Calibrated mixture; diversity is as large
  as the classifier's entropy. This is the setting for the energy score.
- **stratified**: the K samples are spread over the top-M' anchors. Diverse by
  construction; the setting for best-of-K and for a downstream selector.

The remaining failure mode is an overconfident classifier. That is visible directly
(validation NLL, entropy, accuracy) and cheap to fix by early stopping the classifier on
validation NLL, which does not affect the flow.

Alternatives considered and not chosen:
- Winner-takes-all multi-head regression (MCL, annealed WTA, arXiv 2409.11172): diverse,
  but no density inside a mode and it suffers its own hypothesis collapse.
- Diversity losses (DPP, repulsion): diverse sets but no longer a model of the
  distribution; the energy score would penalize them.
- RL fine-tuning (DiffusionDriveV2): needs a simulator in the loop, out of scope.

## Experiments (outputs/17_diversity)

Stress test: the configuration that collapses, width 64 on 10k scenes, 100k steps, wm11_ctx
features, image-only. Every variant runs in the same new trainer with all data on the GPU,
evaluated every 5k steps on 2000 held-out-log scenes with 32 samples and 20 Euler steps.

| run | kinematic head + CV prior | anchors |
|---|---|---|
| plain | no | no (the old probe) |
| kin | yes | no |
| anchor | no | M = 16 |
| full | yes | M = 16 |

Then full data (92k), anchor count M = 8 / 32, and BEV figures of the samples on the map
for the same validation scenes at several checkpoints.
