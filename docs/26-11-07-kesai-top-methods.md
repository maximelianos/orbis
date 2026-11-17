# What the top py123d_garage navhard methods do, and what we can borrow (2026-11-07)

Sources: kesai-labs/py123d_garage README (leaderboard) and docs/research_logs.md; arXiv papers read through alphaXiv:
DrivoR 2601.05083, DriveZero 2609.06055, TerraTransfer++ 2610.30319, SimScale 2511.23369, GTRS 2506.06664,
ZTRS 2510.24108, WoTE 2504.01941, World4Drive 2507.00603, SeerDrive 2510.11092, RFWorld 2611.05072, Orbis 2 2607.15898
(+ HF configs of sud0301/orbis2, + local orbis2_code/). GigaPixel was NOT read (only its table row and the DriveZero/Terra
description: vector-state RL teacher, pixel student by self-play DAgger).
Our reference: 46.0 with a logged goal, 43.7 without (ens5, mg4 null rescoring, minmult). Scores below are the README
table; the README says the older baselines predate a devkit fix that lifts other entries by 2 to 3 points, so rows are not perfectly comparable.

## 1. Leaderboard (navhard two-stage EPDMS), top entries

| rank | model | nav input | metric/sim training | EPDMS | S1 NC/DAC/EP/EC | S2 NC/DAC/EP/EC |
|---|---|---|---|---|---|---|
| 1 | TerraTransfer++ | Goal | yes (self-play teacher, no human traj) | 59.4 | 98.9/100/84.9/70.2 | 94.0/90.2/85.2/53.8 |
| 2 | DriveZero-Scale | Command | yes (RL teacher + 237k SimScale scenes) | 57.1 | 98.7/99.3/76.5/44.4 | 94.6/95.6/76.2/44.0 |
| 3 | DrivoR (+134k SimScale) | Command | yes | 54.5 (paper 54.6) | 99.1/98.2/75.4/70.2 | 92.3/91.6/75.7/44.7 |
| 4 | SimScale (GTRS-Dense V2-99) | Command | yes | 53.2 | 99.5/99.1/69.6/28.4 | 94.5/94.2/75.8/43.2 |
| 5 | GigaPixel | Command | yes | 50.1 | 99.4/95.8/68.1/49.8 | 93.7/92.9/62.6/58.8 |
| 6 | ZTRS | Command | yes (RL, no human demo) | 48.1 | 98.8/97.5/66.6/44.0 | 91.1/90.4/63.5/66.0 |
| 7 | SimScale GTRS-Dense (R34) | Command | yes | 48.0 | 99.3/97.6/71.5/30.7 | 94.9/94.3/79.0/30.9 |
| 8 | GTRS-Dense | Command | yes | 45.3 | 98.9/98.2/73.9/40.0 | 91.5/90.8/70.8/54.2 |
| 9 | RAP | Command | yes | 39.6 | 97.1/94.4/83.8/66.2 | 83.2/83.8/86.8/52.4 |
| ref | TransFuser-K1 (py123d_garage own) | none | no | 34.9 +- 0.4 | 97.2/90.7/83.1/72.7 | 82.6/77.7/86.9/58.7 |
| ref | ours (ens5 + mg4, goal / no goal) | goal / none | yes (10k PDM sub-scores) | 46.0 / 43.7 | | |

We sit between GTRS-Dense and SimScale-GTRS with the goal, and just below GTRS-Dense without. Every entry above 48 either
uses sim data (SimScale 3DGS renders) or a teacher not trained on human logs.

py123d_garage research log (TransFuser recipe; AlpaSim/NuRec tracks, not navhard): front camera only adopted; one past frame 1 s ago
adopted (+7.3 NuRec); 8 register tokens in fusion adopted; flow matching instead of L1 regression adopted (+5.4); frozen encoder in posttraining
no score cost and 1.6x faster; target points with 75 to 95% dropout (+9.5 nuPlan); NAVSIM navtrain only is significantly worse than the larger mix;
learned BEV tokens in place of the BEV ResNet and BEVFormer listed as untried. Not directly comparable to navhard.

## 2. Per-method extraction

| | DrivoR | DriveZero | TerraTransfer++ | SimScale | ZTRS / GTRS |
|---|---|---|---|---|---|
| cameras | 4 (F, FL, FR, B), 672x1148, no lidar | 4 (F0,B0,L0,R0), 960x512, no lidar | camera only, DINOv3 backbone | 3 cams stitched 2048x512, lidar removed | F + cropped FL/FR stitched, 512x2048 (GTRS-Aug 256x1024), image only |
| ego/nav | pose, vel, acc, command into ego token, added to traj queries | ego kinematics + command | goal = end point of the logged trajectory (same as ours) | command | command + ego |
| backbone | DINOv2 ViT-S, LoRA r32 (frozen+registers only drops 90.0 to 84.4 PDMS) | DriveVFM ViT-L (distilled from DINOv3+SigLIP2+SAM+DepthAnythingV2), frozen + LoRA r32 Q/V | frozen DINOv3 + linear adapters (aligned on 1.83M nuPlan frames + 300k sim frames) | R34 / V2-99 | V2-99, EVA-ViT-L, ViT-L (DepthAnything init), fully finetuned |
| BEV / lifting | none. 16 learned register tokens per camera inside the ViT, concatenated to N*16 scene tokens (64 tokens) | same registers as DrivoR plus 3D position embeddings on the visual tokens (PETR-style) | adapters produce "road" and "partner" scene features for a frozen teacher head (21 tokens, 3 self-attn layers) | LTF/DD: TransFuser BEV; GTRS: no BEV | GTRS-DP generator: BEV queries cross-attend image features (TransFuser seg head); scorer: trajectory tokens cross-attend image tokens, no BEV |
| multi-camera | per-camera registers, camera identity from register index; back camera registers collapse to ~1 distinct token, front registers specialise (lead car, light, road edge) | same | shared backbone, per-type adapters | stitched | stitched, separate DP model for rear views |
| candidates | 64 learned queries, WTA L1 (no vocabulary) | 64 queries, WTA vs teacher rollout | flow-matching teacher (16 samples, best by CV forecast); student samples 1 plan, no selection | per planner | 16384 vocabulary (train), 8192 (navhard inference), + 100 DP proposals in GTRS |
| scoring | decoder on the decoded trajectory re-embedded, stop-grad to generator, 6 PDM sub-scores with own BCE head, max aggregate | same as DrivoR (6 comps) | none at test time | GTRS: Hydra heads | 6+ sub-score heads (GTRS), ZTRS adds a policy head over vocabulary with EPDMS-advantage (EPO) |
| training data | navtrain + 65k/134k SimScale | navtrain + RL-teacher rollouts (human traj not used) + 237k SimScale, goal augmentation | navtrain alignment only | navtrain + up to 147k recovery / 237k planner-expert 3DGS scenes | navtrain only |

Key ablation numbers:
- DrivoR (navval PDMS, ViT-S): registers LoRA 90.0 vs frozen registers 84.4; no compression 16k tokens LoRA 90.2 (frozen 88.2); pooling 89.7; decoder-query compression frozen 86.9 / LoRA 89.3 (table columns partly garbled in extraction, read with care);
  registers per camera 5/8/16/32 = 89.7/89.7/90.0/89.8; trajectory queries 1/8/16/32/64/128 = 80.1/87.6/88.1/89.5/90.0/90.0; single decoder 84.7, separate decoders 86.8, + disentangle (stop-grad + re-embed) 90.0, single combined score instead of 6 sub-scores 88.2;
  single-token trajectory 90.0 vs multi-token 83.9; ImageNet-21k init 87.5, DINOv2 90.0, random 70.1. Navhard: 48.3 (navtrain) -> 52.3 (+65k sim) -> 54.6 (+134k sim); 110 ms/forward, 41M params.
- SimScale (navhard EPDMS): GTRS-Dense R34 38.3 -> 46.9 (reward-only on sim data), V2-99 41.9 -> 48.0, multi-expert ensemble 50.9; LTF 24.4 -> 30.2; DiffusionDrive 27.5 -> 32.6.
  Planner-based (PDM-Closed) pseudo-expert beats recovery-to-log expert; reward-only sim supervision beats expert trajectories for the scorer; reactive agents: 43.7 (non-reactive 2 rounds) vs 45.0 (reactive 3 rounds) for R34; gain scales ~log(N) for scoring models, saturates for single-mode regression.
- GTRS (navhard): vocabulary dropout + train on 16384, infer on 8192 + DP proposals 43.4 vs 42.0 without dropout; Aug (view rotation + EMA self-distillation refinement) +2.8 over Hydra scorer; ensemble of 6: 49.4.
  DP-only with random pick 25.6 -> scored 36.7 (scorer is the value).
- ZTRS: EPO (dense advantage over whole vocabulary, normalised) vs IL on V2-99: 45.5 vs 41.7 navhard; reward E alone 84.2 navtest with EC 53.8, E minus comfort-consistency term b: 85.3 and EC 77.2; inference on 8192 of 16384 gives stage-2 +3.6 (and stage-1 -0.7); GRPO and REINFORCE unstable.
- DriveZero (navtest PDMS, ViT-S): DriveVFM 94.41 vs DINOv3 93.88; teacher trajectories alone 93.61 < human 93.92 < teacher + goal augmentation 94.41. Navhard 51.5 -> 57.1 with SimScale (S2 60.8 -> 69.1).
- TerraTransfer++: 59.4, student amortises best-of-16 teacher selection; backbone ablation (closed loop HD score) ResNet-50 0.499 / DINOv2 0.512 / DINOv3 0.548 with linear adapters on frozen features.

Take-aways for us: (i) the biggest lever in the table is data (sim-rendered OOD states), not architecture (DrivoR +6.3, SimScale +8.6, DriveZero +5.6); (ii) all high entries output sub-scores or rewards and pick by weighted aggregation, like us;
(iii) Terra's student is also a frozen-feature plus light adapters design and uses the goal as input, so the frozen-feature route is not the limit.

## 3. Focus questions

### (a) BEV features from frozen per-camera ViT grids without depth
Evidence from the papers:
- No top method needs BEV. DrivoR and DriveZero use learned tokens, and GTRS-Dense scores with trajectory-token cross-attention to image tokens. GTRS' own generator (the weakest part of GTRS-DP) uses TransFuser-style BEV queries.
- With a frozen backbone DrivoR's registers lose a lot (84.4 vs 90.0 LoRA), and feeding all tokens frozen (88.2) beats compression. So for frozen ORBIS tokens: keep all 576 tokens (18x32) as K/V and let anchors or BEV queries cross-attend them directly; do not squeeze to 16 registers. Our 1024-d tokens are cheap to attend (256 anchors x 576 tokens).
- Depth-free geometry: World4Drive and DriveZero add 3D position embeddings to image tokens (PETR style). World4Drive gets the 3D point from metric depth (Metric3Dv2) by forward projection and MLP(sin PE). Their ablation (nuScenes, L2/collision): LAW 0.61/0.30, + intention 0.55/0.25, + spatial depth PE 0.51/0.29, + semantic 0.49/0.26, full 0.50/0.16. So position PE helps L2, semantics helps collisions.
- Cheapest depth-free option for one fixed camera (my suggestion, not tested in the papers): per-token camera-ray direction and its flat-ground intersection (x, y) clipped to a range, passed through sin-MLP and added to K only. Intrinsics are constant for the nuPlan front camera so this is a fixed 576x(2 to 5) table. Next: 8x8 to 16x16 learned BEV queries (TransFuser/WoTE/SeerDrive use 8x8, C=256) cross-attending all tokens with that PE, followed by the semantic head of (c).
- Lift-splat with a predicted depth distribution needs depth supervision (pseudo-depth from Depth Anything / Metric3D, as World4Drive) and none of the top entries uses it; for a single front camera only the near field is covered, so I expect little over cross-attention with ray PE.
- BEVFormer-style deformable attention: RFWorld only uses deformable sampling for readout from BEV (below), not for lifting. py123d_garage lists BEVFormer as untried.

### (b) World-model futures for planning
| method | what is imagined | how scored | gain | cost |
|---|---|---|---|---|
| WoTE | future 8x8x256 BEV state per candidate, 2 recurrent steps (0->2s->4s), 2-layer transformer, action embedding appended as a token | reward model: conv over [B_t, B_t+1, B_t+2] + MLP -> imitation + NC/DAC/TTC/Comf/EP rewards (BCE vs nuPlan simulator labels); BEV semantic decoder (focal loss) supervises states | PDMS 81.0 baseline -> 83.2 with eval -> 85.6 with future states (net +2.4); 0s->4s in one step 84.0 vs two steps 85.6 | 18.7 ms for 256 candidates |
| RFWorld (2611.05072) | no per-candidate rollout. One shared memory; training-time temporal readout (SE-gate per future step) decoded to BEV semantics at up to 8 future times; inference: waypoint-conditioned deformable sampling | reward heads as WoTE | 88.9 baseline -> 89.4 (future sup. only) -> 90.5 (+ shared memory); no future sup. at all 88.4 (world model alone does not help); #future steps 0/2/4/6/8 = 88.4/89.5/90.0/90.3/90.5; beats WoTE (88.3) with 8.7 GB vs 33.9 GB training memory | 37.5 ms for 1024 candidates |
| SeerDrive | one future BEV (last step only) per mode, predicted from current BEV + ego feature, injected into planner via a decoupled decoder + motion-aware LayerNorm; 2 iterations | no scoring, feature injection | 87.1 -> 88.9; future-BEV mIoU only 38.9 | 24 ms |
| World4Drive | future latent per intention (K=18 intents) by cross-attn predictor, n=3 frames ahead | ScoreNet over predicted latents, self-supervised: the intention whose predicted latent is closest (MSE) to the actual future latent is the target (focal loss) | nuScenes collision 0.30 -> 0.16; NAVSIM PDMS 85.1 | cheap |
| LAW (cited only) | single-modal future latent as auxiliary self-supervision | training only | 84.6 | |
Consensus: supervising future semantics in BEV (not pixels) and reading them rollout-free is the cheap, effective form; gains come from the auxiliary future loss plus waypoint readout, not from iterating a heavy video model.
All of these still use NAVSIM v1 PDMS and none reports navhard.

Orbis 2 as an in-loop imaginer (our own estimate, see section 4): conditioned on trajectory, one 4 s rollout costs on the order of 0.3 PFLOP (distilled L2) to 0.8 PFLOP, i.e. 0.3 to 2 s on a H100 per candidate. Fine for re-checking the top 4 to 8 candidates at test time, impossible for the 256 candidates or for training the scorer on 100k scenes.
A scorer would still need a decoder from imagined frames or latents to PDM sub-scores, which would be a new training problem with our own E4 warning (a single learned verifier is exploited by gradient search; 5-member ensemble too).

### (c) BEV semantic auxiliary losses
- TransFuser/LTF/WoTE/SeerDrive BEV decoder: 7 classes: background, road, walkways, centerline, static objects, vehicles, pedestrians (NAVSIM BEV map; WoTE lists exactly these). Focal loss (WoTE, RFWorld). SeerDrive weights: current map lambda 10, future map 0.1 on NAVSIM, trajectory 1.
- RFWorld: shared decoder over timesteps; current + up to 8 future maps; ablation above; semantic supervision alone (no WM) 88.9 -> 89.4.
- GTRS-DP: BEV seg head only on the generator, scorer has no aux loss. DrivoR and DriveZero have no auxiliary perception loss at all (sub-score BCE only). SimScale Table 6: LTF and DiffusionDrive use detection + segmentation, GTRS-Dense none.
- World4Drive (no BEV): image-space Grounded-SAM pseudo labels, CE, loss weight 0.2; adds collision improvements.
- Our repo already has `data/bev_targets/` (untracked), so labels may exist; check class set and horizon before reuse.

## 4. Orbis 2 (arXiv 2607.15898, orbis2_code/, HF configs)
| item | value |
|---|---|
| structure | two levels. L2 "abstract" predictor: 2 Hz, window 6 frames (3 s), 5 context frames -> predicts 1 frame, in a 16-d DINOv2-B-aligned latent (VQ-regularised hybrid ViT tokenizer, 86M enc / 91M dec). L1 "detail" predictor: 10 Hz, window 10 frames (1 s), 5 context -> 5 new frames (0.5 s), conditioned on the L2 prediction (gated spatial cross-attention + adaLN) |
| training | L2: diffusion forcing pretraining then teacher forcing; then steering fine-tune on 500 h of PhysicalAI-AV (IMU speed + yaw rate); 5,890 h video; one epoch; ~6k H100 h, 1.07B params in total (ST-DiT 512M + 43M conditioning for L1) |
| resolution | 512x288 pixels, latent grid 18x32 (patch 16), 16 latent channels. Same 18x32 grid as our tokens |
| DiT (HF config, both L1 and L2) | hidden 768, depth 24 blocks, 16 heads, MLP 4 (SwiGLU), patch 1; ST block (spatial + temporal attention); L2 causal in time, L1 non-causal. (The 1152/28 defaults in `networks/DiT/dit_v2.py` are not used by the released configs.) |
| steering input | speed and yaw-rate sequence (or a 2D trajectory converted to them, `get_trajectory_from_speeds_and_yaw_rates_batch`). `LinearSteeringEmbedder`: each scalar -> its own 1-768-768 MLP, features summed, NaN -> learned/zero "no steering" vector, then mean over K steps (positional embedding, 5 steps per image frame in L2) and ADDED TO THE TIMESTEP EMBEDDING, i.e. adaLN. Only the top (L2) branch is conditioned; L1 inherits through the L2 sub-goal |
| sampler | L2 flow matching, 20 NFE (distilled L2: 4 NFE); L1 Heun x2 + Euler x1 (about 5 NFE). Paper text says 15 NFE (L2) and 30 NFE (L1) for the reported setting |
| steering quality | ADE (m) original/speed x0.5/x1.5/yaw x0.5/x1.5: Orbis 2.23/3.49/1.42/3.02/1.31, Orbis 2 1.20/2.18/0.53/1.65/0.73 |
| cost per rollout step (0.5 s) | no latency table in the paper. My estimate: L2 forward on 6x576 tokens ~3.5 TFLOP x 20 NFE ~70 TFLOP (14 with distilled), L1 forward on 10x576 tokens ~6 TFLOP x ~5 NFE ~30 TFLOP. About 100 TFLOP per 0.5 s step (about 45 with distilled L2); a 4 s plan = 8 steps = 0.35 to 0.8 PFLOP, roughly 1 to 2 s per candidate at 40% MFU on one H100 (not measured) |
| relevance | "L2 abstract" latent is the right object for scoring: 16-d, 2 Hz, steerable. We would need its per-frame latents decoded to something PDM-like |

## 5. Ranked adoptable ideas for our frozen-feature planner

| # | idea | evidence | expected gain (navhard EPDMS) | cost | sketch |
|---|---|---|---|---|---|
| 1 | Co-train on released SimScale sim data (reward-only for the scorer) | DrivoR 48.3 -> 52.3 (+65k) -> 54.6 (+134k); SimScale GTRS-Dense +8.6; DriveZero +5.6; stage-2 DAC and NC rise most. Reward-only sim supervision beats expert trajectories for scoring models | +3 to +6 (my guess from the three papers, lower since our frozen features were never trained on rendered artefacts) | medium: ORBIS feature extraction for 65k to 237k front images (frozen, cheap), PDM sub-scores of our 256 anchors per scene on CPU (we already have a fast scorer pipeline), disk. Need to confirm the release (OpenDriveLab/SimScale) ships front images + states + pseudo-expert labels; not verified | extract ORBIS tokens for sim frames, compute 256-anchor sub-scores with the existing FastEPDMS path, mix sim with real 1:1 per batch (SimScale co-training), sim samples get scorer loss only (no imitation term), goal for sim frames = pseudo-expert end point |
| 2 | Train on a bigger vocabulary with dropout, infer on a subset, plus dense EPO-style loss with a comfort-consistency term | GTRS: dropout on V_XL 42.0 -> 43.4; ZTRS: inference on half the vocabulary +2.1 overall, stage-2 +3.6; EPO vs IL +3.8; E minus b: EC 53.8 -> 77.2, +1.1 EPDMS. Our own E5: candidate expansion gives most of the gain of mg4 | +1 to +3; also targets our weak EC (S1 54.7) | low to medium: sub-scores for 2k to 4k anchors on the 10k-scene set (CPU), scorer already takes anchor queries | cluster 4096 anchors, train scorer on random half per step (queries are order-free), infer on 256 refined + pooled goal-conditioned proposals; add loss sum_a softmax-normalised(EPDMS(a) - lambda*EC_violation(a, previous argmax)) * log pi(a) from the 5 scorers' policy head, or just use the penalty on the argmax-previous-frame anchor |
| 3 | Future-aware auxiliary: BEV semantic map at current + 1..4 s with shared decoder, scorer reads the BEV memory at the waypoints (RFWorld/WoTE) | RFWorld 88.9 -> 90.5 PDMS (8 future steps; no future sup. no gain); WoTE +2.4; SeerDrive +1.8 | +0.5 to +1.5, likeliest on stage-2 DAC/NC (these are our largest losses: DAC2 20.8% fail) | medium: data/bev_targets exists; add 16x16 BEV queries to the trunk, 7-class focal head, deformable (M=4) bilinear sampling of BEV memory at each anchor waypoint feeding the sub-score heads | BEV queries = ray-PE (idea 5) cross-attending 576 tokens; temporal SE-gate per target time; keep the anchor cross-attention as is, concat the sampled waypoint features; do not roll out per candidate |
| 4 | DrivoR-style disentangled scorer: re-embed the generated trajectory, stop-gradient to the generator, separate decoders, sub-score weights tunable at test time | DrivoR: separate decoders 84.7 -> 86.8, + disentangle 90.0, 6 sub-scores vs 1 score 88.2 -> 90.0 (PDMS navval) | +0.5 to +1.5, or 0 if our scorer is already separated from the flow head (check) | low: code only, no new data | verify whether scorer loss gradient reaches the flow head/trunk used to build the offsets; if yes, detach the refined trajectory before the scorer's trajectory MLP; per-run sweep of aggregation weights (EP, TTC, LK weights) like DrivoR's behaviour conditioning |
| 5 | Depth-free geometry in cross-attention: ray / flat-ground 3D positional encoding on the 576 frozen tokens (PETR/World4Drive) | World4Drive spatial PE: L2 0.55 -> 0.51; DriveZero adds 3D PE to ViT tokens; DrivoR: keep all frozen tokens rather than register compression (frozen 88.2 vs 84.4) | +0 to +1 (lateral precision, our main no-goal error: failures end within 1 m of the log) | very low: fixed table from intrinsics (precompute once), add to K only | per token (u,v) -> ray in ego frame -> intersect z = 0 plane (clip 60 m) -> (x, y, ray dir) -> sin-MLP -> add to K; ablate on 24aa trunk |

Honourable mentions, not ranked: (a) DriveZero goal augmentation: query the goal-conditioned teacher/scorer with alternative goal points to multiply training targets (+0.8 PDMS in their ablation); our E3/E5 results already show goal hypotheses are valuable as proposals, so this is a natural training-time extension. (b) Orbis 2 per-candidate rollouts: only plausible as a test-time check of 4 to 8 candidates with a new decoder, 1 to 2 s per candidate, high risk of verifier exploitation. (c) Multi-expert ensemble (SimScale 47.7 -> 50.9 with recovery + planner + reward models) matches our ens5 + minmult idea; diversity in the training target seems to matter more than seeds alone.

Caveats: numbers are from papers on different splits (navval/navtest PDMS vs navhard EPDMS), some rows predate the devkit fix, and expected gains are my judgement. I did not verify the SimScale data release contents or the bev_targets class set.
