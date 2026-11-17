# Test-time inputs of the navhard leaderboard entries and of our planner (2026-11-09)

Table: NAVSIM v2 navhard two-stage EPDMS in kesai-labs/py123d_garage. The README itself only has two input columns, "Navigation Input" (command, goal or none) and "Metric training". It says nothing about sensors. The sensor columns below come from the papers (read through alphaXiv), the official NAVSIM agent code in `navsim/`, the DrivoR repo configs and the py123d_garage `latent_transfuser.py` preset and `docs/policy/transfuser.md`.
"not stated" means I did not find it. "inferred" means it comes from code, config or the method lineage, not from the paper text.

## Short answer

- No entry on the navhard table uses LiDAR at test time, as far as I can verify. The only open case is the DiffusionDrive 27.5 row (see notes).
- No entry uses radar, an HD map or other privileged state at test time. Privileged state (vector scene, HD map, GT agents) appears only in training, as an RL or planning teacher (TerraTransfer++, DriveZero-Scale, GigaPixel).
- Every learned entry uses ego status. Most use velocity, acceleration and the driving command. A few add history (RAP 2 s of ego states, SimWAM yaw rate).
- Navigation: all entries use the NAVSIM driving command, except TerraTransfer++ (goal point = end of the logged trajectory), TransFuser-K1 (nothing) and CV (nothing).
- Ours uses no LiDAR either. It uses one front camera, ego status, past poses, the command and a goal point taken from the log, like TerraTransfer++.

## Leaderboard entries

| Method (EPDMS) | Cameras at test | LiDAR | Ego status | Navigation | Training-only extras | Source |
|---|---|---|---|---|---|---|
| TerraTransfer++ (59.4) | camera only, frozen DINOv3 + linear adapters; camera count not stated | no | kinematics + goal + reward-weight vector of the teacher (fields in the appendix, not read) | goal = end of logged trajectory | self-play RL teacher on nuPlan HD maps (vector state); flow teacher picks best of 16 with GT agent forecasts | 2610.30319 |
| DriveZero-Scale (57.1) | 4 (F0, B0, L0, R0), 960x512, ViT-L + LoRA | no | ego kinematics (fields not stated) | command | PPO teacher on privileged state + goal points; goal augmentation; 237k SimScale scenes; PDM sub-score head | 2609.06055 |
| DrivoR (54.5) | 4 (F0, L0, R0, B0), 1148x672, DINOv2 ViT-S + LoRA | no (`lidar_pc: []`) | pose, velocity, acceleration, current frame only (inferred from config) | command | 134k SimScale scenes; PDM sub-score BCE head | 2601.05083, code |
| SimScale (53.2) | GTRS-Dense V2-99, 3 cams stitched 2048x512 | removed | not stated (NAVSIM default, inferred) | command | 3DGS sim scenes, reward scoring; exact variant not confirmed | 2511.23369 |
| GigaPixel (50.1) | 4 (F, FL, FR, B), 434x252, DINOv2 ViT-S + LoRA | no | speed, accelerations, vehicle size, collision flag, reward vector | command | vector RL teacher (privileged), self-play DAgger, sim-to-real adaptation | 2606.19641 |
| ZTRS (48.1) | F0 + cropped L0/R0, 512x2048 | no | "ego status" (fields not stated) | command | reward-only offline RL over 16384 vocabulary | 2510.24108 |
| SimScale GTRS-Dense (48.0) | 3 cams stitched 2048x512 | removed | not stated (inferred default) | command | 3DGS sim data, reward-only on sim | 2511.23369 |
| GTRS-Dense (45.3) | F0 + cropped L0/R0, 512x2048, ViT-L | no | not stated (Hydra-MDP lineage: vel, acc, cmd, inferred) | command | PDM sub-score heads, vocabulary dropout | 2506.06664 |
| RAP (39.6) | 4 cams, 448x768, frozen DINOv3-H+ | no | 2 s of ego history (fields not stated) | command (from the py123d table) | 3D rasterized boxes and lanes, raster-to-real alignment | 2510.04333 |
| SimWAM (37.6) | 1 front, 384x672 | no | velocity, acceleration, yaw rate | command (text) | video world model co-training | 2608.07468 |
| SUV (36.9) | 1 front, 4 frames over 2 s, 640x384 | no | current ego state | command (prompt) | future RGB, segmentation, depth, track targets | 2608.03084 |
| 4D-WAM (35.9) | 1 front, 4 frames, 352x640 | no | current ego state | command (text) | video DiT, VGGT 4D teacher | 2608.10107 |
| DiffusionDrive-ViRA (35.8) | 3 cams stitched 2048x512 | no ("camera only") | not stated (DD default, inferred) | command | detection + BEV seg, VFM alignment | 2610.09695 |
| TransFuser-K1 (34.9) | front camera, current frame + frame 1 s back, stacked | no (latent) | ego speed | none | nuPlan + PhysicalAI-AV + NuRec mix, BEV seg + boxes | py123d_garage preset and docs |
| ReDrive (34.4) | camera, 4 frames, 256x512 (front only, inferred) | no | ego status (fields not stated) | not stated (table: command) | V-JEPA 2 pretraining on driving video | 2609.33854 |
| LTFv6 (31.9) | camera, setup not stated | no (latent) | not stated | command (table) | 100k CARLA frames | 2512.20563 |
| SimScale DiffusionDrive (32.6) | 3 cams 2048x512 | removed | vel, acc, cmd (inferred) | command | det + seg, sim data | 2511.23369 |
| SimScale LTF (30.2) | 3 cams 2048x512 | removed | vel, acc, cmd (inferred) | command | det + seg, sim data | 2511.23369 |
| DiffusionDrive (27.5) | original: 3 cams 1024x256 + LiDAR BEV | unclear, see notes | velocity, acceleration, command | command | det + BEV seg | DD paper, SimScale Tab. 6 |
| LTF (25.1) | L0/F0/R0 cropped and stitched, 1024x256 | no (learned latent replaces LiDAR BEV) | velocity, acceleration, command | command | BEV seg + boxes | navsim code |
| MLP (14.1) | none | no | velocity, acceleration, command | command | none | navsim code |
| CV (11.4) | none | no | velocity only | none | none | navsim code |

## Ours

| | best with goal | best without goal |
|---|---|---|
| navhard EPDMS | 51.7 (10-model ensemble), 49.2 single | 49.4 (5 models + 4 goal hypotheses), 45.5 single |
| Cameras | front camera CAM_F0 only, 4 history frames (2 s at 2 Hz), resized to 512x288, frozen ORBIS world model, block 11 tokens, PCA 64, 9x16 grid | same |
| LiDAR | no | no |
| Ego status | current velocity and acceleration (4 values) + 4 past poses (x, y, heading) | same |
| Navigation | driving command (one-hot) + goal point. Stage 1: logged pose at the 4 s horizon end. Stage 2: logged pose 4 s after the log frame nearest to the synthetic ego ("near goal") | driving command only; goal token set to the learned "no goal" embedding |
| Map / privileged at test | none besides the goal, which is taken from the logged future | none |
| Training-only extras | PDM sub-scores of 256 anchors on 10k navtrain scenes, SimScale pdm0 rewards (65k sim scenes, scorer only), dense 2048-anchor SimScale scores, goal augmentation | same, with goal dropout |

Side cameras (L0, R0, B0) and a BEV query encoder were built and tested (28c, 28d, 28t) but gave no gain, so the best models read only the front camera.

## How comparable is ours

- Our goal input is the same kind as TerraTransfer++'s: a point from the logged future. It is more information than the driving command that all other entries use, so our with-goal numbers should be compared with TerraTransfer++, and our no-goal numbers (command only) with the rest.
- We use fewer cameras than the top entries (1 front camera against 3 to 4) and no LiDAR, like all of them.
- Our 4-frame history (2 s) is longer than most top entries, which use the current frame only (DrivoR, DriveZero) or the current frame plus one past frame (TransFuser-K1).

## Notes and open points

1. DiffusionDrive 27.5: the original paper uses LiDAR. SimScale reports the same 27.5 for its DiffusionDrive baseline with LiDAR removed, while ZTRS labels the 27.5 as LiDAR + image. Which one the row is could not be resolved.
2. SimScale 53.2 is not in the SimScale paper tables I read (best single 48.0, ensemble 50.9). DrivoR calls it "SimScale (+185k SimScale data, V2-99)".
3. GTRS and ZTRS stitch the front camera with cropped front-left and front-right images. DrivoR's efficiency table counts them as 1 or 2 cameras.
4. TerraTransfer++: camera count, resolution and the exact ego fields were not in the sections read.
5. The py123d_garage docs say official scoring keeps each policy's own inputs, while the README labels baseline rows "Command". So the navigation column is the table's label, not a check of each run.
