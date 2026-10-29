# exp_navsim

Probing frozen image and world-model features with a flow-matching trajectory
model on NAVSIM, in the official protocol: 4 history frames of the front camera
at 2 Hz in, 8 poses (x, y, heading) over the next 4 s out.

```
data/build_index.py      navtrain / navtest / navhard scenes -> data/navsim_index/<split>.pkl
data/resize_frames.py    history frames -> data/frames/<split>/ at 288 x 512 (local copy)
encoders/cache_features.py   frozen encoder -> data/features/<split>/<feature>.npy  (N, C, 9, 16)
train.py                 probe on cached features -> logs_exp/<run>/
eval/evaluate.py         sample metrics per checkpoint and split -> logs_exp/<run>/eval/
eval/pdm.py              EPDMS of saved predictions with the official navsim scripts
eval/baselines.py        constant velocity, train mean, nearest ego status
```

## Layout

| folder | what |
|---|---|
| `data/` | scene index (`build_index`), local frame copies (`resize_frames`), `SceneFeatureDataset` (`scenes`), Lightning data module; raw-log readers and the BEV extraction used by the viewers (`navsim_base`, `navsim_long`, `navtrain_long`, `bev_extract`) |
| `encoders/` | frozen encoders behind one interface: Orbis tokenizer (`q_sem`, `q_rec`), DINOv2, V-JEPA 2, Orbis world model (`wm_runner`); feature caching with PCA |
| `models/` | trajectory codecs (raw poses, polynomial), the probe denoiser, the Lightning probe |
| `eval/` | distribution metrics, offline evaluation, baselines, EPDMS (`pdm`, `navsim_agent`) |
| `viz/` | episode / BEV drawing and dataset visualization |
| `configs/` | `probe.yaml`, the single training config; everything else is a command-line override |
| `episode/` | the earlier whole-episode probe (variable-length episodes, tokenizer latents at 256 x 256); kept for the results in the report and the old checkpoints, see `episode/README.md` |

## Data

**Index.** A NAVSIM scene is a 14-frame window of a log (4 history, 10 future),
sliding by one frame, named by the token of its current frame. `build_index`
rebuilds the navtrain filter exactly and keeps the scenes whose 4 history images
are on disk: 102,056 of 103,288. Per scene it stores the frame ids of the history
images, the future and past poses in the current ego frame (navsim's own SE2
conversion), ego velocity and acceleration, the driving command and the city.
The test splits are read through navsim's `SceneLoader`, which also knows the
synthetic second-stage scenes of navhard.

**Splits.** Validation holds out whole logs (10% of them, by md5 of the log
name). navtrain scenes overlap, so a scene-level split leaks. `cities` keeps only
some cities, `holdout` removes (city, command) pairs from training and evaluates
them separately (compositional test), `max_scenes` subsamples training.

**Features.** Every encoder is reduced to the same interface: an 18 x 32 grid,
average-pooled to 9 x 16, with 64 PCA channels for wide features (DINOv2 768,
V-JEPA 2 1024, world model 768) and the native 16 channels of the tokenizer
latents. The PCA is fitted on training logs only and sorted, so `channels: k`
gives the k-dim PCA without a new cache. Per-frame features (q_sem, q_rec,
dinov2) are stored per image and gathered for the 4 history frames; clip
features (vjepa2, wm*) depend on the whole history and are stored per scene.

World-model features `wm<b>` / `wm<b>_ctx`: the backbone runs once over the 4
encoded history frames plus a pure-noise next frame at t = 1; `wm<b>` is the
next-frame slot after block b (the model's guess of the future), `wm<b>_ctx` the
last observed frame (the history as the model contextualizes it).

## Model

`models/probe.py`: flow matching on a trajectory code z (`models/codec.py`:
the 24 raw numbers, or 3 K polynomial coefficients without a constant term). The
denoiser pools every frame's grid into `image_tokens` tokens, optionally adds an
ego-status and a command token, and a single query token built from (z_t, t)
attends to them through `depth` AdaLN-Zero cross-attention blocks.

## Metrics

`eval/metrics.py`, over K samples per scene: ADE/FDE averaged over samples and
best of K, the error of the sample mean, the energy score (a proper scoring rule
for the whole distribution), the spread E||X - X'||, and the dispersion ratio
spread / E ADE, which is about 1 for a calibrated sampler and below 1 when the
samples collapse. Validation runs on held-out logs (`val`) and on a fixed subset
of training scenes (`train_sub`), so memorization shows up as the two curves
separating.

## Commands

```bash
export PYTHONPATH=.:navsim   # navsim/ holds the devkit; it must win over the repo folder of the same name
python -m exp_navsim.data.build_index --split navtrain
python -m exp_navsim.data.resize_frames --split navtrain
python -m exp_navsim.encoders.cache_features --encoder orbis --split navtrain
python -m exp_navsim.train --name 01_qsem data.features=[q_sem]
python -m exp_navsim.eval.evaluate --run logs_exp/01_qsem --split val --ckpts all
python tools/tensorboard_to_pdf.py --logdir logs_exp/01_qsem --name tb.pdf
# test splits: tools/download_navsim_test.sh, then build_index / cache_features with --split navtest
python -m exp_navsim.eval.pdm cache --split navtest
python -m exp_navsim.eval.evaluate --run logs_exp/01_qsem --split navtest --save-predictions
python -m exp_navsim.eval.pdm score --split navtest --predictions logs_exp/01_qsem/eval/navtest_last.npz --samples 0,1,2,3 --mean
```
