# exp_navsim

Self-contained NAVSIM long-episode trajectory-prediction pipeline. Everything is
driven by one config (`config.yaml`) whose subkeys separate the logic. The
dataloader and model are rewritten here (not inherited from the older nuplan
code); other utilities are imported where allowed.

## Pipeline

```
dataset: navtrain | navhard        (one switch in config.yaml)
      │
      │  NavsimLongBase.from_config(cfg) → NavsimLongDataset (navhard) | NavtrainLongDataset (navtrain)
      │  both subclass NavsimLongBase — one whole episode per index
      ▼
1. visualize_dataset.py ──► navsim_episodes.pdf   (stats page + episodes)
2. cache_latents.py     ──► data/<navsim|navtrain>_latents/<i>.h5   (per-frame latents)
      │
      ├─ 3. test_decode.py ──► navsim_decoded.pdf   (decode latents, sanity check)
      ▼
   NavsimLatentDataset (latent_dataset.py) — fixed-length windows from cache
      │  behind BufferDataset (buffer.py)
      ▼
4. train_nuplan.py -c exp_navsim/config.yaml --max_steps N   (model.py)
      ▼
5. test_model.py --ckpt ... ──► navsim_predictions.pdf   (predicted trajectories)
```

## Choosing the dataset (navtrain vs navhard)

One top-level `dataset:` key in `config.yaml` selects the source long dataset.
`config.data_long.{navhard,navtrain}` hold one params block each, and
`NavsimLongBase.from_config(cfg)` dispatches explicitly on the flag (navhard →
`NavsimLongDataset`, navtrain → `NavtrainLongDataset`), loading that block's
params into the chosen class. The scripts (`visualize_dataset`, `cache_latents`,
`test_decode`) just call `from_config` / `cache_dir`, so they never hard-code a
dataset. To switch:

1. set `dataset: navtrain` (or `navhard`) in `config.yaml`;
2. point the training `data:` block's `cache_dir` at the matching latent dir
   (`data/navtrain_latents` vs `data/navsim_latents`) — the alternative is in a
   comment beside it.

`cache.dir` is a `{navhard, navtrain}` mapping, read by
`NavsimLongBase.cache_dir(cfg)`, so caching/decoding follow the switch too.

- **navhard** (`navhard_two_stage`): one meta pickle == one episode (its longest
  scene, ~5 frames). All camera images are present.
- **navtrain** (`trainval_navsim_logs` + `trainval_sensor_blobs`): each log pickle
  is a full driving log split into many ~40-frame scenes (2 Hz). **One episode ==
  one scene**, restricted to the *longest contiguous run of frames whose
  front-camera image is actually on disk* — the navtrain sensor download is
  usually partial, so most frames have no image. Scenes with fewer than
  `min_frames` usable frames are dropped; missing surround images are rendered as
  black in visualization. The (log, scene) → present-frames index is scanned once
  and cached to `index_cache` (`data/navtrain_episode_index.pkl`); **delete that
  file to rebuild** after downloading more blobs or changing `front_camera`.

## Files

| file | role |
|------|------|
| `data/navsim_base.py`  | `NavsimLongBase`: shared core (image transform, camera reading, frames→sample, subsampling, split) + pose/velocity helpers; `from_config(cfg)` / `cache_dir(cfg)` explicit `dataset:` dispatch |
| `data/navsim_long.py`  | `NavsimLongDataset` (navhard): episode = longest scene of a meta pickle; re-exports the shared helpers |
| `data/navtrain_long.py`| `NavtrainLongDataset` (navtrain): episode = a scene's longest on-disk-present frame run; builds+caches an episode index |
| `draw.py`          | `draw_episode(batch, i)` — field-driven: BEV + 5 evenly spaced cameras + trajectory |
| `data/bev_extract.py` | `extract_bev(long_ds, i)` — map BEV context (nuPlan `map_api` + one navsim `Frame`) and GT path, built straight from the raw log frames (no sensor blobs, no `SceneLoader`) |
| `draw_bev.py`      | `draw_bev_distribution(ax, bev, gt, preds)` — map + agents with the whole sampled trajectory distribution on top |
| `visualize_dataset.py` | stats page (episode count + length distribution) + episode pages → PDF |
| `encoder_io.py`    | one place to load the frozen tokenizer and encode/decode |
| `cache_latents.py` | encode every episode → one HDF of per-frame latents in `data/` |
| `latent_dataset.py`| `NavsimLatentDataset` (cached latents) and `NavsimRawWindowDataset` (raw images); fixed `num_frames` windows |
| `buffer.py`        | ring-buffer dataset copied from `brain_matching` |
| `denoiser.py`      | `TrajectoryDenoiser` transformer — network only, no flow-matching |
| `embeddings.py`    | sin-cos embeddings shared by both denoisers: diffusion time (spread over [0, 1000)) and a tabulated absolute frame index |
| `model.py`         | `NavsimTrajectoryModel` — flow matching, EMA norm, raw/encoded flag, val metrics |
| `norm.py`          | `TrajectoryNorm` — EMA velocity normalization |
| `metrics.py`       | trajectory MSE and STD of total turn |
| `test_decode.py`   | decode cached latents → PDF (BEV + decoded images + trajectory) |
| `test_model.py`    | run a checkpoint on fixed test episodes → PDF of predicted trajectories |
| `config.yaml`      | single config, subkeys `dataset` (switch) / `data_long` / `cache` / `model` / `data` |
| `wm/wm_runner.py`  | `WorldModelRunner` — the frozen stage-2 world model, run in latent space only |
| `wm/denoiser.py`   | `WmTrajectoryDenoiser` — compact trajectory denoiser over world-model latents |
| `wm/model.py`      | `WmTrajectoryModel` — subclasses `NavsimTrajectoryModel`, swaps the conditioning |
| `wm/config.yaml`   | config for the world-model variant (training command on top of the file) |

## Raw vs encoded (config flag)

`model.params.encode_images`:
- **false** (default): read cached latents (`NavsimLatentDataset`); the model
  never loads the tokenizer. Requires running `cache_latents.py` first.
- **true**: read raw images and let the model encode on the fly. A ready
  `data_raw:` block is in `config.yaml` — set `encode_images: true`, then rename
  `data:` → something else and `data_raw:` → `data:`. Batches then carry `images`
  (via `NavsimRawWindowDataset`) instead of `encoded_q_sem`; no latent cache is
  needed.

## What the denoiser sees

Both cached latents condition the trajectory head. `get_encoded_q` concatenates
them over the channel axis as `[q_sem, q_rec]`, so the aggregator's input is
`2 * model.params.image_encoder_channels`: the semantic latent says what the
scene means, the reconstruction latent where things are.

Episodes have different lengths, so batches are zero-padded to the longest one
(`data/collate.py`). Two separate masks follow from that, and they are not the
same mask:

- `frame_mask` marks the frames that exist. It becomes the attention
  `key_padding_mask`, so a real frame never reads a padded one. Without it the
  padded tail leaks into the prediction, and differently in every batch.
- `scored_mask` marks the frames that count towards a loss or a metric: the real
  frames minus frame 0. Velocity is a finite difference, so `v_0` is 0 for every
  episode; scoring it rewards predicting a constant, and feeding it to the EMA
  normalizer pulls the velocity mean towards zero. Frame 0 stays in the input,
  where it is a real observed frame.

Frame positions are embedded by their absolute index, never by `arange(T) / T`.
With padding, `T` is the longest episode in the batch, so the normalized version
gave a frame a different embedding depending on which episodes happened to sit
beside it.

## Validation / metrics

The validation split is a deterministic hash of the scene token
(`data.*.split: val`). Each context is sampled `num_val_samples` times;
`validation_step` logs `val/mse` and `val/std` (STD of the total turn — the summed
turn angle along the trajectory) to TensorBoard, and prints only `val/loss`
(= MSE) on the command line. `test_model.py` reuses the same validation set to
render the predicted-trajectory PDF.

Its top row is: first context view, last context view, and the **map BEV with the
predicted distribution** (`data/bev_extract.py` extracts the BEV + GT path,
`draw_bev.py` draws the samples on it). The BEV is rendered in the ego frame of
episode frame 0, which is exactly the frame `batch["trajectory"]` lives in for the
deterministic (start=0) validation windows — so no transform is needed. It needs
`NUPLAN_MAPS_ROOT` / `NUPLAN_MAP_VERSION`; pass `--no-bev` to drop the panel.

## World-model conditioning (`wm/`)

A second model, in `wm/`, keeps the whole pipeline above and changes only what
conditions the trajectory diffusion:

```
basic model   observed tokenizer latents                     -> aggregator -> tokens
wm model      observed latents + the latent the frozen Orbis world model
              imagines for the NEXT frame                    -> aggregator -> tokens
```

Same cache, same dataloader, same flow-matching schedule and the same
`val/loss` / `val/mse` / `val/std`, so the two runs are comparable curve for
curve. `WmTrajectoryModel` subclasses `NavsimTrajectoryModel` and overrides one
method (`get_encoded_q`); `exp_navsim/model.py` is untouched.

**World-model execution** (`wm/wm_runner.py`) is deliberately separate from the
trainable model. It builds *only* the STDiT backbone out of the stage-2
checkpoint — never the ~3 GB tokenizer inside it, because cached latents go in
and a latent comes out — and it is a plain object rather than an `nn.Module`, so
its ~700 M frozen parameters never join the trajectory model's parameters or
state dict. The next latent comes from `wm_steps` Euler steps of the same
flow-matching ODE the world model was trained with; the step count is low on
purpose (a conditioning signal, not a picture). The rollout is memoized per
batch, so validation's 20 samples pay for it once.

**The denoiser** (`wm/denoiser.py`) is written for few weights and little
overfitting: one aggregator shared by every latent frame (frames differ only by
a learned slot vector), one MLP for context and predicted trajectory frames
separated by a two-row role embedding, and `num_layers` passes over
`num_unique_layers` tied cross-attention blocks — depth without the parameters.
Regularisation is dropout in every block, `cond_dropout` blanking a sample's
entire conditioning during training, a zero-initialised output layer, and no
parameter that scales with the episode length.

Requires `model.params.wm_exp_dir` to hold **both** `checkpoints/last.ckpt` and
the `config.yaml` the world model was trained with (the config supplies the
backbone architecture and the latent scales).

**It needs its own latent cache.** The released world model is
`SwinSTDiTNoExtraMLP`, and every Swin block bakes in one latent resolution — the
window partitioning, the shifted-window attention masks and the
relative-position tables. Unlike a plain DiT it cannot be retargeted to another
grid, so the latents must be exactly 18×32, i.e. cached from 288×512 images
rather than the basic model's 256×256. `wm/config.yaml` is already set up for
that; run its caching pass once:

```bash
python -m exp_navsim.data.cache_latents --config exp_navsim/wm/config.yaml
```

288×512 is also the tokenizer's own declared resolution and the native aspect
ratio of the NAVSIM front camera, so this cache crops nothing, where `size: 256`
cut a square out of the frame. Feeding a mismatched grid raises a
`ValueError` naming the expected size instead of failing deep inside a Swin
block.

## Commands

```bash
# All commands honour the `dataset:` switch in config.yaml (navtrain by default).
python -m exp_navsim.visualize_dataset --config exp_navsim/config.yaml --num 5
python -m exp_navsim.data.cache_latents --config exp_navsim/config.yaml   # first run builds the navtrain index
python -m exp_navsim.test_decode       --config exp_navsim/config.yaml --num 3
python -m exp_navsim.data.bev_extract  --config exp_navsim/config.yaml --num 3   # BEV extraction smoke check
python -m exp_navsim.draw_bev          --config exp_navsim/config.yaml --num 3   # BEV drawing smoke check -> PDF
python train_nuplan.py -c exp_navsim/config.yaml --max_steps 20000 --logdir logs_navsim
python tensorboard_to_pdf.py --logdir ./logs_navsim --last --name navtrain.pdf --from 1000
python -m exp_navsim.test_model --config exp_navsim/config.yaml --ckpt logs_navsim/2026-08-21T14-45-31_config --num 6

python -m exp_navsim.test_model --config exp_navsim/config.yaml \
    --ckpt logs_navsim/2026-08-21T14-45-31_config/checkpoints/last.ckpt --num 5

# --- world-model-conditioned variant (exp_navsim/wm/) ---
# ONCE: its own cache at the world model's 288x512 / 18x32 resolution.
python -m exp_navsim.data.cache_latents --config exp_navsim/wm/config.yaml
python train_nuplan.py -c exp_navsim/wm/config.yaml --max_steps 20000 --logdir logs_navsim_wm
python tensorboard_to_pdf.py --logdir ./logs_navsim_wm --last --name navtrain_wm.pdf --from 1000
python -m exp_navsim.test_model --config exp_navsim/wm/config.yaml \
    --ckpt logs_navsim_wm/<run>/checkpoints/last.ckpt --num 5
```

## Data-format assumption

Both datasets use the **same per-frame dict** (verified from the pickles): keys
`ego2global_translation`, `ego2global_rotation`, `ego_dynamic_state`, `cams`
(camera → `data_path` relative to its sensor root), `scene_token`, `frame_idx`.
The pose/velocity/camera code in `NavsimLongBase` therefore works unchanged for
both; the datasets differ **only** in episode enumeration:

- **navhard** — `NavsimLongDataset._episode_frames`: each
  `openscene_meta_datas/<token>.pkl` is a `list[dict]`; episode = its longest
  scene. Sensor root = `navhard_two_stage/sensor_blobs`.
- **navtrain** — `NavtrainLongDataset._build_index` / `_episode_frames`: each
  `trainval_navsim_logs/trainval/<log>.pkl` is a `list[dict]` spanning many
  scenes; episode = one scene's longest run of frames whose front-camera image
  exists under `trainval_sensor_blobs/trainval/<log>/<CAM>/`. Handles the partial
  sensor download by construction (see "Choosing the dataset" above).

If a future dump differs, only the relevant `_episode_frames` / `_build_index`
(and possibly camera key casing in `NavsimLongBase._read_camera`) needs adjusting
— the rest of the pipeline is agnostic to how a sample is produced.
