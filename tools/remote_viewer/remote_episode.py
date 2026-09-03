"""Remote half of the episode viewer — runs INSIDE the cluster's Jupyter kernel.

view_episode.py (on the local PC) reads this file's source and pushes it into the
kernel, then calls the entry points below over the Jupyter websocket:

    viewer_list(config)                                 -> one JSON line
    viewer_scan(config)                                 -> one JSON line (slow, once)
    viewer_filter(min_distance, min_angle, min_frames)  -> one JSON line
    viewer_model(ckpt, config)                          -> one JSON line
    viewer_open(config, episode, ...)                   -> one JSON line
    viewer_frames(start, stop, ...)                     -> JPEG display_data messages

Nothing here is imported on the local PC; it only has to be importable on the
cluster (orbis repo root on sys.path, navsim env active).

Everything expensive — the dataset index, the per-episode statistics, the loaded
checkpoint, the rasterised BEV, the per-frame predictions — is cached in the
module-level _VIEWER dict, which lives in the kernel between calls. That is the
whole point of driving a kernel instead of spawning a process per request.
"""

import base64
import io
import json

_VIEWER = {}          # kernel-resident state; survives between calls


# --------------------------------------------------------------------------- #
# Dataset + per-episode statistics
# --------------------------------------------------------------------------- #
def _config(config_path):
    """Loaded config, cached alongside the dataset it belongs to."""
    from omegaconf import OmegaConf

    if _VIEWER.get("cfg") is None:
        _VIEWER["cfg"] = OmegaConf.load(config_path)
    return _VIEWER["cfg"]


def _dataset(config_path):
    """Build the long dataset, cached on the config path.

    Rebuilding navtrain's episode index costs tens of seconds, so it happens once
    per kernel, not once per seek.
    """
    from exp_navsim.data.navsim_base import NavsimLongBase

    if _VIEWER.get("config_path") != config_path:
        _VIEWER["config_path"], _VIEWER["cfg"] = config_path, None
        _VIEWER["ds"] = NavsimLongBase.from_config(_config(config_path))
        for stale in ("stats", "cache_index", "cache_traj"):
            _VIEWER.pop(stale, None)                # all belong to the old dataset
    return _VIEWER["ds"]


def _cache_index(config):
    """One pass over the latent cache -> {token: h5 path}, plus its trajectories.

    Both the episode scan and the model need this map, and walking ~10k files
    twice would be pure waste. The file open dominates the cost, so reading each
    episode's trajectory in the same pass adds only ~1.7 s over the whole cache
    and saves the scan a second walk.
    """
    import h5py
    from pathlib import Path

    from exp_navsim.data.navsim_base import NavsimLongBase

    if "cache_index" not in _VIEWER:
        cache_dir = Path(NavsimLongBase.cache_dir(_config(config)))
        index, trajectories = {}, {}
        for path in sorted(cache_dir.glob("*.h5")):
            with h5py.File(path, "r") as f:
                token = f.attrs.get("token", path.stem)
                index[token] = str(path)
                trajectories[token] = f["trajectory"][:]
        _VIEWER["cache_index"], _VIEWER["cache_traj"] = index, trajectories
    return _VIEWER["cache_index"]


def viewer_list(config="exp_navsim/config.yaml"):
    """Episode count — the viewer browses range(num_episodes) until a filter runs."""
    ds = _dataset(config)
    print(json.dumps({"num_episodes": len(ds)}))


def _scan(config):
    """Cache length / distance / steering of every episode, for the GUI filters.

    Trajectories come from the latent cache rather than from
    ds.episode_trajectory, which re-parses the navtrain log pickles and then
    filters all their frames per episode: measured 0.4 ms vs 13 ms per episode,
    i.e. ~4 s instead of ~2 min over the 9577 episodes. The cached trajectory is
    identical — cache_latents.py writes exactly sample["trajectory"] — so the
    statistics are unchanged. Episodes with no cached latents fall back to the
    slow path.

    Prints nothing: callers that talk to the GUI must emit exactly one JSON line
    of their own.
    """
    import numpy as np
    from exp_navsim.visualize_dataset import trajectory_stats, MIN_DISPLACEMENT

    ds = _dataset(config)
    if "stats" not in _VIEWER:
        _cache_index(config)
        cached = _VIEWER["cache_traj"]
        rows = []
        for i in range(len(ds)):
            # episode_token is O(1) for both long datasets (it reads the index
            # record, not the log), so the lookup itself costs nothing.
            traj = cached.get(ds.episode_token(i))
            if traj is None:
                traj = ds.episode_trajectory(i)
            rows.append((len(traj),) + trajectory_stats(traj))
        lengths, distances, steerings, displacements = map(np.asarray, zip(*rows))
        _VIEWER["stats"] = {
            "lengths": lengths, "distances": distances, "steerings": steerings,
            # Below MIN_DISPLACEMENT the start->end vector is too short for its
            # angle to mean anything, so those episodes never pass an angle filter.
            "moved": displacements >= MIN_DISPLACEMENT,
        }
    return _VIEWER["stats"]


def viewer_scan(config="exp_navsim/config.yaml"):
    """Run (or reuse) the scan and report the ranges the filter fields accept."""
    import numpy as np

    s = _scan(config)
    print(json.dumps({
        "num_episodes": len(s["lengths"]),
        "max_frames": int(s["lengths"].max()),
        "max_distance": float(s["distances"].max()),
        "max_angle": float(np.abs(s["steerings"]).max()),
    }))


def viewer_filter(min_distance=0.0, min_angle=0.0, min_frames=0,
                  config="exp_navsim/config.yaml"):
    """Episode indices passing the three filters, same criteria as the PDF report.

    Mirrors visualize_dataset.py / test_model.py's EPISODE_FILTERS: distance
    driven, |angle between the initial heading and start->end|, episode length.
    All-zero filters simply return every episode.
    """
    import numpy as np

    s = _scan(config)
    keep = ((s["distances"] >= min_distance)
            & (np.abs(s["steerings"]) >= min_angle)
            & (s["lengths"] >= min_frames))
    if min_angle > 0:
        keep &= s["moved"]
    print(json.dumps({"episodes": np.flatnonzero(keep).tolist(),
                      "total": len(s["lengths"])}))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def viewer_model(ckpt, config="exp_navsim/config.yaml", split="all"):
    """Load a checkpoint plus the latent cache, cached in the kernel.

    The viewer indexes episodes through the *long* dataset while the model is fed
    from the *cached-latent* dataset, so a scene-token map bridges the two —
    exactly what test_model.py does with _episode_tokens.

    `split` overrides the validation block's own split. It defaults to "all"
    because the two datasets otherwise disagree: data_long runs with split "all"
    (~every episode) while data.params.validation is split "val" at
    val_fraction 0.1, so nine out of ten browsable episodes would silently have
    no latents and therefore no prediction. cache_latents.py caches every
    episode, so "all" is available; viewer_open reports which side of the split
    each episode falls on, since a train episode was seen during training.

    Only the encoded mode is supported; raw mode would need the tokenizer.
    """
    import torch

    from exp_navsim.data.navsim_base import in_split
    from exp_navsim.model import NavsimTrajectoryModel

    if _VIEWER.get("ckpt") != (ckpt, split):
        _dataset(config)                     # binds cfg + cache to this config first
        cfg = _config(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NavsimTrajectoryModel.load_from_checkpoint(ckpt, map_location=device)
        if model.encode_images:
            raise RuntimeError("viewer supports encoded mode only (model.encode_images is set)")

        # The cache index is shared with the episode scan, so whichever runs
        # first pays for the walk and the other gets it free.
        val_fraction = float(cfg.data.params.validation.params.get("val_fraction", 0.1))
        token_to_file = {token: path for token, path in _cache_index(config).items()
                         if split == "all" or in_split(token, split, val_fraction)}
        _VIEWER.update(ckpt=(ckpt, split), model=model.to(device).eval(), device=device,
                       token_to_file=token_to_file, val_fraction=val_fraction)

    model = _VIEWER["model"]
    print(json.dumps({
        "ckpt": ckpt, "device": _VIEWER["device"], "split": split,
        "cached_episodes": len(_VIEWER["token_to_file"]),
        "context_traj": int(model.context_traj),
        "context_images": int(model.context_images),
        "num_samples": int(model.num_val_samples),
    }))


def _heading_at(trajectory, i):
    """Ego heading at frame `i`, taken from the previous velocity step.

    Walks back to the most recent non-zero step, since the ego may be standing
    still at `i`; if nothing precedes `i` it uses the first move instead. The
    final fallback of 0 is the episode-frame-0 heading, which
    poses_to_local_traj puts along +x by construction.
    """
    import numpy as np

    traj = np.asarray(trajectory, dtype=np.float64)
    order = list(range(min(i, len(traj) - 1), 0, -1)) + list(range(1, len(traj)))
    for j in order:
        step = traj[j] - traj[j - 1]
        if np.linalg.norm(step) > 1e-6:
            return float(np.arctan2(step[1], step[0]))
    return 0.0


def _rotate_path(path, heading):
    """Rotate a path about its origin by `heading` radians. Shape (..., T, 2).

    Written as the polar round trip the fix calls for: every step becomes a
    (heading, distance) pair, `heading` is added to each, and the path is
    re-integrated. That is a rigid rotation, so step lengths are untouched — only
    the direction the whole path sets off in changes.
    """
    import numpy as np

    path = np.asarray(path, dtype=np.float64)
    steps = np.diff(path, axis=-2, prepend=np.zeros_like(path[..., :1, :]))
    angle = np.arctan2(steps[..., 1], steps[..., 0]) + heading
    distance = np.linalg.norm(steps, axis=-1)
    return np.cumsum(np.stack([distance * np.cos(angle),
                               distance * np.sin(angle)], axis=-1), axis=-2)


def _window_batch(starts, total_len):
    """Build one padded batch holding the model window that starts at each frame.

    Each window is [start, start + total_len) of the cached episode, re-origined
    at its own first frame — the same construction as
    latent_dataset._window_traj, just at an explicit start instead of the
    dataset's own choice.

    Windows running off the end of the episode are padded by repeating the last
    pose (zero velocity). That padding is never read: the denoiser fills only
    `true_deltas[:, :context_traj]` from the batch and predicts everything after,
    so the final frames still get a genuine future prediction rather than an
    echo of the ground truth.
    """
    import h5py
    import numpy as np
    import torch

    from exp_navsim.data.collate import pad_collate
    from exp_navsim.data.navsim_base import traj_to_velocity

    model = _VIEWER["model"]
    context_images = int(model.context_images)
    samples = []
    with h5py.File(_VIEWER["latent_file"], "r") as f:
        trajectory = f["trajectory"][:]
        frame_rate = int(f.attrs.get("frame_rate", 2))
        for start in starts:
            # Only encoded_q_sem is read: get_encoded_q uses nothing else, and this
            # runs once per displayed frame, so the skipped q_rec read matters.
            n_img = min(context_images, total_len)
            q_sem = f["encoded_q_sem"][start:start + n_img]

            traj = np.asarray(trajectory[start:start + total_len], dtype=np.float32)
            traj = traj - traj[0]
            # Rotate into the ego frame at `start`. Training only ever saw windows
            # beginning at episode frame 0 (num_frames=0 makes _window_start
            # return 0), where poses_to_local_traj leaves the ego heading along
            # +x — so that, not the frame-0 heading, is the frame the model was
            # fitted in. Feeding a mid-episode window unrotated hands it context
            # velocities pointing the wrong way.
            #traj = _rotate_path(traj, -_heading_at(trajectory, start)).astype(np.float32)
            pad = total_len - len(traj)
            if pad > 0:
                traj = np.concatenate([traj, np.repeat(traj[-1:], pad, axis=0)], axis=0)
            samples.append({
                "encoded_q_sem": torch.from_numpy(np.asarray(q_sem)).float(),
                "trajectory": torch.from_numpy(traj).float(),
                "velocity": torch.from_numpy(traj_to_velocity(traj)).float(),
                "frame_rate": frame_rate,
                # length = total_len: every frame of the window is predicted, and
                # sample() ignores the mask anyway.
                "metadata": {"length": total_len},
            })

    batch = pad_collate(samples)
    return {k: (v.to(_VIEWER["device"]) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def _predict(starts, total_len, num_samples):
    """{start: (N, total_len, 2)} sampled trajectories, in episode frame-0 coords.

    All requested starts go through the model as ONE batch, so a batch of frames
    costs `num_samples` forward passes rather than num_samples * len(starts).
    Results are memoised per (total_len, start) so scrubbing backwards is free.

    The model returns window-local paths starting at the origin, in the ego frame
    of the window start (see _window_batch). Placing one back into episode
    coordinates therefore takes a rotation by the ego heading at `start` followed
    by a translation to the ego position there — the cached trajectory is aligned
    to episode frame 0, so the two frames differ by exactly that heading.
    """
    import torch

    if "model" not in _VIEWER or _VIEWER.get("latent_file") is None:
        return {}

    cache = _VIEWER.setdefault("preds", {})
    todo = [s for s in starts if (total_len, s) not in cache]
    if todo:
        model = _VIEWER["model"]
        n = num_samples or int(model.num_val_samples)
        batch = _window_batch(todo, total_len)
        with torch.no_grad():
            preds = torch.stack([model.sample(batch) for _ in range(n)], 1)  # (B, N, T, 2)
        preds = preds.float().cpu().numpy()
        gt = _VIEWER["bev_gt"]
        for j, start in enumerate(todo):
            # Undo the input rotation, then place the path at the ego position.
            heading = _heading_at(gt, start)
            cache[(total_len, start)] = (_rotate_path(preds[j], heading)
                                         + gt[min(start, len(gt) - 1)])
    return {s: cache[(total_len, s)] for s in starts if (total_len, s) in cache}


def _max_start(total_len):
    """Last frame whose model context is real rather than padding.

    The window's leading context_traj poses and context_images latents are the
    model's *input*; once they would run off the end of the episode there is
    nothing genuine to condition on, so those final frames get no prediction.
    """
    model = _VIEWER.get("model")
    if model is None:
        return -1
    context = max(int(model.context_traj), int(model.context_images), 1)
    return len(_VIEWER["frames"]) - min(context, total_len)


# --------------------------------------------------------------------------- #
# BEV panel
# --------------------------------------------------------------------------- #
def _render_bev_base(episode, size):
    """Rasterise the static part of the BEV once: map, agents, whole GT path.

    Drawing nuPlan lanes and agents costs on the order of a second, so it cannot
    happen per frame — and it does not have to, because the map is fixed for the
    episode. Only the GT window and the predicted distribution change with the
    playhead, and those are drawn as cheap PIL overlays on top of this bitmap
    using the recorded data->pixel affine.

    Uses Figure/FigureCanvasAgg directly rather than pyplot, so it cannot disturb
    the inline backend of a notebook sharing this kernel.

    Returns (image, affine, gt_path) or (None, None, None).
    """
    import numpy as np
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from exp_navsim.data.bev_extract import extract_bev
    from exp_navsim.draw_bev import draw_bev_distribution

    # anchor=0: the GT path from poses_to_local_traj is rooted at episode frame 0
    # in both origin and heading, so BEV and path share one frame with no
    # transform (see exp_navsim/data/bev_extract.py).
    bev = extract_bev(_VIEWER["ds"], episode, anchor=0)
    if bev is None:
        return None, None, None
    gt = np.asarray(bev["trajectory"], dtype=np.float64)

    dpi = 100
    fig = Figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])          # full-bleed: no wasted margin
    # No predictions here — they are per-frame. Extra margin leaves room for the
    # sampled paths, which can leave the GT's bounding box.
    draw_bev_distribution(ax, bev, gt, None, title=None)
    canvas.draw()

    image = Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3])
    # The BEV ax plots (y, x) with x inverted, and matplotlib's display origin is
    # bottom-left while PIL's is top-left. The axes are linear, so two probes give
    # the whole mapping: px = px0 + sx * ego_y, py = py0 + sy * ego_x.
    origin = ax.transData.transform([[0.0, 0.0]])[0]
    unit = ax.transData.transform([[1.0, 1.0]])[0]
    affine = (float(origin[0]), float(unit[0] - origin[0]),
              float(image.height - origin[1]), float(origin[1] - unit[1]))
    return image, affine, gt


def _to_pixels(points, affine):
    """(T, 2) ego-frame path -> list of PIL (x, y) pixel tuples."""
    import numpy as np

    px0, sx, py0, sy = affine
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return [(px0 + sx * p[1], py0 + sy * p[0]) for p in pts]


def _bev_overlay(i, total_len, num_samples):
    """The BEV bitmap with this frame's GT window, predictions and ego marker."""
    from PIL import ImageDraw

    panel = _VIEWER["bev_img"].copy()
    affine, gt = _VIEWER["bev_affine"], _VIEWER["bev_gt"]
    draw = ImageDraw.Draw(panel, "RGBA")

    # Predicted distribution for this frame, translucent so overlap reads as density.
    # A failure degrades to a plain BEV rather than losing the frame, but it is
    # recorded — swallowing it silently looks exactly like "the model predicted
    # nothing", which is the one thing this panel exists to show.
    preds = None
    if i <= _max_start(total_len):
        try:
            preds = _predict([i], total_len, num_samples).get(i)
        except Exception as e:
            _VIEWER["pred_error"] = f"prediction failed: {type(e).__name__}: {e}"
    if preds is not None:
        # Same opacity ramp as draw_bev.draw_bev_distribution, so the density
        # reads the same here as in the PDF report.
        alpha = int(255 * min(max(3.0 / len(preds), 0.06), 0.8))
        for pred in preds:
            draw.line(_to_pixels(pred, affine), fill=(255, 40, 40, alpha), width=2)
        for x, y in _to_pixels(preds[:, -1], affine):     # endpoint cloud
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 40, 40, 110),
                         outline=(180, 0, 0, 180))

    # The GT window the prediction is scored against. It is shorter than the
    # window near the end of the episode — that is the point: those frames are a
    # genuine forecast with no ground truth to compare against.
    window = gt[i:i + total_len]
    if len(window) > 1:
        draw.line(_to_pixels(window, affine), fill=(30, 140, 255, 255), width=3)

    x, y = _to_pixels(gt[min(i, len(gt) - 1)], affine)[0]
    draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(255, 220, 0, 255),
                 outline=(0, 0, 0, 255), width=2)
    return panel


# --------------------------------------------------------------------------- #
# Opening an episode / streaming frames
# --------------------------------------------------------------------------- #
def viewer_open(config="exp_navsim/config.yaml", episode=0, cameras="surround",
                bev=True, bev_size=512):
    """Select one episode, rasterise its BEV, and report the frame geometry.

    The camera frames themselves are NOT decoded here — _episode_frames only
    parses the pose pickle — so opening stays fast even for long episodes.
    """
    from exp_navsim.data.navsim_base import SURROUND_CAMERAS, in_split

    ds = _dataset(config)
    frames, meta = ds._episode_frames(ds.episodes[episode])
    frames = frames[::ds.frame_interval]             # same subsampling as load_sample
    names = list(SURROUND_CAMERAS) if cameras == "surround" else [ds.front_camera]
    _VIEWER.update(frames=frames, cams=names, episode=episode, preds={},
                   bev_img=None, bev_affine=None, bev_gt=None,
                   # Only validation episodes have cached latents to predict from.
                   latent_file=_VIEWER.get("token_to_file", {}).get(meta["token"]))

    note = ""
    if bev:
        try:
            (_VIEWER["bev_img"], _VIEWER["bev_affine"],
             _VIEWER["bev_gt"]) = _render_bev_base(episode, bev_size)
            if _VIEWER["bev_img"] is None:
                note = "no nuPlan map for this episode"
        except Exception as e:                       # missing maps / navsim
            note = f"BEV unavailable: {type(e).__name__}: {e}"
    if "model" not in _VIEWER:
        note = note or "no --ckpt given — BEV shows ground truth only"
    elif _VIEWER["latent_file"] is None:
        note = note or "no cached latents for this episode — no prediction"

    h, w = ds.size                                   # per-camera crop; cameras side by side
    strip_w, strip_h = w * len(names), h
    panel = _VIEWER["bev_img"]
    print(json.dumps({
        "episode": episode, "token": meta["token"], "num_frames": len(frames),
        "num_episodes": len(ds), "cameras": names, "frame_rate": ds.frame_rate,
        "width": max(strip_w, panel.width if panel is not None else 0),
        "height": strip_h + (panel.height if panel is not None else 0),
        "bev": panel is not None,
        "predicted": panel is not None and _VIEWER["latent_file"] is not None,
        # Which side of the train/val split this episode falls on: predictions on
        # a train episode were fitted on that very trajectory.
        "split": ("val" if in_split(meta["token"], "val", _VIEWER.get("val_fraction", 0.1))
                  else "train"),
        "note": note,
    }))


def _camera_strip(i):
    """Frame `i` as a PIL image: the selected cameras concatenated left to right.

    Reuses NavsimLongBase._read_camera, so the resize / center-crop / scaling match
    the training pipeline exactly — what you watch is what the model is fed. It
    returns (3, H, W) in [-1, 1]; required=False tolerates the surround cameras
    that navtrain only partially downloads (they come back black).
    """
    import torch
    from PIL import Image

    ds, frame = _VIEWER["ds"], _VIEWER["frames"][i]
    strip = torch.cat([ds._read_camera(frame, c, required=False)
                       for c in _VIEWER["cams"]], dim=2)          # concat along width
    strip = ((strip.clamp(-1, 1) + 1) * 127.5).round().byte()     # [-1,1] -> uint8
    return Image.fromarray(strip.permute(1, 2, 0).numpy())        # CHW -> HWC


def _render(i, total_len, num_samples):
    """Composite frame `i`: the camera strip above the BEV panel."""
    from PIL import Image

    strip = _camera_strip(i)
    if _VIEWER.get("bev_img") is None:
        return strip
    panel = _bev_overlay(i, total_len, num_samples)

    width = max(strip.width, panel.width)
    out = Image.new("RGB", (width, strip.height + panel.height), (0, 0, 0))
    out.paste(strip, ((width - strip.width) // 2, 0))
    out.paste(panel, ((width - panel.width) // 2, strip.height))
    return out


def _encode(img, quality, max_width):
    """JPEG-encode, downscaling first if the frame is wider than the viewer needs."""
    if max_width and img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def viewer_frames(start, stop, quality=80, max_width=0, total_len=20, num_samples=0):
    """Emit frames [start, stop) as display_data, one message per frame.

    The whole span's predictions are computed up front as a single batch — that is
    where per-frame inference becomes affordable — and then each frame is drawn
    from the memoised result.

    The frame index rides in the message metadata rather than in the image, so the
    local side can drop them straight into its cache. One message per frame (as
    opposed to one big reply) lets the viewer start showing a batch before the
    batch has finished rendering. A frame that fails to render is reported in-band
    so one bad frame cannot stall playback.
    """
    from IPython.display import display

    start, stop = max(start, 0), min(stop, len(_VIEWER["frames"]))
    if _VIEWER.get("bev_img") is not None:
        limit = _max_start(total_len)
        batched = [i for i in range(start, stop) if i <= limit]
        if batched:
            try:
                _predict(batched, total_len, num_samples)
            except Exception as e:
                _VIEWER["pred_error"] = f"prediction failed: {type(e).__name__}: {e}"

    for i in range(start, stop):
        try:
            jpeg = _encode(_render(i, total_len, num_samples), quality, max_width)
            display({"image/jpeg": base64.b64encode(jpeg).decode()},
                    raw=True, metadata={"viewer": {"index": i}})
        except Exception as e:
            display({"text/plain": repr(e)}, raw=True,
                    metadata={"viewer": {"index": i, "error": f"{type(e).__name__}: {e}"}})

    # Anything that went wrong without costing us a frame travels back on its own
    # message, so the viewer can show it in the status line.
    note = _VIEWER.pop("pred_error", None)
    if note:
        display({"text/plain": note}, raw=True, metadata={"viewer": {"note": note}})
