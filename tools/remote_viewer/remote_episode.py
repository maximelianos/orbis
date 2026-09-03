"""Remote half of the episode viewer — runs INSIDE the cluster's Jupyter kernel.

view_episode.py (on the local PC) reads this file's source and pushes it into the
kernel, then calls the entry points below over the Jupyter websocket:

    viewer_config(**options)                            -> one JSON line
    viewer_filter(min_distance, min_angle, min_frames)  -> one JSON line
    viewer_model(ckpt, config, split)                   -> one JSON line
    viewer_prepare(episode)                             -> one JSON line
    viewer_frames(episode, start, stop, ...)            -> JPEG display_data

Nothing here is imported on the local PC; it only has to be importable on the
cluster (orbis repo root on sys.path, navsim env active).

Two levels of caching, both living in the kernel between calls — which is the
whole point of driving a kernel instead of spawning a process per request:

  _VIEWER    dataset index, episode statistics, checkpoint, latent-cache index
  _EPISODES  per-episode render state (frames, rasterised BEV, predictions),
             LRU-bounded so the viewer can keep neighbouring episodes warm and
             switch between them without paying the BEV render again.
"""

import base64
import io
import json
from collections import OrderedDict

_VIEWER = {}          # dataset-wide state
_EPISODES = OrderedDict()   # episode -> render state (LRU)

# Rendering options, set once from the client's SETTINGS dict via viewer_config.
_OPTS = {
    "config": "exp_navsim/config.yaml",
    "cameras": "front",       # "front" | "surround"
    "bev": True,
    "bev_size": 512,
    "quality": 80,
    "width": 0,               # final frame width in px (0 = leave as composited)
}

# Episodes kept warm. The viewer buffers the current episode plus 3 either side,
# so 8 leaves one slot of slack before the least-recently-used one is dropped.
MAX_EPISODES = 8


def viewer_config(**options):
    """Set the rendering options and drop every cached panel drawn with the old ones."""
    _OPTS.update(options)
    _EPISODES.clear()
    print(json.dumps(_OPTS))


# --------------------------------------------------------------------------- #
# Dataset, latent cache, per-episode statistics
# --------------------------------------------------------------------------- #
def _config(config_path):
    """Loaded config, cached alongside the dataset it belongs to."""
    from omegaconf import OmegaConf

    if _VIEWER.get("cfg") is None:
        _VIEWER["cfg"] = OmegaConf.load(config_path)
    return _VIEWER["cfg"]


def _dataset():
    """Build the long dataset, cached on the config path.

    Rebuilding navtrain's episode index costs tens of seconds, so it happens once
    per kernel, not once per seek.
    """
    from exp_navsim.data.navsim_base import NavsimLongBase

    config_path = _OPTS["config"]
    if _VIEWER.get("config_path") != config_path:
        _VIEWER["config_path"], _VIEWER["cfg"] = config_path, None
        _VIEWER["ds"] = NavsimLongBase.from_config(_config(config_path))
        for stale in ("stats", "cache_index", "cache_traj"):
            _VIEWER.pop(stale, None)                # all belong to the old dataset
        _EPISODES.clear()
    return _VIEWER["ds"]


def _cache_index():
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
        cache_dir = Path(NavsimLongBase.cache_dir(_config(_OPTS["config"])))
        index, trajectories = {}, {}
        for path in sorted(cache_dir.glob("*.h5")):
            with h5py.File(path, "r") as f:
                token = f.attrs.get("token", path.stem)
                index[token] = str(path)
                trajectories[token] = f["trajectory"][:]
        _VIEWER["cache_index"], _VIEWER["cache_traj"] = index, trajectories
    return _VIEWER["cache_index"]


def _scan():
    """Cache length / distance / steering of every episode, for the GUI filters.

    Trajectories come from the latent cache rather than from
    ds.episode_trajectory, which re-parses the navtrain log pickles and then
    filters all their frames per episode: measured 0.4 ms vs 13 ms per episode,
    i.e. ~4 s instead of ~2 min over the 9577 episodes. The cached trajectory is
    identical — cache_latents.py writes exactly sample["trajectory"] — so the
    statistics are unchanged. Episodes with no cached latents fall back to the
    slow path.
    """
    import numpy as np
    from exp_navsim.visualize_dataset import trajectory_stats, MIN_DISPLACEMENT

    ds = _dataset()
    if "stats" not in _VIEWER:
        _cache_index()
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


def viewer_filter(min_distance=0.0, min_angle=0.0, min_frames=0):
    """Episode indices passing the three filters, same criteria as the PDF report.

    Mirrors visualize_dataset.py / test_model.py's EPISODE_FILTERS: distance
    driven, |angle between the initial heading and start->end|, episode length.
    All-zero filters simply return every episode.
    """
    import numpy as np

    s = _scan()
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
    episode, so "all" is available; viewer_prepare reports which side of the
    split each episode falls on, since a train episode was seen during training.

    Only the encoded mode is supported; raw mode would need the tokenizer.
    """
    import torch

    from exp_navsim.data.navsim_base import in_split
    from exp_navsim.model import NavsimTrajectoryModel

    if _VIEWER.get("ckpt") != (ckpt, split):
        _dataset()                           # binds cfg + cache to this config first
        cfg = _config(_OPTS["config"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NavsimTrajectoryModel.load_from_checkpoint(ckpt, map_location=device)
        if model.encode_images:
            raise RuntimeError("viewer supports encoded mode only (model.encode_images is set)")

        # The cache index is shared with the episode scan, so whichever runs
        # first pays for the walk and the other gets it free.
        val_fraction = float(cfg.data.params.validation.params.get("val_fraction", 0.1))
        token_to_file = {token: path for token, path in _cache_index().items()
                         if split == "all" or in_split(token, split, val_fraction)}
        _VIEWER.update(ckpt=(ckpt, split), model=model.to(device).eval(), device=device,
                       token_to_file=token_to_file, val_fraction=val_fraction)
        _EPISODES.clear()                    # states were built without predictions

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


def _window_batch(state, starts, total_len):
    """Build one padded batch holding the model window that starts at each frame.

    Each window is [start, start + total_len) of the cached episode, re-origined
    at its own first frame and rotated into the ego frame there — the frame the
    model was fitted in, since num_frames=0 makes _window_start return 0 and so
    every training window began at episode frame 0, where poses_to_local_traj
    leaves the ego heading along +x.

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
    with h5py.File(state["latent_file"], "r") as f:
        trajectory = f["trajectory"][:]
        frame_rate = int(f.attrs.get("frame_rate", 2))
        for start in starts:
            # Only encoded_q_sem is read: get_encoded_q uses nothing else, and this
            # runs once per displayed frame, so the skipped q_rec read matters.
            q_sem = f["encoded_q_sem"][start:start + min(context_images, total_len)]

            traj = np.asarray(trajectory[start:start + total_len], dtype=np.float32)
            traj = traj - traj[0]
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


def _predict(state, starts, total_len, num_samples):
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

    if "model" not in _VIEWER or state.get("latent_file") is None:
        return {}

    cache = state["preds"]
    todo = [s for s in starts if (total_len, s) not in cache]
    if todo:
        model = _VIEWER["model"]
        n = num_samples or int(model.num_val_samples)
        batch = _window_batch(state, todo, total_len)
        with torch.no_grad():
            preds = torch.stack([model.sample(batch) for _ in range(n)], 1)  # (B, N, T, 2)
        preds = preds.float().cpu().numpy()
        gt = state["bev_gt"]
        for j, start in enumerate(todo):
            # Undo the input rotation, then place the path at the ego position.
            heading = _heading_at(gt, start)
            cache[(total_len, start)] = (preds[j] #(_rotate_path(preds[j], heading)
                                         + gt[min(start, len(gt) - 1)])
    return {s: cache[(total_len, s)] for s in starts if (total_len, s) in cache}


def _max_start(state, total_len):
    """Last frame that gets a prediction — every frame, once a model is loaded.

    For the final context_traj-1 frames part of the model's input context is the
    padding _window_batch adds (a repeated pose, so zero velocity, and a short
    latent stack the denoiser already tolerates via min(shape[1], context)).
    Those forecasts are weaker rather than meaningless, and the heading they are
    placed at still comes from real motion, so they are drawn: seeing the model's
    last call is more useful than a blank panel at the end of every episode.
    """
    if _VIEWER.get("model") is None or state.get("latent_file") is None:
        return -1
    return len(state["frames"]) - 1


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
    # No predictions here — they are per-frame, drawn as overlays.
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


def _bev_overlay(state, i, total_len, num_samples):
    """The BEV bitmap with this frame's GT window, predictions and ego marker."""
    from PIL import ImageDraw

    panel = state["bev_img"].copy()
    affine, gt = state["bev_affine"], state["bev_gt"]
    draw = ImageDraw.Draw(panel, "RGBA")

    # Predicted distribution for this frame, translucent so overlap reads as density.
    # A failure degrades to a plain BEV rather than losing the frame, but it is
    # recorded — swallowing it silently looks exactly like "the model predicted
    # nothing", which is the one thing this panel exists to show.
    preds = None
    if i <= _max_start(state, total_len):
        try:
            preds = _predict(state, [i], total_len, num_samples).get(i)
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
# Per-episode state
# --------------------------------------------------------------------------- #
def _state(episode):
    """Per-episode render state, LRU-cached so neighbouring episodes stay warm.

    Holds the parsed frame list, the rasterised BEV with its data->pixel affine,
    the GT path, the episode's latent file and its prediction memo. Building it
    is the expensive part of showing an episode (the BEV render is ~a second), so
    keeping MAX_EPISODES of them is what lets the viewer buffer the episodes
    either side and make n/p instant.
    """
    from exp_navsim.data.navsim_base import SURROUND_CAMERAS, in_split

    if episode in _EPISODES:
        _EPISODES.move_to_end(episode)
        return _EPISODES[episode]

    ds = _dataset()
    frames, meta = ds._episode_frames(ds.episodes[episode])
    frames = frames[::ds.frame_interval]              # same subsampling as load_sample
    cams = (list(SURROUND_CAMERAS) if _OPTS["cameras"] == "surround"
            else [ds.front_camera])
    state = {
        "episode": episode, "token": meta["token"], "frames": frames, "cams": cams,
        "bev_img": None, "bev_affine": None, "bev_gt": None, "preds": {},
        # Only episodes with cached latents can be predicted.
        "latent_file": _VIEWER.get("token_to_file", {}).get(meta["token"]),
        "split": ("val" if in_split(meta["token"], "val", _VIEWER.get("val_fraction", 0.1))
                  else "train"),
        "note": "",
    }

    if _OPTS["bev"]:
        try:
            (state["bev_img"], state["bev_affine"],
             state["bev_gt"]) = _render_bev_base(episode, _OPTS["bev_size"])
            if state["bev_img"] is None:
                state["note"] = "no nuPlan map for this episode"
        except Exception as e:                        # missing maps / navsim
            state["note"] = f"BEV unavailable: {type(e).__name__}: {e}"
    if "model" not in _VIEWER:
        state["note"] = state["note"] or "no --ckpt given — BEV shows ground truth only"
    elif state["latent_file"] is None:
        state["note"] = state["note"] or "no cached latents for this episode — no prediction"

    _EPISODES[episode] = state
    while len(_EPISODES) > MAX_EPISODES:
        _EPISODES.popitem(last=False)                 # drop the least recently used
    return state


def viewer_prepare(episode):
    """Build (or reuse) an episode's state and report its frame geometry.

    The viewer calls this for the episodes it wants buffered, not just the one on
    screen, so switching to a neighbour needs no round trip at all.
    """
    ds = _dataset()
    state = _state(episode)
    h, w = ds.size                                    # per-camera crop, cameras side by side
    strip_w, strip_h = w * len(state["cams"]), h
    panel = state["bev_img"]
    width = max(strip_w, panel.width if panel is not None else 0)
    height = strip_h + (panel.height if panel is not None else 0)
    if _OPTS["width"]:                                # the frame is rescaled on encode
        height, width = round(height * _OPTS["width"] / width), _OPTS["width"]
    print(json.dumps({
        "episode": episode, "token": state["token"], "num_frames": len(state["frames"]),
        "num_episodes": len(ds), "cameras": state["cams"], "frame_rate": ds.frame_rate,
        "width": width, "height": height,
        "bev": panel is not None,
        "predicted": panel is not None and state["latent_file"] is not None,
        # Which side of the train/val split: predictions on a train episode were
        # fitted on that very trajectory.
        "split": state["split"], "note": state["note"],
    }))


# --------------------------------------------------------------------------- #
# Frame rendering
# --------------------------------------------------------------------------- #
def _camera_strip(state, i):
    """Frame `i` as a PIL image: the selected cameras concatenated left to right.

    Reuses NavsimLongBase._read_camera, so the resize / center-crop / scaling match
    the training pipeline exactly — what you watch is what the model is fed. It
    returns (3, H, W) in [-1, 1]; required=False tolerates the surround cameras
    that navtrain only partially downloads (they come back black).
    """
    import torch
    from PIL import Image

    ds, frame = _VIEWER["ds"], state["frames"][i]
    strip = torch.cat([ds._read_camera(frame, c, required=False)
                       for c in state["cams"]], dim=2)             # concat along width
    strip = ((strip.clamp(-1, 1) + 1) * 127.5).round().byte()      # [-1,1] -> uint8
    return Image.fromarray(strip.permute(1, 2, 0).numpy())         # CHW -> HWC


def _render(state, i, total_len, num_samples):
    """Composite frame `i`: the camera strip above the BEV panel."""
    from PIL import Image

    strip = _camera_strip(state, i)
    if state["bev_img"] is None:
        return strip
    panel = _bev_overlay(state, i, total_len, num_samples)

    width = max(strip.width, panel.width)
    out = Image.new("RGB", (width, strip.height + panel.height), (0, 0, 0))
    out.paste(strip, ((width - strip.width) // 2, 0))
    out.paste(panel, ((width - panel.width) // 2, strip.height))
    return out


def _encode(img, quality, width):
    """JPEG-encode, resizing to `width` first (0 = leave as composited).

    The width applies to the finished composite, so it scales in both directions.
    Asking for more than the composite's natural size only buys bytes, not
    detail — raise bev_size (and so the panel's own resolution) instead.
    """
    from PIL import Image

    if width and width != img.width:
        img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def viewer_frames(episode, start, stop, total_len=20, num_samples=0):
    """Emit frames [start, stop) of `episode` as display_data, one per frame.

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

    state = _state(episode)
    start, stop = max(start, 0), min(stop, len(state["frames"]))
    if state["bev_img"] is not None:
        limit = _max_start(state, total_len)
        batched = [i for i in range(start, stop) if i <= limit]
        if batched:
            try:
                _predict(state, batched, total_len, num_samples)
            except Exception as e:
                _VIEWER["pred_error"] = f"prediction failed: {type(e).__name__}: {e}"

    for i in range(start, stop):
        try:
            jpeg = _encode(_render(state, i, total_len, num_samples),
                           _OPTS["quality"], _OPTS["width"])
            display({"image/jpeg": base64.b64encode(jpeg).decode()},
                    raw=True, metadata={"viewer": {"episode": episode, "index": i}})
        except Exception as e:
            display({"text/plain": repr(e)}, raw=True,
                    metadata={"viewer": {"episode": episode, "index": i,
                                         "error": f"{type(e).__name__}: {e}"}})

    # Anything that went wrong without costing us a frame travels back on its own
    # message, so the viewer can show it in the status line.
    note = _VIEWER.pop("pred_error", None)
    if note:
        display({"text/plain": note}, raw=True, metadata={"viewer": {"note": note}})
