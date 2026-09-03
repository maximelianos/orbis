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
def _dataset(config_path):
    """Build the long dataset, cached on the config path.

    Rebuilding navtrain's episode index costs tens of seconds, so it happens once
    per kernel, not once per seek.
    """
    from omegaconf import OmegaConf
    from exp_navsim.data.navsim_base import NavsimLongBase

    if _VIEWER.get("config_path") != config_path:
        _VIEWER["ds"] = NavsimLongBase.from_config(OmegaConf.load(config_path))
        _VIEWER["config_path"] = config_path
        _VIEWER.pop("stats", None)                  # stats belong to the old dataset
    return _VIEWER["ds"]


def viewer_list(config="exp_navsim/config.yaml"):
    """Episode count — the viewer browses range(num_episodes) until a filter runs."""
    ds = _dataset(config)
    print(json.dumps({"num_episodes": len(ds)}))


def _scan(config):
    """Cache length / distance / steering of every episode, for the GUI filters.

    Image-free (episode_trajectory only parses the pose pickles) but still a full
    pass over the dataset, so it is minutes on navtrain — hence cached, and only
    triggered when the user actually applies a filter. Prints nothing: callers
    that talk to the GUI must emit exactly one JSON line of their own.
    """
    import numpy as np
    from exp_navsim.visualize_dataset import trajectory_stats, MIN_DISPLACEMENT

    ds = _dataset(config)
    if "stats" not in _VIEWER:
        rows = [(len(t),) + trajectory_stats(t)
                for t in (ds.episode_trajectory(i) for i in range(len(ds)))]
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
    import h5py
    import torch
    from omegaconf import OmegaConf

    from util import instantiate_from_config
    from exp_navsim.model import NavsimTrajectoryModel

    if _VIEWER.get("ckpt") != (ckpt, split):
        cfg = OmegaConf.load(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NavsimTrajectoryModel.load_from_checkpoint(ckpt, map_location=device)
        if model.encode_images:
            raise RuntimeError("viewer supports encoded mode only (model.encode_images is set)")

        latent_cfg = OmegaConf.to_container(cfg.data.params.validation, resolve=True)
        latent_cfg["params"]["split"] = split
        latent_ds = instantiate_from_config(latent_cfg)
        # token -> h5 path, so a long-dataset episode can find its cached latents.
        token_to_file = {}
        for path in latent_ds.files:
            with h5py.File(path, "r") as f:
                token_to_file[f.attrs.get("token", path.stem)] = path
        _VIEWER.update(ckpt=(ckpt, split), model=model.to(device).eval(), device=device,
                       latent_key=latent_ds.latent_key, token_to_file=token_to_file,
                       val_fraction=float(latent_cfg["params"].get("val_fraction", 0.1)))

    model = _VIEWER["model"]
    print(json.dumps({
        "ckpt": ckpt, "device": _VIEWER["device"], "split": split,
        "cached_episodes": len(_VIEWER["token_to_file"]),
        "context_traj": int(model.context_traj),
        "context_images": int(model.context_images),
        "num_samples": int(model.num_val_samples),
    }))


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

    The model returns window-local paths starting at the origin. The cached
    trajectory is local to episode frame 0 in both origin *and* heading, and the
    windowing only translates it, so putting a prediction back into episode
    coordinates is a pure translation by the ego position at `start` — no
    rotation (see exp_navsim/data/bev_extract.py on why the heading matters).
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
            cache[(total_len, start)] = preds[j] + gt[min(start, len(gt) - 1)]
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
