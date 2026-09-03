"""Remote half of the episode viewer — runs INSIDE the cluster's Jupyter kernel.

view_episode.py (on the local PC) reads this file's source and pushes it into the
kernel, then calls the entry points below over the Jupyter websocket:

    viewer_list(config)                             -> one JSON line
    viewer_scan(config)                             -> one JSON line (slow, once)
    viewer_filter(min_distance, min_angle, min_frames)  -> one JSON line
    viewer_model(ckpt, config)                      -> one JSON line
    viewer_open(config, episode, ...)               -> one JSON line
    viewer_frames(start, stop, quality, max_width)  -> JPEG display_data messages

Nothing here is imported on the local PC; it only has to be importable on the
cluster (orbis repo root on sys.path, navsim env active).

Everything expensive — the dataset index, the per-episode statistics, the loaded
checkpoint, the rendered BEV panel — is cached in the module-level _VIEWER dict,
which lives in the kernel between calls. That is the whole point of driving a
kernel instead of spawning a process per request.
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
    """Episode count, so the local side can validate --episode before opening."""
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
    """
    import numpy as np

    s = _scan(config)
    keep = ((s["distances"] >= min_distance)
            & (np.abs(s["steerings"]) >= min_angle)
            & (s["lengths"] >= min_frames))
    if min_angle > 0:
        keep &= s["moved"]
    episodes = np.flatnonzero(keep).tolist()
    print(json.dumps({"episodes": episodes, "total": len(s["lengths"])}))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def viewer_model(ckpt, config="exp_navsim/config.yaml"):
    """Load a checkpoint plus the validation latent dataset, cached in the kernel.

    The viewer indexes episodes through the *long* dataset while the model is fed
    from the *cached-latent* validation dataset, so a scene-token map bridges the
    two — exactly what test_model.py does with _episode_tokens.
    """
    import h5py
    import torch
    from omegaconf import OmegaConf

    from util import instantiate_from_config
    from exp_navsim.model import NavsimTrajectoryModel

    if _VIEWER.get("ckpt") != ckpt:
        cfg = OmegaConf.load(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NavsimTrajectoryModel.load_from_checkpoint(ckpt, map_location=device)
        latent_ds = instantiate_from_config(cfg.data.params.validation)
        token_to_latent = {}
        for i, path in enumerate(latent_ds.files):
            with h5py.File(path, "r") as f:
                token_to_latent[f.attrs.get("token", path.stem)] = i
        _VIEWER.update(ckpt=ckpt, model=model.to(device).eval(), device=device,
                       latent_ds=latent_ds, token_to_latent=token_to_latent)

    print(json.dumps({
        "ckpt": ckpt, "device": _VIEWER["device"],
        "val_episodes": len(_VIEWER["latent_ds"]),
        "num_samples": int(_VIEWER["model"].num_val_samples),
    }))


def _predict(token, num_samples):
    """(N, T, 2) sampled trajectories for one episode token, or None.

    None means "no prediction available": either no checkpoint is loaded, or this
    episode is not in the validation split (the latent cache only holds val
    episodes, while the viewer browses the whole long dataset).
    """
    import torch

    from exp_navsim.data.collate import pad_collate

    index = _VIEWER.get("token_to_latent", {}).get(token)
    if index is None:
        return None

    # pad_collate on a batch of one: same code path as training, so metadata
    # ["length"] rides along and we can strip the padded tail.
    batch = pad_collate([_VIEWER["latent_ds"][index]])
    batch = {k: (v.to(_VIEWER["device"]) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
    model = _VIEWER["model"]
    n = num_samples or int(model.num_val_samples)
    with torch.no_grad():
        preds = torch.stack([model.sample(batch)[0] for _ in range(n)], 0)  # (N, T, 2)
    length = int(batch["metadata"]["length"][0])
    return preds[:, :length].float().cpu().numpy()


# --------------------------------------------------------------------------- #
# BEV panel
# --------------------------------------------------------------------------- #
def _render_bev_panel(episode, size, num_samples):
    """Rasterise the episode's BEV once: map + agents + GT + predicted distribution.

    Drawing nuPlan lanes and agents costs on the order of a second, so it cannot
    happen per frame. Instead the panel is rendered once here and every frame only
    pastes it and stamps a marker at the ego's current position — which is why the
    data->pixel mapping of the GT path is recorded alongside the image.

    Uses Figure/FigureCanvasAgg directly rather than pyplot, so it cannot disturb
    the inline backend of a notebook sharing this kernel.

    Returns (PIL image, (T, 2) pixel coordinates of the GT path) or (None, None).
    """
    import numpy as np
    from PIL import Image
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from exp_navsim.data.bev_extract import extract_bev
    from exp_navsim.draw_bev import draw_bev_distribution

    ds = _VIEWER["ds"]
    # anchor=0: the GT path from poses_to_local_traj is rooted at episode frame 0
    # in both origin and heading, so BEV and path share one frame with no
    # transform (see exp_navsim/data/bev_extract.py).
    bev = extract_bev(ds, episode, anchor=0)
    if bev is None:
        return None, None
    gt = bev["trajectory"]
    preds = _predict(bev["token"], num_samples)

    dpi = 100
    fig = Figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])          # full-bleed: no wasted margin
    draw_bev_distribution(ax, bev, gt, preds, title=None)
    canvas.draw()

    image = Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3])
    # The BEV ax plots (y, x) and matplotlib's display origin is bottom-left,
    # while PIL's is top-left — hence the swap and the flip.
    pts = ax.transData.transform(np.column_stack([gt[:, 1], gt[:, 0]]))
    pixels = np.column_stack([pts[:, 0], image.height - pts[:, 1]])
    return image, pixels


# --------------------------------------------------------------------------- #
# Opening an episode / streaming frames
# --------------------------------------------------------------------------- #
def viewer_open(config="exp_navsim/config.yaml", episode=0, cameras="surround",
                bev=True, bev_size=512, num_samples=0):
    """Select one episode, render its BEV panel, and report the frame geometry.

    The camera frames themselves are NOT decoded here — _episode_frames only
    parses the pose pickle — so opening stays fast even for long episodes. The
    BEV panel (and with it the model inference) is the one up-front cost.
    """
    from exp_navsim.data.navsim_base import SURROUND_CAMERAS

    ds = _dataset(config)
    frames, meta = ds._episode_frames(ds.episodes[episode])
    frames = frames[::ds.frame_interval]             # same subsampling as load_sample
    names = list(SURROUND_CAMERAS) if cameras == "surround" else [ds.front_camera]
    _VIEWER.update(frames=frames, cams=names, episode=episode,
                   bev_img=None, bev_pix=None)

    note = ""
    if bev:
        try:
            _VIEWER["bev_img"], _VIEWER["bev_pix"] = _render_bev_panel(
                episode, bev_size, num_samples)
            if _VIEWER["bev_img"] is None:
                note = "no nuPlan map for this episode"
        except Exception as e:                       # missing maps / navsim / ckpt
            note = f"BEV unavailable: {type(e).__name__}: {e}"

    h, w = ds.size                                   # per-camera crop; cameras go side by side
    strip_w, strip_h = w * len(names), h
    panel = _VIEWER["bev_img"]
    print(json.dumps({
        "episode": episode, "token": meta["token"], "num_frames": len(frames),
        "num_episodes": len(ds), "cameras": names, "frame_rate": ds.frame_rate,
        "width": max(strip_w, panel.width if panel is not None else 0),
        "height": strip_h + (panel.height if panel is not None else 0),
        "bev": panel is not None,
        "predicted": panel is not None and "model" in _VIEWER,
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


def _render(i):
    """Composite frame `i`: the camera strip above the BEV panel.

    The BEV is the cached rasterisation, so the only per-frame work is pasting it
    and stamping the ego marker at this frame's position along the GT path.
    """
    from PIL import Image, ImageDraw

    strip = _camera_strip(i)
    panel = _VIEWER.get("bev_img")
    if panel is None:
        return strip

    panel = panel.copy()
    pixels = _VIEWER["bev_pix"]
    x, y = pixels[min(i, len(pixels) - 1)]
    r = 7
    ImageDraw.Draw(panel).ellipse([x - r, y - r, x + r, y + r],
                                  fill=(255, 220, 0), outline=(0, 0, 0), width=2)

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


def viewer_frames(start, stop, quality=80, max_width=0):
    """Emit frames [start, stop) as display_data, one message per frame.

    The frame index rides in the message metadata rather than in the image, so the
    local side can drop them straight into its cache. One message per frame (as
    opposed to one big reply) lets the viewer start showing a batch before the
    batch has finished rendering. A frame that fails to decode is reported in-band
    so one bad blob cannot stall playback.
    """
    from IPython.display import display

    for i in range(max(start, 0), min(stop, len(_VIEWER["frames"]))):
        try:
            jpeg = _encode(_render(i), quality, max_width)
            display({"image/jpeg": base64.b64encode(jpeg).decode()},
                    raw=True, metadata={"viewer": {"index": i}})
        except Exception as e:
            display({"text/plain": repr(e)}, raw=True,
                    metadata={"viewer": {"index": i, "error": f"{type(e).__name__}: {e}"}})
