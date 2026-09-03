"""Remote half of the episode viewer — runs INSIDE the cluster's Jupyter kernel.

view_episode.py (on the local PC) reads this file's source and pushes it into the
kernel, then calls the three entry points below over the websocket:

    viewer_list(config)                             -> prints one JSON line
    viewer_open(config, episode, cameras)           -> prints one JSON line
    viewer_frames(start, stop, quality, max_width)  -> emits JPEG display_data

Nothing here is imported on the local PC; it only has to be importable on the
cluster (orbis repo root on sys.path, navsim env active).

Everything expensive is cached in the module-level _VIEWER dict, which lives in
the kernel between calls — that is the point of using a kernel at all.
"""

import base64
import io
import json

_VIEWER = {}          # kernel-resident state: dataset + the open episode's frames


def _dataset(config_path):
    """Build the long dataset, cached on the config path.

    Rebuilding navtrain's episode index costs tens of seconds, so it must happen
    once per kernel, not once per seek.
    """
    from omegaconf import OmegaConf
    from exp_navsim.data.navsim_base import NavsimLongBase

    if _VIEWER.get("config_path") != config_path:
        _VIEWER["ds"] = NavsimLongBase.from_config(OmegaConf.load(config_path))
        _VIEWER["config_path"] = config_path
    return _VIEWER["ds"]


def viewer_list(config="exp_navsim/config.yaml"):
    """Episode count, so the local side can validate --episode before opening."""
    ds = _dataset(config)
    print(json.dumps({"num_episodes": len(ds)}))


def viewer_open(config="exp_navsim/config.yaml", episode=0, cameras="surround"):
    """Select one episode and report its geometry.

    Metadata only: _episode_frames parses the pose pickle but decodes no images,
    so opening a 1000-frame episode is still fast. Subsampling matches
    NavsimLongBase.load_sample so frame indices agree with the rest of the repo.
    """
    from exp_navsim.data.navsim_base import SURROUND_CAMERAS

    ds = _dataset(config)
    frames, meta = ds._episode_frames(ds.episodes[episode])
    frames = frames[::ds.frame_interval]
    names = list(SURROUND_CAMERAS) if cameras == "surround" else [ds.front_camera]
    _VIEWER.update(frames=frames, cams=names, episode=episode)

    h, w = ds.size                                  # per-camera crop, cameras go side by side
    print(json.dumps({
        "episode": episode, "token": meta["token"], "num_frames": len(frames),
        "num_episodes": len(ds), "width": w * len(names), "height": h,
        "cameras": names, "frame_rate": ds.frame_rate,
    }))


def _render(i):
    """Frame `i` as a PIL image: the selected cameras concatenated left to right.

    Reuses NavsimLongBase._read_camera so the resize / center-crop / scaling match
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


def _encode(img, quality, max_width):
    """JPEG-encode, downscaling first if the strip is wider than the viewer needs."""
    if max_width and img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def viewer_frames(start, stop, quality=80, max_width=0):
    """Emit frames [start, stop) as display_data, one message per frame.

    The frame index rides in the message metadata rather than in the image, so the
    local side can drop them straight into its cache. Streaming one message per
    frame (instead of one big reply) lets the viewer start showing the batch
    before the batch has finished rendering. A frame that fails to decode is
    reported in-band so one bad blob cannot stall playback.
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
