"""Run arguments -> features and planner, shared by the scripts that reload a v22 run.

    feats = feature_names(a)                 # ["wm11_ctx", "wm11_ctx_L0", ...], camera-major
    ds = SceneFeatureDataset(index, features=feats, feature_dir=feature_dir(a), ...)
    m = build_planner(a, C, T, vocab)        # C = channels of all cameras together

`a` is the args dict of a run (model.pt "args" or config.json). Runs from before the
multi-camera options have none of the new keys and get exactly the old model.
"""

import numpy as np


def cams_of(a):
    return (a.get("cams") or "F0").split(",")


def feature_names(a):
    """Cached feature names of every camera, in the order train.py concatenates them."""
    base = [f for f in a["features"].split(",") if f]
    return [f if cam == "F0" else f"{f}_{cam}" for cam in cams_of(a) for f in base]


def feature_dir(a):
    return a.get("feature_dir") or "data/features"


def bev_extent(a):
    """(x0, x1, y) of the BEV query grid: as saved by train.py, else from the targets file."""
    if a.get("bev_extent") is not None:
        return tuple(a["bev_extent"])
    from exp_navsim.v22 import bev_targets as bev_mod
    if a.get("bev_targets"):
        bt = np.load(a["bev_targets"])
        return float(bt["x0"]), float(bt["x0"]) + float(bt["cell"]) * int(bt["shape"][2]), float(bt["y"])
    return bev_mod.X0, bev_mod.X1, bev_mod.Y


def grid_of(a):
    """Feature grid of the run (saved in model.pt since the bottleneck ablation; before: 9 x 16)."""
    return tuple(a.get("grid") or (9, 16))


def camera_kwargs(a):
    """GaragePlanner kwargs of the multi-camera / BEV-query options ({} for a front-only run)."""
    cams = cams_of(a)
    kw = {}
    if len(cams) > 1:
        from exp_navsim.v22.cameras import CALIB, load_calib, token_rays
        kw.update(cams=len(cams), rays=token_rays(load_calib(a.get("calib") or CALIB), cams, grid_of(a)))
    if a.get("bev_queries"):
        kw.update(bev_queries=a["bev_queries"], bev_extent=bev_extent(a))
    return kw


def build_planner(a, C, T, vocab=None, **override):
    """The GaragePlanner of run args `a` (not loaded); C = channels of all cameras together."""
    from exp_navsim.v22.model import GaragePlanner
    kw = dict(grid=grid_of(a), dim=a["dim"], depth=a["depth"], heads=a["heads"], flow_depth=a["flow_depth"],
              registers=a["registers"], dropout=a["dropout"], vocab=vocab, groups=a["groups"].split(","),
              mask_path=not a["no_mask_path"], score_head=bool(a["score_targets"] or a.get("sim_targets")),
              use_image=bool(a["features"]), use_status=not a["no_status"], use_past=not a["no_past"],
              use_command=not a["no_command"], flow=not a["no_flow"], bev_shape=a.get("bev_shape"),
              goal=a.get("goal", "none") != "none", anchor_depth=a.get("anchor_depth", 2))
    kw.update(camera_kwargs(a))
    kw.update(override)
    return GaragePlanner(C // len(cams_of(a)), T, **kw)
