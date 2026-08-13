"""Extract the NAVSIM birds-eye-view context (map + agents) and GT trajectory.

    python -m exp_navsim.data.bev_extract --config exp_navsim/config.yaml --num 3

This is an independent, minimal version of what `visualize_cv_agent.py` does via
the NAVSIM devkit `SceneLoader`: instead of building a full `Scene` (which loads
cameras and LiDAR blobs), we build only the two objects
`navsim.visualization.bev.add_configured_bev_on_ax` actually needs — a nuPlan
`map_api` and one navsim `Frame` — straight from the raw log frame dicts that the
long dataloaders already hold. Nothing here touches sensor blobs, so it is cheap.

Coordinate frames (the reason the anchor frame matters)
------------------------------------------------------
* The BEV is rendered in the **ego frame of the anchor frame**: annotation boxes
  are already ego-local in the logs, and the map is transformed into that frame by
  `add_map_to_bev_ax` using the anchor's global pose. x is forward, y is left.
* `NavsimLongBase` builds `batch["trajectory"]` with `poses_to_local_traj`, i.e.
  local to **episode frame 0** (both origin and heading). The windowed datasets
  re-origin with `traj - traj[0]`, which is a pure translation, so the trajectory
  is heading-aligned to frame 0 only when the window starts at 0.

So with the default `anchor=0` — which is what the deterministic validation
windows (`deterministic: true` -> start 0) and `test_model.py` use — the batch
trajectories and the BEV share one frame and can be plotted with no transform.
Pass a different `anchor` only together with a window that starts there.

Requires the nuPlan maps: NUPLAN_MAPS_ROOT (and NUPLAN_MAP_VERSION). If they are
missing, `extract_bev` returns None with a warning rather than raising, so callers
can degrade to the mapless plots.
"""

import os

import numpy as np

from exp_navsim.data.navsim_base import poses_from_frames, poses_to_local_traj

MAP_VERSION = os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

# Loading a map api is expensive and there are only a handful of locations, so
# every episode of the same city reuses one instance.
_MAP_API_CACHE = {}


def get_map_api(map_name):
    """Cached nuPlan map api for a `map_location` string (e.g. us-nv-las-vegas-strip)."""
    if map_name not in _MAP_API_CACHE:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

        maps_root = os.environ.get("NUPLAN_MAPS_ROOT")
        if not maps_root:
            raise RuntimeError("NUPLAN_MAPS_ROOT is not set — cannot load nuPlan maps")
        _MAP_API_CACHE[map_name] = get_maps_api(maps_root, MAP_VERSION, map_name)
    return _MAP_API_CACHE[map_name]


def frame_to_navsim_frame(raw):
    """Build the minimal navsim `Frame` that the BEV renderer consumes.

    Only `ego_status.ego_pose` (map origin) and `annotations` (ego-local boxes) are
    read by `add_configured_bev_on_ax` under the default BEV layers
    ("map", "annotations"). Sensors are set to empty objects, so no blob is read.
    """
    from pathlib import Path

    from navsim.common.dataclasses import Cameras, Frame, Lidar, Scene

    empty = Path(".")   # never read: the empty sensor_names below skip all I/O
    return Frame(
        token=raw["token"],
        timestamp=raw["timestamp"],
        roadblock_ids=raw["roadblock_ids"],
        traffic_lights=raw["traffic_lights"],
        # Reuse navsim's own builders so we stay in sync with the log format.
        annotations=Scene._build_annotations(raw),
        ego_status=Scene._build_ego_status(raw),
        lidar=Lidar.from_paths(empty, Path(raw["lidar_path"]), []),
        cameras=Cameras.from_camera_dict(empty, raw["cams"], []),
    )


def extract_bev(long_ds, episode_idx, anchor=0):
    """BEV context + GT trajectory for one episode of a long dataset.

    Args:
        long_ds: a `NavsimLongBase` (navhard or navtrain).
        episode_idx: index into `long_ds`.
        anchor: index into the *subsampled* episode used as the BEV origin.
            Keep 0 unless the trajectory window also starts there (see module doc).

    Returns:
        dict with
            "map_api"       nuPlan map interface for this episode's location
            "frame"         navsim Frame at the anchor (ego-local annotations)
            "trajectory"    (T, 2) GT ego path, local to the anchor frame
            "map_name"      map_location string
            "token"         episode token (matches the latent cache `token` attr)
            "ego_speed"     |ego velocity| at the anchor, m/s
        or None if the maps are unavailable.
    """
    frames, meta = long_ds._episode_frames(long_ds.episodes[episode_idx])
    frames = frames[:: long_ds.frame_interval]
    raw = frames[anchor]

    # GT path in the anchor's frame: `poses_to_local_traj` roots the path at its
    # first pose (origin *and* heading), so slicing from the anchor is all we need.
    poses = poses_from_frames(frames)                       # (T, 3) global [x, y, yaw]
    trajectory = poses_to_local_traj(poses[anchor:])         # (T - anchor, 2)

    try:
        map_api = get_map_api(raw["map_location"])
    except Exception as e:                                   # missing/corrupt maps
        print(f"[bev_extract] no map for {meta['token']}: {e}")
        return None

    ego_vel = np.asarray(raw["ego_dynamic_state"][:2], dtype=np.float32)
    return {
        "map_api": map_api,
        "frame": frame_to_navsim_frame(raw),
        "trajectory": trajectory,
        "map_name": raw["map_location"],
        "token": meta["token"],
        "ego_speed": float(np.linalg.norm(ego_vel)),
    }


def extract_bev_for_tokens(long_ds, tokens, anchor=0, token_to_idx=None):
    """`extract_bev` for a list of episode tokens; entries are None where it failed.

    Pass `token_to_idx` to reuse a token->episode map the caller already built.
    """
    if token_to_idx is None:
        token_to_idx = {long_ds.episode_token(i): i for i in range(len(long_ds))}
    out = []
    for token in tokens:
        idx = token_to_idx.get(token)
        out.append(None if idx is None else extract_bev(long_ds, idx, anchor))
    return out


def _main():
    """Smoke check: extract a few BEVs and report what came out."""
    import argparse

    from omegaconf import OmegaConf

    from exp_navsim.data.navsim_base import NavsimLongBase

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_navsim/config.yaml")
    ap.add_argument("--num", type=int, default=3)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    ds = NavsimLongBase.from_config(cfg, load_surround=False)
    print(f"episodes: {len(ds)}")
    for i in range(min(args.num, len(ds))):
        bev = extract_bev(ds, i)
        if bev is None:
            print(f"[{i}] no BEV")
            continue
        print(f"[{i}] token={bev['token']} map={bev['map_name']} "
              f"speed={bev['ego_speed']:.2f} m/s "
              f"traj={bev['trajectory'].shape} boxes={len(bev['frame'].annotations.boxes)}")


if __name__ == "__main__":
    _main()
