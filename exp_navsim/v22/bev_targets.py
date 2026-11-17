"""Coarse BEV targets of every navtrain scene: road, walkway and agent occupancy.

    python -m exp_navsim.v22.bev_targets --split navtrain --workers 8
    python -m exp_navsim.v22.bev_targets --split navtrain --scenes 2000 --workers 4   # timing probe

TransFuser supervises its encoder with a BEV semantic map and agent boxes, and the
py123d_garage recipe keeps that as the pretraining stage. Our image encoder is frozen, so
the only thing an auxiliary BEV loss can shape is the small transformer that reads the
cached grid — but that is also what the plan and the metric head read, and road layout is
exactly what the stage-two drivable-area score is missing.

The target is a 3 x H x W occupancy grid in the current ego frame, ahead of the car:

    road      LANE and INTERSECTION polygons of the nuPlan map
    walkway   WALKWAYS polygons (what the car must not enter)
    agents    the annotated vehicle, pedestrian and static-object boxes

It needs no simulation, only the log frame and the nuPlan map, so it can be built for all
92k training scenes, unlike the PDM sub-scores of the metric head (10k scenes).

-> data/bev_targets/<split>.npz: rows (index rows), bits (packed uint8), grid parameters.
"""

import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from exp_navsim.data.bev_extract import get_map_api
from exp_navsim.data.scenes import load_index

OUT = Path("data/bev_targets")
DOWNLOAD = Path("/scratch/global/velikanm/navsim/download")
LOGS = {"navtrain": DOWNLOAD / "trainval_navsim_logs/trainval",
        "navhard": DOWNLOAD / "test_navsim_logs/test",
        "navtest": DOWNLOAD / "test_navsim_logs/test"}
# grid in the ego frame: x forward in [X0, X1), y left in [-Y, Y), one cell = CELL metres
X0, X1, Y, CELL = 0.0, 64.0, 24.0, 2.0
H, W = int(2 * Y / CELL), int((X1 - X0) / CELL)        # rows = lateral, cols = longitudinal
CHANNELS = ("road", "walkway", "agents")
_STATE = {}


def _layers():
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    return {"road": [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION],
            "walkway": [SemanticMapLayer.WALKWAYS]}


def _rasterize(points):
    """Ego-frame polygon (N, 2) -> pixel polygon of the grid, or None if it is outside."""
    col = (points[:, 0] - X0) / CELL
    row = (Y - points[:, 1]) / CELL
    if col.max() < 0 or col.min() > W or row.max() < 0 or row.min() > H:
        return None
    return np.stack([col, row], -1).reshape(-1, 1, 2).astype(np.int32)


def _scene_grid(frame, map_api):
    """-> (3, H, W) uint8 occupancy of one log frame."""
    from nuplan.common.actor_state.state_representation import StateSE2

    pose = StateSE2(*_STATE["pose"](frame))
    c, s = np.cos(-pose.heading), np.sin(-pose.heading)
    rot = np.array([[c, -s], [s, c]])
    out = np.zeros((len(CHANNELS), H, W), np.uint8)
    radius = float(np.hypot(X1, Y)) + 10.0
    objects = map_api.get_proximal_map_objects(point=pose.point, radius=radius,
                                               layers=[l for ls in _layers().values() for l in ls])
    for ci, (name, layers) in enumerate(_layers().items()):
        for layer in layers:
            for obj in objects.get(layer, []):
                ext = np.asarray(obj.polygon.exterior.coords)
                poly = _rasterize((ext - np.array([pose.x, pose.y])) @ rot.T)
                if poly is not None:
                    cv2.fillPoly(out[ci], [poly], 1)
    boxes = frame.get("anns", {}).get("gt_boxes")
    if boxes is not None and len(boxes):
        boxes = np.asarray(boxes, np.float64)
        for b in boxes:
            # box = (x, y, z, length, width, height, yaw), already in the ego frame
            bx, by, length, width, h = b[0], b[1], b[3], b[4], b[-1]
            ch, sh = np.cos(h), np.sin(h)
            corners = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                                [-length / 2, -width / 2], [-length / 2, width / 2]])
            corners = corners @ np.array([[ch, sh], [-sh, ch]]) + np.array([bx, by])
            poly = _rasterize(corners)
            if poly is not None:
                cv2.fillPoly(out[2], [poly], 1)
    return out


def _init(split, index):
    _STATE["split"], _STATE["index"] = split, index
    _STATE["logs"] = {}
    _STATE["pose"] = lambda f: (f["ego2global_translation"][0], f["ego2global_translation"][1],
                                _yaw(f["ego2global_rotation"]))


def _yaw(q):
    from exp_navsim.data.navsim_base import quat_to_yaw
    return float(quat_to_yaw(np.asarray(q)))


def _log_frames(log):
    if log not in _STATE["logs"]:
        _STATE["logs"].clear()                                    # one log at a time, rows are sorted
        with open(LOGS[_STATE["split"]] / f"{log}.pkl", "rb") as f:
            frames = pickle.load(f)
        _STATE["logs"][log] = {f["token"]: f for f in frames}
    return _STATE["logs"][log]


def _one(row):
    ix = _STATE["index"]
    token, log = str(ix["token"][row]), str(ix["log"][row])
    try:
        frame = _log_frames(log)[token]
        grid = _scene_grid(frame, get_map_api(frame["map_location"]))
    except Exception as e:                                        # missing map or frame
        return row, None, f"{type(e).__name__}: {e}"
    return row, np.packbits(grid), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtrain")
    ap.add_argument("--scenes", type=int, default=0, help="0 = every scene of the split")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ix = load_index(f"data/navsim_index/{args.split}.pkl")
    rows = np.arange(len(ix["token"]))
    if args.scenes:
        rows = np.sort(np.random.default_rng(0).choice(rows, min(args.scenes, len(rows)), replace=False))
    # group by log so each worker keeps one log pickle in memory
    order = np.argsort(np.asarray(ix["log"])[rows], kind="stable")
    rows = rows[order]
    print(f"{len(rows)} scenes, grid {len(CHANNELS)} x {H} x {W} at {CELL} m", flush=True)

    out_rows, bits, errors = [], [], {}
    import time
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.split, ix)) as pool:
        for i, (row, packed, err) in enumerate(pool.imap(_one, rows, chunksize=32)):
            if err is not None:
                errors[err] = errors.get(err, 0) + 1
                continue
            out_rows.append(row)
            bits.append(packed)
            if (i + 1) % 5000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"{i + 1}/{len(rows)}  {rate:.0f} scenes/s  "
                      f"eta {(len(rows) - i - 1) / rate / 60:.0f} min", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.split}.npz"
    order = np.argsort(out_rows)
    np.savez_compressed(path, rows=np.asarray(out_rows)[order], bits=np.stack(bits)[order],
                        shape=np.array([len(CHANNELS), H, W]), cell=CELL, x0=X0, y=Y)
    occ = np.unpackbits(np.stack(bits), axis=1)[:, :len(CHANNELS) * H * W].reshape(-1, len(CHANNELS), H, W)
    print(f"-> {path}  {len(out_rows)} scenes, {path.stat().st_size / 1e6:.0f} MB, "
          f"occupancy " + "  ".join(f"{c} {occ[:, i].mean():.3f}" for i, c in enumerate(CHANNELS)))
    if errors:
        print("errors:", errors)


if __name__ == "__main__":
    main()


def check(split="navtrain", n=4000):
    """Sanity check: the logged future should lie on the road channel.

        python -c "from exp_navsim.v22.bev_targets import check; check()"
    """
    d = np.load(OUT / f"{split}.npz")
    shape = tuple(int(v) for v in d["shape"])
    grids = np.unpackbits(d["bits"][:n], axis=1)[:, :int(np.prod(shape))].reshape(-1, *shape)
    ix = load_index(f"data/navsim_index/{split}.pkl")
    fut = np.asarray(ix["future"])[d["rows"][:n]]
    col = ((fut[..., 0] - X0) / CELL).astype(int)
    row = ((Y - fut[..., 1]) / CELL).astype(int)
    inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    on = grids[np.arange(len(grids))[:, None].repeat(fut.shape[1], 1)[inside], 0,
               row[inside], col[inside]]
    print(f"{len(grids)} scenes: {inside.mean():.3f} of the future poses are inside the grid, "
          f"{on.mean():.3f} of those fall on a road cell")
    print("occupancy " + "  ".join(f"{c} {grids[:, i].mean():.3f}" for i, c in enumerate(CHANNELS)))
