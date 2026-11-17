"""History image paths of any of the 8 cameras for the scenes of an index.

    python -m exp_navsim.encoders.camera_paths --split navhard --cams L0,R0,B0   # count missing files

The indices of build_index.py only keep the front camera (frame_paths for navtrain,
pre-resized copies for navhard). The other cameras are looked up again in the logs:

    navtrain       log pkl of the scene; the frame whose CAM_F0 path is the history image
                   gives the path of the other camera at the same time
    navhard real   test log pkl; the 4 frames ending at the scene token
    navhard synth  synthetic scene pickle (named by scene token, indexed by initial token)
    other splits   a per-camera index data/navsim_index/<split>_<cam>.pkl with the same rows, if it
                   exists (SimScale: simscale/stream.py index --cams F0,L0,...)

history_paths(index, cam, rows) -> list of 4 absolute paths per scene (oldest first).
A missing image takes the nearest image of the same scene that exists (as for the front
camera in build_index.build_eval); a scene without any image of that camera gets None
entries and a blank image in cache_features.
"""

import argparse
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from exp_navsim.data.build_index import EVAL_SPLITS, NUM_HISTORY, SPLITS

CAMS = ("F0", "L0", "L1", "L2", "R0", "R1", "R2", "B0")


def cam_name(cam):
    """'L0' or 'CAM_L0' -> 'CAM_L0'."""
    cam = cam.upper()
    return cam if cam.startswith("CAM_") else f"CAM_{cam}"


def cam_short(cam):
    return cam_name(cam)[4:]


def _cam_entry(cams, cam):
    return cams.get(cam) or cams.get(cam.lower()) or {}


def _navtrain_log(job):
    """(log pkl, cam) -> {CAM_F0 data_path: cam data_path} of every frame of the log."""
    path, cam = job
    with open(path, "rb") as f:
        frames = pickle.load(f)
    return {_cam_entry(fr["cams"], "CAM_F0").get("data_path"): _cam_entry(fr["cams"], cam).get("data_path")
            for fr in frames}


def _navhard_real(job):
    """(log pkl, tokens, cam) -> {token: 4 relative paths}."""
    path, tokens, cam = job
    with open(path, "rb") as f:
        frames = pickle.load(f)
    pos = {fr["token"]: i for i, fr in enumerate(frames)}
    out = {}
    for t in tokens:
        i = pos[t]
        out[t] = [_cam_entry(fr["cams"], cam).get("data_path") for fr in frames[i - NUM_HISTORY + 1:i + 1]]
    return out


def _navhard_synthetic(job):
    """(scene pickle, cam) -> (initial token, 4 relative paths)."""
    path, cam = job
    with open(path, "rb") as f:
        s = pickle.load(f)
    paths = [_cam_entry(fr["camera_dict"], cam).get("data_path") for fr in s["frames"][:NUM_HISTORY]]
    return s["scene_metadata"]["initial_token"], paths


class Present:
    """File existence by one listdir per directory (network storage, no recursive scans)."""

    def __init__(self):
        self.cache = {}

    def __call__(self, path):
        d = os.path.dirname(path)
        if d not in self.cache:
            self.cache[d] = set(os.listdir(d)) if os.path.isdir(d) else set()
        return os.path.basename(path) in self.cache[d]


def _resolve(rel, roots, present):
    """Relative paths of a scene -> absolute paths; missing -> nearest present, else None."""
    out = [None] * len(rel)
    for i, r in enumerate(rel):
        for root in roots:
            if r is not None and present(str(root / r)):
                out[i] = str(root / r)
                break
    ok = [i for i, p in enumerate(out) if p is not None]
    missing = len(rel) - len(ok)
    if ok and missing:
        out = [out[min(ok, key=lambda j: abs(j - i))] for i in range(len(rel))]
    return out, missing


def history_paths(index, cam, rows=None, workers=16):
    """-> (paths: list of 4 absolute paths or None per row, missing: (len(rows),) count of
    history images that were not on disk before the nearest-image fill)."""
    cam = cam_name(cam)
    split = index["split"]
    rows = np.arange(len(index["token"])) if rows is None else np.asarray(rows)
    present = Present()
    side = Path("data/navsim_index") / f"{split}_{cam_short(cam)}.pkl"
    if split not in SPLITS and split not in EVAL_SPLITS and side.exists():
        with open(side, "rb") as f:
            s = pickle.load(f)
        assert list(s["token"]) == list(index["token"]), f"{side} has other rows than the {split} index"
        rel = [[s["frame_paths"][f] if f >= 0 else None for f in s["frame_ids"][r]] for r in rows]
        roots = [Path(s["sensor_root"])]
    elif split in SPLITS:
        cfg = SPLITS[split]
        logs = sorted(set(np.asarray(index["log"])[rows]))
        with Pool(workers) as pool:
            lookup = {}
            for m in pool.imap_unordered(_navtrain_log, [(cfg["logs"] / f"{l}.pkl", cam) for l in logs]):
                lookup.update(m)
        rel = [[lookup.get(index["frame_paths"][f]) for f in index["frame_ids"][r]] for r in rows]
        roots = [Path(cfg["sensors"])]
    else:
        cfg = EVAL_SPLITS[split]
        tokens = np.asarray(index["token"])[rows]
        synthetic = np.asarray(index.get("synthetic", np.zeros(len(index["token"]), bool)))[rows]
        found = {}
        real = {}
        for t, l in zip(tokens[~synthetic], np.asarray(index["log"])[rows][~synthetic]):
            real.setdefault(l, []).append(t)
        jobs = [(cfg["logs"] / f"{l}.pkl", ts, cam) for l, ts in real.items()]
        with Pool(workers) as pool:
            for m in pool.imap_unordered(_navhard_real, jobs):
                found.update(m)
            if synthetic.any():
                want = set(tokens[synthetic])
                pkls = sorted(Path(cfg["synthetic_scenes"]).iterdir())
                for t, p in pool.imap_unordered(_navhard_synthetic, [(p, cam) for p in pkls], chunksize=16):
                    if t in want:
                        found[t] = p
        rel = [found.get(t, [None] * NUM_HISTORY) for t in tokens]
        # real navhard scenes: the navhard download has all cameras of some of them, the
        # test download only the front camera
        roots = [Path(cfg["synthetic_sensors"]), Path(cfg["sensors"])] if "synthetic_sensors" in cfg \
            else [Path(cfg["sensors"])]
    paths, missing = zip(*[_resolve(r, roots, present) for r in rel])
    return list(paths), np.asarray(missing)


def main():
    from exp_navsim.data.scenes import load_index
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navhard")
    ap.add_argument("--cams", default="L0,R0,B0")
    ap.add_argument("--max-scenes", type=int, default=None)
    args = ap.parse_args()
    index = load_index(f"data/navsim_index/{args.split}.pkl")
    rows = None if args.max_scenes is None else np.arange(args.max_scenes)
    syn = np.asarray(index.get("synthetic", np.zeros(len(index["token"]), bool)))
    syn = syn if rows is None else syn[rows]
    for cam in args.cams.split(","):
        paths, missing = history_paths(index, cam, rows)
        none = np.array([p[0] is None for p in paths])
        for name, m in (("real", ~syn), ("synthetic", syn)):
            if m.any():
                print(f"[{args.split} {cam_name(cam)} {name}] scenes {m.sum()}, missing history images "
                      f"{missing[m].sum()} / {NUM_HISTORY * m.sum()}, scenes with any missing "
                      f"{(missing[m] > 0).sum()}, scenes with none on disk {none[m].sum()}", flush=True)


if __name__ == "__main__":
    main()
