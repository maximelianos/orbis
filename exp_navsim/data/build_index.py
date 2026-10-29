"""Index of NAVSIM scenes in the official protocol: 4 history frames in, 8 future poses out.

    python -m exp_navsim.data.build_index --split navtrain
    python -m exp_navsim.data.build_index --split navhard

A NAVSIM scene is a window of 4 history + 10 future frames of a log, sliding by one
frame, named by the token of its current frame (index 3); a split is a scene filter
that keeps some of those windows (navsim/common/dataloader.py, filter_scenes). This
script rebuilds the windows the same way, keeps the tokens of the split, and stores
everything a probe needs without opening a log again:

    frames.csv-like table (one row per distinct history image):
        frame_path   CAM_F0 path relative to the sensor root
    per scene:
        token, log, city
        frame_ids    (4,)    rows of the frame table, oldest first, -1 = image missing
        future       (8, 3)  x, y, heading of the next 4 s at 2 Hz, in the current ego frame
        past         (4, 3)  the history poses in the same frame (the last one is 0)
        velocity     (2,)    ego velocity at the current frame (ego frame)
        acceleration (2,)    ego acceleration at the current frame
        command      (4,)    one-hot driving command (left, straight, right, unknown)

The future poses come from navsim's own SE2 conversion, the same one
Scene.get_future_trajectory uses, so they match what the PDM score compares against.
The logs are complete; only the camera download is partial, so a scene is kept only
when all 4 history front images are on disk.

navtest / navhard: the test splits go through navsim's own SceneLoader, which also
knows navhard's synthetic second-stage scenes. Their history images are written
straight to the local frame cache (resize_frames.py), 4 per scene, and the index
gets a "synthetic" flag. These need the test logs and the test front camera
(tools/download_navsim_test.sh).
"""

import argparse
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from nuplan.common.actor_state.state_representation import StateSE2

from exp_navsim.data.navsim_base import quat_to_yaw

NUM_HISTORY, NUM_FUTURE, NUM_POSES = 4, 10, 8
FRONT = "CAM_F0"
NAVSIM = Path("navsim")
DOWNLOAD = Path("/scratch/global/velikanm/navsim/download")
FILTERS = NAVSIM / "navsim/planning/script/config/common/train_test_split/scene_filter"
EVAL_SPLITS = {
    "navtest": dict(logs=DOWNLOAD / "test_navsim_logs/test", sensors=DOWNLOAD / "test_sensor_blobs/test",
                    filter=FILTERS / "navtest.yaml"),
    # Stage-1 navhard scenes are test-log scenes; the navhard download misses some of
    # their history images, the full test front camera has them.
    "navhard": dict(logs=DOWNLOAD / "test_navsim_logs/test", sensors=DOWNLOAD / "test_sensor_blobs/test",
                    filter=FILTERS / "navhard_two_stage.yaml",
                    synthetic_sensors=DOWNLOAD / "navhard_two_stage/sensor_blobs",
                    synthetic_scenes=DOWNLOAD / "navhard_two_stage/synthetic_scene_pickles"),
}
SPLITS = {
    "navtrain": dict(
        logs=NAVSIM / "download/trainval_navsim_logs/trainval",
        sensors=NAVSIM / "download/trainval_sensor_blobs/trainval",
        filter=NAVSIM / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml",
    ),
}


def _pose(frame):
    return (frame["ego2global_translation"][0], frame["ego2global_translation"][1],
            quat_to_yaw(frame["ego2global_rotation"]))


def scene_record(frames, log):
    """One 14-frame window -> the record of its current frame (see the module doc)."""
    cur = NUM_HISTORY - 1
    poses = np.array([_pose(f) for f in frames], dtype=np.float64)
    origin = StateSE2(*poses[cur])
    future = convert_absolute_to_relative_se2_array(origin, poses[cur + 1:cur + 1 + NUM_POSES])
    past = convert_absolute_to_relative_se2_array(origin, poses[:NUM_HISTORY])
    dyn = np.asarray(frames[cur]["ego_dynamic_state"], dtype=np.float32)
    cams = [f["cams"].get(FRONT) or f["cams"][FRONT.lower()] for f in frames[:NUM_HISTORY]]
    return {
        "token": frames[cur]["token"],
        "log": log,
        "city": frames[cur]["map_location"],
        "frame_paths": [c["data_path"] for c in cams],
        "future": future.astype(np.float32),
        "past": past.astype(np.float32),
        "velocity": dyn[:2],
        "acceleration": dyn[2:4],
        "command": np.asarray(frames[cur]["driving_command"], dtype=np.float32),
    }


def _navtrain_log(args):
    """Records of the navtrain scenes of one log; images not checked yet."""
    path, tokens = args
    with open(path, "rb") as f:
        frames = pickle.load(f)
    n = NUM_HISTORY + NUM_FUTURE
    out = []
    for i in range(len(frames) - n + 1):
        window = frames[i:i + n]
        if window[NUM_HISTORY - 1]["token"] in tokens and len(window[NUM_HISTORY - 1]["roadblock_ids"]):
            out.append(scene_record(window, path.stem))
    return out


def build(split, workers):
    cfg = SPLITS[split]
    with open(cfg["filter"]) as f:
        scene_filter = yaml.load(f, Loader=yaml.CSafeLoader)
    tokens = set(scene_filter.get("tokens") or [])
    logs = [cfg["logs"] / f"{name}.pkl" for name in scene_filter["log_names"]]
    jobs = [(p, tokens) for p in logs if p.exists()]
    with Pool(workers) as pool:
        records = [r for rs in tqdm(pool.imap(_navtrain_log, jobs, chunksize=4), total=len(jobs)) for r in rs]

    # Frame table: one row per distinct history image that is on disk.
    present_cache = {}

    def present(rel):
        d = cfg["sensors"] / os.path.dirname(rel)
        if d not in present_cache:
            present_cache[d] = set(os.listdir(d)) if d.is_dir() else set()
        return os.path.basename(rel) in present_cache[d]

    frame_ids, frame_paths = {}, []
    for r in records:
        ids = []
        for rel in r.pop("frame_paths"):
            if not present(rel):
                ids.append(-1)
                continue
            if rel not in frame_ids:
                frame_ids[rel] = len(frame_paths)
                frame_paths.append(rel)
            ids.append(frame_ids[rel])
        r["frame_ids"] = np.asarray(ids, dtype=np.int64)
    complete = [r for r in records if (r["frame_ids"] >= 0).all()]
    print(f"[{split}] scenes in filter: {len(tokens)}, rebuilt: {len(records)}, "
          f"with all {NUM_HISTORY} history images: {len(complete)}, distinct images: {len(frame_paths)}")

    stack = lambda key, dtype=None: np.stack([r[key] for r in complete]).astype(dtype or np.float32)
    index = {
        "split": split,
        "sensor_root": str(cfg["sensors"]),
        "frame_paths": frame_paths,
        "token": [r["token"] for r in complete],
        "log": [r["log"] for r in complete],
        "city": [r["city"] for r in complete],
        "frame_ids": stack("frame_ids", np.int64),
        "future": stack("future"),
        "past": stack("past"),
        "velocity": stack("velocity"),
        "acceleration": stack("acceleration"),
        "command": stack("command"),
    }
    return index


def build_eval(split, workers):
    """navtest / navhard through navsim's SceneLoader; images go to the frame cache."""
    from concurrent.futures import ThreadPoolExecutor
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from PIL import Image
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    from exp_navsim.data.resize_frames import SIZE, frame_file

    cfg = EVAL_SPLITS[split]
    sensors = SensorConfig.build_no_sensors()
    sensors.cam_f0 = list(range(NUM_HISTORY))
    loader = SceneLoader(data_path=cfg["logs"], original_sensor_path=cfg["sensors"],
                         scene_filter=instantiate(OmegaConf.load(cfg["filter"])),
                         synthetic_sensor_path=cfg.get("synthetic_sensors"),
                         synthetic_scenes_path=cfg.get("synthetic_scenes"), sensor_config=sensors)
    tokens = loader.tokens
    synthetic = set(getattr(loader, "synthetic_scenes_tokens", ()))

    def one(i):
        token = tokens[i]
        try:
            scene = loader.get_scene_from_token(token)
        except FileNotFoundError as e:
            print(f"[{split}] skip {token}: {e}")
            return None
        ids = []
        for h in range(NUM_HISTORY):
            image = scene.frames[h].cameras.cam_f0.image
            if image is None:
                ids.append(-1)
                continue
            dst = frame_file(split, NUM_HISTORY * i + h)
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).resize(SIZE, Image.BILINEAR, reducing_gap=2.0).save(dst, quality=95)
            ids.append(NUM_HISTORY * i + h)
        status = scene.get_agent_input().ego_statuses
        try:
            future = scene.get_future_trajectory(NUM_POSES).poses.astype(np.float32)
        except Exception:                      # synthetic scenes may stop at the current frame
            future = np.full((NUM_POSES, 3), np.nan, dtype=np.float32)
        return {
            "token": token, "log": scene.scene_metadata.log_name, "city": scene.scene_metadata.map_name,
            "synthetic": token in synthetic, "frame_ids": np.asarray(ids, dtype=np.int64), "future": future,
            "past": np.stack([s.ego_pose for s in status]).astype(np.float32),
            "velocity": status[-1].ego_velocity.astype(np.float32),
            "acceleration": status[-1].ego_acceleration.astype(np.float32),
            "command": np.asarray(status[-1].driving_command, dtype=np.float32),
        }

    with ThreadPoolExecutor(workers) as pool:
        records = list(tqdm(pool.map(one, range(len(tokens))), total=len(tokens)))
    print(f"[{split}] skipped scenes: {sum(r is None for r in records)}")
    records = [r for r in records if r is not None]
    # A missing history image takes the nearest image of the same scene, so every
    # test scene still gets a prediction (the score needs one per token).
    for r in records:
        ids = r["frame_ids"]
        ok = np.flatnonzero(ids >= 0)
        if len(ok) and len(ok) < len(ids):
            r["frame_ids"] = ids[ok[np.abs(np.arange(len(ids))[:, None] - ok[None]).argmin(1)]]
    print(f"[{split}] scenes: {len(records)}, synthetic: {sum(r['synthetic'] for r in records)}, "
          f"missing images: {sum((r['frame_ids'] < 0).any() for r in records)}")
    stack = lambda key: np.stack([r[key] for r in records])
    return {
        "split": split, "sensor_root": None,
        "frame_paths": [None] * (NUM_HISTORY * len(records)),
        "token": [r["token"] for r in records], "log": [r["log"] for r in records],
        "city": [r["city"] for r in records], "synthetic": np.array([r["synthetic"] for r in records]),
        "frame_ids": stack("frame_ids"), "future": stack("future"), "past": stack("past"),
        "velocity": stack("velocity"), "acceleration": stack("acceleration"), "command": stack("command"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS) + list(EVAL_SPLITS), default="navtrain")
    ap.add_argument("--out", default="data/navsim_index")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    index = (build if args.split in SPLITS else build_eval)(args.split, args.workers)
    out = Path(args.out) / f"{args.split}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(index, f)
    cities, counts = np.unique(index["city"], return_counts=True)
    print("cities:", dict(zip(cities, counts)))
    print("commands:", index["command"].sum(0))
    print(f"logs: {len(set(index['log']))}  -> {out}")


if __name__ == "__main__":
    main()
