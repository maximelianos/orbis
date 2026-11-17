"""Resumable download of the SimScale history cameras, kept only as 512 x 288 JPEGs.

    # 1. scene records of a round from its meta_datas.tar.gz (one pass, no extraction)
    python -m exp_navsim.simscale.stream meta --round pdm_v1.0-0
    # 2. shards: download, keep the 4 history frames of the chosen cameras, delete the shard
    python -m exp_navsim.simscale.stream shards --round pdm_v1.0-0 --shards 0-65 --cams F0
    # 3. scene index in the navtrain format + frame links for cache_features
    python -m exp_navsim.simscale.stream index --round pdm_v1.0-0 --split simscale_pdm0 --cams F0,L0,R0,B0

Release layout (HF dataset OpenDriveLab/SimScale, SimScale_data/synthetic_reaction_<round>/):
    simscale_<round>_meta_datas.tar.gz      one OpenScene log pkl per synthetic scene, 12 frames at
                                            2 Hz (4 history + 8 future), all frames simulated
    simscale_<round>_sensor_blobs_hist/simscale_<round>_sensor_blobs_hist_<i>.tar.gz
                                            1000 scenes per shard, <log_token>/<CAM>/<frame_token>.jpg
    simscale_<round>_sensor_blobs_hist.json shard name -> list of log tokens

A scene is named by the token of its current frame (frame 3), as in NAVSIM. Frames are written
to data/simscale/frames/<round>/<log_token>/<CAM>/<frame_token>.jpg, i.e. the data_path of the
log, so the frame root works as the sensor_root of the index. Finished shards are appended to
data/simscale/frames/<round>/done_shards.txt as "<shard>:<cameras>" and skipped on a restart when
they already hold all requested cameras; a shard is processed again
from scratch after a crash (frames already on disk are overwritten).

`--mode stream` (default) reads the shard over HTTP straight into the tar reader, so the 9.4 GB
archive never touches the disk; `--mode disk` downloads it with wget -c first, then extracts and
deletes it. Before every shard the free space of the data disk is checked; below
--min-free-gb (+ the shard size in disk mode) the run sleeps until space is back.
"""

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import tarfile
import time
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from exp_navsim.simscale.safe_pickle import safe_loads

HF = "https://huggingface.co/datasets/OpenDriveLab/SimScale/resolve/main/SimScale_data"
ROOT = Path("data/simscale")
SIZE = (512, 288)                       # PIL (W, H), as exp_navsim/data/resize_frames.py
NUM_HISTORY, NUM_POSES = 4, 8
SAFE_REL = re.compile(r"^[0-9a-f]{16}-\d{3}/CAM_[A-Z]\d/[0-9a-f]{16}-\d{3}\.jpg$")


def cam_name(cam):
    cam = cam.upper()
    return cam if cam.startswith("CAM_") else f"CAM_{cam}"


def round_dir(rnd):
    return f"synthetic_reaction_{rnd}"


def shard_url(rnd, i):
    return f"{HF}/{round_dir(rnd)}/simscale_{rnd}_sensor_blobs_hist/simscale_{rnd}_sensor_blobs_hist_{i}.tar.gz"


def meta_path(rnd):
    return ROOT / "raw" / rnd / "meta_datas.tar.gz"


def records_path(rnd):
    return ROOT / "meta" / f"{rnd}_records.pkl"


def frame_root(rnd):
    return ROOT / "frames" / rnd


# ---------------------------------------------------------------- meta

def _pose(frame):
    from exp_navsim.data.navsim_base import quat_to_yaw
    return (frame["ego2global_translation"][0], frame["ego2global_translation"][1],
            quat_to_yaw(frame["ego2global_rotation"]))


def scene_record(frames):
    """12 frames of a synthetic log -> one scene record (as build_index.scene_record, all cameras)."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
        convert_absolute_to_relative_se2_array)
    from nuplan.common.actor_state.state_representation import StateSE2
    cur = NUM_HISTORY - 1
    poses = np.array([_pose(f) for f in frames], dtype=np.float64)
    origin = StateSE2(*poses[cur])
    dyn = np.asarray(frames[cur]["ego_dynamic_state"], dtype=np.float32)
    sim_log = frames[cur]["log_name"]
    return {
        "token": frames[cur]["token"],
        "log": sim_log[:-len(frames[cur]["log_token"]) - 1],     # the real nuPlan log it was simulated in
        "sim_log": sim_log,
        "log_token": frames[cur]["log_token"],
        "city": frames[cur]["map_location"],
        "cam_paths": {c: [f["cams"][c]["data_path"] for f in frames[:NUM_HISTORY]] for c in frames[cur]["cams"]},
        "future": convert_absolute_to_relative_se2_array(origin, poses[cur + 1:cur + 1 + NUM_POSES]).astype(np.float32),
        "past": convert_absolute_to_relative_se2_array(origin, poses[:NUM_HISTORY]).astype(np.float32),
        "velocity": dyn[:2],
        "acceleration": dyn[2:4],
        "command": np.asarray(frames[cur]["driving_command"], dtype=np.float32),
        "n_frames": len(frames),
        "has_route": len(frames[cur]["roadblock_ids"]) > 0,
    }


def cmd_meta(args):
    src = meta_path(args.round)
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF}/{round_dir(args.round)}/simscale_{args.round}_meta_datas.tar.gz"
        subprocess.run(["wget", "-q", "-c", "-O", str(src), url], check=True)
    records, t0 = [], time.time()
    with tarfile.open(src, "r|gz") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".pkl"):
                frames = safe_loads(tf.extractfile(m).read())
                if len(frames) >= NUM_HISTORY + NUM_POSES:
                    records.append(scene_record(frames))
                if len(records) % 5000 == 0:
                    print(f"[meta] {len(records)} scenes  {time.time() - t0:.0f}s", flush=True)
    out = records_path(args.round)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(records, f)
    print(f"[meta] {len(records)} scenes -> {out}  {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- shards

def _resize(job):
    data, dst = job
    img = Image.open(BytesIO(data))
    orig = img.size
    img.draft("RGB", (SIZE[0] * 2, SIZE[1] * 2))
    img = img.convert("RGB").resize(SIZE, Image.BILINEAR, reducing_gap=2.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    img.save(tmp, format="JPEG", quality=95)
    os.replace(tmp, dst)
    return orig


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def wait_for_space(path, need_gb):
    while free_gb(path) < need_gb:
        print(f"[disk] {free_gb(path):.0f} GB free < {need_gb:.0f} GB, waiting", flush=True)
        time.sleep(120)


def open_shard(rnd, i, mode):
    """-> (file object of the .tar.gz, path to delete afterwards or None)."""
    if mode == "disk":
        path = ROOT / "raw" / rnd / "shards" / f"hist_{i}.tar.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["wget", "-q", "-c", "-O", str(path), shard_url(rnd, i)], check=True)
        return open(path, "rb"), path
    import requests
    resp = requests.get(shard_url(rnd, i), stream=True, timeout=60)
    resp.raise_for_status()
    resp.raw.decode_content = False
    return resp.raw, None


def process_shard(rnd, i, wanted, mode, pool):
    """Keep the wanted history frames of shard i. -> stats dict."""
    t0 = time.time()
    fobj, path = open_shard(rnd, i, mode)
    t_download = time.time() - t0
    out = frame_root(rnd)
    n_members, n_kept, cams_seen, sizes, batch = 0, 0, {}, {}, []

    def flush():
        for size in pool.imap_unordered(_resize, batch, chunksize=8):
            sizes[size] = sizes.get(size, 0) + 1
        batch.clear()

    with tarfile.open(fileobj=fobj, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            n_members += 1
            parts = m.name.split("/")
            rel = "/".join(parts[-3:])
            if len(parts) >= 3:
                cams_seen[parts[-2]] = cams_seen.get(parts[-2], 0) + 1
            if rel not in wanted or not SAFE_REL.match(rel):
                continue
            batch.append((tf.extractfile(m).read(), out / rel))
            n_kept += 1
            if len(batch) >= 256:
                flush()
        flush()
    fobj.close()
    if path is not None:
        path.unlink()
    return {"shard": i, "mode": mode, "t_download": round(t_download, 1), "t_total": round(time.time() - t0, 1),
            "members": n_members, "kept": n_kept, "cams": cams_seen,
            "source_sizes": {f"{w}x{h}": c for (w, h), c in sizes.items()}}


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def cmd_shards(args):
    with open(records_path(args.round), "rb") as f:
        records = pickle.load(f)
    cams = [cam_name(c) for c in args.cams.split(",")]
    wanted = {p for r in records for c in cams for p in r["cam_paths"][c]}
    out = frame_root(args.round)
    out.mkdir(parents=True, exist_ok=True)
    done_file, log_file = out / "done_shards.txt", out / "shard_stats.jsonl"
    done = {}                                   # shard -> cameras already written
    for line in (done_file.read_text().split() if done_file.exists() else []):
        i, _, cs = line.partition(":")
        done.setdefault(int(i), set()).update(cs.split(","))
    shard_gb = 10.0 if args.mode == "disk" else 0.0
    with Pool(args.workers) as pool:
        for i in parse_range(args.shards):
            if set(cams) <= done.get(i, set()):
                continue
            wait_for_space(ROOT, args.min_free_gb + shard_gb)
            for attempt in range(args.retries):
                try:
                    stats = process_shard(args.round, i, wanted, args.mode, pool)
                    break
                except Exception as e:          # dropped connection, truncated archive
                    print(f"[shard {i}] attempt {attempt + 1} failed: {e!r}", flush=True)
                    time.sleep(30)
            else:
                raise RuntimeError(f"shard {i} failed {args.retries} times")
            stats["free_gb"] = round(free_gb(ROOT), 1)
            print(f"[shard {i}] {json.dumps(stats)}", flush=True)
            with open(log_file, "a") as f:
                f.write(json.dumps(stats) + "\n")
            with open(done_file, "a") as f:
                f.write(f"{i}:{','.join(cams)}\n")


# ---------------------------------------------------------------- index

def cmd_index(args):
    """Navtrain-format index of the scenes whose 4 history frames of every --cams camera are on disk.

    data/navsim_index/<split>.pkl holds CAM_F0 (the first camera); every further camera C gets
    <split>_<C>.pkl with the same rows. data/frames/<split>[_C]/<id // 1000>/<id>.jpg are
    symlinks to the frames, which is the layout exp_navsim/encoders/cache_features.py reads.
    """
    from exp_navsim.data.resize_frames import frame_file
    with open(records_path(args.round), "rb") as f:
        records = pickle.load(f)
    root = frame_root(args.round).resolve()
    cams = [cam_name(c) for c in args.cams.split(",")]
    present = {}

    def on_disk(rel):
        d = os.path.dirname(rel)
        if d not in present:
            present[d] = set(os.listdir(root / d)) if (root / d).is_dir() else set()
        return os.path.basename(rel) in present[d]

    keep = [r for r in records if all(on_disk(p) for c in cams for p in r["cam_paths"][c])]
    print(f"[index] scenes in meta: {len(records)}, with all {len(cams)} cameras on disk: {len(keep)}")
    stack = lambda key, dtype=np.float32: np.stack([r[key] for r in keep]).astype(dtype)
    base = {
        "round": args.round, "token": [r["token"] for r in keep], "log": [r["log"] for r in keep],
        "sim_log": [r["sim_log"] for r in keep], "log_token": [r["log_token"] for r in keep],
        "city": [r["city"] for r in keep], "synthetic": np.ones(len(keep), dtype=bool),
        "has_route": np.array([r["has_route"] for r in keep]),
        "future": stack("future"), "past": stack("past"), "velocity": stack("velocity"),
        "acceleration": stack("acceleration"), "command": stack("command"),
    }
    for k, cam in enumerate(cams):
        split = args.split if k == 0 else f"{args.split}_{cam[4:]}"
        paths = [p for r in keep for p in r["cam_paths"][cam]]      # 4 distinct frames per scene
        index = dict(base, split=split, camera=cam, sensor_root=str(root), frame_paths=paths,
                     frame_ids=np.arange(len(paths), dtype=np.int64).reshape(-1, NUM_HISTORY))
        for fid, rel in enumerate(paths):
            link = frame_file(split, fid)
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(root / rel)
        out = Path("data/navsim_index") / f"{split}.pkl"
        with open(out, "wb") as f:
            pickle.dump(index, f)
        print(f"[index] {split}: {len(keep)} scenes, {len(paths)} frames -> {out}")
    print("commands:", base["command"].sum(0), " cities:", dict(zip(*np.unique(base["city"], return_counts=True))))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("meta", "shards", "index"):
        p = sub.add_parser(name)
        p.add_argument("--round", default="pdm_v1.0-0")
        if name == "shards":
            p.add_argument("--shards", default="0-65", help="e.g. 0-65 or 0,1,5-9")
            p.add_argument("--cams", default="F0")
            p.add_argument("--mode", choices=["stream", "disk"], default="stream")
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--retries", type=int, default=3)
            p.add_argument("--min-free-gb", type=float, default=120)
        if name == "index":
            p.add_argument("--split", default="simscale_pdm0")
            p.add_argument("--cams", default="F0")
    args = ap.parse_args()
    {"meta": cmd_meta, "shards": cmd_shards, "index": cmd_index}[args.cmd](args)


if __name__ == "__main__":
    main()
