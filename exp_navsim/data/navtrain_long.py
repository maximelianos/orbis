"""Long (whole-episode) dataloader for the NAVSIM navtrain subset.

    python -m exp_navsim.data.navtrain_long   # smoke check (needs config: dataset: navtrain)

navtrain differs from navhard only in *packaging*, so this reuses everything in
navsim_base.NavsimLongBase and only re-implements episode enumeration:

  * Logs live in   trainval_navsim_logs/trainval/<log>.pkl  — each pickle is a
    full driving log (hundreds of frames @ 2 Hz) split into many ~40-frame
    scenes (scene_token). One episode == one scene.
  * Sensor blobs   trainval_sensor_blobs/trainval/<log>/<CAM>/<img>.jpg, and the
    download is typically PARTIAL: only some frames of some scenes have images
    on disk. An episode is therefore the *longest contiguous run* of frames of a
    scene whose front-camera image is actually present (contiguity keeps the
    finite-difference velocity valid). Scenes with fewer than `min_frames` usable
    frames are dropped.

Because scanning every log for present images is slow, the episode index is
built once and cached to `index_cache` (a pickle). Delete that file to rebuild
(e.g. after downloading more sensor blobs, or changing `front_camera`).

The return dict, split logic, subsampling and camera reading are all inherited.
Surround-camera images that happen to be missing are returned as black frames
(see NavsimLongBase._read_camera) so visualization still works.
"""

import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from exp_navsim.data.navsim_base import NavsimLongBase, FRONT_CAMERA, in_split


def _longest_true_run(flags):
    """Return (start, end) inclusive indices of the longest run of True in `flags`,
    or None if there is none."""
    best = cur_start = None
    best_len = 0
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            if j - i > best_len:
                best_len, best, cur_start = j - i, (i, j - 1), i
            i = j
        else:
            i += 1
    return best


class NavtrainLongDataset(NavsimLongBase):
    """One navtrain scene (its longest on-disk-present frame run) per index."""

    def __init__(
        self,
        logs_root,                     # .../trainval_navsim_logs/trainval
        sensor_root,                   # .../trainval_sensor_blobs/trainval
        index_cache=None,              # pickle path to cache the episode index
        size=256,
        stored_data_frame_rate=2,
        frame_rate=2,
        front_camera=FRONT_CAMERA,
        load_surround=True,
        min_frames=5,                  # drop scenes with fewer usable frames
        split="all",                   # "all" | "train" | "val"
        val_fraction=0.1,
        max_episodes=None,
    ):
        super().__init__(
            sensor_root=sensor_root,
            size=size,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            front_camera=front_camera,
            load_surround=load_surround,
        )
        self.logs_root = Path(logs_root)
        self.min_frames = min_frames
        self._cached_log_path = None   # tiny 1-entry cache: episodes of a log are adjacent
        self._cached_log = None

        # Build (or load) the full index, then filter by usable length and split.
        # The index is split-independent, so one cache serves train/val/all.
        index = self._load_or_build_index(index_cache)
        self.episodes = [
            rec for rec in index
            if len(rec["frame_idxs"]) >= min_frames
            and in_split(rec["token"], split, val_fraction)
        ]
        self.episodes.sort(key=lambda r: r["token"])   # deterministic, cache-friendly
        if max_episodes is not None:
            self.episodes = self.episodes[:max_episodes]
        assert len(self.episodes) > 0, (
            f"No navtrain episodes under {self.logs_root} for split={split} "
            f"(min_frames={min_frames}). Are sensor blobs downloaded?"
        )

    # --- index construction (the only navtrain-specific, disk-scanning part) - #
    def _load_or_build_index(self, index_cache):
        if index_cache and Path(index_cache).exists():
            with open(index_cache, "rb") as f:
                return pickle.load(f)
        index = self._build_index()
        if index_cache:
            Path(index_cache).parent.mkdir(parents=True, exist_ok=True)
            with open(index_cache, "wb") as f:
                pickle.dump(index, f)
        return index

    def _build_index(self):
        """Scan every log for scenes with contiguous on-disk front-camera frames.

        Each record: {"log": <stem>, "scene_token": <st>,
                      "frame_idxs": [contiguous present frame_idx...],
                      "token": "<log>_<scene_token>"}  (token is globally unique).
        """
        records = []
        log_paths = sorted(self.logs_root.glob("*.pkl"))
        print(f"[navtrain] building episode index from {len(log_paths)} logs "
              f"(front camera: {self.front_camera}) ...")
        for lp in tqdm(log_paths, desc="[navtrain] indexing logs"):
            cam_dir = self.sensor_root / lp.stem / self.front_camera
            if not cam_dir.is_dir():
                continue                          # this log's sensors not downloaded
            present = set(os.listdir(cam_dir))
            with open(lp, "rb") as f:
                frames = pickle.load(f)
            scenes = {}
            for fr in frames:
                scenes.setdefault(fr["scene_token"], []).append(fr)
            for st, frs in scenes.items():
                frs.sort(key=lambda fr: fr["frame_idx"])
                flags = [os.path.basename(self._cam_entry(fr)["data_path"]) in present
                         for fr in frs]
                run = _longest_true_run(flags)
                if run is None:
                    continue
                s, e = run
                records.append({
                    "log": lp.stem,
                    "scene_token": st,
                    "frame_idxs": [frs[i]["frame_idx"] for i in range(s, e + 1)],
                    "token": f"{lp.stem}_{st}",
                })
        print(f"[navtrain] indexed {len(records)} candidate scenes.")
        return records

    def _cam_entry(self, frame):
        cams = frame["cams"]
        key = self.front_camera if self.front_camera in cams else self.front_camera.lower()
        return cams[key]

    # --- episode -> frames (inherited machinery does the rest) --------------- #
    def _load_log(self, path):
        if self._cached_log_path != path:
            with open(path, "rb") as f:
                self._cached_log = pickle.load(f)
            self._cached_log_path = path
        return self._cached_log

    def _episode_frames(self, rec):
        path = self.logs_root / f"{rec['log']}.pkl"
        want = set(rec["frame_idxs"])
        frames = [fr for fr in self._load_log(path)
                  if fr["scene_token"] == rec["scene_token"] and fr["frame_idx"] in want]
        frames.sort(key=lambda fr: fr["frame_idx"])
        return frames, {"path": str(path), "token": rec["token"]}

    def episode_length(self, idx):
        return len(self.episodes[idx]["frame_idxs"][:: self.frame_interval])

    def episode_token(self, idx):
        return self.episodes[idx]["token"]      # "<log>_<scene_token>"


def _main():
    from omegaconf import OmegaConf
    from exp_navsim.data.navsim_base import NavsimLongBase

    cfg = OmegaConf.load("exp_navsim/config.yaml")
    ds = NavsimLongBase.from_config(cfg)
    print(f"episodes: {len(ds)}")
    s = ds[0]
    for k in ("images", "trajectory", "velocity"):
        print(f"  {k}: {np.asarray(s[k]).shape}")
    print(f"  metadata: {s['metadata']}")


if __name__ == "__main__":
    _main()
