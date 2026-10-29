"""NAVSIM scenes with cached encoder features: the probe's training and evaluation data.

One item is one scene of an index built by build_index.py:
    "features"     (T, C, 9, 16)  cached grids (encoders/cache_features.py); T = the
                                  `history` newest frames for per-frame encoders, 1 for
                                  clip encoders. Several features are concatenated over C.
    "future"       (8, 3)         x, y, heading over the next 4 s, current ego frame
    "past"         (4, 3)         history poses in the same frame
    "status"       (4,)           ego velocity and acceleration (ego frame)
    "command"      (4,)           one-hot driving command
    "index"        int            row in the index, to find the scene again

Splits never cut a log: navtrain scenes overlap (a 14-frame window slides by one
frame), so a scene-level split would put near-copies of a validation scene into
training. `log_in_split` hashes the log name instead. On top of that:
    cities          keep only these cities
    holdout         (city, command) pairs left out of training and validation; split
                    "holdout" gives their scenes in all logs, "holdout_val" only
                    those in validation logs, which a model trained without the
                    holdout has not seen either (compositional test)
    max_scenes      random subset of the split, for the data-size experiment
"""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

COMMANDS = ("left", "straight", "right", "unknown")
# Loaded feature arrays, shared by the train and validation datasets of a run.
_ARRAYS = {}


def log_in_split(log, split, val_fraction):
    """Deterministic train/val membership of a whole log."""
    if split == "all":
        return True
    bucket = int(hashlib.md5(log.encode()).hexdigest(), 16) % 1000
    return (bucket < val_fraction * 1000) == (split == "val")


def load_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class SceneFeatureDataset(Dataset):
    def __init__(self, index_path, features, feature_dir="data/features", split="train",
                 val_fraction=0.1, history=4, channels=None, cities=None, holdout=None,
                 max_scenes=None, seed=0, preload=True, require_future=True):
        """
        Args:
            features: list of cached feature names, e.g. ["dinov2"] or ["q_sem", "q_rec"];
                      [] = no image input (status / command only baselines).
            split: train | val | all | holdout | holdout_val. The test indices use all.
            channels: keep the first k channels of every feature (PCA prefix).
            holdout: list of [city, command] pairs; removed from train/val, the only
                     scenes of split=holdout.
        """
        super().__init__()
        index = load_index(index_path)
        name = index["split"]
        self.index = index
        n = len(index["token"])

        log_split = {"holdout": "all", "holdout_val": "val"}.get(split, split)
        keep = np.array([log_in_split(log, log_split, val_fraction) for log in index["log"]])
        city = np.asarray(index["city"])
        command = index["command"].argmax(1)
        # Synthetic navhard scenes have no logged future; they only exist for the PDM
        # score, which needs a prediction for them (require_future=False).
        if require_future:
            keep &= np.isfinite(index["future"]).all(axis=(1, 2))
        if cities is not None:
            keep &= np.isin(city, list(cities))
        if holdout:
            held = np.zeros(n, dtype=bool)
            for c, cmd in holdout:
                held |= (city == c) & (command == COMMANDS.index(cmd))
            keep &= held if split.startswith("holdout") else ~held
        rows = np.flatnonzero(keep)
        if max_scenes is not None and max_scenes < len(rows):
            rows = np.sort(np.random.default_rng(seed).choice(rows, max_scenes, replace=False))
        self.rows = rows

        self.history = history
        self.arrays, self.kinds = [], []
        for f in features:
            path = Path(feature_dir) / name / f"{f}.npy"
            meta = json.loads(path.with_suffix(".json").read_text())
            key = (str(path), channels, preload)
            if key not in _ARRAYS:
                a = np.load(path, mmap_mode="r")
                a = a[:, :channels] if channels is not None else a
                _ARRAYS[key] = np.ascontiguousarray(a) if preload else a
            self.arrays.append(_ARRAYS[key])
            self.kinds.append(meta["items"])
        assert len(set(self.kinds)) <= 1, f"cannot mix frame and scene features: {features}"

        self.future = index["future"]
        self.past = index["past"]
        self.status = np.concatenate([index["velocity"], index["acceleration"]], 1)
        self.command = index["command"]
        cities_n = {c: int((city[rows] == c).sum()) for c in np.unique(city[rows])}
        print(f"[scenes] {name}/{split}: {len(rows)} scenes, {len(set(np.asarray(index['log'])[rows]))} logs, "
              f"features {features}, cities {cities_n}")

    def __len__(self):
        return len(self.rows)

    def features_of(self, row):
        """(T, C, 9, 16) for one index row, or (B, T, C, 9, 16) for an array of rows."""
        rows = np.atleast_1d(row)
        if not self.arrays:
            x = np.zeros((len(rows), 1, 0, 1, 1), dtype=np.float16)
        elif self.kinds[0] == "frames":
            ids = self.index["frame_ids"][rows][:, -self.history:]                 # (B, T)
            x = np.concatenate([a[ids] for a in self.arrays], axis=2)
        else:
            x = np.concatenate([a[rows][:, None] for a in self.arrays], axis=2)
        return x if np.ndim(row) else x[0]

    def __getitem__(self, i):
        """i: a position, or a list of positions (with a BatchSampler) for one vectorized gather."""
        row = self.rows[i]
        return {
            "features": torch.from_numpy(self.features_of(row).astype(np.float32)),
            "future": torch.from_numpy(self.future[row]),
            "past": torch.from_numpy(self.past[row]),
            "status": torch.from_numpy(self.status[row]),
            "command": torch.from_numpy(self.command[row]),
            "index": torch.as_tensor(row),
        }
