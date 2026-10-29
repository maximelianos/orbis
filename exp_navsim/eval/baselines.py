"""Reference predictors that need no training, scored like the probe.

    python -m exp_navsim.eval.baselines --split val

constant_velocity  the current speed held for 4 s, straight ahead (navsim's
                   ConstantVelocityAgent)
train_mean         the mean future of the training scenes, the same for every scene
nearest_status     the future of the training scene with the closest ego status
                   (velocity, acceleration) and the same driving command: a lookup
                   table, i.e. what pure memorization of the status would give

Deterministic predictors have one sample, so their spread is 0 and their energy
score equals the ADE-like term E||X - y||. Results: outputs/03_baselines/<split>.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.eval.evaluate import summarize
from exp_navsim.eval.metrics import sample_metrics

DT = 0.5


def constant_velocity(status):
    speed = np.linalg.norm(status[:, :2], axis=1)
    t = DT * np.arange(1, 9)
    return np.stack([speed[:, None] * t, np.zeros((len(speed), 8)), np.zeros((len(speed), 8))], -1)


def nearest_status(train_status, train_cmd, train_future, status, cmd):
    out = np.empty((len(status), 8, 3), dtype=np.float32)
    scale = train_status.std(0) + 1e-6
    for c in np.unique(cmd):
        ref = np.flatnonzero(train_cmd == c)
        q = np.flatnonzero(cmd == c)
        d = torch.cdist(torch.from_numpy(status[q] / scale), torch.from_numpy(train_status[ref] / scale))
        out[q] = train_future[ref[d.argmin(1).numpy()]]
    return out


def score(pred, ds, rows):
    gt = torch.from_numpy(ds.future[rows])
    m = {k: v.numpy() for k, v in sample_metrics(torch.from_numpy(pred)[None].float(), gt).items()}
    l2 = np.linalg.norm(pred[..., :2] - gt.numpy()[..., :2], axis=-1)
    for t in (1, 2, 3, 4):
        m[f"l2_{t}s"] = l2[:, 2 * t - 1]
    m["command"] = ds.command[rows].argmax(1)
    m["row"] = rows
    return summarize(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--index", default="data/navsim_index/navtrain.pkl")
    args = ap.parse_args()

    train = SceneFeatureDataset(args.index, [], split="train")
    if args.split in ("navtest", "navhard"):
        ds = SceneFeatureDataset(f"data/navsim_index/{args.split}.pkl", [], split="all")
    else:
        ds = SceneFeatureDataset(args.index, [], split=args.split)
    rows = ds.rows
    status, cmd = ds.status[rows], ds.command[rows].argmax(1)
    preds = {
        "constant_velocity": constant_velocity(status),
        "train_mean": np.broadcast_to(train.future[train.rows].mean(0), (len(rows), 8, 3)).copy(),
        "nearest_status": nearest_status(train.status[train.rows], train.command[train.rows].argmax(1),
                                         train.future[train.rows], status, cmd),
    }
    table = pd.DataFrame([{"baseline": k, **score(p.astype(np.float32), ds, rows)} for k, p in preds.items()])
    out = Path("outputs/03_baselines") / f"{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table[["baseline", "ade_mean", "fde_mean", "l2_1s", "l2_2s", "l2_3s", "l2_4s", "scenes"]].to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
