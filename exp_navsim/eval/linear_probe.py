"""Linear read-out of driving quantities from cached features (ridge regression).

    python -m exp_navsim.eval.linear_probe --features q_sem q_rec dinov2 dinov2:16 vjepa2 wm23 --out outputs/13_linear_probes

The flow-matching probe answers "can a small network turn these features into a
distribution of futures". This asks the simpler question of what is linearly
there. Input: every history frame's grid average-pooled to 3 x 4 cells and
concatenated (per-frame features, 4 frames) or the one scene grid (clip
features), z-scored. Targets, all from the index:

    speed        |v| at the current frame             (needs motion: two frames)
    accel        longitudinal acceleration
    progress_4s  x after 4 s
    lateral_4s   y after 4 s
    heading_4s   heading after 4 s
    command      left / straight / right (argmax of a ridge fit to the one-hot)

Ridge with the penalty picked on a held-out part of the training logs, scored on
the validation logs: R^2 for the regressions, accuracy for the command.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from exp_navsim.data.scenes import SceneFeatureDataset, log_in_split

TARGETS = ("speed", "accel", "progress_4s", "lateral_4s", "heading_4s")
LAMBDAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)


def design(ds, rows, pool=(3, 4), batch=4096):
    out = []
    for i in range(0, len(rows), batch):
        x = torch.from_numpy(np.stack([ds.features_of(r) for r in rows[i:i + batch]]).astype(np.float32))
        x = F.adaptive_avg_pool2d(x.flatten(0, 1), pool).unflatten(0, x.shape[:2])
        out.append(x.flatten(1))
    return torch.cat(out)


def targets(ds, rows):
    fut, st = ds.future[rows], ds.status[rows]
    y = np.stack([np.linalg.norm(st[:, :2], axis=1), st[:, 2], fut[:, -1, 0], fut[:, -1, 1], fut[:, -1, 2]], 1)
    return torch.from_numpy(y.astype(np.float32)), torch.from_numpy(ds.command[rows][:, :3].astype(np.float32))


def ridge(x, y, lam):
    d = x.shape[1]
    return torch.linalg.solve(x.T @ x + lam * len(x) * torch.eye(d, device=x.device), x.T @ y)


def r2(pred, y):
    return 1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def probe(feature, index, device):
    """feature: a cached name, "name:k" for its first k (PCA) channels, or "none"."""
    name, _, k = feature.partition(":")
    ds = SceneFeatureDataset(index, [name] if name != "none" else [], split="all", channels=int(k) if k else None)
    logs = np.asarray(ds.index["log"])
    is_val = np.array([log_in_split(l, "val", 0.1) for l in logs])
    # A second log-level cut inside training picks the ridge penalty.
    is_sel = np.array([log_in_split(l + "_sel", "val", 0.1) for l in logs]) & ~is_val
    parts = {k: np.flatnonzero(m) for k, m in
             {"fit": ~is_val & ~is_sel, "sel": is_sel, "val": is_val}.items()}
    X = {k: design(ds, r).to(device) for k, r in parts.items()}
    mu, sd = X["fit"].mean(0), X["fit"].std(0) + 1e-6
    X = {k: torch.cat([(v - mu) / sd, torch.ones(len(v), 1, device=device)], 1) for k, v in X.items()}
    Y = {k: tuple(t.to(device) for t in targets(ds, r)) for k, r in parts.items()}

    row = {"feature": feature, "dim": X["fit"].shape[1] - 1}
    for j, name in enumerate(TARGETS + ("command",)):
        y_of = (lambda k: Y[k][0][:, j:j + 1]) if name != "command" else (lambda k: Y[k][1])
        best = None
        for lam in LAMBDAS:
            w = ridge(X["fit"], y_of("fit"), lam)
            if name == "command":
                s = (X["sel"] @ w).argmax(1).eq(y_of("sel").argmax(1)).float().mean().item()
            else:
                s = r2(X["sel"] @ w, y_of("sel")).item()
            if best is None or s > best[0]:
                best = (s, lam, w)
        w = best[2]
        if name == "command":
            row[name] = (X["val"] @ w).argmax(1).eq(y_of("val").argmax(1)).float().mean().item()
        else:
            row[name] = r2(X["val"] @ w, y_of("val")).item()
    print(row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--index", default="data/navsim_index/navtrain.pkl")
    ap.add_argument("--out", default="outputs/13_linear_probes")
    args = ap.parse_args()
    rows = [probe(f, args.index, "cuda") for f in args.features]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = out / "linear_probe.csv"
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old.feature.isin(df.feature)], df])
    df.to_csv(path, index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
