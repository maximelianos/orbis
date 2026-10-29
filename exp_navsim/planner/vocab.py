"""Trajectory vocabulary: k-means centres of the logged navtrain futures (VADv2 / Hydra-MDP).

    python -m exp_navsim.planner.vocab --sizes 64,256,1024,4096

Clusters the 8 future poses of the training logs (xy and heading, heading scaled by
`heading_scale` metres per radian so a turn and a lateral shift weigh about the same).
Writes outputs/16_fast_planner/vocab_<V>.npy (V, 8, 3) and a coverage table: the error
of the nearest vocabulary entry to the logged future on held-out logs, i.e. the
best a pure classifier over the vocabulary can do.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from exp_navsim.data.scenes import load_index, log_in_split

OUT = Path("outputs/16_fast_planner")


def flat(traj, heading_scale):
    t = traj.copy()
    t[..., 2] *= heading_scale
    return t.reshape(len(t), -1)


def kmeans(x, k, iters=50, seed=0):
    """Plain Lloyd on the GPU with k-means++ init. x (N, D) float tensor."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = [x[torch.randint(len(x), (1,), generator=g, device=x.device)]]
    d = ((x - c[0]) ** 2).sum(1)
    for _ in range(k - 1):   # k-means++, keeping the squared distance to the nearest centre
        c.append(x[torch.multinomial(d / d.sum(), 1, generator=g)])
        d = torch.minimum(d, ((x - c[-1]) ** 2).sum(1))
    c = torch.cat(c)
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        s = torch.zeros_like(c).index_add_(0, a, x)
        n = torch.bincount(a, minlength=k).clamp(min=1)[:, None]
        c = torch.where(n > 0, s / n, c)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64,256,1024,4096")
    ap.add_argument("--heading-scale", type=float, default=5.0)
    args = ap.parse_args()

    ix = load_index("data/navsim_index/navtrain.pkl")
    fut = ix["future"].astype(np.float32)
    ok = np.isfinite(fut).all((1, 2))
    train = ok & np.array([log_in_split(l, "train", 0.1) for l in ix["log"]])
    val = ok & ~train
    x = torch.from_numpy(flat(fut[train], args.heading_scale)).cuda()
    gt_val = torch.from_numpy(fut[val]).cuda()

    rows = []
    fig, axes = plt.subplots(1, len(args.sizes.split(",")), figsize=(4 * len(args.sizes.split(",")), 4))
    for ax, V in zip(np.atleast_1d(axes), map(int, args.sizes.split(","))):
        c = kmeans(x, V)
        vocab = c.reshape(V, 8, 3).clone()
        vocab[..., 2] /= args.heading_scale
        np.save(OUT / f"vocab_{V}.npy", vocab.cpu().numpy())
        d = (gt_val[:, None, :, :2] - vocab[None, :, :, :2]).norm(dim=-1)        # (N, V, 8)
        best = d.mean(-1).argmin(1)
        ade = d.mean(-1).min(1).values
        fde = d[torch.arange(len(d)), best, -1]
        rows.append(dict(V=V, oracle_ade=ade.mean().item(), oracle_fde=fde.mean().item(),
                         oracle_ade_p90=ade.quantile(.9).item()))
        print(rows[-1])
        v = vocab.cpu().numpy()
        for t in v:
            ax.plot(-t[:, 1], t[:, 0], lw=.5, alpha=.5)
        ax.set_title(f"V = {V}, nearest-entry ADE {rows[-1]['oracle_ade']:.2f} m")
        ax.set_aspect("equal"); ax.set_xlabel("-y (m)"); ax.set_ylabel("x (m)")
    fig.tight_layout(); fig.savefig(OUT / "vocab.png", dpi=120)
    pd.DataFrame(rows).to_csv(OUT / "vocab_coverage.csv", index=False)


if __name__ == "__main__":
    main()
