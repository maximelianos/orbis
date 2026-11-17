"""Intent anchors: k-means centres of the training residuals future - CV(v).

    python -m exp_navsim.diverse.anchors --num 8,16,32 [--no-kin]

Clusters on the 8 (x, y) positions of the residual (16 numbers, ADE-like distance); the
heading of an anchor is the mean heading of its members. Only training logs are used.
-> outputs/17_diversity/anchors/anchors_<kin|raw>_<M>.npy and a figure of the anchors.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.diverse.model import anchor_distance, constant_velocity

OUT = Path("outputs/17_diversity/anchors")


def kmeans(x, k, iters=50, seed=0):
    """x (N, d) -> centres (k, d), labels (N,). k-means++ init."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    centres = x[torch.randint(len(x), (1,), generator=g, device=x.device)]
    for _ in range(k - 1):
        d = torch.cdist(x, centres).min(1).values ** 2
        centres = torch.cat([centres, x[torch.multinomial(d / d.sum(), 1, generator=g)]])
    for _ in range(iters):
        labels = torch.cdist(x, centres).argmin(1)
        for j in range(k):
            m = labels == j
            if m.any():
                centres[j] = x[m].mean(0)
    return centres, torch.cdist(x, centres).argmin(1)


def residuals(future, status, use_kin):
    return future - constant_velocity(status) if use_kin else future


def build_anchors(future, status, num, use_kin, seed=0):
    resid = residuals(future, status, use_kin)
    centres, labels = kmeans(resid[..., :2].flatten(1), num, seed=seed)
    head = torch.stack([resid[labels == j, :, 2].mean(0) if (labels == j).any() else torch.zeros(8, device=resid.device)
                        for j in range(num)])
    return torch.cat([centres.view(num, 8, 2), head[..., None]], -1), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", default="8,16,32")
    ap.add_argument("--no-kin", action="store_true")
    args = ap.parse_args()
    use_kin = not args.no_kin
    tag = "kin" if use_kin else "raw"

    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=[], split="train")
    val = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=[], split="val")
    fut = lambda d: torch.from_numpy(d.future[d.rows]).float().cuda()
    stat = lambda d: torch.from_numpy(d.status[d.rows]).float().cuda()
    OUT.mkdir(parents=True, exist_ok=True)
    nums = [int(n) for n in args.num.split(",")]
    fig, axes = plt.subplots(1, len(nums), figsize=(4 * len(nums), 4.5), squeeze=False)
    for ax, num in zip(axes[0], nums):
        anchors, labels = build_anchors(fut(ds), stat(ds), num, use_kin)
        np.save(OUT / f"anchors_{tag}_{num}.npy", anchors.cpu().numpy())
        d = anchor_distance(residuals(fut(val), stat(val), use_kin), anchors)
        share = torch.bincount(labels, minlength=num).float() / len(labels)
        print(f"M={num} {tag}: val ADE to nearest anchor {d.min(1).values.mean():.3f} m, "
              f"cluster shares {np.round(share.cpu().numpy(), 3).tolist()}")
        a = anchors.cpu().numpy()
        for j in range(num):
            ax.plot(np.r_[0, a[j, :, 1]], np.r_[0, a[j, :, 0]], lw=0.5 + 8 * share[j].item())
        ax.set_aspect("equal")
        ax.invert_xaxis()
        ax.set_title(f"M = {num}, residual to {'CV' if use_kin else 'origin'}\nline width = cluster share", fontsize=9)
        ax.set_xlabel("y (left) [m]")
        ax.set_ylabel("x (forward) [m]")
    fig.tight_layout()
    fig.savefig(OUT / f"anchors_{tag}.png", dpi=130)


if __name__ == "__main__":
    main()
