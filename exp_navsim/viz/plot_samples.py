"""Sampled futures of a few scenes at several checkpoints of a run.

    python -m exp_navsim.viz.plot_samples --run logs_exp/02_qsem_10k --split val \
        --ckpts step0005000,step0050000,last --num 6 --out outputs/02_collapse/samples_10k.png

One row per scene: the current front image, then one panel per checkpoint with the
K sampled futures (thin) and the logged future (black), in the current ego frame
(x forward, drawn upwards). Scenes are picked among those whose logged future
turns or stops, where a distribution has something to show (seeded, so the same
scenes come back for every run).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from exp_navsim.data.resize_frames import frame_file
from exp_navsim.data.scenes import COMMANDS
from exp_navsim.eval.evaluate import checkpoints, dataset_for
from exp_navsim.models.probe import TrajectoryProbe
from omegaconf import OmegaConf


def pick_scenes(ds, num, seed=0):
    fut = ds.future[ds.rows]
    interesting = (np.abs(fut[:, -1, 2]) > 0.3) | (np.abs(fut[:, -1, 1]) > 2.0) | (fut[:, -1, 0] < 3.0)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(interesting)
    return np.sort(rng.choice(pos, num, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ckpts", default="last")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = OmegaConf.load(run / "config.yaml")
    ds = dataset_for(cfg, args.split)
    pos = pick_scenes(ds, args.num)
    batch = ds[pos.tolist()]
    batch = {k: v.cuda() for k, v in batch.items()}
    ckpts = checkpoints(run, args.ckpts)
    preds = []
    for ckpt in ckpts:
        model = TrajectoryProbe.load_from_checkpoint(ckpt, map_location="cuda")
        g = torch.Generator(device="cuda").manual_seed(0)
        preds.append(model.sample(batch, args.samples, generator=g).cpu().numpy())    # (K, B, 8, 3)
    gt = batch["future"].cpu().numpy()

    cols = 1 + len(ckpts)
    fig, axes = plt.subplots(args.num, cols, figsize=(2.6 * cols, 2.6 * args.num), squeeze=False)
    for i, p in enumerate(pos):
        row = ds.rows[p]
        fid = ds.index["frame_ids"][row][-1]
        axes[i, 0].imshow(Image.open(frame_file(ds.index["split"], fid)))
        cmd = COMMANDS[int(ds.command[row].argmax())]
        axes[i, 0].set_title(f"{ds.index['city'][row][:12]}, cmd {cmd}", fontsize=8)
        axes[i, 0].axis("off")
        for j, s in enumerate(preds):
            ax = axes[i, j + 1]
            for k in range(s.shape[0]):
                ax.plot(-s[k, i, :, 1], s[k, i, :, 0], color="tab:blue", alpha=0.35, lw=0.8)
            ax.plot(-np.r_[0, gt[i, :, 1]], np.r_[0, gt[i, :, 0]], color="black", lw=2)
            ax.scatter([0], [0], color="red", s=10, zorder=3)
            lim = max(8.0, np.abs(gt[i, :, :2]).max() * 1.3, np.percentile(np.abs(s[:, i, :, :2]), 95) * 1.1)
            ax.set_xlim(-lim / 2, lim / 2)
            ax.set_ylim(-lim * 0.15, lim * 0.85)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(ckpts[j].stem, fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
