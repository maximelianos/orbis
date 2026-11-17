"""Figures and tables of 20_norm_hydra: normalization / prediction target of the flow and
Hydra-MDP-style PDM targets.

    python -m exp_navsim.hybrid.report20 --runs 19_flow,20_flow_raw,... --labels std,raw,...

forward.png   the forward process z_t = (1 - t) x + t eps of one held-out trajectory under each
              normalization, decoded back to metres (20 noise draws per t)
snr.png       per pose and coordinate, the t at which noise and signal have the same std
              in metres, t* = s / (s + n) (s: data std of that coordinate, n: noise std the
              normalization maps the unit noise to). Small t*: the coordinate is destroyed
              early in the schedule; t* near 1: the high-noise end is meaningless for it
horizon.png   per horizon step on 2000 held-out-log scenes (16 samples, 10 Euler steps):
              longitudinal and lateral std of the samples and L2 of the sample mean
modes.csv     mode recall on the navtest 500 subset: share of the good vocabulary entries
              (EPDMS >= 0.9, 256 k-means trajectories, true PDM scores from 19) that have a
              sample within 2 m ADE, and the number of 4 s endpoint clusters (2 m grid) of
              the samples, from evaluate.py's per_scene/<label>.npz
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from exp_navsim.hybrid.report import GRID, INK, MUTED, SERIES, style

OUT = Path("outputs/20_norm_hydra")
VOCAB = "outputs/19_hybrid/per_scene/vocab256_scorer.npz"
TS = (0.0, 0.25, 0.5, 0.75, 0.95)


def load(runs):
    from exp_navsim.hybrid.train import load_model
    return [load_model(Path("logs_exp") / r / "model.pt")[0] for r in runs]


def val_data(n=2000):
    from exp_navsim.data.scenes import SceneFeatureDataset
    from exp_navsim.hybrid.train import to_gpu
    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=["wm11_ctx"], split="val",
                             feature_dir="data/features", history=4)
    return to_gpu(ds, np.sort(np.random.default_rng(0).permutation(ds.rows)[:n]))


@torch.no_grad()
def forward_fig(models, labels, val):
    fut = val["future"]
    j = int(fut[:, -1, 1].abs().argsort()[-200])        # a clear turn
    x = fut[j:j + 1]
    fig, axes = plt.subplots(len(models), len(TS), figsize=(2.3 * len(TS), 2.3 * len(models)), squeeze=False)
    g = torch.Generator(device="cuda").manual_seed(0)
    for i, (m, lab) in enumerate(zip(models, labels)):
        z0 = m.to_z(x, None)
        for k, t in enumerate(TS):
            eps = torch.randn(20, 24, device="cuda", generator=g)
            zt = (1 - t) * z0 + t * eps
            p = m.from_z(zt, None).cpu().numpy()
            ax = axes[i, k]
            for q in p:
                ax.plot(np.r_[0, q[:, 1]], np.r_[0, q[:, 0]], color=SERIES[0], lw=0.7, alpha=0.5)
            g0 = x[0].cpu().numpy()
            ax.plot(np.r_[0, g0[:, 1]], np.r_[0, g0[:, 0]], color=INK, lw=1.6)
            ax.set_xlim(30, -30)
            ax.set_ylim(-15, 45)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6, colors=MUTED)
            if i == 0:
                ax.set_title(f"t = {t}", fontsize=9, color=INK)
            if k == 0:
                ax.set_ylabel(lab, fontsize=9, color=INK)
    fig.suptitle("forward process in metres (black: logged future, blue: z_t decoded)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "forward.png", dpi=110)


@torch.no_grad()
def snr_fig(models, labels, val):
    from exp_navsim.planner.model import scaled
    s = scaled(val["future"]).std(0).cpu().numpy()                    # (24,) data std in metres
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for m, lab, c in zip(models, labels, SERIES * 2):
        eps = torch.randn(8192, 24, device="cuda")
        zero = m.from_z(torch.zeros(1, 24, device="cuda"), None)
        n = scaled(m.from_z(eps, None) - zero).std(0).cpu().numpy()    # noise std in metres at t = 1
        tstar = s / (s + n)
        ls = "--" if len(rows) >= len(SERIES) else "-"
        for ax, off, name in ((axes[0], 0, "longitudinal x"), (axes[1], 1, "lateral y")):
            ax.plot(np.arange(1, 9) * 0.5, tstar[off::3], color=c, lw=1.6, ls=ls, marker="o", ms=3, label=lab)
            ax.set_title(f"{name}: t where noise std = data std", fontsize=9, color=INK)
        rows.append({"label": lab, **{f"tstar_x_{k}": tstar[3 * k] for k in (0, 7)},
                     **{f"tstar_y_{k}": tstar[3 * k + 1] for k in (0, 7)}})
    for ax in axes:
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("horizon (s)", fontsize=8, color=MUTED)
        ax.set_ylim(0, 1)
        style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "snr.png", dpi=120)
    return pd.DataFrame(rows)


@torch.no_grad()
def horizon_fig(models, labels, val):
    from exp_navsim.hybrid.train import batch_of
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    rows = []
    for m, lab, c in zip(models, labels, SERIES * 2):
        g = torch.Generator(device="cuda").manual_seed(0)
        fut = []
        for i in range(0, len(val["row"]), 100):
            b = batch_of(val, torch.arange(i, min(i + 100, len(val["row"])), device="cuda"))
            fut.append(m.sample(b, 16, 10, generator=g)[0])
        fut = torch.cat(fut, 1)                                           # (K, N, 8, 3)
        sd = fut[..., :2].std(0).mean(0).cpu().numpy()                    # (8, 2)
        l2 = (fut.mean(0)[..., :2] - val["future"][..., :2]).norm(dim=-1).mean(0).cpu().numpy()
        ls = "--" if len(rows) >= len(SERIES) else "-"
        h = np.arange(1, 9) * 0.5
        axes[0].plot(h, sd[:, 0], color=c, lw=1.6, ls=ls, label=lab)
        axes[1].plot(h, sd[:, 1], color=c, lw=1.6, ls=ls)
        axes[2].plot(h, l2, color=c, lw=1.6, ls=ls)
        rows.append({"label": lab, "std_x_4s": sd[-1, 0], "std_y_4s": sd[-1, 1], "std_x_0.5s": sd[0, 0],
                     "std_y_0.5s": sd[0, 1], "l2_mean_0.5s": l2[0], "l2_mean_4s": l2[-1]})
    for ax, t in zip(axes, ("longitudinal std of samples (m)", "lateral std of samples (m)", "L2 of sample mean (m)")):
        ax.set_title(t, fontsize=9, color=INK)
        ax.set_xlabel("horizon (s)", fontsize=8, color=MUTED)
        style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "horizon.png", dpi=120)
    return pd.DataFrame(rows)


def modes(labels, thr=0.9, radius=2.0):
    v = np.load(VOCAB)
    vocab = np.load("outputs/16_fast_planner/vocab_256.npy")[..., :2]          # (V, 8, 2)
    good = v["score"][:, 5:] >= thr                                            # (N, V)
    rows = []
    for lab in labels:
        f = OUT / "per_scene" / f"{lab}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        assert (d["tokens"] == v["tokens"]).all()
        xy = d["samples"][..., :2].astype(np.float32)                         # (N, K, 8, 2)
        rec, clusters = [], []
        for i in range(len(xy)):
            if good[i].any():
                ade = np.linalg.norm(vocab[good[i]][:, None] - xy[i][None], axis=-1).mean(-1)   # (G, K)
                rec.append((ade.min(1) < radius).mean())
            clusters.append(len({tuple(c) for c in np.floor(xy[i][:, -1] / 2.0).astype(int)}))
        rows.append({"label": lab, "good_mode_recall": np.mean(rec), "endpoint_cells": np.mean(clusters),
                     "good_vocab_per_scene": good.sum(1).mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="norm / target runs for the figures")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--mode-labels", default="", help="evaluate.py labels for modes.csv")
    args = ap.parse_args()
    runs, labels = args.runs.split(","), args.labels.split(",")
    models = load(runs)
    val = val_data()
    forward_fig(models, labels, val)
    snr = snr_fig(models, labels, val)
    hor = horizon_fig(models, labels, val)
    snr.merge(hor, on="label").to_csv(OUT / "norm_stats.csv", index=False)
    print(snr.merge(hor, on="label").round(3).to_string())
    if args.mode_labels:
        m = modes(args.mode_labels.split(","))
        m.to_csv(OUT / "modes.csv", index=False)
        print(m.round(3).to_string())
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
