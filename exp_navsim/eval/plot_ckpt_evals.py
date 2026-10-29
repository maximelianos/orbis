"""Metrics over checkpoints from eval/evaluate.py CSVs (--ckpts all), several runs on one figure.

    python -m exp_navsim.eval.plot_ckpt_evals \
        --csvs logs_exp/02_qsem_10k/eval/val.csv logs_exp/02_qsem_10k/eval/val_s50.csv \
        --labels "D16, 10 steps" "D16, 50 steps" --out outputs/02_collapse/offline.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TITLES = {"energy": "energy score [m]", "ade_min": "minADE@32 [m]", "ade_mean": "ADE, mean over samples [m]",
          "spread": "spread E|X-X'| [m]", "dispersion": "dispersion (1 = calibrated)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--metrics", nargs="+", default=list(TITLES))
    ap.add_argument("--styles", nargs="+", default=None, help="matplotlib format per csv, e.g. C0- C0--")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(1, len(args.metrics), figsize=(3.4 * len(args.metrics), 3.1))
    for i, (csv, label) in enumerate(zip(args.csvs, args.labels)):
        d = pd.read_csv(csv)
        d = d[d.step > 0].sort_values("step")
        fmt = args.styles[i] if args.styles else "-"
        for ax, m in zip(axes, args.metrics):
            ax.plot(d.step, d[m], fmt, label=label, marker=".")
    for ax, m in zip(axes, args.metrics):
        ax.set_title(TITLES.get(m, m), fontsize=9)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        if m == "dispersion":
            ax.axhline(1, ls="--", c="gray", lw=0.8)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
