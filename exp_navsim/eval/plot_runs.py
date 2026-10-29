"""Validation curves of several runs on one figure.

    python -m exp_navsim.eval.plot_runs --runs logs_exp/a logs_exp/b --labels A B \
        --metrics energy spread dispersion ade_min --loaders val train_sub --out outputs/02_collapse/curves.png

One column per metric, one row per validation loader, one line per run. Reads the
TensorBoard scalars "<loader>/<metric>" that TrajectoryProbe logs. Also returns
the curves as a DataFrame (written next to the figure as .csv).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LABELS = {"energy": "energy score [m]", "spread": "spread E|X-X'| [m]", "dispersion": "dispersion (1 = calibrated)",
          "ade_min": "minADE@16 [m]", "ade_mean": "ADE (mean over samples) [m]", "loss": "flow-matching loss",
          "ade_of_mean": "ADE of sample mean [m]", "fde_min": "minFDE@16 [m]", "lat_std": "lateral end std [m]"}


def load_scalars(run):
    acc = EventAccumulator(str(Path(run) / "tb"), size_guidance={"scalars": 0})
    acc.Reload()
    rows = [{"tag": t, "step": e.step, "value": e.value} for t in acc.Tags()["scalars"] for e in acc.Scalars(t)]
    return pd.DataFrame(rows)


def plot(runs, labels, metrics, loaders, out, logx=False):
    frames = []
    for run, label in zip(runs, labels):
        df = load_scalars(run)
        df["run"] = label
        frames.append(df)
    data = pd.concat(frames)
    fig, axes = plt.subplots(len(loaders), len(metrics), figsize=(3.6 * len(metrics), 2.8 * len(loaders)),
                             squeeze=False, sharex=True)
    for i, loader in enumerate(loaders):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for label in labels:
                d = data[(data.run == label) & (data.tag == f"{loader}/{metric}")].sort_values("step")
                ax.plot(d.step, d.value, label=label, lw=1.5)
            if metric == "dispersion":
                ax.axhline(1.0, color="gray", lw=0.8, ls="--")
            ax.set_title(f"{loader}: {LABELS.get(metric, metric)}", fontsize=9)
            if logx:
                ax.set_xscale("log")
            ax.grid(alpha=0.3)
            if i == len(loaders) - 1:
                ax.set_xlabel("training step")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    wide = data[data.tag.str.contains("/")].pivot_table(index=["run", "step"], columns="tag", values="value")
    wide.to_csv(Path(out).with_suffix(".csv"))
    print(f"-> {out}")
    return wide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--metrics", nargs="+", default=["loss", "energy", "spread", "dispersion", "ade_min"])
    ap.add_argument("--loaders", nargs="+", default=["val", "train_sub"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--logx", action="store_true")
    args = ap.parse_args()
    plot(args.runs, args.labels or [Path(r).name for r in args.runs], args.metrics, args.loaders, args.out, args.logx)


if __name__ == "__main__":
    main()
