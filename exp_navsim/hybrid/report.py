"""Figures and the markdown table of 19_hybrid from results.csv and the run logs.

    python -m exp_navsim.hybrid.report --labels regress,flow,flow_raw,resid,...

comparison.png   plan EPDMS per method and selection rule (dots), with PDM-Closed and the
                 16 regression planner as reference lines; sample quality next to it
curves.pdf       TensorBoard curves of the runs: training losses, val energy, navtest100
                 EPDMS of the sample mean
table.md         the main table
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("outputs/19_hybrid")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a63d2", "#7a7a7a"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
RULES = [("plan_mean", "mean of samples"), ("plan_ref", "regression output"),
         ("plan_scorer", "scorer top-1"), ("plan_oracle", "oracle best of K")]
PDM_CLOSED, REGRESS16 = 0.871, 0.887
XMIN = 0.74


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def comparison(df):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.42 * len(df) + 1.6), gridspec_kw={"width_ratios": [2, 1]},
                                 sharey=True)
    y = np.arange(len(df))[::-1]
    for (col, name), c in zip(RULES, SERIES):
        v = df[col].to_numpy(dtype=float)
        if col == "plan_ref":
            v = np.where(df["mode"].isin(["regress", "resid"]), v, np.nan)
        a1.scatter(np.clip(v, XMIN, None), y, s=40, color=c, label=name, zorder=3, edgecolor="white", linewidth=1.5)
        for yi, vi in zip(y, v):                      # off-scale values are drawn at the edge with their value
            if vi < XMIN:
                a1.text(XMIN + 0.004, yi + 0.18, f"{vi:.2f}", color=INK, fontsize=7)
    for x, name, ha in ((PDM_CLOSED, "PDM-Closed", "right"), (REGRESS16, "16 regression", "left")):
        a1.axvline(x, color=MUTED, lw=1, ls="--", zorder=1)
        a1.text(x, len(df) - 0.4, f" {name} ", color=MUTED, fontsize=7, ha=ha)
    a1.set_xlim(XMIN, 1.0)
    a1.set_yticks(y, df["label"])
    a1.set_xlabel("EPDMS, navtest 500 subset", color=INK, fontsize=9)
    a1.legend(fontsize=8, frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0, 1.02))
    style(a1)
    a2.barh(y + 0.18, df["sample_bad"] * 100, height=0.34, color=SERIES[0], label="bad samples (EPDMS < 0.5)")
    a2.barh(y - 0.18, df["sample_offroad"] * 100, height=0.34, color=SERIES[1], label="off-road samples")
    a2.set_xlabel("% of samples", color=INK, fontsize=9)
    a2.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02))
    style(a2)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=130)


def scalars(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(Path("logs_exp") / run / "tb"))
    ea.Reload()
    return {t: pd.Series({e.step: e.value for e in ea.Scalars(t)}) for t in ea.Tags()["scalars"]}


def curves(runs):
    tags = [("train/fm", "flow matching loss (train)"), ("train/l1", "regression L1 (train)"),
            ("val/energy", "energy score (val, 16 samples)"), ("navtest100/epdms_mean", "EPDMS of sample mean (navtest 100)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for run, c in zip(runs, SERIES):
        s = scalars(run)
        for ax, (tag, title) in zip(axes, tags):
            if tag in s:
                v = s[tag]
                if tag.startswith("train/"):
                    v = v.rolling(10, min_periods=1).mean()
                ax.plot(v.index, v.values, color=c, lw=1.6, label=run, marker="o" if len(v) < 15 else None, ms=4)
    for ax, (_, title) in zip(axes, tags):
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("step", fontsize=8, color=MUTED)
        style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "curves.pdf")
    fig.savefig(OUT / "curves.png", dpi=110)


def table(df):
    cols = [("label", "method"), ("plan_mean", "mean"), ("plan_ref", "reg. out"), ("plan_medoid", "medoid"),
            ("plan_scorer", "scorer"), ("plan_verifier", "verifier"), ("plan_oracle", "oracle@K"),
            ("sample_epdms", "sample EPDMS"), ("sample_bad", "bad"), ("sample_offroad", "off-road"),
            ("val_energy", "val energy"), ("val_ade_min", "minADE"), ("val_spread", "spread"),
            ("ms_per_scene", "ms/scene")]
    lines = ["| " + " | ".join(n for _, n in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c, _ in cols:
            v = r[c]
            if c == "label":
                cells.append(str(v))
            elif c == "plan_ref" and r["mode"] == "flow":
                cells.append("")
            elif c in ("sample_bad", "sample_offroad"):
                cells.append(f"{100 * v:.1f}%")
            elif c == "ms_per_scene":
                cells.append(f"{v:.1f}")
            else:
                cells.append(f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None, help="rows of results.csv in this order (default: all)")
    ap.add_argument("--runs", default=None, help="runs for curves.pdf")
    ap.add_argument("--out", default=str(OUT), help="directory with results.csv, for the outputs too")
    args = ap.parse_args()
    OUT = Path(args.out)
    df = pd.read_csv(OUT / "results.csv")
    if args.labels:
        df = df.set_index("label").loc[args.labels.split(",")].reset_index()
    comparison(df)
    table(df)
    if args.runs:
        curves(args.runs.split(","))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
