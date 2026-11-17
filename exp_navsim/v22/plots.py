"""Plots of the v22 ablation: EPDMS per run, the stage components, and the training curves.

    python -m exp_navsim.v22.plots

Reads outputs/24_v22/results.csv (one row per training run, written by train.py) and
outputs/24_v22/navhard.csv (full-split scores of saved predictions, written by
score_runs.py) and writes ladder.png, components.png and curves.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("outputs/24_v22")
BLUE, GREY, GREEN, ORANGE = "#3b6ea5", "#8a8f98", "#4f9a6e", "#c9792e"
# py123d_garage, NAVSIM v2 navhard two-stage; italic rows there are metric-trained
LITERATURE = [("CV (garage)", 11.4, False), ("LTF (Kesai 2026)", 25.1, False),
              ("DiffusionDrive (Liao 2025)", 27.5, False), ("LTFv6 (Kesai 2026)", 31.9, False),
              ("TransFuser-K1 (Kesai 2026)", 34.9, False), ("SUV (2026)", 36.9, False),
              ("GTRS-Dense (Li 2025)", 45.3, True), ("ZTRS (2026)", 48.1, True)]


def epdms_column(df):
    cols = [c for c in df.columns if c.endswith("/epdms")]
    if not cols:
        return None
    # the largest group count is the most reliable
    best = max(cols, key=lambda c: int(c.split("/")[0].replace("hard", "")))
    return best


def ladder():
    path = OUT / "results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    col = epdms_column(df)
    df = df.dropna(subset=[col]).drop_duplicates("name", keep="last")
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(df) + 1.6), dpi=140)
    y = np.arange(len(df))
    ax.barh(y, df[col], color=BLUE, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(df["name"], fontsize=8)
    ax.invert_yaxis()
    for i, (v, t) in enumerate(zip(df[col], df["train_min"])):
        ax.text(v + 0.3, i, f"{v:.1f}   ({t:.1f} min)", va="center", fontsize=7.5, color="#333")
    ax.set_xlabel(f"navhard two-stage EPDMS ({col.split('/')[0].replace('hard', '')} groups)")
    ax.set_xlim(0, max(df[col].max() * 1.22, 10))
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("v22 ladder on the navhard proxy")
    fig.tight_layout()
    fig.savefig(OUT / "ladder.png")
    print(f"-> {OUT / 'ladder.png'}")


def components():
    path = OUT / "navhard.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df.groups == 225].drop_duplicates("name", keep="last")
    if df.empty:
        return
    metrics = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
               "ego_progress", "time_to_collision_within_bound", "lane_keeping",
               "two_frame_extended_comfort"]
    short = ["NC", "DAC", "DDC", "EP", "TTC", "LK", "EC"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), dpi=140, sharey=True)
    x = np.arange(len(metrics))
    w = 0.8 / max(len(df), 1)
    for ax, stage in zip(axes, ("s1", "s2")):
        for j, (_, r) in enumerate(df.iterrows()):
            ax.bar(x + j * w, [r.get(f"{m}_{stage}", np.nan) for m in metrics], w,
                   label=r["name"] if stage == "s1" else None)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(short)
        ax.set_title("stage one (real frames)" if stage == "s1" else "stage two (synthetic frames)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("sub-score [%]")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("navhard two-stage sub-scores, whole split", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "components.png")
    print(f"-> {OUT / 'components.png'}")


def literature():
    """Our best next to the py123d_garage NAVSIM v2 table."""
    path = OUT / "navhard.csv"
    rows = list(LITERATURE)
    if path.exists():
        df = pd.read_csv(path)
        df = df[df.groups == 225].drop_duplicates("name", keep="last")
        for _, r in df.iterrows():
            rows.append((f"ours: {r['name']}", r["epdms"], False))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 0.4 * len(rows) + 1.2), dpi=140)
    y = np.arange(len(rows))
    colors = [ORANGE if n.startswith("ours") else (GREY if m else BLUE) for n, _, m in rows]
    ax.barh(y, [v for _, v, _ in rows], color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in rows], fontsize=8)
    ax.invert_yaxis()
    for i, (_, v, _) in enumerate(rows):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=7.5, color="#333")
    ax.set_xlabel("navhard two-stage EPDMS (NAVSIM v2.2)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Published entries (py123d_garage table) and this work\n"
                 "grey = trained on the benchmark metric", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "literature.png")
    print(f"-> {OUT / 'literature.png'}")


if __name__ == "__main__":
    ladder()
    components()
    literature()
