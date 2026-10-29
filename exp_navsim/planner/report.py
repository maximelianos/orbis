"""Comparison table and plot of the fast planners against the reference planners.

    python -m exp_navsim.planner.report

Reads outputs/16_fast_planner/results.csv (train.py) and baselines.csv (baselines.py),
writes comparison.csv and comparison.png (EPDMS on the 500-scene navtest subset).
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs/16_fast_planner")
SUB = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound",
       "two_frame_extended_comfort"]
COLORS = {"ours": "#2a78d6", "earlier probe": "#eb6834", "reference": "#8c8b85"}


def main():
    res = pd.read_csv(OUT / "results.csv").drop_duplicates("name", keep="last")
    base = pd.read_csv(OUT / "baselines.csv")
    base = base[base.subset == 500]

    rows = []
    for _, r in base.iterrows():
        group = "earlier probe" if r.planner.startswith("diffusion") else "reference"
        rows.append({"planner": r.planner, "group": group, "epdms": r.score, "epdms_best8": None,
                     "train_min": None, **{c: r[c] for c in SUB}})
    for _, r in res.iterrows():
        rows.append({"planner": r["name"], "group": "ours", "epdms": r["navtest500/epdms"],
                     "epdms_best8": r["navtest500/epdms_best8"], "train_min": r["train_min"],
                     **{c: r[f"navtest500/{c}"] for c in SUB}})
    df = pd.DataFrame(rows).sort_values("epdms", ascending=False)
    df.round(4).to_csv(OUT / "comparison.csv", index=False)
    print(df.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 0.28 * len(df) + 1))
    y = range(len(df))[::-1]
    ax.barh(list(y), df.epdms, height=0.6, color=[COLORS[g] for g in df.group])
    for yi, v in zip(y, df.epdms):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=7, color="#3d3d3a")
    ax.set_yticks(list(y), df.planner, fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("EPDMS, navtest subset of 500 scenes")
    ax.grid(axis="x", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()], labels=list(COLORS),
              fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=150)
    print(f"-> {OUT / 'comparison.csv'}, {OUT / 'comparison.png'}")
    curves()


def curves():
    """Training curves (TensorBoard) of a few runs: train L1, val L2 at 4 s, navtest100 EPDMS."""
    from exp_navsim.eval.plot_runs import load_scalars
    runs = {"16_regress_wm11ctx_20k": "regression, 20k steps", "16_regress_wm11ctx_d256": "regression, dim 256",
            "16_v256_wm11": "vocab 256, wm11", "16_v256_noimg": "vocab 256, no image"}
    tags = {"train/l1": "train L1 (scaled poses)", "val/l2_4s": "val L2 at 4 s [m]",
            "navtest100/epdms": "EPDMS, 100 navtest scenes"}
    fig, axes = plt.subplots(1, len(tags), figsize=(10, 2.8))
    for (run, label), color in zip(runs.items(), ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")):
        df = load_scalars(Path("logs_exp") / run)
        for ax, tag in zip(axes, tags):
            d = df[df.tag == tag].sort_values("step")
            ax.plot(d.step, d.value, color=color, lw=2, marker="o" if len(d) < 30 else None, ms=3, label=label)
    for ax, title in zip(axes, tags.values()):
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(color="#e5e4df", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yscale("log")
    axes[-1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=150)
    print(f"-> {OUT / 'curves.png'}")


if __name__ == "__main__":
    main()
