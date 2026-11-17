"""Figures for docs/26-11-12-diffusion-vs-regression: diffusion probe vs regression planners.

    python -m exp_navsim.hybrid.plot_doc

probe_dynamics.png         the small attentive diffusion probe of 15 (100k steps): train loss,
                           val error, spread and energy vs step, with the 19 regression planner
same_backbone_dynamics.png 19 runs on the same backbone: val L2 at 4 s, energy, spread, EPDMS
end_results.png            full navtest EPDMS of all planners; EPDMS difference to regression
                           on the navtest 500 subset with paired bootstrap intervals
bev_examples.png           copy of outputs/19_hybrid/bev.png

The val subsets are the same: both trainers take np.random.default_rng(0) over the 9675 log-split
val scenes of navtrain and keep the first 2000 (checked numerically).
"""

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("docs/26-11-12-diffusion-vs-regression")
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
# Okabe-Ito colors, one per method family in all figures
BLUE, VERMILLION, ORANGE, BROWN = "#0072B2", "#D55E00", "#E69F00", "#8C510A"
GREEN, PURPLE, GRAY, DARK = "#009E73", "#7B4FA6", "#7a7a7a", "#3d3d3d"

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9.5,
                     "ytick.labelsize": 9.5, "legend.fontsize": 10, "axes.titlelocation": "left"})

PROBES = [("WM11+status+cmd", "diffusion probe: image + ego status + command", VERMILLION),
          ("WM11", "diffusion probe: image only", ORANGE),
          ("status+cmd", "diffusion probe: ego status + command only", BROWN)]
RUNS19 = [("19_regress", "regression", BLUE),
          ("19_flow", "flow matching (predict v)", VERMILLION),
          ("19_flow_x0", "flow matching (predict x0)", GREEN),
          ("19_resid", "residual flow on top of regression", PURPLE),
          ("19_flow_eps", "flow matching (predict noise)", GRAY)]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def ema(v, alpha=0.05):
    return pd.Series(v).ewm(alpha=alpha).mean().to_numpy()


def probe_dynamics():
    df = pd.read_csv("outputs/15_planner/curves.csv")
    ref = pd.read_csv("logs_exp/19_regress/evals.csv").iloc[-1]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for run, label, c in PROBES:
        d = df[df.run == run]
        tr = d.dropna(subset=["train/loss"])
        axes[0].plot(tr.step, ema(tr["train/loss"].to_numpy()), color=c, lw=1.6, label=label)
        va = d.dropna(subset=["val/energy"])
        axes[1].plot(va.step, va["val/ade_of_mean"], color=c, lw=1.8)
        axes[1].plot(va.step, va["val/ade_min"], color=c, lw=1.4, ls="--")
        axes[2].plot(va.step, va["val/spread"], color=c, lw=1.8)
        axes[3].plot(va.step, va["val/energy"], color=c, lw=1.8)
    for ax, col in ((axes[1], "val/ade_mean"), (axes[3], "val/energy")):
        ax.axhline(ref[col], color=BLUE, lw=1.4, ls=":")
        ax.text(1.0, ref[col], "regression planner (final)", transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", color=BLUE, fontsize=9)
    axes[0].set_yscale("log")
    titles = ["(a) training loss (smoothed)", "(b) val error over 4 s",
              "(c) val spread of samples", "(d) val energy score (lower is better)"]
    ylabels = ["flow matching loss", "average displacement error [m]", "spread [m]", "energy score [m]"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_title(t, color=INK)
        ax.set_xlabel("training step", color=INK)
        ax.set_ylabel(yl, color=INK)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
        style(ax)
    # line-style key for panel (b), placed outside the data
    axes[1].plot([], [], color=MUTED, lw=1.8, label="mean of samples")
    axes[1].plot([], [], color=MUTED, lw=1.4, ls="--", label="best of 16 samples")
    axes[1].legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2,
                   handlelength=2.2, borderaxespad=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "probe_dynamics.png", dpi=150)
    plt.close(fig)


def same_backbone_dynamics():
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for run, label, c in RUNS19:
        d = pd.read_csv(f"logs_exp/{run}/evals.csv")
        l2 = d["val/l2_4s_ref"] if run == "19_regress" else d["val/l2_4s_mean"]
        for ax, v in zip(axes, (l2, d["val/energy"], d["val/spread"], d["navtest100/epdms_mean"])):
            ax.plot(d.step, v, color=c, lw=1.8, marker="o", ms=6, mec="white", mew=1.2, label=label,
                    zorder=5 if run == "19_regress" else 3)
    titles = ["(a) val L2 error at 4 s", "(b) val energy score (lower is better)",
              "(c) val spread of samples", "(d) EPDMS of the plan, 100 navtest scenes"]
    ylabels = ["L2 error at 4 s [m]", "energy score [m]", "spread [m]", "EPDMS"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_title(t, color=INK)
        ax.set_xlabel("training step", color=INK)
        ax.set_ylabel(yl, color=INK)
        ax.set_xlim(0, 32000)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
        style(ax)
    axes[0].text(0.98, 0.98, "flows: mean of 16 samples\nregression: its single output", ha="right", va="top",
                 transform=axes[0].transAxes, fontsize=9, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "same_backbone_dynamics.png", dpi=150)
    plt.close(fig)


# full navtest, official EPDMS (docs/26-10-29-results-summary.md, outputs/16_*, report)
FULL = [("constant velocity", 0.259, GRAY),
        ("diffusion probe, ego status + command only", 0.645, BROWN),
        ("diffusion probe, image only", 0.788, ORANGE),
        ("diffusion probe, image + ego status + command", 0.810, VERMILLION),
        ("vocabulary planner (classification over 256 anchors)", 0.857, DARK),
        ("regression planner", 0.881, BLUE)]
PROBE_ORACLE = 0.900
# navtest 500 subset: (results.csv label, row name, color, diff, low, high) from outputs/19_hybrid/README.md,
# paired bootstrap over the 500 scenes, plan = mean of samples
DIFFS = [("flow", "flow matching (predict v)", VERMILLION, 0.004, -0.014, 0.022),
         ("flow_x0", "flow matching (predict x0)", GREEN, 0.003, -0.016, 0.021),
         ("flow_eps", "flow matching (predict noise)", GRAY, -0.092, -0.113, -0.072),
         ("flow_nonorm", "flow, no normalization", VERMILLION, -0.017, -0.037, 0.001),
         ("flow_mmgt", "flow, multi-modal targets", VERMILLION, 0.001, -0.016, 0.018),
         ("flow_rw1", "flow + reward fine-tuning", VERMILLION, 0.017, 0.001, 0.034),
         ("resid", "residual flow", PURPLE, -0.002, -0.018, 0.014),
         ("resid_x0", "residual flow (predict x0)", PURPLE, 0.004, -0.011, 0.018),
         ("resid_irp1", "residual + reference perturbation", PURPLE, -0.014, -0.033, 0.003),
         ("resid_mmgt", "residual, multi-modal targets", PURPLE, -0.001, -0.017, 0.015),
         ("resid_rw2", "residual + 2 rounds reward fine-tuning", PURPLE, -0.001, -0.016, 0.015)]


def end_results():
    res = pd.read_csv("outputs/19_hybrid/results.csv").set_index("label")
    base = res.loc["regress", "plan_mean"]
    for key, _, _, d, _, _ in DIFFS:                    # the hard-coded differences match results.csv
        assert abs(res.loc[key, "plan_mean"] - base - d) < 0.0015, key
    oracle = res.loc["flow", "plan_oracle"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1, 1]})
    # (a) full navtest
    rows = FULL + [("diffusion probe, best of 4 samples (oracle)", PROBE_ORACLE, None)]
    y = np.arange(len(rows))[::-1]
    for yi, (name, v, c) in zip(y, rows):
        if c is None:
            a1.scatter(v, yi, s=80, facecolor="white", edgecolor=VERMILLION, linewidth=2, zorder=3)
        else:
            a1.barh(yi, v, height=0.6, color=c)
        a1.text(v + (0.022 if c is None else 0.012), yi, f"{v:.3f}", va="center", color=INK, fontsize=10)
    a1.set_yticks(y, [r[0] for r in rows])
    a1.set_xlim(0, 1.0)
    a1.set_xlabel("EPDMS (higher is better)", color=INK)
    a1.set_title("(a) Full navtest (12,146 scenes), official EPDMS", color=INK)
    style(a1)
    a1.grid(axis="y", visible=False)

    # (b) difference to regression, navtest 500 subset
    y = np.arange(len(DIFFS) + 1)[::-1]
    for yi, (key, name, c, d, lo, hi) in zip(y, DIFFS):
        a2.plot([lo, hi], [yi, yi], color=c, lw=2, solid_capstyle="round")
        a2.scatter(d, yi, s=60, color=c, edgecolor="white", linewidth=1.2, zorder=3)
        a2.text(hi + 0.004, yi, f"{res.loc[key, 'plan_mean']:.3f}", va="center", color=MUTED, fontsize=9)
    yo = y[-1]
    a2.scatter(oracle - base, yo, s=60, color=GRAY, edgecolor="white", linewidth=1.2, zorder=3)
    a2.text(oracle - base + 0.004, yo, f"{oracle:.3f}", va="center", color=MUTED, fontsize=9)
    a2.axvline(0, color=BLUE, lw=1.4)
    a2.set_yticks(y, [r[1] for r in DIFFS] + ["flow, best of 32 samples (oracle)"])
    a2.set_xlim(-0.125, 0.07)
    a2.set_ylim(-0.7, len(DIFFS) + 0.7)
    a2.set_xlabel("EPDMS difference to regression, with 95% bootstrap interval\n"
                  "(gray numbers next to the dots: absolute EPDMS)", color=INK)
    a2.set_title(f"(b) Same backbone and inputs, navtest 500 subset:\n"
                 f"EPDMS difference to regression ({base:.3f})", color=INK)
    style(a2)
    a2.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=3)
    fig.savefig(OUT / "end_results.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    probe_dynamics()
    same_backbone_dynamics()
    end_results()
    shutil.copy("outputs/19_hybrid/bev.png", OUT / "bev_examples.png")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
