"""Memorization onset vs training-set size from the validation curves of several runs.

    python -m exp_navsim.eval.scaling --runs logs_exp/04_qsem_n2500 ... logs_exp/02_qsem_full \
        --sizes 2500 5000 10000 20000 46000 92381 --out outputs/04_data_scaling

Per run, from the TensorBoard scalars (2000 validation scenes, 16 samples, every
2000 steps):
    best_step     step of the lowest val energy score (the best distribution)
    loss_step     step of the lowest val flow-matching loss (fixed t and noise, the
                  cleanest curve): after it the model fits seen scenes at the cost of
                  unseen ones
    gap_step      first step where the val - train_sub energy gap has grown by `gap`
                  over its value at the first validation (the two splits come from
                  different logs and differ a little from the start)
    censored      the optimum is the last validation: onset not reached in 100k steps
    final_*       val energy, dispersion and train_sub / val minADE at the end
Bonnaire et al. (NeurIPS 2025) predict a memorization time proportional to n, so
gap_step and best_step should grow linearly; the fitted log-log slope is printed.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp_navsim.eval.plot_runs import load_scalars


def curve(df, tag):
    d = df[df.tag == tag].sort_values("step")
    return d.step.to_numpy(), d.value.to_numpy()


def onset(run, gap):
    df = load_scalars(run)
    steps, val = curve(df, "val/energy")
    _, tr = curve(df, "train_sub/energy")
    _, disp = curve(df, "val/dispersion")
    _, vmin = curve(df, "val/ade_min")
    _, tmin = curve(df, "train_sub/ade_min")
    # A little smoothing: one validation round is 2000 scenes and noisy.
    smooth = lambda x: np.convolve(np.pad(x, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    diff = smooth(val) - smooth(tr)
    over = np.flatnonzero(diff - diff[0] > gap)
    lsteps, loss = curve(df, "val/loss")
    loss_step = int(lsteps[np.argmin(smooth(loss))])
    return {
        "loss_step": loss_step,
        "censored": loss_step >= lsteps[-3],
        "best_step": int(steps[np.argmin(smooth(val))]),
        "best_energy": float(smooth(val).min()),
        "gap_step": int(steps[over[0]]) if len(over) else np.nan,
        "final_energy": float(val[-1]), "final_dispersion": float(disp[-1]),
        "final_val_minade": float(vmin[-1]), "final_train_minade": float(tmin[-1]),
        "final_gap": float(diff[-1]),
    }, (steps, val, tr, disp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, required=True)
    ap.add_argument("--gap", type=float, default=0.3, help="growth of the energy gap [m] that counts as memorization")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, curves = [], {}
    for run, n in zip(args.runs, args.sizes):
        row, c = onset(run, args.gap)
        rows.append({"run": Path(run).name, "n": n, **row})
        curves[n] = c
    table = pd.DataFrame(rows).sort_values("n")
    table.to_csv(out / "onset.csv", index=False)
    print(table.to_string())
    for key in ("loss_step", "gap_step", "best_step"):
        ok = table[key].notna() & (table[key] > 0) & ~table.censored
        if ok.sum() >= 2:
            slope = np.polyfit(np.log(table.n[ok]), np.log(table[key][ok]), 1)[0]
            print(f"log-log slope of {key} vs n: {slope:.2f} (1 = linear)")

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    cmap = plt.get_cmap("viridis")
    for i, n in enumerate(sorted(curves)):
        steps, val, tr, disp = curves[n]
        color = cmap(i / max(len(curves) - 1, 1))
        axes[0].plot(steps, val, color=color, label=f"n={n}")
        axes[0].plot(steps, tr, color=color, ls=":")
        axes[1].plot(steps, (val - tr) - (val - tr)[0], color=color)
        axes[2].plot(steps, disp, color=color)
    axes[0].set_title("energy score (solid val, dotted train)", fontsize=9)
    axes[1].set_title("growth of val - train energy gap", fontsize=9)
    axes[1].axhline(args.gap, color="gray", ls="--", lw=0.8)
    axes[2].set_title("val dispersion (1 = calibrated)", fontsize=9)
    axes[2].axhline(1.0, color="gray", ls="--", lw=0.8)
    for ax in axes[:3]:
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    ok = ~table.censored
    axes[3].loglog(table.n[ok], table.loss_step[ok], "o-", label="min val loss")
    axes[3].loglog(table.n[ok], table.gap_step[ok], "s--", label="gap growth > %.1f" % args.gap)
    axes[3].loglog(table.n[~ok], table.loss_step[~ok], "x", color="gray", label="not reached (censored)")
    axes[3].set_xlabel("training scenes n")
    axes[3].set_ylabel("step")
    axes[3].grid(alpha=0.3, which="both")
    axes[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "scaling.png", dpi=150)
    print(f"-> {out / 'scaling.png'}")


if __name__ == "__main__":
    main()
