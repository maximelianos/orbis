"""Evaluation curves of several IntentFlow runs (their evals.csv) side by side.

    python -m exp_navsim.diverse.plot_curves --runs 17_plain_h64_10k,17_full_h64_10k \
        --out outputs/17_diversity/curves_10k.png

Solid: held-out logs (val), dashed: training scenes (train_sub). Sample metrics use the
proportional mode (k ~ p(k | scene)); the best-of-K panel also shows the stratified mode
(dotted) for runs with anchors. Prints the last and best-energy rows as a table.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PANELS = [("prop/energy", "energy score (lower better)"), ("prop/ade_min", "minADE@32 [m]"),
          ("prop/dispersion", "dispersion = spread / mean ADE"), ("prop/spread", "spread [m]"),
          ("intent/nll", "intent NLL of nearest anchor"), ("comfort/any", "samples breaking a comfort bound")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = args.runs.split(",")
    labels = args.labels.split(",") if args.labels else runs
    dfs = [pd.read_csv(Path("logs_exp") / r / "evals.csv") for r in runs]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title) in zip(axes.flat, PANELS):
        for i, (df, lab) in enumerate(zip(dfs, labels)):
            c = f"C{i}"
            if f"val/{key}" not in df:
                continue
            ax.plot(df.step, df[f"val/{key}"], color=c, label=lab)
            ax.plot(df.step, df[f"train_sub/{key}"], color=c, ls="--", alpha=0.6)
            strat = f"val/{key.replace('prop/', 'strat/')}"
            if key == "prop/ade_min" and strat in df:
                ax.plot(df.step, df[strat], color=c, ls=":", lw=2)
        if key == "prop/dispersion":
            ax.axhline(1, color="gray", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, title="solid val, dashed train_sub\ndotted: stratified", title_fontsize=7)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)

    cols = ["prop/energy", "prop/ade_min", "prop/ade_mean", "prop/ade_of_mean", "prop/spread", "prop/dispersion",
            "strat/energy", "strat/ade_min", "strat/dispersion", "intent/acc", "intent/nll", "intent/entropy",
            "intent/distinct", "kin/speed_r2", "comfort/any"]
    out = []
    for df, lab in zip(dfs, labels):
        for tag, r in (("best", df.loc[df["val/prop/energy"].idxmin()]), ("last", df.iloc[-1])):
            out.append({"run": lab, "at": f"{tag} {int(r.step)}",
                        **{c: round(r[f"val/{c}"], 3) for c in cols if f"val/{c}" in r}})
    table = pd.DataFrame(out)
    table.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print(table.to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
