"""One table over runs from their TensorBoard validation curves.

    python -m exp_navsim.eval.summarize_runs --runs logs_exp/07_* --out outputs/07_encoders/summary.csv

Per run and validation loader (val, train_sub, navtest, ...): the step and value of
the best (smoothed) energy score, and the final energy, minADE@16, mean ADE, spread
and dispersion. The loaders are 2000 fixed scenes with 16 samples each, so the
numbers are comparable between runs but noisier than eval/evaluate.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.eval.plot_runs import load_scalars

FINAL = ("energy", "ade_min", "ade_mean", "ade_of_mean", "spread", "dispersion", "fde_min")


def summarize(run):
    df = load_scalars(run)
    rows = []
    loaders = sorted({t.split("/")[0] for t in df.tag if t.endswith("/energy")})
    for loader in loaders:
        d = df[df.tag == f"{loader}/energy"].sort_values("step")
        smooth = np.convolve(np.pad(d.value.to_numpy(), 1, mode="edge"), np.ones(3) / 3, mode="valid")
        row = {"run": Path(run).name, "loader": loader, "best_step": int(d.step.iloc[np.argmin(smooth)]),
               "best_energy": float(smooth.min())}
        for m in FINAL:
            v = df[df.tag == f"{loader}/{m}"].sort_values("step")
            row[f"final_{m}"] = float(v.value.iloc[-1]) if len(v) else np.nan
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    table = pd.DataFrame([r for run in sorted(args.runs) for r in summarize(run)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
