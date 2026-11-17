"""How well a few navhard groups stand in for the whole split.

    python -m exp_navsim.v22.proxy

The EPDMS of navhard two-stage is a plain mean over the 225 groups, so the per-group files
that score_runs.py writes allow any subset to be re-evaluated without scoring again. For
every run that has been scored on the whole split, this compares the subset mean with the
full mean over many random draws, and also the *paired* difference between two runs, which
is what an ablation actually needs.

-> outputs/24_v22/proxy.csv, proxy.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("outputs/24_v22")
SIZES = (8, 16, 32, 64, 112, 160)
DRAWS = 200


def group_scores():
    """{run: (225,) EPDMS per group} from the per-group files of the full-split runs."""
    out = {}
    for path in sorted((OUT / "per_group").glob("*.csv")):
        df = pd.read_csv(path)
        df = df[df.k == 0]
        s = df.groupby("group").score.mean()
        if len(s) == 225:
            out[path.stem] = s.to_numpy()
    return out


def main():
    runs = group_scores()
    if len(runs) < 2:
        print(f"need at least two runs scored on all 225 groups, have {list(runs)}")
        if not runs:
            return
    rng = np.random.default_rng(0)
    names = list(runs)
    rows = []
    for n in SIZES:
        abs_err, rank_err = [], []
        for _ in range(DRAWS):
            idx = rng.choice(225, n, replace=False)
            for a in names:
                abs_err.append(abs(runs[a][idx].mean() - runs[a].mean()))
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    full = runs[a].mean() - runs[b].mean()
                    sub = runs[a][idx].mean() - runs[b][idx].mean()
                    rank_err.append(abs(sub - full))
        rows.append({"groups": n, "tokens_approx": round(n * 5912 / 225),
                     "abs_error": 100 * np.mean(abs_err), "abs_error_p95": 100 * np.quantile(abs_err, 0.95),
                     "paired_error": 100 * np.mean(rank_err),
                     "paired_error_p95": 100 * np.quantile(rank_err, 0.95)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "proxy.csv", index=False)
    print(f"runs: {', '.join(f'{n} {100 * runs[n].mean():.2f}' for n in names)}")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=140)
    ax.plot(df.groups, df.abs_error, "o-", label="absolute EPDMS of one run")
    ax.plot(df.groups, df.paired_error, "s-", label="difference between two runs")
    ax.fill_between(df.groups, 0, df.abs_error_p95, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels(SIZES)
    ax.set_xlabel("navhard groups in the subset (225 = whole split)")
    ax.set_ylabel("mean absolute error [EPDMS points]")
    ax.set_title("A group subset as a proxy for navhard two-stage")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "proxy.png")
    print(f"-> {OUT / 'proxy.png'}")


def select(n, draws=4000, seed=0):
    """Choose the `n` groups that stand in best for the whole split.

    A plain random draw can be far off: the first 32 groups of the training permutation give
    the fast planner 17.7 instead of 25.98. The groups are therefore drawn stratified by the
    decile of that planner's per-group score, so easy and hard groups keep their proportion,
    and of many such draws the one with the smallest largest error over every run scored on
    the whole split is kept.
    """
    from exp_navsim.v22.fast_epdms import all_groups, save_groups
    runs = group_scores()
    if not runs:
        print("no run scored on all 225 groups yet")
        return
    ref = runs[max(runs, key=lambda k: runs[k].mean())]            # the strongest run
    decile = np.minimum((ref.argsort().argsort() * 10) // len(ref), 9)
    rng = np.random.default_rng(seed)
    best, best_err = None, np.inf
    for _ in range(draws):
        idx = np.concatenate([rng.choice(np.flatnonzero(decile == d),
                                         max(1, round(n * (decile == d).mean())), replace=False)
                              for d in range(10)])
        idx = rng.permutation(idx)[:n]
        err = max(abs(s[idx].mean() - s.mean()) for s in runs.values())
        if err < best_err:
            best, best_err = idx, err
    groups = all_groups()
    path = OUT / f"groups_{n}.json"
    save_groups([groups[i] for i in sorted(best)], path)
    print(f"-> {path}  largest error over {len(runs)} runs {100 * best_err:.2f} EPDMS points")
    for name, s in sorted(runs.items(), key=lambda kv: kv[1].mean()):
        print(f"  {name:34s} full {100 * s.mean():6.2f}   subset {100 * s[best].mean():6.2f}")


def on_subset(n, seed=0):
    """EPDMS of every full-split run on the first n groups of train.pick_groups(n)."""
    from exp_navsim.v22.fast_epdms import all_groups
    order = np.random.default_rng(seed).permutation(len(all_groups()))[:n] if n else None
    runs = group_scores()
    for name, s in sorted(runs.items(), key=lambda kv: kv[1].mean()):
        sub = s.mean() if order is None else s[order].mean()
        print(f"{name:34s} full {100 * s.mean():6.2f}   {n or 225} groups {100 * sub:6.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "select":
        select(int(sys.argv[2]))
    elif len(sys.argv) > 1:
        on_subset(int(sys.argv[1]))
    else:
        main()
