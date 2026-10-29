"""Pick a small navtest subset whose mean EPDMS tracks the full navtest score.

    python -m exp_navsim.planner.subset --sizes 50,100,200,500 --out outputs/16_fast_planner/subset

The full one-stage navtest runs of earlier probes (logs_exp/*/eval/pdm/navtest_*/*.csv
and the constant-velocity baseline) give per-scene EPDMS for ~20 planners. A subset
is good when the subset mean of every planner is close to its full mean and the
ranking of the planners is kept. Two ways of drawing it are compared:
    random       uniform over scenes
    stratified   equal share of every (command, difficulty quintile) cell, where the
                 difficulty of a scene is its mean score over all planners, and at
                 most `per_log` scenes from one log
Only scenes with a predecessor in navtest are drawn (their two-frame extended
comfort is defined), the fast scorer then scores the predecessor as well.

Stratifying does not beat random draws, so the saved subsets are random.
Writes <out>_<size>.json (tokens and their predecessors), <out>_proxy.csv and a plot.
"""

import argparse
import glob
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_runs():
    files = sorted(glob.glob("logs_exp/*/eval/pdm/navtest_*/*.csv")) + sorted(glob.glob("outputs/11_epdms/cv_navtest/*.csv"))
    scores, comfort = {}, None
    for f in files:
        name = f.split("/")[1] + ":" + f.split("/")[-2] if f.startswith("logs_exp") else "constant_velocity"
        d = pd.read_csv(f)
        d = d[d["token"] != "average_all_frames"].set_index("token")
        scores[name] = d["score"]
        if comfort is None:
            comfort = d["two_frame_extended_comfort"]
    return pd.DataFrame(scores), comfort


def predecessors(tokens):
    """token -> previous token of the same log in navtest (frames 0.5 s apart)."""
    ix = pickle.load(open("data/navsim_index/navtest.pkl", "rb"))
    order = {}
    for i, (t, log) in enumerate(zip(ix["token"], ix["log"])):
        order.setdefault(log, []).append(t)
    # build_index keeps the scene order of the log, consecutive scenes are 1 frame apart
    prev = {}
    for log, ts in order.items():
        for a, b in zip(ts[:-1], ts[1:]):
            prev[b] = a
    command = dict(zip(ix["token"], ix["command"].argmax(1)))
    logs = dict(zip(ix["token"], ix["log"]))
    return {t: prev.get(t) for t in tokens}, command, logs


def draw(rng, cands, size, how, command, difficulty, logs, per_log=3):
    if how == "random":
        return list(rng.choice(cands, size, replace=False))
    q = pd.qcut(difficulty[cands], 5, labels=False, duplicates="drop").to_numpy()
    cells = {}
    for t, d in zip(cands, q):
        cells.setdefault((command[t], d), []).append(t)
    picked, per = [], {}
    # round robin over cells, each cell in random order, so every cell gets its share
    queues = [list(rng.permutation(v)) for v in cells.values()]
    weights = np.array([len(v) for v in queues], float)
    quota = np.maximum(1, np.round(size * weights / weights.sum())).astype(int)
    for qu, k in zip(queues, quota):
        n = 0
        for t in qu:
            if n >= k:
                break
            if per.get(logs[t], 0) < per_log:
                picked.append(t)
                per[logs[t]] = per.get(logs[t], 0) + 1
                n += 1
    return picked[:size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="25,50,100,200,500,1000")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--out", default="outputs/16_fast_planner/subset")
    args = ap.parse_args()

    S, comfort = load_runs()
    S = S.dropna()
    print(f"{S.shape[1]} planners, {len(S)} scenes")
    full = S.mean()
    prev, command, logs = predecessors(S.index)
    cands = np.array([t for t in S.index if prev[t] in S.index and not np.isnan(comfort.get(t, np.nan))])
    difficulty = S.mean(1)
    print(f"{len(cands)} candidate scenes with a scored predecessor")

    rows = []
    for how in ("random", "stratified"):
        for size in map(int, args.sizes.split(",")):
            for r in range(args.repeats):
                sub = draw(np.random.default_rng(r), cands, size, how, command, difficulty, logs)
                m = S.loc[sub].mean()
                rows.append(dict(how=how, size=size, seed=r, mae=float((m - full).abs().mean()),
                                 max_err=float((m - full).abs().max()), spearman=float(spearmanr(m, full)[0])))
    df = pd.DataFrame(rows)
    agg = df.groupby(["how", "size"]).agg(mae=("mae", "mean"), max_err=("max_err", "mean"),
                                         spearman=("spearman", "mean"), spearman_p10=("spearman", lambda x: x.quantile(.1)))
    print(agg.to_string())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(f"{args.out}_proxy.csv")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for how, g in agg.reset_index().groupby("how"):
        ax[0].plot(g["size"], g["mae"], "o-", label=how)
        ax[1].plot(g["size"], g["spearman"], "o-", label=how)
    for a, t in zip(ax, ("mean |subset - full| EPDMS", "Spearman rank vs full")):
        a.set_xscale("log"); a.set_xlabel("scenes in subset"); a.set_title(t); a.grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout(); fig.savefig(f"{args.out}_proxy.png", dpi=120)

    # the subsets used later: random, seed 0 (stratifying did not help, see the table)
    for size in map(int, args.sizes.split(",")):
        sub = draw(np.random.default_rng(0), cands, size, "random", command, difficulty, logs)
        json.dump({"tokens": [str(t) for t in sub], "previous": [str(prev[t]) for t in sub]},
                  open(f"{args.out}_{size}.json", "w"))
        m = S.loc[sub].mean()
        print(f"size {size}: mae {float((m - full).abs().mean()):.4f}  spearman {spearmanr(m, full)[0]:.3f}")


if __name__ == "__main__":
    main()
