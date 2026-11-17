"""Score saved predictions on navhard two-stage with the fast scorer, and keep the per-group rows.

    python -m exp_navsim.v22.score_runs --predictions logs_exp/<run>/navhard_top8.npz --name <run>
    python -m exp_navsim.v22.score_runs --baseline cv --name cv

Writes outputs/24_v22/per_group/<name>.csv (one row per group, half and candidate index) and
appends a summary row to outputs/24_v22/navhard.csv. Because the EPDMS is a plain mean over
groups, any group subset can be re-evaluated from the per-group file without scoring again
(`proxy.py`), which is how the proxy sizes are compared.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.eval.navsim_agent import mean_trajectory
from exp_navsim.v22.fast_epdms import COLS, FastEPDMS, all_groups, group_tokens

OUT = Path("outputs/24_v22")


def constant_velocity(status, dt=0.5, horizon=8):
    """(N, 4) ego status -> (N, 8, 3), exactly navsim's ConstantVelocityAgent: the current
    *speed* (the norm of the 2D velocity) held straight ahead, no lateral motion."""
    speed = np.linalg.norm(status[:, :2], axis=-1)
    t = np.arange(1, horizon + 1)[None] * dt
    x = speed[:, None] * t
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], -1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--baseline", default=None, choices=("cv", "human"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--sample", default="0", help="sample index, or 'mean' over the saved samples")
    ap.add_argument("--groups", type=int, default=0, help="number of groups; 0 = the whole split")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    groups = all_groups()
    groups = groups if not args.groups else groups[:args.groups]
    tokens = group_tokens(groups)

    if args.baseline == "cv":
        from exp_navsim.data.scenes import load_index
        ix = load_index("data/navsim_index/navhard.pkl")
        status = np.concatenate([ix["velocity"], ix["acceleration"]], 1)
        poses = constant_velocity(status)
        lookup = dict(zip([str(t) for t in ix["token"]], poses))
        n_samples = 1
    else:
        data = np.load(args.predictions)
        s = data["samples"].astype(np.float32)
        poses = mean_trajectory(s) if args.sample == "mean" else s[int(args.sample)]
        lookup = dict(zip(data["token"].tolist(), poses))
        n_samples = 1

    scorer = FastEPDMS(groups, workers=args.workers)
    t0 = time.time()
    df = scorer.score({t: lookup[t][None] for t in tokens})
    s = scorer.summary(df)
    (OUT / "per_group").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_group" / f"{args.name}.csv", index=False)
    row = {"name": args.name, "groups": len(groups), "n_samples": n_samples,
           "scored_tokens": scorer.n_scored, "seconds": round(time.time() - t0),
           "epdms": round(100 * s.loc[0, "score"], 3)}
    for m in COLS[:-1]:
        row[f"{m}_s1"] = round(100 * s.loc[0, f"{m}_stage_one"], 2)
        row[f"{m}_s2"] = round(100 * s.loc[0, f"{m}_stage_two"], 2)
    path = OUT / "navhard.csv"
    out = pd.concat([pd.read_csv(path), pd.DataFrame([row])]) if path.exists() else pd.DataFrame([row])
    out.drop_duplicates(subset=["name", "groups"], keep="last").to_csv(path, index=False)
    print(f"{args.name}: EPDMS {row['epdms']:.2f} on {len(groups)} groups "
          f"({scorer.n_scored} tokens, {row['seconds']} s)")
    print("  stage one " + " ".join(f"{m[:4]} {row[f'{m}_s1']:.1f}" for m in COLS[:-1]))
    print("  stage two " + " ".join(f"{m[:4]} {row[f'{m}_s2']:.1f}" for m in COLS[:-1]))
    scorer.close()


if __name__ == "__main__":
    main()
