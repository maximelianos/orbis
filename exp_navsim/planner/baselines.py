"""Reference planners on the navtest subsets, scored with the fast EPDMS (fast_pdm.py).

    python -m exp_navsim.planner.baselines

constant_velocity   current speed held for 4 s straight ahead (navsim ConstantVelocityAgent)
human               the logged future (upper reference, privileged)
pdm_closed          PDM-Closed from the metric cache: rule-based planner with the map,
                    the route and the true agents (privileged)
diffusion probes    earlier flow-matching probes (logs_exp/<run>/eval/navtest_last.npz),
                    sample 0 and the mean of the samples

Writes outputs/16_fast_planner/baselines.csv: EPDMS and sub-scores per planner and subset.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.eval.navsim_agent import mean_trajectory
from exp_navsim.planner.fast_pdm import FastPDM, load_subset

OUT = Path("outputs/16_fast_planner")
PROBES = ("08_wm11_ctx", "07_status_cmd")


def main():
    subsets = {n: load_subset(OUT / f"subset_{n}.json") for n in (100, 500)}
    tokens = sorted({t for toks, prev in subsets.values() for t in [*toks, *prev.values()]})
    scorer = FastPDM("navtest", tokens)

    ix = pickle.load(open("data/navsim_index/navtest.pkl", "rb"))
    pos = {t: i for i, t in enumerate(ix["token"])}
    speed = np.linalg.norm(ix["velocity"], axis=1)
    t = 0.5 * np.arange(1, 9)
    cv = {tok: np.stack([speed[pos[tok]] * t, np.zeros(8), np.zeros(8)], 1) for tok in tokens}
    ref = scorer.reference_poses(tokens)
    planners = {"constant_velocity": cv,
                "human": {tok: ref[tok][1] for tok in tokens},
                "pdm_closed": {tok: ref[tok][0] for tok in tokens}}
    for run in PROBES:
        d = np.load(f"logs_exp/{run}/eval/navtest_last.npz")
        s = d["samples"].astype(np.float32)
        for name, poses in ((f"diffusion_{run}_s0", s[0]), (f"diffusion_{run}_mean", mean_trajectory(s))):
            lookup = dict(zip(d["token"].tolist(), poses))
            planners[name] = {tok: lookup[tok] for tok in tokens}

    rows = []
    for name, traj in planners.items():
        for n, (toks, prev) in subsets.items():
            df = scorer.score({tok: traj[tok][None] for tok in traj}, prev)
            r = {"planner": name, "subset": n, **df.drop(columns=["token", "k"]).mean().round(4).to_dict()}
            rows.append(r)
            print(f"{name:32s} subset {n}: EPDMS {r['score']:.4f}", flush=True)
    scorer.close()
    pd.DataFrame(rows).to_csv(OUT / "baselines.csv", index=False)
    print(f"-> {OUT / 'baselines.csv'}")


if __name__ == "__main__":
    main()
