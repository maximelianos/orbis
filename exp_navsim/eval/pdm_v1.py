"""NAVSIM v1.1 PDMS on navtest, for comparison with v1 tables (e.g. py123d_garage).

The v1.1 devkit is checked out at navsim_v1/ (tag v1.1). Its scorer differs from v2.2:
PDMS = NC * DAC * (5 EP + 5 TTC + 2 C) / 12, no human-penalty filter, its own metric cache.
Run with the v1.1 devkit first on the path:

    export PYTHONPATH=navsim_v1:.
    python -m exp_navsim.eval.pdm_v1 cache
    python -m exp_navsim.eval.pdm_v1 score --predictions logs_exp/x/navtest_top8.npz --sample 0
    python -m exp_navsim.eval.pdm_v1 score --predictions cv          # constant velocity check (paper: 20.6)

score writes a submission pickle {"predictions": [{token: Trajectory}]} next to the
predictions and runs the v1.1 run_pdm_score_from_submission.py on it, as py123d_garage
does. Tokens without a prediction get constant velocity. The result CSV goes to
<predictions dir>/pdm_v1/<stem>_<sample>/; its "average" row is the PDMS.
"""

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEVKIT = Path("navsim_v1/navsim/planning/script")
LOGS = Path("/scratch/global/velikanm/navsim/download/test_navsim_logs/test")
CACHE = Path("outputs/navsim_exp/metric_cache_v1/navtest")
OUT_CV = Path("outputs/21_navsim_metrics/cv_navtest_v1")
COMPONENTS = ("no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
              "comfort", "ego_progress", "driving_direction_compliance", "score")


def run(script, overrides):
    env = {**os.environ, "NAVSIM_EXP_ROOT": str(Path("outputs/navsim_exp").resolve())}
    cmd = [sys.executable, str(DEVKIT / script), *overrides]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def constant_velocity(metric_cache):
    speed = float(metric_cache.ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
    t = 0.5 * np.arange(1, 9)
    return np.stack([speed * t, np.zeros(8), np.zeros(8)], 1).astype(np.float32)


def build_submission(predictions, sample, path):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from navsim.common.dataclasses import Trajectory
    from navsim.common.dataloader import MetricCacheLoader
    from exp_navsim.eval.navsim_agent import mean_trajectory

    poses = {}
    if predictions != "cv":
        data = np.load(predictions)
        samples = data["samples"].astype(np.float32)                    # (K, N, 8, 3)
        chosen = mean_trajectory(samples) if sample == "mean" else samples[int(sample)]
        poses = dict(zip(data["token"].tolist(), chosen))
    sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    loader = MetricCacheLoader(CACHE)
    out, fallbacks = {}, 0
    for token in loader.tokens:
        p = poses.get(token)
        if p is None:
            p = constant_velocity(loader.get_from_token(token))
            fallbacks += 1
        out[token] = Trajectory(p, sampling)
    print(f"{len(out)} tokens, {fallbacks} constant-velocity fallbacks", flush=True)
    with open(path, "wb") as f:
        pickle.dump({"team_name": "orbis", "authors": [], "email": "", "institution": "",
                     "country / region": "", "predictions": [out]}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["cache", "score"])
    ap.add_argument("--predictions", help="npz from evaluate.py / planner, or 'cv'")
    ap.add_argument("--sample", default="0", help="sample index or 'mean'")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    import navsim
    assert "navsim_v1" in navsim.__file__, f"v1.1 devkit must come first on PYTHONPATH, got {navsim.__file__}"

    if args.action == "cache":
        run("run_metric_caching.py", ["train_test_split=navtest", f"navsim_log_path={LOGS}",
                                      f"cache.cache_path={CACHE.resolve()}",
                                      "worker=single_machine_thread_pool", "worker.use_process_pool=true",
                                      f"worker.max_workers={args.workers}"])
        return

    if args.predictions == "cv":
        out = OUT_CV
    else:
        out = Path(args.predictions).parent / "pdm_v1" / f"{Path(args.predictions).stem}_{args.sample}"
    out.mkdir(parents=True, exist_ok=True)
    if not list(out.glob("*.csv")):
        submission = out / "submission.pkl"
        build_submission(args.predictions, args.sample, submission)
        run("run_pdm_score_from_submission.py", ["train_test_split=navtest", f"submission_file_path={submission.resolve()}",
                                                 f"metric_cache_path={CACHE.resolve()}",
                                                 f"output_dir={out.resolve()}", "worker=sequential"])
    df = pd.read_csv(sorted(out.glob("*.csv"))[-1], index_col=0)
    avg = df[df["token"] == "average"].iloc[0]
    rows = df[df["token"] != "average"]
    print(f"PDMS {100 * avg['score']:.1f} on {int(rows['valid'].sum())} / {len(rows)} valid tokens")
    print(" ".join(f"{c} {100 * float(avg[c]):.1f}" for c in COMPONENTS if c in avg))


if __name__ == "__main__":
    main()
