"""EPDMS of probe predictions with the official navsim scoring scripts.

    # once per split: metric cache (CPU, map + agents of every scene)
    python -m exp_navsim.eval.pdm cache --split navtest
    # score saved predictions: every sample index in --samples, plus their mean
    python -m exp_navsim.eval.pdm score --split navtest --predictions logs_exp/x/eval/navtest_last.npz \
        --samples 0,1,2,3 --mean

navtest is scored one-stage (run_pdm_score_one_stage.py), navhard_two_stage with the
two-stage aggregation (run_pdm_score.py). Each scoring run writes the navsim CSV
under <predictions dir>/pdm/<split>_<sample>/; the summary <stem>_pdm_<samples>.csv holds
one row per sample index with EPDMS and its components, a row for the mean
trajectory, and two rows built per scene from the sample rows:
    best    the best sample of every scene (oracle best-of-K)
    avg     the average over samples of every scene (expected score of one sample)
"""

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd

DEVKIT = Path("navsim/navsim/planning/script")
DOWNLOAD = Path("/scratch/global/velikanm/navsim/download")
SPLITS = {
    "navtest": dict(script="run_pdm_score_one_stage.py", split="navtest", logs=DOWNLOAD / "test_navsim_logs/test",
                    sensors=DOWNLOAD / "test_sensor_blobs/test"),
    "navhard": dict(script="run_pdm_score.py", split="navhard_two_stage", logs=DOWNLOAD / "test_navsim_logs/test",
                    sensors=DOWNLOAD / "navhard_two_stage/sensor_blobs"),
}
CACHE = Path("outputs/navsim_exp/metric_cache")
COMPONENTS = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
              "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
              "history_comfort", "two_frame_extended_comfort", "score")


def common_overrides(split, workers):
    s = SPLITS[split]
    return [f"train_test_split={s['split']}", f"navsim_log_path={s['logs']}", f"original_sensor_path={s['sensors']}",
            f"synthetic_sensor_path={DOWNLOAD / 'navhard_two_stage/sensor_blobs'}",
            f"synthetic_scenes_path={DOWNLOAD / 'navhard_two_stage/synthetic_scene_pickles'}",
            f"metric_cache_path={CACHE / split}", "worker=single_machine_thread_pool",
            "worker.use_process_pool=true", f"worker.max_workers={workers}",
            "logger_level=warning"]  # hides the per-scenario "Processing scenario i / n" lines


def run(script, overrides):
    env = {**os.environ, "NAVSIM_EXP_ROOT": str(Path("outputs/navsim_exp").resolve())}
    cmd = ["python", str(DEVKIT / script), *overrides]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def score_one(split, predictions, sample, workers):
    out = Path(predictions).parent / "pdm" / f"{Path(predictions).stem}_{sample}"
    if not list(out.glob("*.csv")):
        run(SPLITS[split]["script"], common_overrides(split, workers) + [
            f"experiment_name=probe_{sample}", f"output_dir={out}",
            "agent._target_=exp_navsim.eval.navsim_agent.PrecomputedAgent",
            f"+agent.predictions={Path(predictions).resolve()}", f"+agent.sample={sample}"])
    csv = sorted(out.glob("*.csv"))[-1]
    df = pd.read_csv(csv)
    return df[df["token"] != "average_all_frames"] if "token" in df else df


def summary_row(df, name):
    cols = [c for c in COMPONENTS if c in df.columns]
    row = {"sample": name, "scenes": int(df["valid"].sum()) if "valid" in df else len(df)}
    row.update({c: float(df[c].mean()) for c in cols})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["cache", "score"])
    ap.add_argument("--split", default="navtest", choices=list(SPLITS))
    ap.add_argument("--predictions")
    ap.add_argument("--samples", default="0")
    ap.add_argument("--mean", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    if args.action == "cache":
        run("run_metric_caching.py", common_overrides(args.split, args.workers))
        return

    rows, per_sample = [], []
    for k in args.samples.split(","):
        df = score_one(args.split, args.predictions, int(k), args.workers)
        rows.append(summary_row(df, k))
        per_sample.append(df.set_index("token")["score"].rename(k))
    if len(per_sample) > 1:
        scores = pd.concat(per_sample, axis=1)
        rows.append({"sample": "best", "score": float(scores.max(1).mean())})
        rows.append({"sample": "avg", "score": float(scores.mean(1).mean())})
    if args.mean:
        rows.append(summary_row(score_one(args.split, args.predictions, "mean", args.workers), "mean"))
    tag = "s" + args.samples.replace(",", "") + ("_mean" if args.mean else "")
    out = Path(args.predictions).with_name(f"{Path(args.predictions).stem}_pdm_{tag}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
