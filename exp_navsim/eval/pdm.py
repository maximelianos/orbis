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
trajectory, and two rows built from the sample rows:
    best    the best sample of every scene (oracle best-of-K; navtest only)
    avg     the average over samples (expected score of one sample)

navhard EPDMS is the devkit's extended_pdm_score_combined row: per (current, previous)
frame pair, the stage-one score times the weighted stage-two score, averaged. Components
are reported per stage (<metric>_stage_one / _stage_two). This is the number of the
navhard leaderboard and of py123d_garage's NAVSIM v2 table; the plain mean of the
per-token scores (what this script reported before 11-04) is a different, higher number.

    # rebuild summaries from CSVs already on disk, no scoring
    python -m exp_navsim.eval.pdm reaggregate --split navhard --predictions logs_exp/x/eval/navhard_last.npz \
        --samples 0 --mean
"""

import argparse
import os
import subprocess
import sys
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
    cmd = [sys.executable, str(DEVKIT / script), *overrides]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


TWO_STAGE_ROWS = ("extended_pdm_score_stage_one", "extended_pdm_score_stage_two", "extended_pdm_score_combined")


def read_csv(csv):
    """navsim CSV -> (per-token rows, two-stage summary rows indexed by name or None)."""
    df = pd.read_csv(csv, index_col=0)
    if "token" not in df:
        return df, None
    summary = df[df["token"].isin(TWO_STAGE_ROWS)].set_index("token")
    df = df[~df["token"].isin(TWO_STAGE_ROWS + ("average_all_frames", "average"))]
    return df, (summary if len(summary) else None)


def score_one(split, predictions, sample, workers, rescore=True):
    out = Path(predictions).parent / "pdm" / f"{Path(predictions).stem}_{sample}"
    if not list(out.glob("*.csv")):
        assert rescore, f"no scoring CSV in {out}"
        run(SPLITS[split]["script"], common_overrides(split, workers) + [
            f"experiment_name=probe_{sample}", f"output_dir={out}",
            "agent._target_=exp_navsim.eval.navsim_agent.PrecomputedAgent",
            f"+agent.predictions={Path(predictions).resolve()}", f"+agent.sample={sample}"])
    return read_csv(sorted(out.glob("*.csv"))[-1])


def summary_row(scored, name):
    df, two_stage = scored
    row = {"sample": name, "scenes": int(df["valid"].sum()) if "valid" in df else len(df)}
    if two_stage is None:
        row.update({c: float(df[c].mean()) for c in COMPONENTS if c in df.columns})
        return row
    combined = two_stage.loc["extended_pdm_score_combined"]
    row["score"] = float(combined["score"])
    row["score_stage_one"] = float(two_stage.loc["extended_pdm_score_stage_one", "score"])
    row["score_stage_two"] = float(two_stage.loc["extended_pdm_score_stage_two", "score"])
    for stage in ("stage_one", "stage_two"):
        row.update({f"{c}_{stage}": float(combined[f"{c}_{stage}"]) for c in COMPONENTS[:-1]
                    if f"{c}_{stage}" in combined})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["cache", "score", "reaggregate"])
    ap.add_argument("--split", default="navtest", choices=list(SPLITS))
    ap.add_argument("--predictions")
    ap.add_argument("--samples", default="0")
    ap.add_argument("--mean", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    if args.action == "cache":
        run("run_metric_caching.py", common_overrides(args.split, args.workers))
        return

    rescore = args.action == "score"
    rows, per_sample = [], []
    for k in args.samples.split(","):
        scored = score_one(args.split, args.predictions, int(k), args.workers, rescore)
        rows.append(summary_row(scored, k))
        per_sample.append(scored[0].set_index("token")["score"].rename(k))
    if len(per_sample) > 1:
        if scored[1] is None:  # per-scene oracle only makes sense for one-stage scores
            scores = pd.concat(per_sample, axis=1)
            rows.append({"sample": "best", "score": float(scores.max(1).mean())})
        rows.append({"sample": "avg", "score": float(pd.DataFrame(rows)["score"].mean())})
    if args.mean:
        rows.append(summary_row(score_one(args.split, args.predictions, "mean", args.workers, rescore), "mean"))
    tag = "s" + args.samples.replace(",", "") + ("_mean" if args.mean else "")
    out = Path(args.predictions).with_name(f"{Path(args.predictions).stem}_pdm_{tag}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
