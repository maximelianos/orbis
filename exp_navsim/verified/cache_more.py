"""Metric cache for more navtrain training scenes (verifier training data).

    python -m exp_navsim.verified.cache_more --scenes 5000 --workers 8
    python -m exp_navsim.verified.cache_more --scenes 15000 --name navtrain_sub3 --seed 2 \
        --exclude outputs/16_fast_planner/navtrain_sub_tokens.json,outputs/18_verified_sampling/navtrain_sub2_tokens.json

Like planner/cache_navtrain.py, but a new split name (navtrain_sub2) and token list, and
the 5000 scenes of navtrain_sub are left out. Training logs of the probe split only.
navsim's cache_data gives each worker a quarter of all logs at once (OOM at the 40 GB job
limit for 1051 logs), so the logs go to the workers in small chunks instead, scenes that
are already cached are skipped (resume), and the metadata csv is written from the files.
-> outputs/navsim_exp/metric_cache/navtrain_sub2, outputs/18_verified_sampling/navtrain_sub2_tokens.json
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from exp_navsim.data.scenes import load_index, log_in_split
from exp_navsim.eval.pdm import CACHE, DOWNLOAD
from exp_navsim.planner.fast_pdm import CONFIG_DIR

OUT = Path("outputs/18_verified_sampling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--logs-per-task", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default="navtrain_sub2", help="split name of the new metric cache")
    ap.add_argument("--exclude", default="outputs/16_fast_planner/navtrain_sub_tokens.json",
                    help="comma-separated token jsons whose scenes are left out")
    args = ap.parse_args()

    ix = load_index("data/navsim_index/navtrain.pkl")
    old = {r for f in args.exclude.split(",") for r in json.load(open(f))["rows"]}
    train = np.array([r for r in np.flatnonzero([log_in_split(l, "train", 0.1) for l in ix["log"]]) if r not in old])
    rows = np.sort(np.random.default_rng(args.seed).choice(train, args.scenes, replace=False))
    tokens = [str(ix["token"][r]) for r in rows]
    logs = sorted({str(ix["log"][r]) for r in rows})
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"tokens": tokens, "rows": rows.tolist()}, open(OUT / f"{args.name}_tokens.json", "w"))
    print(f"{len(tokens)} scenes from {len(logs)} logs", flush=True)

    os.environ["NAVSIM_EXP_ROOT"] = str(Path("outputs/navsim_exp").resolve())
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(DOWNLOAD))
    from hydra import compose, initialize_config_dir

    from navsim.planning.metric_caching.caching import cache_scenarios
    from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry, save_cache_metadata

    out = CACHE / args.name
    with initialize_config_dir(config_dir=str(CONFIG_DIR.parent / "metric_caching"), version_base=None):
        cfg = compose(config_name="default_metric_caching", overrides=[
            "train_test_split=navtrain", f"navsim_log_path={DOWNLOAD / 'trainval_navsim_logs/trainval'}",
            f"metric_cache_path={out.resolve()}", "force_feature_computation=false"])

    # tokens per log that have no cache file yet
    per_log = {}
    for r, t in zip(rows, tokens):
        per_log.setdefault(str(ix["log"][r]), []).append(t)
    todo = {l: [t for t in ts if cache_file(out, l, t) is None] for l, ts in per_log.items()}
    todo = {l: ts for l, ts in todo.items() if ts}
    n_todo = sum(len(ts) for ts in todo.values())
    print(f"{len(tokens) - n_todo} scenes already cached, {n_todo} to do in {len(todo)} logs", flush=True)

    items = [{"cfg": cfg, "log_file": l, "tokens": ts} for l, ts in todo.items()]
    tasks = [items[i:i + args.logs_per_task] for i in range(0, len(items), args.logs_per_task)]
    t0, done, fails = time.time(), 0, 0
    with mp.get_context("fork").Pool(args.workers, maxtasksperchild=4) as pool:
        for res in pool.imap_unordered(cache_scenarios, tasks):
            done += sum(r.successes + r.failures for r in res)
            fails += sum(r.failures for r in res)
            rate = done / (time.time() - t0)
            print(f"{done}/{n_todo} scenes  {fails} failed  {rate * 60:.0f} scenes/min  "
                  f"eta {(n_todo - done) / max(rate, 1e-6) / 60:.0f} min", flush=True)

    files = [cache_file(out, l, t) for l, ts in per_log.items() for t in ts]
    files = [f for f in files if f is not None]
    save_cache_metadata([CacheMetadataEntry(f) for f in files], out, 0)
    print(f"metadata: {len(files)}/{len(tokens)} scenes cached", flush=True)


def cache_file(out, log, token):
    """Path of a scene's metric_cache.pkl (<out>/<log>/<scenario type>/<token>/), None if missing."""
    log_dir = out / log
    if not log_dir.is_dir():
        return None
    for kind in os.listdir(log_dir):
        f = log_dir / kind / token / "metric_cache.pkl"
        if f.is_file() and f.stat().st_size > 0:
            return f
    return None

if __name__ == "__main__":
    main()
