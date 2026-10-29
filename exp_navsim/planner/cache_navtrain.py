"""Metric cache for a random subset of navtrain training scenes (targets for PDM-score distillation).

    python -m exp_navsim.planner.cache_navtrain --scenes 20000 --workers 24

Same as `run_metric_caching.py train_test_split=navtrain`, but the scene filter only
keeps the chosen tokens (training logs of the probe split, so validation logs stay
unseen). Writes outputs/navsim_exp/metric_cache/navtrain_sub and the token list
outputs/16_fast_planner/navtrain_sub_tokens.json.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from exp_navsim.data.scenes import load_index, log_in_split
from exp_navsim.eval.pdm import CACHE, DOWNLOAD
from exp_navsim.planner.fast_pdm import CONFIG_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ix = load_index("data/navsim_index/navtrain.pkl")
    train = np.flatnonzero([log_in_split(l, "train", 0.1) for l in ix["log"]])
    rows = np.sort(np.random.default_rng(args.seed).choice(train, args.scenes, replace=False))
    tokens = [str(ix["token"][r]) for r in rows]
    logs = sorted({str(ix["log"][r]) for r in rows})
    Path("outputs/16_fast_planner").mkdir(parents=True, exist_ok=True)
    json.dump({"tokens": tokens, "rows": rows.tolist()}, open("outputs/16_fast_planner/navtrain_sub_tokens.json", "w"))
    print(f"{len(tokens)} scenes from {len(logs)} logs")

    os.environ["NAVSIM_EXP_ROOT"] = str(Path("outputs/navsim_exp").resolve())
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(DOWNLOAD))
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from navsim.planning.metric_caching.caching import cache_data
    from navsim.planning.script.builders.worker_pool_builder import build_worker

    out = CACHE / "navtrain_sub"
    with initialize_config_dir(config_dir=str(CONFIG_DIR.parent / "metric_caching"), version_base=None):
        cfg = compose(config_name="default_metric_caching", overrides=[
            "train_test_split=navtrain", f"navsim_log_path={DOWNLOAD / 'trainval_navsim_logs/trainval'}",
            f"metric_cache_path={out.resolve()}", "worker=single_machine_thread_pool",
            "worker.use_process_pool=true", f"worker.max_workers={args.workers}"])
    with open_dict(cfg):
        cfg.train_test_split.scene_filter.tokens = tokens
        cfg.train_test_split.scene_filter.log_names = logs
    cache_data(cfg=cfg, worker=build_worker(cfg))


if __name__ == "__main__":
    main()
