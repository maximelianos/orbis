"""Per-sample quality and calibration of sampler specs (diverse/score_samples.py with sampler.Sampler).

    python -m exp_navsim.verified.sample_eval --specs 17_full_h64_full/step0100000@prop@-@g1,... --labels ...

Navtest 500 subset: every one of the K samples scored as a plan (sample EPDMS, share bad
< 0.5, off-road, collision, best of K, spread); 2000 held-out navtrain scenes: energy,
minADE, dispersion against the logged future. Also the sampling time per scene.
-> outputs/18_verified_sampling/samples.csv (rows of the same label are replaced)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.diverse.score_samples import SUBSET, save, summarize
from exp_navsim.diverse.train import to_gpu
from exp_navsim.eval.metrics import sample_metrics
from exp_navsim.planner.fast_pdm import FastPDM, load_subset
from exp_navsim.verified.methods import METHODS, VERIFIER_V3
from exp_navsim.verified.sampler import VERIFIER, Sampler

OUT = Path("outputs/18_verified_sampling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=None)
    ap.add_argument("--methods", action="store_true", help="the methods of methods.py, verifier v3")
    ap.add_argument("--out", default=None, help="csv path (default: the script's csv)")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--verifier", default=VERIFIER)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.methods:
        specs, labels, args.verifier = [m[2] for m in METHODS], [m[0] for m in METHODS], VERIFIER_V3
    else:
        specs = args.specs.split(",")
        labels = args.labels.split(",") if args.labels else specs

    tokens, _ = load_subset(SUBSET)
    ds = SceneFeatureDataset("data/navsim_index/navtest.pkl", features=["wm11_ctx"], split="all",
                             require_future=False, feature_dir="data/features", history=4)
    pos = {t: r for r, t in zip(ds.rows, np.asarray(ds.index["token"])[ds.rows])}
    rows = np.array([pos[t] for t in tokens])
    test = to_gpu(ds, rows)
    v0 = np.linalg.norm(ds.status[rows][:, :2], axis=1)
    val_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=["wm11_ctx"], split="val")
    val = to_gpu(val_ds, np.sort(np.random.default_rng(0).permutation(val_ds.rows)[:2000]))
    scorer = FastPDM("navtest", tokens, workers=args.workers)

    results = []
    for spec, label in zip(specs, labels):
        sampler = Sampler(spec, args.verifier)
        torch.cuda.synchronize()
        t0 = time.time()
        fut = sampler(test, args.samples)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / len(rows) * 1000
        fut = fut.cpu().numpy()
        d = scorer.candidate_scores({t: fut[:, i].astype(np.float32) for i, t in enumerate(tokens)})
        r, _ = summarize(np.stack([d[t] for t in tokens], 1), fut, v0)
        vm = {k: v.mean().item() for k, v in sample_metrics(sampler(val, args.samples), val["future"]).items()}
        r.update({"val/energy": vm["energy"], "val/ade_min": vm["ade_min"], "val/spread": vm["spread"],
                  "val/dispersion": vm["spread"] / vm["ade_mean"], "ms_per_scene": ms})
        results.append({"model": label, "spec": spec, **r})
        print(label, {k: round(float(v), 3) for k, v in r.items()}, flush=True)
        save(results, Path(args.out or OUT / "samples.csv"))
    scorer.close()
    print(f"-> {args.out or OUT / 'samples.csv'}")


if __name__ == "__main__":
    main()
