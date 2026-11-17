"""One plan out of the sampled distribution: EPDMS of selection rules on the navtest subset.

    python -m exp_navsim.verified.plan_eval --specs 17_full_h64_full/step0100000@nucleus0.8@0.5,... \
        --labels full16_nuc_rej,...

For every scene of the 500-scene navtest subset (and its previous frame, for two-frame
comfort) the sampler draws K futures and each rule picks one plan:

    sample     the first sample (a random draw)
    mean       the mean of the samples
    medoid     the sample with the smallest mean ADE to the others (densest mode)
    verifier   the sample with the highest predicted EPDMS of the verifier

All rules of one spec are scored in one FastPDM call (candidate k = rule k, also for the
previous frame, so two-frame comfort compares a rule with itself). This is the official
EPDMS of fast_pdm.score. References from 16_fast_planner/comparison.csv: regression
planner 0.887, diffusion probe sample mean 0.782, PDM-Closed 0.871.
-> outputs/18_verified_sampling/plans.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.diverse.train import to_gpu
from exp_navsim.diverse.verifier import load_verifier
from exp_navsim.planner.fast_pdm import FastPDM, load_subset
from exp_navsim.verified.methods import METHODS, VERIFIER_V3
from exp_navsim.verified.sampler import VERIFIER, Sampler

OUT = Path("outputs/18_verified_sampling")
SUBSET = Path("outputs/16_fast_planner/subset_500.json")
RULES = ("sample", "mean", "medoid", "verifier")


def select(samples, verifier, features):
    """samples (K, N, 8, 3) -> (len(RULES), N, 8, 3)."""
    xy = samples[..., :2]
    d = (xy[:, None] - xy[None]).norm(dim=-1).mean(-1)                      # (K, K, N)
    medoid = d.mean(1).argmin(0)                                            # (N,)
    pred = verifier.predict_epdms({"features": features}, samples)          # (K, N)
    pick = lambda idx: samples[idx, torch.arange(samples.shape[1], device=samples.device)]
    mean = samples.mean(0)
    mean[..., 2] = torch.atan2(torch.sin(samples[..., 2]).mean(0), torch.cos(samples[..., 2]).mean(0))
    return torch.stack([samples[0], mean, pick(medoid), pick(pred.argmax(0))])


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

    tokens, prev = load_subset(SUBSET)
    need = list(dict.fromkeys(tokens + list(prev.values())))
    ds = SceneFeatureDataset("data/navsim_index/navtest.pkl", features=["wm11_ctx"], split="all",
                             require_future=False, feature_dir="data/features", history=4)
    pos = {t: r for r, t in zip(ds.rows, np.asarray(ds.index["token"])[ds.rows])}
    need = [t for t in need if t in pos]
    data = to_gpu(ds, np.array([pos[t] for t in need]))
    data["future"] = torch.nan_to_num(data["future"])
    features = data["features"].float()
    verifier = load_verifier(args.verifier)
    scorer = FastPDM("navtest", need, workers=args.workers)

    rows = []
    path = Path(args.out or OUT / "plans.csv")
    for spec, label in zip(specs, labels):
        samples = Sampler(spec, args.verifier)(data, args.samples)
        with torch.no_grad():
            plans = torch.cat([select(samples[:, i:i + 125], verifier, features[i:i + 125])
                               for i in range(0, samples.shape[1], 125)], 1).cpu().numpy()
        df = scorer.score({t: plans[:, j] for j, t in enumerate(need)}, prev)
        for k, rule in enumerate(RULES):
            sub = df[df.k == k]
            rows.append({"spec": label, "rule": rule, "epdms": sub.score.mean(),
                         **sub.drop(columns=["token", "k", "score"]).mean().round(4).to_dict()})
            print(f"{label:28s} {rule:9s} EPDMS {rows[-1]['epdms']:.4f}", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["spec"])
        pd.concat([old[~old.spec.isin([label])], pd.DataFrame([r for r in rows if r["spec"] == label])],
                  ignore_index=True).to_csv(path, index=False)
    scorer.close()
    print(f"-> {path}")


if __name__ == "__main__":
    main()
