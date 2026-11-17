"""PDM sub-scores of a FlowPlanner's own samples on the cached navtrain scenes.

    python -m exp_navsim.hybrid.sample_scores --run 19_resid --samples 16

The 10000 training scenes with a metric cache (navtrain_sub of planner/cache_navtrain.py,
navtrain_sub2 of verified/cache_more.py) get --samples futures each (10 Euler steps) and
FastPDM scores every one (sub-scores without two-frame comfort). The file has the layout
of diverse/verifier_data.py, so it is reward data for train.py --rw, extra candidates for
the scorer (scorer.py) and for multi-modal targets (train.py --mmgt).
-> outputs/19_hybrid/scores/<run>[_<tag>].npz: tokens, rows, samples (N, G, 8, 3), scores (N, G, 8)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from exp_navsim.planner.fast_pdm import CACHE, FastPDM, epdms

OUT = Path("outputs/19_hybrid/scores")
SPLITS = {"navtrain_sub": "outputs/16_fast_planner/navtrain_sub_tokens.json",
          "navtrain_sub2": "outputs/18_verified_sampling/navtrain_sub2_tokens.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--raw", action="store_true", help="raw weights instead of the EMA")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--irp", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-scenes", type=int, default=None, help="per split (for tests)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.run}{'_' + args.tag if args.tag else ''}.npz"

    parts = []
    for split, path in SPLITS.items():
        d = json.load(open(path))
        cached = {p.name for p in (CACHE / split).glob("*/*/*")}
        parts.append((split, [(t, r) for t, r in zip(d["tokens"], d["rows"]) if t in cached][:args.max_scenes]))
    # scoring pools fork, so they start before CUDA
    scorers = {s: FastPDM(s, [t for t, _ in tr], workers=args.workers // 2, preload=False) for s, tr in parts}

    import torch
    from exp_navsim.data.scenes import SceneFeatureDataset
    from exp_navsim.hybrid.train import batch_of, load_model, to_gpu

    model, a = load_model(Path("logs_exp") / args.run / "model.pt", raw=args.raw)
    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=a["features"].split(","), split="all",
                             feature_dir="data/features", history=4)
    tokens, rows, samples, scores = [], [], [], []
    t0 = time.time()
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    total = sum(len(tr) for _, tr in parts)
    for split, tr in parts:
        for i in range(0, len(tr), args.chunk):
            chunk = tr[i:i + args.chunk]
            r = np.array([x for _, x in chunk])
            data = to_gpu(ds, r)
            with torch.no_grad():
                fut = torch.cat([model.sample(batch_of(data, torch.arange(j, min(j + 100, len(r)), device="cuda")),
                                              args.samples, 10, irp=args.irp, generator=g)[0]
                                 for j in range(0, len(r), 100)], 1)
            fut = fut.transpose(0, 1).cpu().numpy().astype(np.float32)                 # (N, G, 8, 3)
            res = scorers[split].candidate_scores({t: fut[j] for j, (t, _) in enumerate(chunk)})
            tokens += [t for t, _ in chunk]
            rows.append(r)
            samples.append(fut.astype(np.float16))
            scores.append(np.stack([res[t] for t, _ in chunk]).astype(np.float16))
            np.savez(out, tokens=np.array(tokens), rows=np.concatenate(rows), samples=np.concatenate(samples),
                     scores=np.concatenate(scores))
            n = len(tokens)
            rate = n / (time.time() - t0)
            e = epdms(np.concatenate(scores).astype(np.float64))
            print(f"{n}/{total}  {rate:.1f} scenes/s  eta {(total - n) / rate / 60:.0f} min  "
                  f"sample EPDMS {e.mean():.3f}  best-of-G {e.max(1).mean():.3f}", flush=True)
    for s in scorers.values():
        s.close()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
