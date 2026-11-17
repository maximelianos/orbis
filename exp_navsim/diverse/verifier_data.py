"""PDM sub-scores of generator samples on the cached navtrain scenes: extra verifier training data.

    python -m exp_navsim.diverse.verifier_data --panels 17_full64_h64_10k/step0100000,17_full_h64_full/step0100000 \
        --samples 16

The vocabulary of planner/score_vocab.py is 256 fixed trajectories; the verifier also
has to judge what a generator actually draws (kinks, sideways jumps, near misses). Here
every panel draws --samples futures for each of the 5000 cached navtrain scenes
(cache_navtrain.py) and FastPDM scores them. A generator trained on 10k scenes has not
seen most of these scenes, so its samples there fail as often as on test.
-> outputs/17_diversity/verifier/gen_scores_navtrain_sub.npz: tokens, rows,
   samples (N, C, 8, 3), scores (N, C, len(METRICS)); saved after every chunk.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.diverse.plot_bev import load_model
from exp_navsim.diverse.train import batch_of, to_gpu
from exp_navsim.planner.fast_pdm import CACHE, FastPDM

OUT = Path("outputs/17_diversity/verifier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", required=True)
    ap.add_argument("--samples", type=int, default=16, help="per panel and scene")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--split", default="navtrain_sub", help="metric cache split")
    ap.add_argument("--tokens", default="outputs/16_fast_planner/navtrain_sub_tokens.json")
    ap.add_argument("--out", default=str(OUT / "gen_scores_navtrain_sub.npz"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    d = json.load(open(args.tokens))
    tokens, rows = d["tokens"], np.asarray(d["rows"])
    cached = {p.name for p in (CACHE / args.split).glob("*/*/*")}              # <log>/<type>/<token>
    keep = [i for i, t in enumerate(tokens) if t in cached]
    if len(keep) < len(tokens):
        print(f"[cache] {len(tokens) - len(keep)} of {len(tokens)} scenes have no metric cache, skipped")
    tokens, rows = [tokens[i] for i in keep], rows[keep]
    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=["wm11_ctx"], split="all",
                             feature_dir="data/features", history=4)
    data = to_gpu(ds, rows)
    samples = []
    for p in args.panels.split(","):
        run, ck = p.split("/")
        model, _ = load_model(Path("logs_exp") / run / "checkpoints" / f"{ck}.pt")
        g = torch.Generator(device="cuda").manual_seed(1)
        fut = [model.sample(batch_of(data, torch.arange(i, min(i + 250, len(rows)), device="cuda")), args.samples,
                            20, generator=g)[0].cpu() for i in range(0, len(rows), 250)]
        samples.append(torch.cat(fut, 1))
        print(f"[sampled] {p}", flush=True)
    samples = torch.cat(samples).transpose(0, 1).numpy().astype(np.float32)                     # (N, C, 8, 3)

    scorer = FastPDM(args.split, tokens, workers=args.workers, preload=False)
    scores, t0 = [], time.time()
    for i in range(0, len(tokens), args.chunk):
        chunk = tokens[i:i + args.chunk]
        res = scorer.candidate_scores({t: samples[i + j] for j, t in enumerate(chunk)})
        scores.append(np.stack([res[t] for t in chunk]).astype(np.float16))
        n = i + len(chunk)
        np.savez(out, tokens=np.array(tokens[:n]), rows=rows[:n], samples=samples[:n].astype(np.float16),
                 scores=np.concatenate(scores))
        rate = n / (time.time() - t0)
        print(f"{n}/{len(tokens)}  {rate:.1f} scenes/s  eta {(len(tokens) - n) / rate / 60:.0f} min", flush=True)
    scorer.close()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
