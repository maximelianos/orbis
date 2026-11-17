"""Orbis 2 L2 block features of every scene, on the ORBIS-1 cache interface.

    # fit the PCA on 20k navtrain scenes (train logs only), then encode navtrain
    python -m exp_navsim.orbis2.cache_l2 --split navtrain
    # navhard reuses the navtrain projection
    python -m exp_navsim.orbis2.cache_l2 --split navhard
    # smoke test: fit on 64 scenes, encode 64 scenes, write to a scratch dir
    python -m exp_navsim.orbis2.cache_l2 --split navhard --fit-from navhard --fit-scenes 64 \
        --limit 64 --out /tmp/o2l2_smoke

The L2 DiT runs once per scene over [4 history latents (2 Hz, tokenizer sem branch),
pure-noise next frame at flow time t = 1] with the null steering (NaN goals: at test
time there is no future). For every block b in --blocks two token sets are kept:
    o2l2_b<b>_ctx    the last history frame (t = 0 slot)
    o2l2_b<b>_next   the next-frame slot at the first denoising step
The noise of the next-frame slot is a fixed function of the index row, so a scene
always gets the same features.

Each token grid (18 x 32 x 768) is avg-pooled 2 x 2 to 9 x 16 and projected with a
PCA to 64 channels and one global scale (as in encoders/cache_features.py). The PCA
is fitted from the exact covariance of all pooled vectors of --fit-scenes scenes.

Output: <out>/<split>/<name>.npy (N, 64, 9, 16) float16, <name>.json, <name>.pt.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp_navsim.data.scenes import log_in_split
from exp_navsim.orbis2.common import (Clips, CovarianceFit, NpyAppender, load_index, project, scene_noise,
                                      tokens_to_grid, write_meta)
from exp_navsim.orbis2.loader import L2Model, null_goals

BLOCKS = (5, 11, 17, 23)


def feature_names(blocks):
    return [f"o2l2_b{b}_{s}" for b in blocks for s in ("ctx", "next")]


@torch.no_grad()
def scene_features(l2, clips, rows, blocks):
    """clips (B, 4, 3, H, W) -> {name: pooled (B, 768, 9, 16) float32}."""
    lat = l2.encode(clips)
    noise = scene_noise(rows, 1).to(l2.device)
    seq = torch.cat([lat, noise], 1)
    B = len(rows)
    _, feats = l2.velocity(seq, torch.ones(B, device=l2.device), null_goals(B, l2.device), blocks)
    out = {}
    for b in blocks:
        out[f"o2l2_b{b}_ctx"] = tokens_to_grid(feats[b][:, -2])
        out[f"o2l2_b{b}_next"] = tokens_to_grid(feats[b][:, -1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtrain")
    ap.add_argument("--index-dir", default="data/navsim_index")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--fit-from", default="navtrain", help="split whose projection is reused")
    ap.add_argument("--fit-scenes", type=int, default=20000)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--blocks", default=",".join(map(str, BLOCKS)))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="encode only the first N scenes (smoke test)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--reuse-fit", action="store_true", help="keep an existing projection of this split")
    args = ap.parse_args()

    blocks = tuple(int(b) for b in args.blocks.split(","))
    names = feature_names(blocks)
    index = load_index(args.split, args.index_dir)
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    fit_dir = Path(args.out) / args.fit_from
    fit_dir.mkdir(parents=True, exist_ok=True)
    loader_kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    l2 = L2Model("cuda", compile=not args.no_compile)

    fitted = all((fit_dir / f"{k}.pt").exists() for k in names)
    if args.split == args.fit_from and not (fitted and args.reuse_fit):
        train = np.flatnonzero([log_in_split(log, "train", args.val_fraction) for log in index["log"]])
        ids = np.sort(np.random.default_rng(0).permutation(train)[:args.fit_scenes])
        fits = {k: CovarianceFit(768, l2.device) for k in names}
        t0 = time.time()
        for clips, rows in tqdm(DataLoader(Clips(index, ids), **loader_kw), desc="fit"):
            for k, v in scene_features(l2, clips, rows.tolist(), blocks).items():
                fits[k].add(v)
        print(f"fit pass: {len(ids)} scenes in {time.time() - t0:.0f} s")
        for k, fit in fits.items():
            p = fit.projection(args.pca_dim)
            p["fit_scenes"] = len(ids)
            torch.save(p, fit_dir / f"{k}.pt")
            print(f"[{k}] PCA {args.pca_dim}/768, explained variance {p['explained']:.3f}, scale {p['scale']:.3f}")
        del fits
    proj = {k: torch.load(fit_dir / f"{k}.pt") for k in names}

    n = len(index["frame_ids"]) if args.limit is None else min(args.limit, len(index["frame_ids"]))
    items = Clips(index, np.arange(n))
    writers = {k: NpyAppender(out / f"{k}.npy", (n, args.pca_dim, 9, 16)) for k in names}
    t0, n0 = time.time(), 0
    for it, (clips, rows) in enumerate(tqdm(DataLoader(items, **loader_kw), desc=f"o2l2 {args.split}")):
        if it == 1:
            torch.cuda.synchronize()
            t0, n0 = time.time(), args.batch
        for k, v in scene_features(l2, clips, rows.tolist(), blocks).items():
            writers[k].append(project(v, proj[k]).half().cpu().numpy())
    torch.cuda.synchronize()
    secs = time.time() - t0
    rate = (n - n0) / secs
    print(f"encode pass: {rate:.1f} scenes/s after the first batch")
    for k, w in writers.items():
        w.close()
        write_meta(out / f"{k}.json", items="scenes", shape=list(w.shape), encoder="orbis2_l2",
                   pca=True, fit_from=args.fit_from, fit_scenes=int(proj[k].get("fit_scenes", 0)),
                   steering="null (NaN goals)", noise="scene_noise(row)", scenes_per_s=round(rate, 2))
        print(f"{k}: {w.shape} -> {out / k}.npy")


if __name__ == "__main__":
    main()
