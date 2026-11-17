"""Sanity check of the L2 steering: decode dreams of left / straight / right candidates.

    python -m exp_navsim.orbis2.dream_check --horizon 4

For 2 navhard scenes (real, moving ego) three vocabulary anchors with about the
current speed and the most left, straightest and most right final heading are rolled
out, plus the null steering (NaN goals). Each row of the grid: the last real
history frame with a BEV inset of the candidate (x up, y left), then the decoded
dream frames at +0.5 s, +1 s, ... Writes outputs/28_inputs/dream_check/scene<i>.png
and prints the mean latent distance between the candidates per step.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from exp_navsim.orbis2.common import Clips, load_index
from exp_navsim.orbis2.dream import Dreamer, load_vocab
from exp_navsim.orbis2.loader import L2Model

OUT = Path("outputs/28_inputs/dream_check")


def pick_scenes(index, n, seed=0):
    speed = np.linalg.norm(index["velocity"], axis=1)
    ok = (~index["synthetic"]) & (speed > 6) & (speed < 12)
    return np.sort(np.random.default_rng(seed).choice(np.flatnonzero(ok), n, replace=False)), speed


def pick_candidates(vocab, speed):
    """Anchors whose first step matches the speed; most left, straightest, most right."""
    near = np.argsort(np.abs(vocab[:, 0, 0] - 0.5 * speed))[:48]
    head = vocab[near, -1, 2]
    left, right = near[np.argmax(head)], near[np.argmin(head)]
    straight = near[np.argmin(np.abs(head) + 0.2 * np.abs(vocab[near, -1, 1]))]
    return {"left": left, "straight": straight, "right": right}


def to_pil(img):
    return Image.fromarray(((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8))


def bev_inset(tile, poses, size=110):
    """Draw the candidate path (x forward = up, y left = left) in the top-left corner."""
    d = ImageDraw.Draw(tile)
    d.rectangle([0, 0, size, size], fill=(255, 255, 255))
    if not np.isfinite(poses).all():
        d.text((8, size // 2), "no steering", fill=(0, 0, 0))
        return
    scale = (size - 16) / 40.0                                # 40 m across
    pts = [(size / 2 - y * scale, size - 8 - x * scale) for x, y in np.concatenate([[[0, 0]], poses[:, :2]])]
    d.line(pts, fill=(220, 30, 30), width=3)
    d.ellipse([size / 2 - 3, size - 11, size / 2 + 3, size - 5], fill=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--sampler", default="euler", choices=["euler", "cd"])
    ap.add_argument("--scenes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    index = load_index("navhard")
    vocab = load_vocab()
    rows, speed = pick_scenes(index, args.scenes, args.seed)
    l2 = L2Model("cuda", compile=False)          # few passes: compiling would take longer
    dreamer = Dreamer(l2, None, args.horizon, args.nfe, args.sampler, feat_mode="noise")
    clips = Clips(index, rows)
    W, Hh = 256, 144                                            # tile size (half resolution)
    for i, row in enumerate(rows):
        clip, _ = clips[i]
        cands = pick_candidates(vocab, speed[row])
        names = list(cands) + ["null"]
        poses = np.stack([vocab[cands[n]] for n in cands] + [np.full((8, 3), np.nan, np.float32)])
        _, lat = dreamer(clip[None], [int(row)], poses[None])               # (1, 4, H, 16, 18, 32)
        lat = lat[0]
        imgs = l2.decode(lat)                                                # (4, H, 3, 288, 512)
        dist = [(lat[0, h] - lat[2, h]).pow(2).mean().sqrt().item() for h in range(args.horizon)]
        print(f"scene {row} ({index['token'][row]}, {speed[row]:.1f} m/s): anchors {cands}, "
              f"RMS latent diff left vs right per step {np.round(dist, 3).tolist()}")

        grid = Image.new("RGB", ((args.horizon + 1) * W, len(names) * Hh + 20), (255, 255, 255))
        d = ImageDraw.Draw(grid)
        d.text((4, 4), f"navhard {index['token'][row]}  {speed[row]:.1f} m/s   columns: last real frame, "
                       f"dream +0.5 s ... +{0.5 * args.horizon:.1f} s   (L2, {args.sampler} {args.nfe} NFE)",
               fill=(0, 0, 0))
        for r, name in enumerate(names):
            tile = to_pil(clip[-1]).resize((W, Hh))
            bev_inset(tile, poses[r], size=70)
            grid.paste(tile, (0, 20 + r * Hh))
            d.text((4, 20 + r * Hh + Hh - 14), name, fill=(255, 255, 0))
            for h in range(args.horizon):
                grid.paste(to_pil(imgs[r, h]).resize((W, Hh)), ((h + 1) * W, 20 + r * Hh))
        path = OUT / f"scene{i}_{index['token'][row]}.png"
        grid.save(path)
        print("saved", path)


if __name__ == "__main__":
    main()
