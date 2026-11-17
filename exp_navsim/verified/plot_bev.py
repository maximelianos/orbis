"""BEV of sampler specs (sampler.py) on the same validation scenes as diverse/plot_bev.py.

    python -m exp_navsim.verified.plot_bev --specs 17_full_h64_full/step0100000,...@nucleus0.8@-@g0.3 \
        --labels "full16;full16 nucleus + guidance" --out outputs/18_verified_sampling/bev_guidance.png

Samples red, logged future black. Labels are separated by ";".
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from exp_navsim.data.resize_frames import frame_file
from exp_navsim.data.scenes import COMMANDS, SceneFeatureDataset
from exp_navsim.diverse.plot_bev import draw, load_scenes
from exp_navsim.diverse.train import to_gpu
from exp_navsim.verified.sampler import Sampler
from exp_navsim.viz.plot_samples import pick_scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    specs = args.specs.split(",")
    labels = args.labels.split(";") if args.labels else specs

    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=["wm11_ctx"], split="val")
    rows = ds.rows[pick_scenes(ds, args.num, args.seed)]
    data = to_gpu(ds, rows)
    tokens = [ds.index["token"][r] for r in rows]
    scenes = load_scenes(tokens, [ds.index["log"][r] for r in rows])
    preds = [Sampler(s)(data, args.samples).cpu().numpy() for s in specs]
    gt = data["future"].cpu().numpy()

    cols = 1 + len(specs)
    fig, axes = plt.subplots(args.num, cols, figsize=(3.4 * cols, 3.4 * args.num), squeeze=False)
    for i, r in enumerate(rows):
        ax = axes[i, 0]
        ax.imshow(Image.open(frame_file(ds.index["split"], ds.index["frame_ids"][r][-1])))
        ax.set_title(f"cmd {COMMANDS[int(ds.command[r].argmax())]}, v = {ds.status[r][0]:.1f} m/s", fontsize=9)
        ax.axis("off")
        for j, fut in enumerate(preds):
            ax = axes[i, j + 1]
            draw(ax, scenes[tokens[i]], fut[:, i], np.zeros(fut.shape[0], int), gt[i], 1)
            if i == 0:
                ax.set_title(labels[j], fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
