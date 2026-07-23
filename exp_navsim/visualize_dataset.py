"""Visualize the NAVSIM long dataloader.

    python -m exp_navsim.visualize_dataset --config exp_navsim/config.yaml --num 5

Saves ONE pdf whose first page reports dataset statistics (number of episodes
and the distribution of episode lengths) followed by one page per visualized
episode (BEV + 5 surround-camera observations + trajectory).
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf

from exp_navsim.data.navsim_base import NavsimLongBase
from exp_navsim.draw import draw_episode


def _stats_page(pdf, lengths):
    fig = plt.figure(figsize=(11, 8))
    fig.suptitle("NAVSIM long dataloader — dataset statistics", fontsize=16)
    lengths = np.asarray(lengths)
    txt = (
        f"episodes: {len(lengths)}\n"
        f"episode length (frames)  min={lengths.min()}  max={lengths.max()}  "
        f"mean={lengths.mean():.1f}  median={np.median(lengths):.0f}"
    )
    fig.text(0.1, 0.9, txt, fontsize=12, va="top", family="monospace")
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.65])
    ax.hist(lengths, bins=min(30, max(len(set(lengths.tolist())), 1)),
            color="tab:blue", alpha=0.8)
    ax.set_xlabel("episode length (frames)"); ax.set_ylabel("count")
    ax.set_title("Distribution of episode lengths")
    ax.grid(True, alpha=0.3)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_navsim/config.yaml")
    ap.add_argument("--num", type=int, default=5, help="episodes to visualize")
    ap.add_argument("--out", default="exp_navsim/navsim_episodes.pdf")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    ds = NavsimLongBase.from_config(cfg)
    print(f"Dataset: {len(ds)} episodes")

    # Episode lengths (cheap: only parse frames / read the index, no images).
    lengths = [ds.episode_length(i) for i in range(len(ds))]

    with PdfPages(args.out) as pdf:
        _stats_page(pdf, lengths)
        for i in range(min(args.num, len(ds))):
            print(f"  visualizing episode {i + 1}/{args.num}")
            sample = ds[i]
            # draw_episode expects a batch -> add a batch dim of size 1
            batch = {k: [v] if not isinstance(v, dict) else [v] for k, v in sample.items()}
            batch = {k: (np.stack(v) if isinstance(v[0], np.ndarray) else v)
                     for k, v in batch.items()}
            fig = draw_episode(batch, 0, title=f"Episode {i} — {sample['metadata']['token']}")
            pdf.savefig(fig, dpi=120, bbox_inches="tight"); plt.close(fig)

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
