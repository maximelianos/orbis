"""Produce a PDF of predicted trajectories on a fixed set of test episodes.

    python -m exp_navsim.test_model --config exp_navsim/config.yaml \
        --ckpt logs_navsim/<run>/checkpoints/last.ckpt --num 6

Loads a trained checkpoint, runs inference `num_val_samples` times on the SAME
fixed test episodes (deterministic windows), and plots the predicted
trajectories (all runs) overlaid on the ground truth with the shared episode
visualization. One PDF page per episode.
"""

import argparse

import h5py
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf

from util import instantiate_from_config
from exp_navsim.model import NavsimTrajectoryModel


def _build_batch(dataset, indices, device):
    """Stack samples from `dataset` at `indices` into a batched dict of tensors."""
    batch = {}
    for idx in tqdm(indices):
        sample = dataset[int(idx)]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                batch.setdefault(k, []).append(v)
    return {k: torch.stack(v, 0).to(device) for k, v in batch.items()}


def _attach_context_views(batch, dataset, indices, cfg):
    """Attach the first & last context front-camera views for each episode.

    The cached-latent validation dataset carries no raw images, so we reload the
    raw front camera from the long dataset (matched by scene token) and pick the
    first and last *context* frames of the deterministic (start=0) window.
    """
    from exp_navsim.data.navsim_base import NavsimLongBase

    context_images = cfg.model.params.context_images
    ctx_frames = sorted({0, context_images - 1})            # first & last context frame

    long_ds = NavsimLongBase.from_config(cfg, load_surround=False, split="all")
    token_to_idx = {long_ds.episode_token(i): i for i in range(len(long_ds))}

    views = []
    for idx in tqdm(indices):
        with h5py.File(dataset.files[int(idx)], "r") as f:
            token = f.attrs["token"]
        images = long_ds[token_to_idx[token]]["images"]     # (T, 3, H, W), window starts at 0
        views.append(torch.from_numpy(images[ctx_frames]).float())

    batch["context_views"] = torch.stack(views, 0)          # (B, len(ctx_frames), 3, H, W)
    batch["context_view_labels"] = [f"context frame {i}" for i in ctx_frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_navsim/config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=6, help="test episodes to visualize")
    ap.add_argument("--out", default="exp_navsim/navsim_predictions.pdf")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Same validation dataset the training used (deterministic windows).
    dataset = instantiate_from_config(cfg.data.params.validation)

    model = NavsimTrajectoryModel.load_from_checkpoint(args.ckpt, map_location=device)
    model = model.to(device).eval()

    # Fixed test episodes: evenly spaced across the validation set.
    n = min(args.num, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    batch = _build_batch(dataset, indices, device)
    print("attach context views")
    _attach_context_views(batch, dataset, indices, cfg)

    # N samples of the same context -> (B, N, T, 2) for the drawing overlay.
    print("predict")
    N = model.num_val_samples
    preds = torch.stack([model.sample(batch) for _ in range(N)], dim=1)  # (B, N, T, 2)
    batch["pred_trajectories"] = preds

    print("save pdf")
    with PdfPages(args.out) as pdf:
        for i in tqdm(range(n)):
            fig = _page(batch, i)
            pdf.savefig(fig, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {args.out}")


def _page(batch, i):
    from exp_navsim.draw import draw_episode
    return draw_episode(batch, i, title=f"Test episode {i} — predicted trajectories")


if __name__ == "__main__":
    main()
