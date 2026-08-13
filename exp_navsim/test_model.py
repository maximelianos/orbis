"""Produce a PDF of predicted trajectories on a fixed set of test episodes.

    python -m exp_navsim.test_model --config exp_navsim/config.yaml \
        --ckpt logs_navsim/2026-07-22T23-01-41_config/checkpoints/last.ckpt --num 6

Loads a trained checkpoint, runs inference `num_val_samples` times on the SAME
fixed test episodes (deterministic windows, drawn from the validation episodes
passing EPISODE_FILTERS below), and plots the predicted
trajectories (all runs) overlaid on the ground truth with the shared episode
visualization. One PDF page per episode.

The top row holds the first & last context front views plus a third panel: the
NAVSIM map BEV (lanes + surrounding agents) with the whole predicted distribution
drawn on it (exp_navsim/data/bev_extract.py + exp_navsim/draw_bev.py). Pass
--no-bev to skip it if the nuPlan maps are not available.
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
from exp_navsim.visualize_dataset import trajectory_stats

# Which validation episodes are eligible for a page. Same three filters as
# exp_navsim/visualize_dataset.py, fixed here rather than exposed on the CLI.
# Distance/angle are measured over the window the model actually sees (the
# deterministic window starts at frame 0); "frames" is the whole cached episode.
EPISODE_FILTERS = {
    "min_distance": 5.0,   # m driven
    "min_angle": 10.0,       # deg between the initial heading and start->end
    "min_frames": 0,         # cached episode length
}


def _build_batch(dataset, indices, device):
    """Stack samples from `dataset` at `indices` into a batched dict of tensors."""
    batch = {}
    for idx in tqdm(indices):
        sample = dataset[int(idx)]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                batch.setdefault(k, []).append(v)
    return {k: torch.stack(v, 0).to(device) for k, v in batch.items()}


def _filter_episodes(dataset, filters):
    """Indices of the validation episodes passing `filters` (all of them if none do).

    Reads only the cached `trajectory` of each episode — no latents, no images.
    """
    keep = []
    for i, path in enumerate(tqdm(dataset.files, desc="filtering episodes")):
        with h5py.File(path, "r") as f:
            traj = f["trajectory"][:dataset.num_frames]
            total = f[dataset.latent_key].shape[0]
        distance, steering, _ = trajectory_stats(traj)
        if (distance >= filters["min_distance"]
                and abs(steering) >= filters["min_angle"]
                and total >= filters["min_frames"]):
            keep.append(i)

    print(f"{len(keep)}/{len(dataset.files)} validation episodes pass {filters}")
    if not keep:
        print("  no episode passes the filters — falling back to the whole split")
        return list(range(len(dataset.files)))
    return keep


def _episode_tokens(dataset, indices):
    """Scene token of each selected latent-cache window (matches the long dataset)."""
    tokens = []
    for idx in indices:
        with h5py.File(dataset.files[int(idx)], "r") as f:
            tokens.append(f.attrs["token"])
    return tokens


def _attach_context_views(batch, long_ds, tokens, token_to_idx, cfg):
    """Attach the first & last context front-camera views for each episode.

    The cached-latent validation dataset carries no raw images, so we reload the
    raw front camera from the long dataset (matched by scene token) and pick the
    first and last *context* frames of the deterministic (start=0) window.
    """
    context_images = cfg.model.params.context_images
    ctx_frames = sorted({0, context_images - 1})            # first & last context frame

    views = []
    for token in tqdm(tokens):
        images = long_ds[token_to_idx[token]]["images"]     # (T, 3, H, W), window starts at 0
        views.append(torch.from_numpy(images[ctx_frames]).float())

    batch["context_views"] = torch.stack(views, 0)          # (B, len(ctx_frames), 3, H, W)
    batch["context_view_labels"] = [f"context frame {i}" for i in ctx_frames]


def _attach_bev(batch, long_ds, tokens, token_to_idx):
    """Attach the map-BEV context used by the third top-row panel.

    anchor=0 because the validation windows are deterministic (start=0), so the
    batch trajectories and the BEV share the episode-frame-0 frame — see
    exp_navsim/data/bev_extract.py for why that matters.
    """
    from exp_navsim.data.bev_extract import extract_bev_for_tokens

    batch["bev"] = extract_bev_for_tokens(long_ds, tokens, anchor=0,
                                          token_to_idx=token_to_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_navsim/config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=6, help="test episodes to visualize")
    ap.add_argument("--out", default="exp_navsim/navsim_predictions.pdf")
    ap.add_argument("--no-bev", action="store_true",
                    help="skip the map-BEV panel (needs navsim + nuPlan maps)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Same validation dataset the training used (deterministic windows).
    dataset = instantiate_from_config(cfg.data.params.validation)

    model = NavsimTrajectoryModel.load_from_checkpoint(args.ckpt, map_location=device)
    model = model.to(device).eval()

    # Fixed test episodes: evenly spaced across the filtered validation set.
    eligible = np.asarray(_filter_episodes(dataset, EPISODE_FILTERS))
    n = min(args.num, len(eligible))
    indices = eligible[np.linspace(0, len(eligible) - 1, n, dtype=int)]
    batch = _build_batch(dataset, indices, device)

    # One long dataset + token map, shared by the context views and the BEV.
    from exp_navsim.data.navsim_base import NavsimLongBase
    tokens = _episode_tokens(dataset, indices)
    long_ds = NavsimLongBase.from_config(cfg, load_surround=False, split="all")
    token_to_idx = {long_ds.episode_token(i): i for i in range(len(long_ds))}

    print("attach context views")
    _attach_context_views(batch, long_ds, tokens, token_to_idx, cfg)
    if not args.no_bev:
        print("attach BEV")
        try:
            _attach_bev(batch, long_ds, tokens, token_to_idx)
        except Exception as e:      # missing navsim/nuPlan maps -> drop the panel
            print(f"skipping BEV panel: {e}")

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
