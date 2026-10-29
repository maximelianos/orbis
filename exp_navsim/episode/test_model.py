"""Produce a PDF of predicted trajectories on a fixed set of test episodes.

    python -m exp_navsim.episode.test_model --config exp_navsim/episode/config.yaml \
        --ckpt logs_navsim/2026-08-21T14-45-31_config/checkpoints/last.ckpt --num 6

Loads a trained checkpoint, runs inference `num_val_samples` times on the SAME
fixed test episodes (deterministic windows, drawn from the validation episodes
passing EPISODE_FILTERS below), and plots the predicted
trajectories (all runs) overlaid on the ground truth with the shared episode
visualization. One PDF page per episode.

The top row holds the first & last context front views plus a third panel: the
NAVSIM map BEV (lanes + surrounding agents) with the whole predicted distribution
drawn on it (exp_navsim/data/bev_extract.py + exp_navsim/viz/draw_bev.py). Pass
--no-bev to skip it if the nuPlan maps are not available.
"""

import argparse
from pathlib import Path

import h5py
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf

from util import instantiate_from_config, get_obj_from_str
from evaluate.checkpoint import resolve_checkpoint
from exp_navsim.episode.collate import pad_collate
from exp_navsim.viz.visualize_dataset import trajectory_stats

# Which validation episodes are eligible for a page. Same three filters as
# exp_navsim/viz/visualize_dataset.py, fixed here rather than exposed on the CLI.
# Distance/angle are measured over the window the model actually sees (the
# deterministic window starts at frame 0); "frames" is the whole cached episode.
EPISODE_FILTERS = {
    "min_distance": 5.0,   # m driven
    "min_angle": 10.0,       # deg between the initial heading and start->end
    "min_frames": 0,         # cached episode length
}


def _build_batch(dataset, indices, device):
    """Collate samples from `dataset` at `indices` into a batched dict of tensors.

    Uses the same padding collate as training, so the episode-length settings of
    the config (num_frames=0 -> whole, variable-length episodes) work here too.
    """
    samples = [dataset[int(idx)] for idx in tqdm(indices)]
    batch = pad_collate(samples)
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def _episode_length(batch, i):
    """Valid (unpadded) number of frames of sample `i`."""
    length = batch.get("metadata", {}).get("length")
    return int(length[i]) if length is not None else batch["trajectory"].shape[1]


def _filter_episodes(dataset, filters):
    """Indices of the validation episodes passing `filters` (all of them if none do).

    Reads only the cached `trajectory` of each episode — no latents, no images.
    """
    keep = []
    for i, path in enumerate(tqdm(dataset.files, desc="filtering episodes")):
        with h5py.File(path, "r") as f:
            total = f[dataset.latent_key].shape[0]
            # Same window the model will see (num_frames=0 -> the whole episode).
            traj = f["trajectory"][:dataset.window_length(total)]
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
    ap.add_argument("--config", default="exp_navsim/episode/config.yaml")
    ap.add_argument("--ckpt", default="logs_navsim",
                    help="checkpoint file, a run/checkpoints dir to use its last.ckpt, "
                         "or a logdir (default) to use the latest run's last.ckpt")
    ap.add_argument("--num", type=int, default=6, help="test episodes to visualize")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-bev", action="store_true",
                    help="skip the map-BEV panel (needs navsim + nuPlan maps)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Same validation dataset the training used (deterministic windows).
    dataset = instantiate_from_config(cfg.data.params.validation)

    # The checkpoint's class is taken from the config, so this script serves
    # both exp_navsim/episode/config.yaml (NavsimTrajectoryModel) and
    # exp_navsim/episode/wm/config.yaml (WmTrajectoryModel).
    ckpt_path = resolve_checkpoint(ckpt=args.ckpt)
    print("loading checkpoint", ckpt_path)
    model = get_obj_from_str(cfg.model.target).load_from_checkpoint(ckpt_path, map_location=device,
                                                                  weights_only=False)
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
    preds = torch.stack([model.sample(batch, num_diffusion_steps=10) for _ in range(N)], dim=1)  # (B, N, T, 2)
    batch["pred_trajectories"] = preds

    print("save pdf")
    run_dir = Path(ckpt_path).resolve().parent.parent       # <run_dir>/checkpoints/<name>.ckpt
    pdf_path = Path(args.out) if args.out else run_dir / "test_trajectories.pdf"
    with PdfPages(pdf_path) as pdf:
        for i in tqdm(range(n)):
            fig = _page(batch, i)
            pdf.savefig(fig, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {pdf_path}")


def _page(batch, i):
    from exp_navsim.viz.draw import draw_episode

    # Drop the padding of this episode so the plots stop at its real last frame.
    # The model predicts the full padded length for every sample.
    length = _episode_length(batch, i)
    page_batch = dict(batch)
    page_batch["trajectory"] = batch["trajectory"][:, :length]
    if "pred_trajectories" in batch:
        page_batch["pred_trajectories"] = batch["pred_trajectories"][:, :, :length]

    return draw_episode(page_batch, i, title=f"Test episode {i} — predicted trajectories")


if __name__ == "__main__":
    main()
