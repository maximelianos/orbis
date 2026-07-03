"""
Test script for game-conditioned whole trajectory diffusion model.

This script loads a trained `networks.whole_s_att.DiffusionModel`, builds
synthetic game scenes from `GameTrajectoryDataset`, samples trajectories while
conditioning on the input image, and saves a multi-page PDF with model outputs.

Example usage:
    python evaluate/test_game.py --config configs/game.yaml --ckpt <path/to/last.ckpt>

    # Use latest checkpoint in a logdir
    python evaluate/exp_game/test_game.py --config evaluate/exp_game/game.yaml --last_ckpt --logdir logs_nuplan
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util import instantiate_from_config


def _resolve_checkpoint(args):
    if args.last_ckpt:
        if not os.path.exists(args.logdir):
            raise ValueError(f"Logdir {args.logdir} does not exist")

        subdirs = [
            d
            for d in os.listdir(args.logdir)
            if os.path.isdir(os.path.join(args.logdir, d))
        ]
        if not subdirs:
            raise ValueError(f"No subdirectories found in {args.logdir}")

        subdirs.sort()
        last_subdir = subdirs[-1]
        ckpt_path = os.path.join(args.logdir, last_subdir, "checkpoints", "last.ckpt")

        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint not found at {ckpt_path}")

        print(f"Using last checkpoint from {ckpt_path}")
        return ckpt_path

    if args.ckpt is None:
        raise ValueError("Either --ckpt or --last_ckpt must be provided")

    if not os.path.exists(args.ckpt):
        raise ValueError(f"Checkpoint not found at {args.ckpt}")

    return args.ckpt


def _build_eval_batch(dataset, turns_array, T, device):
    gt_trajectories = []
    position_batch = []
    image_batch = []

    for turn_val in turns_array:
        sample = dataset.generate_trajectory(turn_val, T=T, r=dataset.r, draw_trajectory=False)
        gt_trajectories.append(sample["position"].numpy())
        position_batch.append(sample["position"].unsqueeze(0))
        image_batch.append(sample["images"].unsqueeze(0))

    gt_trajectories = np.array(gt_trajectories)  # (N, T, 2)

    turn_tensor = torch.tensor(turns_array, dtype=torch.float32, device=device)
    position_tensor = torch.cat(position_batch, dim=0).to(device)  # (N, T, 2)
    image_tensor = torch.cat(image_batch, dim=0).to(device)  # (N, H, W, 3)

    batch = {
        "turn": turn_tensor,
        "position": position_tensor,
        "images": image_tensor,
    }
    return batch, gt_trajectories


def sample_with_images(model, dataset, num_samples, T, context_length, num_diffusion_steps):
    model.eval()
    device = next(model.parameters()).device

    turns_array = np.linspace(0, 1, num_samples)
    batch, gt_trajectories = _build_eval_batch(dataset, turns_array, T, device)

    with torch.no_grad():
        pred_trajectories = model.sample(
            batch=batch,
            context_length=context_length,
            T=T,
            num_diffusion_steps=num_diffusion_steps,
            dataset=dataset,
        )

    pred_trajectories = pred_trajectories.cpu().numpy()
    images = batch["images"].cpu().numpy()

    return images, gt_trajectories, pred_trajectories, turns_array


def _draw_square_numpy(image, px, py, color, half_size=1):
    h, w = image.shape[:2]
    x0 = max(0, px - half_size)
    x1 = min(w, px + half_size + 1)
    y0 = max(0, py - half_size)
    y1 = min(h, py + half_size + 1)
    image[y0:y1, x0:x1, :] = color


def _draw_trajectory_numpy(image, trajectory, dataset, color, half_size=1):
    for i in range(len(trajectory) - 1):
        p0 = trajectory[i]
        p1 = trajectory[i + 1]
        steps = max(abs(float(p1[0] - p0[0])), abs(float(p1[1] - p0[1]))) * 200
        steps = max(2, int(steps))
        for j in range(steps + 1):
            alpha = j / steps
            x = (1.0 - alpha) * float(p0[0]) + alpha * float(p1[0])
            y = (1.0 - alpha) * float(p0[1]) + alpha * float(p1[1])
            px, py = dataset._world_to_pixel(x, y)
            _draw_square_numpy(image, px, py, color=color, half_size=half_size)


def save_outputs_pdf(images, gt_trajectories, pred_trajectories, turns, context_length, output_pdf, dataset):
    output_pdf = str(output_pdf)
    with PdfPages(output_pdf) as pdf:
        for idx in range(len(turns)):
            img = images[idx]
            gt = gt_trajectories[idx]
            pred = pred_trajectories[idx]

            gt_img = np.clip(img.copy(), 0.0, 1.0)
            pred_img = np.clip(img.copy(), 0.0, 1.0)

            _draw_trajectory_numpy(gt_img, gt, dataset, color=np.array([0.4, 0.4, 0.4], dtype=np.float32), half_size=1)
            _draw_trajectory_numpy(pred_img, pred, dataset, color=np.array([0.0, 0.2, 1.0], dtype=np.float32), half_size=1)

            if context_length > 0:
                prefix = gt[:context_length]
                _draw_trajectory_numpy(pred_img, prefix, dataset, color=np.array([1.0, 0.6, 0.0], dtype=np.float32), half_size=1)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Left: GT trajectory rendered with numpy
            axes[0].imshow(gt_img)
            axes[0].set_title(f"GT (turn={turns[idx]:.2f})")
            axes[0].axis("off")

            # Right: Pred trajectory rendered with numpy
            axes[1].imshow(pred_img)
            axes[1].set_title("Model Output")
            axes[1].axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved multi-page PDF to {output_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test game-conditioned whole trajectory diffusion model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--last_ckpt", action="store_true", help="Use last checkpoint from logdir")
    parser.add_argument("--logdir", type=str, default="logs_nuplan", help="Directory for logs")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of diffusion sampling steps")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples/pages")
    parser.add_argument("--T", type=int, default=20, help="Number of timesteps per trajectory")
    parser.add_argument("--context_length", type=int, default=1, help="Number of GT prefix points to condition on")
    parser.add_argument("--output_dir", type=str, default="evaluate", help="Output directory")
    parser.add_argument("--output_pdf", type=str, default="game_outputs.pdf", help="Output PDF filename")
    args = parser.parse_args()

    ckpt_path = _resolve_checkpoint(args)

    config = OmegaConf.load(args.config)

    model = instantiate_from_config(config.model)

    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    print("Model loaded successfully")

    print("\nLoading dataset...")
    dataset_config = config.data.params.train
    dataset_config["params"]["image_height"] = 64
    dataset_config["params"]["image_width"] = 64
    dataset = instantiate_from_config(dataset_config)

    print(f"\nSampling {args.num_samples} game trajectories with {args.num_steps} diffusion steps...")
    images, gt_trajectories, pred_trajectories, turns = sample_with_images(
        model=model,
        dataset=dataset,
        num_samples=args.num_samples,
        T=args.T,
        context_length=args.context_length,
        num_diffusion_steps=args.num_steps,
    )

    print(f"Predicted trajectories shape: {pred_trajectories.shape}")
    print(f"Mean endpoint X: {pred_trajectories[:, -1, 0].mean():.4f}")
    print(f"Mean endpoint Y: {pred_trajectories[:, -1, 1].mean():.4f}")

    
    output_dir = Path(os.path.abspath(__file__)).parent
    output_dir.mkdir(exist_ok=True)

    pdf_path = output_dir / args.output_pdf
    save_outputs_pdf(
        images=images,
        gt_trajectories=gt_trajectories,
        pred_trajectories=pred_trajectories,
        turns=turns,
        context_length=args.context_length,
        output_pdf=pdf_path,
        dataset=dataset,
    )

    print("\nTesting complete!")
