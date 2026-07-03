"""
Concise NuPlan test script

Usage:
    python evaluate/test_nuplan.py --last_ckpt --logdir logs_nuplan --num_samples 10
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import instantiate_from_config
from evaluate.checkpoint import resolve_checkpoint, load_model_checkpoint
from loaders_for_projects.draw_nuplan import save_pictures


def _sample_indices(total_samples, num_samples):
    if total_samples <= 0:
        raise ValueError("Dataset is empty")
    num_samples = min(num_samples, total_samples)
    return np.linspace(0, total_samples - 1, num_samples, dtype=int)


def _build_batch(dataset, indices, device):
    # return normalized velocities, if enabled in config
    batch = {}
    
    # create a list of tensors for each key
    for idx in indices:
        sample = dataset[int(idx)]
        
        for k, v in sample.items():
            if not k in batch:
                batch[k] = []
            batch[k].append(v)
        
    # stack tensors
    for k in batch:
        batch[k] = torch.stack(batch[k], dim=0).to(device)

    return batch


@torch.no_grad()
def run_eval(model, dataset, indices, num_steps):
    device = next(model.parameters()).device
    batch = _build_batch(dataset, indices, device)

    predictions = model.sample(
        batch=batch,
        num_diffusion_steps=num_steps,
        dataset=dataset,
        denormalize=True,
    )
    
    # print(batch["trajectory"][0])
    # print(predictions[0])
    # input()

    return batch["images"].cpu(), batch["trajectory"].cpu(), predictions.cpu()


def save_pdf(images, gt_trajectories, pred_trajectories, indices, output_pdf):
    with PdfPages(str(output_pdf)) as pdf:
        for local_idx, dataset_idx in enumerate(indices):
            fig = save_pictures(
                images=None,  # TODO images?
                trajectories=[gt_trajectories[local_idx], pred_trajectories[local_idx]],
                labels=["GT", "Prediction"],
                title=f"NuPlan sample #{int(dataset_idx)}",
                output_path=None,
            )
            pdf.savefig(fig, dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved multi-page PDF to {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Concise NuPlan evaluation")
    parser.add_argument("--config", type=str, default="configs/nuplan.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--last_ckpt", action="store_true")
    parser.add_argument("--logdir", type=str, default="logs_nuplan")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="evaluate")
    parser.add_argument("--output_pdf", type=str, default="nuplan_v2_outputs.pdf")
    args = parser.parse_args()

    # find checkpoint path
    ckpt_path = resolve_checkpoint(ckpt=args.ckpt, last_ckpt=args.last_ckpt, logdir=args.logdir)

    # load model from checkpoint
    config = OmegaConf.load(args.config)
    model_target = str(config.model.target)
    if "whole_context" not in model_target:
        raise ValueError(f"Expected model target containing 'whole_context', got: {model_target}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate_from_config(config.model).to(device)
    model = load_model_checkpoint(model, ckpt_path, map_location=device, strict=False)

    # load dataset
    dataset_config = OmegaConf.load("configs/eval_dataset.yaml")
    dataset = instantiate_from_config(dataset_config.data.params.train)

    # evaluate
    indices = _sample_indices(len(dataset), args.num_samples)
    images, gt_trajectories, pred_trajectories = run_eval(
        model=model,
        dataset=dataset,
        indices=indices,
        num_steps=args.num_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / args.output_pdf

    save_pdf(
        images=images,
        gt_trajectories=gt_trajectories,
        pred_trajectories=pred_trajectories,
        indices=indices,
        output_pdf=output_pdf,
    )


if __name__ == "__main__":
    main()
