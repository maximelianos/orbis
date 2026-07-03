#!/usr/bin/env python3
"""
NuPlan image encoding/decoding visualization script.
Loads images, applies encoder/decoder, and saves before/after visualizations.

Usage:
    python evaluate/nuplan_recode.py --exp_dir logs_tk/tokenizer_288x512
"""
import argparse
import sys
from pathlib import Path
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from util import instantiate_from_config
from loaders_for_projects.custom_multiframe_odo import MultiHDF5DatasetMultiFrameIdxMappingOdometry
from loaders_for_projects.encoder import Encoder


def load_config(config_path='loaders_for_projects/example_data_config.yaml'):
    """Load parameters from YAML config file"""
    config_path = os.path.expandvars(config_path)
    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)

    train_config = config['data']['params']['train']
    if isinstance(train_config, list):
        train_config = train_config[0]

    params = train_config['params']

    # Fix module paths
    if 'odo_transform_config' in params and params['odo_transform_config']:
        odo_config = params['odo_transform_config']
        if 'target' in odo_config:
            odo_config['target'] = odo_config['target'].replace(
                'data.custom_multiframe_odo',
                'loaders_for_projects.custom_multiframe_odo'
            )

    return params


def create_dataset(params):
    """Create NuPlan dataset instance"""
    dataset = MultiHDF5DatasetMultiFrameIdxMappingOdometry(
        size=params['size'],
        hdf5_paths_file=params['hdf5_paths_file'],
        num_frames=params['num_frames'],
        stored_data_frame_rate=params['stored_data_frame_rate'],
        frame_rate=params['frame_rate'],
        aug=params['aug'],
        scale_min=params['scale_min'],
        scale_max=params['scale_max'],
        odo_transform_config=params.get('odo_transform_config')
    )
    return dataset


def save_images(images, batch_idx, output_dir, prefix='before'):
    """Save 5 images from a batch as a single plot"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i in range(5):
        # Convert from [-1, 1] to [0, 1]
        img = (images[i].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f'Sample {i}')
        axes[i].axis('off')

    plt.suptitle(f'{prefix.capitalize()} - Batch {batch_idx}', fontsize=16)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'batch_{batch_idx:02d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {prefix} visualization: {output_path}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="NuPlan recode visualization")
    parser.add_argument("--exp_dir", type=str, required=True, help="Tokenizer experiment directory")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt", help="Checkpoint path")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    seed_everything(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build & load model
    print("Loading model via Encoder class...")
    encoder = Encoder(
        exp_dir=args.exp_dir,
        ckpt=args.ckpt,
        config=args.config,
        device=args.device
    )
    print("Model loaded successfully")

    # Load dataset
    print("\nLoading NuPlan dataset...")
    params = load_config()
    dataset = create_dataset(params)
    print(f"Dataset created: {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    # Create output directories
    before_dir = 'visualizations/before'
    after_dir = 'visualizations/after'
    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)

    # Evenly spaced batch indices from 32 batches
    total_batches = 32
    selected_batches = [0, 7, 15, 23, 31]  # 5 evenly spaced batches

    print(f"\nProcessing {total_batches} batches...")
    print(f"Will save visualizations for batches: {selected_batches}")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= total_batches:
            break

        # Get images - handle multi-frame data
        x = batch["images"].to(device, non_blocking=True)

        # If multi-frame, take first frame
        if x.dim() == 5:  # [B, T, C, H, W]
            x = x[:, 0]  # [B, C, H, W]

        # Save before images for selected batches
        if batch_idx in selected_batches:
            save_images(x[:5], batch_idx, before_dir, prefix='before')

        # Encode and decode using Encoder class
        continuous_latents = encoder.encode(x)
        decoded = encoder.decode(continuous_latents)

        # Convert to float32 for matplotlib compatibility
        decoded = decoded.float()

        # Save after images for selected batches
        if batch_idx in selected_batches:
            save_images(decoded[:5], batch_idx, after_dir, prefix='after')

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{total_batches} batches")

    print(f"\n✓ Processing complete!")
    print(f"Before images saved to: {before_dir}/")
    print(f"After images saved to: {after_dir}/")


if __name__ == "__main__":
    main()
