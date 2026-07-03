"""
Test script for SimpleDiffusionModel (trajectory prediction).

This script instantiates the SimpleDiffusionModel, loads weights from a checkpoint,
runs sampling to generate trajectories, and visualizes the results.

Example usage:
    # Train
    python train_nuplan.py --config exp/whole_trajectory.yaml --logdir log_exp --max_steps 5000
    
    python exp/test_trajectory_diffusion.py --config exp/whole_trajectory.yaml --logdir log_exp --last_ckpt --num_samples 50

    # Evaluate
    python exp/test_trajectory_diffusion.py --config configs/simple_trajectory.yaml --ckpt logs_nuplan/2026-01-30T07-26-17_simple_trajectory/checkpoints/last.ckpt

    # Use last checkpoint
    python exp/test_trajectory_diffusion.py --config configs/simple_trajectory.yaml --last_ckpt --logdir logs_nuplan

    python exp/test_trajectory_diffusion.py --config exp/whole_trajectory.yaml --last_ckpt --logdir logs_nuplan --num_samples 1000
    
    
    
"""

import argparse
import sys
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import instantiate_from_config
from loaders_for_projects.exp_loaders.synthetic_trajectory_dataloader_IMPORTS import generate_trajectory


def compare_with_dataset(model, dataset, num_samples, T, num_diffusion_steps=20, context_length=0, output_path='comparison.png'):
    """
    Compare model predictions with ground truth from dataset.
    
    Args:
        model: Trained model
        dataset: Dataset with ground truth trajectories
        num_samples: Number of samples to compare
        T: Number of timesteps
        num_diffusion_steps: Number of diffusion steps
        context_length: Number of GT trajectory points to use as prefix conditioning (1 = start at (0, 0))
        output_path: Path to save comparison plot
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Sample turn values uniformly with linspace
    turns_array = np.linspace(0, 1, num_samples)
    
    # Generate ground truth trajectories and prefix batches using dataset.generate_trajectory
    gt_trajectories = []
    position_batch = []
    for turn_val in turns_array:
        sample = dataset.generate_trajectory(turn_val, T=T, r=dataset.r)
        gt_trajectories.append(sample['position'].numpy())  # (T, 2)
        position_batch.append(sample['position'].unsqueeze(0))
    gt_trajectories = np.array(gt_trajectories)  # (N, T, 2)

    # Prepare batch for model - position contains full trajectory with prefix
    turn_tensor = torch.tensor(turns_array, dtype=torch.float32, device=device)
    position_tensor = torch.cat(position_batch, dim=0).to(device)  # (N, T, 2)
    batch = {
        'turn': turn_tensor,
        'position': position_tensor
    }

    with torch.no_grad():
        pred_trajectories = model.sample(batch, context_length=context_length, T=T, num_diffusion_steps=num_diffusion_steps, dataset=dataset)
    pred_trajectories = pred_trajectories.cpu().numpy()
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_samples))
    
    # Plot ground truth
    for i in range(num_samples):
        traj = gt_trajectories[i]
        x, y = traj[:, 0], traj[:, 1]
        ax1.plot(x, y, color=colors[i], linewidth=2, alpha=0.7, label=f'turn={turns_array[i]:.2f}')
        ax1.scatter(x[0], y[0], color=colors[i], s=100, marker='o', 
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax1.scatter(x[-1], y[-1], color=colors[i], s=100, marker='s', 
                   edgecolors='black', linewidths=1.5, zorder=5)
    
    ax1.scatter(0, 0, color='green', s=200, marker='*', 
               edgecolors='black', linewidths=2, zorder=10)
    ax1.set_xlabel('X position (m)', fontsize=12)
    ax1.set_ylabel('Y position (m)', fontsize=12)
    ax1.set_title('Ground Truth Trajectories', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    #ax1.legend(fontsize=8)
    
    # Plot predictions
    for i in range(num_samples):
        traj = pred_trajectories[i]
        x, y = traj[:, 0], traj[:, 1]
        ax2.plot(x, y, color=colors[i], linewidth=2, alpha=0.7)
        ax2.scatter(x[0], y[0], color=colors[i], s=100, marker='o', 
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax2.scatter(x[-1], y[-1], color=colors[i], s=100, marker='s', 
                   edgecolors='black', linewidths=1.5, zorder=5)
    
    ax2.scatter(0, 0, color='green', s=200, marker='*', 
               edgecolors='black', linewidths=2, zorder=10)
    ax2.set_xlabel('X position (m)', fontsize=12)
    ax2.set_ylabel('Y position (m)', fontsize=12)
    ax2.set_title('Generated Trajectories (Model)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SimpleDiffusionModel for trajectory prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--last_ckpt", action="store_true", help="Use last checkpoint from logdir")
    parser.add_argument("--logdir", type=str, default="logs_nuplan", help="Directory for logs")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of diffusion sampling steps")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of trajectories to generate")
    parser.add_argument("--T", type=int, default=20, help="Number of timesteps per trajectory")
    parser.add_argument("--context_length", type=int, default=0, help="Number of GT trajectory points to use as prefix conditioning")
    parser.add_argument("--output_dir", type=str, default="evaluate", help="Output directory for plots")
    args = parser.parse_args()

    # Handle --last_ckpt option
    if args.last_ckpt:
        if not os.path.exists(args.logdir):
            raise ValueError(f"Logdir {args.logdir} does not exist")
        
        # Find all subdirectories in logdir
        subdirs = [d for d in os.listdir(args.logdir) 
                   if os.path.isdir(os.path.join(args.logdir, d))]
        
        if not subdirs:
            raise ValueError(f"No subdirectories found in {args.logdir}")
        
        # Sort by name and get the last one
        subdirs.sort()
        last_subdir = subdirs[-1]
        
        # Construct path to last checkpoint
        args.ckpt = os.path.join(args.logdir, last_subdir, "checkpoints", "last.ckpt")
        
        if not os.path.exists(args.ckpt):
            raise ValueError(f"Checkpoint not found at {args.ckpt}")
        
        print(f"Using last checkpoint from {args.ckpt}")
    
    if args.ckpt is None:
        raise ValueError("Either --ckpt or --last_ckpt must be provided")

    # Load config
    config = OmegaConf.load(args.config)

    # Instantiate the model from config
    model = instantiate_from_config(config.model)

    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    print("Model loaded successfully")

    # Load dataset for denormalization
    print("\nLoading dataset for denormalization...")
    dataset = instantiate_from_config(config.data.params.train)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Compare with ground truth
    print("\nCreating comparison plot...")
    comparison_path = output_dir / "trajectory_comparison.png"
    compare_with_dataset(model, dataset, num_samples=min(100, args.num_samples), 
                       T=args.T, num_diffusion_steps=args.num_steps,
                       context_length=args.context_length,
                       output_path=str(comparison_path))
    
    print("\nTesting complete!")
