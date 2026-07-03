"""
Test script for ExampleModel (simple diffusion model).

This script instantiates the SimpleDiffusionModel, loads weights from a checkpoint,
runs sampling with dummy input, and creates an animation of the diffusion process.

Example usage:
    python evaluate/test_simple_diffusion.py --config configs/nuplan_simple_diffusion.yaml --ckpt logs_nuplan/2026-01-29T16-26-36_nuplan_simple_diffusion/checkpoints/last.ckpt
    
    python evaluate/test_simple_diffusion.py --config configs/nuplan_simple_diffusion.yaml --logdir logs_nuplan --last_ckpt
    
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from util import instantiate_from_config
from PIL import Image
import io
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SimpleDiffusionModel with checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--last_ckpt", action="store_true", help="Use last checkpoint from logdir")
    parser.add_argument("--logdir", type=str, default="logs_nuplan", help="Directory for logs")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of diffusion sampling steps")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for testing")
    parser.add_argument("--output_dir", type=str, default="evaluate", help="Output directory for animation")
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

    # Create dummy batch
    dummy_batch = {
        'value': torch.randn(args.batch_size, 2)  # Assume 2D for visualization
    }

    # Run sampling with animation
    print(f"\nGenerating animation with {args.num_steps} diffusion steps...")
    
    B, D = dummy_batch['value'].shape
    device = dummy_batch['value'].device
    
    # Start from noise at t=1
    sampled_value = torch.randn(B, D, device=device)
    
    # Diffusion steps from t=1 to t=0
    t_steps = torch.linspace(1, 0, args.num_steps + 1, device=device)
    
    # Store frames for animation
    frames = []
    
    # Determine plot bounds based on initial noise
    samples_np = sampled_value.cpu().numpy()
    plot_lim = max(abs(samples_np).max() * 1.2, 3.0)
    
    for i in range(args.num_steps + 1):
        # Convert to numpy for plotting
        samples_np = sampled_value.cpu().numpy()
        
        # Compute color based on progress (red -> green)
        progress = i / args.num_steps  # 0 (red/start) to 1 (green/end)
        color = plt.cm.RdYlGn(progress)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 2D scatter plot
        ax.scatter(samples_np[:, 0], samples_np[:, 1], 
                    alpha=0.6, s=20, c=[color], edgecolors='none')
        ax.set_xlim(-plot_lim, plot_lim)
        ax.set_ylim(-plot_lim, plot_lim)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        
        # Add title with timestep info
        t_value = t_steps[i].item()
        ax.set_title(f'Diffusion Step {i}/{args.num_steps} (t={t_value:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frame = Image.open(buf).copy()
        frames.append(frame)
        if i == args.num_steps: # pause in the end
            for j in range(10):
                frames.append(frame)
        plt.close(fig)
        buf.close()
        
        # Perform denoising step (except for last iteration)
        if i < args.num_steps:
            t = t_steps[i].repeat(B)
            dt = t_steps[i] - t_steps[i + 1]
            sampled_value = model.sample_step(sampled_value, t, dt)
    
    # Save animation as GIF
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    gif_path = output_dir / "diffusion_animation.gif"
    
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,  # milliseconds per frame
        loop=0
    )
    
    print(f"Saved animation to {gif_path}")
    
    # Create comparison plot with ground truth
    print("\nCreating comparison plot with ground truth...")
    dataset = instantiate_from_config(config.data.params.train)
    
    # Sample from dataset
    gt_samples = []
    for i in range(len(dataset)):
        sample = dataset[i]
        gt_samples.append(sample['value'])
    gt_samples = np.array(gt_samples)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Ground truth
    ax1 = axes[0]
    ax1.scatter(gt_samples[:, 0], gt_samples[:, 1], 
            alpha=0.6, s=20, c='green', edgecolors='none')
    ax1.set_xlim(-plot_lim, plot_lim)
    ax1.set_ylim(-plot_lim, plot_lim)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('Ground Truth (Dataset)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Generated samples
    ax2 = axes[1]
    ax2.scatter(samples_np[:, 0], samples_np[:, 1], 
            alpha=0.6, s=20, c='blue', edgecolors='none')
    ax2.set_xlim(-plot_lim, plot_lim)
    ax2.set_ylim(-plot_lim, plot_lim)
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Generated (Model)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    comparison_path = output_dir / "comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved comparison plot to {comparison_path}")
    
    # Measure distance between distributions
    print("\nMeasuring distribution distance...")
    
    # Compute statistics for both distributions
    gt_mean = gt_samples.mean(axis=0)
    gt_std = gt_samples.std(axis=0)
    gen_mean = samples_np.mean(axis=0)
    gen_std = samples_np.std(axis=0)
    
    # Mean squared error between means
    mean_distance = np.sqrt(((gt_mean - gen_mean) ** 2).sum())
    
    # Mean squared error between standard deviations
    std_distance = np.sqrt(((gt_std - gen_std) ** 2).sum())
    
    # Compute 2-Wasserstein distance approximation using Gaussian assumption
    # W2^2 = ||mu1 - mu2||^2 + ||sigma1 - sigma2||_F^2
    wasserstein_distance = np.sqrt(mean_distance**2 + std_distance**2)
    
    # Compute KL divergence approximation (assuming Gaussian distributions)
    # For diagonal covariances: KL(P||Q) = 0.5 * (tr(Sigma_Q^-1 Sigma_P) + (mu_Q - mu_P)^T Sigma_Q^-1 (mu_Q - mu_P) - k + ln(det(Sigma_Q)/det(Sigma_P)))
    try:
        gt_cov = np.cov(gt_samples.T)
        gen_cov = np.cov(samples_np.T)
        
        # Add small regularization for numerical stability
        gt_cov += np.eye(D) * 1e-6
        gen_cov += np.eye(D) * 1e-6
        
        gt_cov_inv = np.linalg.inv(gt_cov)
        mean_diff = gen_mean - gt_mean
        
        kl_divergence = 0.5 * (
            np.trace(gt_cov_inv @ gen_cov) +
            mean_diff.T @ gt_cov_inv @ mean_diff -
            D +
            np.log(np.linalg.det(gt_cov) / np.linalg.det(gen_cov))
        )
    except:
        kl_divergence = float('nan')
    
    print(f"\nDistribution Distance Metrics:")
    print(f"  Mean Distance: {mean_distance:.4f}")
    print(f"  Std Distance: {std_distance:.4f}")
    print(f"  Approximate Wasserstein-2 Distance: {wasserstein_distance:.4f}")
    if not np.isnan(kl_divergence):
        print(f"  Approximate KL Divergence (Gen || GT): {kl_divergence:.4f}")
    
    print(f"\nGround Truth Statistics:")
    print(f"  Mean: {gt_mean}")
    print(f"  Std: {gt_std}")
    print(f"\nGenerated Statistics:")
    print(f"  Mean: {gen_mean}")
    print(f"  Std: {gen_std}")
    
    # Final statistics
    final_samples = sampled_value.cpu().numpy()
    print(f"\nFinal sampled values statistics:")
    if D == 2:
        print(f"  X - Mean: {final_samples[:, 0].mean():.4f}, Std: {final_samples[:, 0].std():.4f}")
        print(f"  Y - Mean: {final_samples[:, 1].mean():.4f}, Std: {final_samples[:, 1].std():.4f}")
    else:
        print(f"  Mean: {final_samples.mean():.4f}")
        print(f"  Std: {final_samples.std():.4f}")
        print(f"  Min: {final_samples.min():.4f}")
        print(f"  Max: {final_samples.max():.4f}")
