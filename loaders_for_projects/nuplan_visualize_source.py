"""
NuPlan Dataset Visualization

Usage:
    python loaders_for_projects/nuplan_visualize.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import instantiate_from_config
from loaders_for_projects.nuplan_dataloader import NuPlanVelocityDataset
from loaders_for_projects.draw_nuplan import save_pictures

def create_dataset(config_path):
    """Create NuPlan dataset instance from config.
    Use loaders_for_projects.nuplan_dataloader.NuPlanVelocityDataset
    """
    config = OmegaConf.load(config_path)
    return instantiate_from_config(config.data.params.train)

def main():
    print(f"\n{'='*60}")
    print("NuPlan Dataset Visualization")
    print(f"{'='*60}")
    
    # Sampling parameters
    num_samples = 100
    indices = np.linspace(0, len(dataset) - 1, num_samples).astype(int)
    #indices = np.linspace(0, 1000, num_samples).astype(int)
    

    # Create dataset directly from config
    print("\nCreating dataset from configs/nuplan_v1.yaml...")
    dataset = create_dataset('configs/nuplan_v1.yaml')
    print(f"✓ Dataset created: {len(dataset)} samples")

    # Load a sample
    print("\nLoading sample...")
    sample = dataset[0]
    print(f"  images shape: {sample['images'].shape}")
    print(f"  trajectory shape: {sample['trajectory'].shape}")

    # Test encode/decode functionality
    print("\nTesting encode/decode...")
    trajectory = sample['trajectory'][:, :2]  # Extract (x, y) only
    print(f"  Original trajectory shape: {trajectory.shape}")
    
    # Encode to velocity
    velocity = NuPlanVelocityDataset.traj_to_velocity(trajectory)
    print(f"  Encoded velocity shape: {velocity.shape}")
    print(f"  Velocity range: [{velocity.min():.6f}, {velocity.max():.6f}]")
    
    # Decode back to trajectory
    start_pos = trajectory[0].cpu().numpy() if isinstance(trajectory, torch.Tensor) else trajectory[0]
    reconstructed = NuPlanVelocityDataset.velocity_to_traj(velocity, start_position=start_pos)
    print(f"  Reconstructed trajectory shape: {reconstructed.shape}")
    
    # Compute error
    error = np.abs(trajectory - reconstructed).max()
    print(f"  Reconstruction error: {error:.6e} meters")
    
    if error < 1e-5:
        print("  ✓ Encode/decode round-trip successful!")
    else:
        print("  ⚠ Warning: reconstruction error detected")

    # Visualize
    print("\nCreating visualization...")
    pdf_output_path = "nuplan_pdf.pdf"
    
    with PdfPages(pdf_output_path) as pdf:
        for i, idx in enumerate(indices):
            print(f"  Processing sample {idx} ({i+1}/{num_samples})...")
            
            # Load sample
            sample = dataset[idx]
            
            fig_pdf = save_pictures([sample['trajectory']], title=f"Sample #{idx}", images=sample['images'], output_path=None)
            pdf.savefig(fig_pdf, dpi=150, bbox_inches='tight')
            
            plt.close(fig_pdf)
    

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
