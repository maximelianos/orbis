#!/usr/bin/env python3
"""
NuPlan HDF5 Dataset Visualization
Loads data strictly from HDF5 cache and generates a PDF containing 5 visualization samples.

Usage:
    python loaders_for_projects/nuplan_visualize_hdf.py --num_samples 10
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders_for_projects.nuplan_dataloader import NuplanHDF, decode_trajectory
from loaders_for_projects.draw_nuplan import save_pictures

def main():
    print("=" * 60)
    print("NuPlan HDF5 Visualization")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Save samples to single HDF")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_hdf_index", type=int, default=1000)
    parser.add_argument("--data_config", type=str, default="configs/nuplan_export_config.yaml")
    parser.add_argument("--pdf_output", type=str, default="nuplan_hdf_visualizations.pdf")
    args = parser.parse_args()
    
    num_samples = args.num_samples
    pdf_output = args.pdf_output

    # File paths
    hdf5_path = "data/cache/nuplan_precomputed.h5"
    normalization_cache_dir = "data/cache"
    
    print(f"\nLoading NuplanHDF directly from: {hdf5_path}")
    if not os.path.exists(hdf5_path):
        print(f"Error: HDF5 file not found at {hdf5_path}.")
        print("Please run nuplan_precompute_hdf.py first.")
        return

    # Load dataset strictly from HDF5 (no source dataset configured)
    dataset = NuplanHDF(
        hdf5_path=hdf5_path,
        return_image=True,
        output_normalization=True,
        normalization_cache_dir=normalization_cache_dir
    )
    
    total_samples = len(dataset)
    print(f"✓ HDF5 Dataset loaded with {total_samples} samples")

    print(f"\nGenerating PDF with {num_samples} samples at {pdf_output}...")
    
    # Take evenly spaced samples from the subset available
    indices = np.linspace(0, total_samples - 1, num_samples, dtype=int)
    
    with PdfPages(pdf_output) as pdf:
        for i, idx in enumerate(indices):
            print(f"  Processing sample {idx} ({i+1}/{num_samples})...")
            
            # Load sample
            hdf_idx = dataset._resolve_hdf_index(idx)
            sample = dataset.load_sample(hdf_idx)
            sample = dataset.normalize(sample)
            
            def print_shape(arr):
                print(arr.shape, arr.dtype)
            
            print_shape(sample["images"])
            print_shape(sample["encoded_q_rec"])
            print_shape(sample["encoded_q_sem"])
            # input()
            
            denorm = dataset.denormalize(sample)
            velocity = denorm["velocity"]
            trajectory = decode_trajectory(velocity)
            
            fig_pdf = save_pictures([trajectory], title=f"HDF sample #{idx}", images=sample['images'], output_path=None)
            pdf.savefig(fig_pdf, dpi=150, bbox_inches='tight')
            plt.close(fig_pdf)
                
    print(f"\n✓ Completed! Generated multi-page PDF visualization.")

if __name__ == "__main__":
    main()
