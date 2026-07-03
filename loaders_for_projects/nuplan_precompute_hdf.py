#!/usr/bin/env python3
"""
Script to precompute and cache NuPlan dataset samples into an HDF5 file.
Loads the encoder, loads the NuPlan source dataset, and saves 100 samples to HDF5.

Usage:
    python loaders_for_projects/nuplan_precompute_hdf.py --num_samples 1 --max_index 1000
    
    python loaders_for_projects/nuplan_precompute_hdf.py --num_samples 10 --max_index -1
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import instantiate_from_config
from loaders_for_projects.encoder import Encoder
from loaders_for_projects.nuplan_dataloader import NuplanHDF

def main():
    print("=" * 60)
    print("NuPlan HDF5 Precomputation")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Save samples to single HDF")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_index", type=int, default=1000)  # -1 for original dataset length
    parser.add_argument("--data_config", type=str, default="configs/nuplan_export_config.yaml")
    args = parser.parse_args()

    # File paths
    hdf5_output_path = "data/cache/nuplan_precomputed.h5"
    normalization_cache_dir = "data/cache"
    num_samples = args.num_samples
    max_index = args.max_index
    
    print(f"\n1. Output configuration:")
    print(f"   HDF5 Path: {hdf5_output_path}")
    print(f"   Samples: {num_samples}")

    # Load Encoder
    print("\n2. Loading Encoder model...")
    encoder = Encoder(
        exp_dir="logs_tk/tokenizer_288x512",
        ckpt="checkpoints/last.ckpt",
        config="config.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("   ✓ Encoder loaded")

    # Load Source Dataset
    data_config = args.data_config
    print(f"\n3. Loading source dataset from {data_config}...")
    config = OmegaConf.load(data_config)
    source_dataset = instantiate_from_config(config.data.params.train)
    print(f"   ✓ Source dataset loaded with {len(source_dataset)} total samples")

    # Remove existing cache if any for a clean run
    cache_file = Path(hdf5_output_path)
    if cache_file.exists():
        print(f"\n   Removing existing cache file: {hdf5_output_path}")
        cache_file.unlink()

    # Create HDF wrapper
    print("\n4. Initializing NuplanHDF wrapper...")
    hdf_dataset = NuplanHDF(
        hdf5_path=str(cache_file),
        source_dataset=source_dataset,
        return_image=True,
        encoder=encoder,
        output_normalization=True,
        normalization_cache_dir=normalization_cache_dir
    )
    
    print("\n5. Starting precomputation loop...")
    # Force access to save samples and precompute
    if max_index == -1:
        max_index = len(source_dataset) # use original length
        print("source dataset length:", max_index)
    indices = np.linspace(0, max_index-1, num_samples).astype(int)
    for i in tqdm(indices, desc="Processing samples"):
        # Accessing the item triggers `load_sample` which triggers `save_sample` internally
        hdf_dataset.save_sample(i)
        
    print(f"\n✓ Precomputation complete! Cached {num_samples} samples sequentially.")
    
    # Validation step
    print("\n6. Validating HDF5 cache independently...")
    validation_dataset = NuplanHDF(
        hdf5_path=str(cache_file),
        return_image=True,
        output_normalization=True,
        normalization_cache_dir=normalization_cache_dir
    )
    print(f"   Found {len(validation_dataset)} items in HDF5")
    
    print("Precompute normalization...")
    validation_dataset.precompute_normalization()
    print("Precomputation complete!")
    
    assert len(validation_dataset) == num_samples, f"Expected {num_samples} items, got {len(validation_dataset)}"
    print("   ✓ Validation successful!")

if __name__ == "__main__":
    main()
