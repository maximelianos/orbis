"""
Synthetic Trajectory DataLoader

Generates synthetic trajectories that follow Bezier curves from (0, 0) to (X, Y),
where the endpoint is determined by a turn parameter (0 = left turn, 1 = right turn).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from pathlib import Path


def bezier_curve(p0, p1, p2, p3, t):
    """
    Calculate point on cubic Bezier curve.
    
    Args:
        p0, p1, p2, p3: Control points (2D arrays)
        t: Parameter from 0 to 1
        
    Returns:
        Point on Bezier curve at parameter t
    """
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def generate_trajectory(turn, T, r):
    """
    Generate synthetic trajectory using Bezier curve.
    
    Args:
        turn: Turn parameter from 0 (left, 30°) to 1 (right, -30°)
        T: Number of timesteps
        r: Radius/distance to endpoint
        
    Returns:
        trajectory: (T, 2) array of (x, y) positions
    """
    # Convert turn to angle: turn=0 -> 30°, turn=1 -> -30°
    alpha_deg = 30 - 60 * turn
    alpha_rad = np.deg2rad(alpha_deg)
    
    # Calculate endpoint
    X = r * np.cos(alpha_rad)
    Y = r * np.sin(alpha_rad)
    
    # Define control points for Bezier curve
    p0 = np.array([0.0, 0.0])  # Start at origin
    p3 = np.array([X, Y])       # End at target
    
    # Control points for smooth curve that starts more straight
    # First control point close to start for straighter beginning
    p1 = p0 + np.array([r, 0]) * 0.5
    p2 = p1
    #p1 = p0 + (p3 - p0) * 0.15 + np.array([0, 0.1 * r * (0.5 - turn)])
    # Second control point closer to end for gradual turn
    #p2 = p0 + (p3 - p0) * 0.75 + np.array([0, 0.4 * r * (0.5 - turn)])
    
    # Generate T points along the Bezier curve
    t_values = np.linspace(0, 1, T)
    trajectory = np.array([bezier_curve(p0, p1, p2, p3, t) for t in t_values])
    
    return trajectory


class SyntheticTrajectoryDataset(Dataset):
    """
    Dataset of synthetic trajectories with different turn parameters.
    
    Args:
        dataset_size: Number of samples in the dataset
        T: Number of timesteps per trajectory
        r: Radius/distance to trajectory endpoint
        turn_min: Minimum turn value (0 = left turn at 30°)
        turn_max: Maximum turn value (1 = right turn at -30°)
        normalization_cache_dir: Directory to cache normalization statistics
    """
    
    def __init__(self, dataset_size, T, r, turn_min=0.0, turn_max=1.0,
                 output_normalization=True,
                 normalization_cache_dir="data/cache", *args, **kwargs):
        super().__init__()
        self.dataset_size = dataset_size
        self.T = T
        self.r = r
        self.output_normalization = output_normalization
        self.normalization_cache_dir = Path(normalization_cache_dir)
        self.normalization_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-generate random turn parameters
        self.turns = np.random.uniform(turn_min, turn_max, size=dataset_size)
        
        # Precompute or load normalization statistics
        self.precompute_normalization()
    
    def precompute_normalization(self):
        """
        Precompute mean and std of velocity for normalization.
        Saves to disk and loads from cache if available.
        """
        # Create unique cache filename based on dataset parameters
        cache_filename = f"norm_stats_size{self.dataset_size}_T{self.T}_r{self.r}.npz"
        cache_path = self.normalization_cache_dir / cache_filename
        
        if cache_path.exists():
            # Load from cache
            print(f"Loading normalization stats from {cache_path}")
            data = np.load(cache_path)
            self.velocity_mean = data['mean']
            self.velocity_std = data['std']
        else:
            # Compute statistics
            print(f"Computing normalization stats for {self.dataset_size} samples...")
            velocities = []
            
            for idx in range(self.dataset_size):
                turn = self.turns[idx]
                sample = self.generate_trajectory(turn, T=self.T, r=self.r)
                velocities.append(sample['velocity'].numpy())
            
            velocities = np.concatenate(velocities, axis=0)  # (dataset_size * (T-1), 2)
            
            self.velocity_mean = velocities.mean(axis=0)
            self.velocity_std = velocities.std(axis=0)
            
            # Save to cache
            np.savez(cache_path, mean=self.velocity_mean, std=self.velocity_std)
            print(f"Saved normalization stats to {cache_path}")
            print(f"  Velocity mean: {self.velocity_mean}")
            print(f"  Velocity std: {self.velocity_std}")
    
    def generate_trajectory(self, turn, T, r):
        """
        Generate a single unnormalized trajectory sample.
        
        Args:
            turn: Turn parameter from 0 to 1
            T: Number of timesteps
            r: Radius/distance to endpoint
            
        Returns:
            sample: dict with keys:
                - 'turn': float in [0, 1]
                - 'position': (T, 2) array of (x, y) positions
                - 'velocity': (T, 2) array of (dx, dy) velocities
        """
        trajectory = generate_trajectory(turn, T=T, r=r)  # (T, 2)
        
        # Compute velocity: x_{t+1} - x_t
        velocity = np.zeros_like(trajectory)
        velocity[:-1] = trajectory[1:] - trajectory[:-1]  # (T, 2)
        
        sample = {
            'turn': torch.tensor(turn, dtype=torch.float32),
            'position': torch.tensor(trajectory, dtype=torch.float32),
            'velocity': torch.tensor(velocity, dtype=torch.float32),
        }
        
        return sample
    
    def normalize(self, batch):
        """
        Normalize velocity in batch.
        
        Args:
            batch: dict with 'velocity' key
            
        Returns:
            batch: dict with normalized 'velocity'
        """
        if not self.output_normalization:
            return batch
        
        batch = batch.copy()
        velocity = batch['velocity']
        
        # Normalize: (v - mean) / std
        mu = torch.tensor(self.velocity_mean, dtype=velocity.dtype)
        sigma = torch.tensor(self.velocity_std, dtype=velocity.dtype) + 1e-8
        velocity_normalized = (velocity - mu) / sigma
        
        batch['velocity'] = velocity_normalized
        return batch

    def denormalize(self, batch):
        """
        Args:
            batch: dict with 'velocity' key
            
        Returns:
            batch: dict with denormalized 'velocity'
        """
        if not self.output_normalization:
            return batch

        batch = batch.copy()
        velocity = batch['velocity']
        
        # Denormalize: v * std + mean
        mu = torch.tensor(self.velocity_mean, dtype=velocity.dtype)
        sigma = torch.tensor(self.velocity_std, dtype=velocity.dtype) + 1e-8
        velocity_denormalized = velocity * sigma + mu
        
        batch['velocity'] = velocity_denormalized
        return batch
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        """
        Get a single trajectory sample with normalized velocity.
        
        Returns:
            sample: dict with keys:
                - 'turn': float in [0, 1]
                - 'position': (T, 2) array of (x, y) positions
                - 'velocity': (T, 2) array of normalized (dx, dy) velocities
        """
        turn = self.turns[idx]
        sample = self.generate_trajectory(turn, T=self.T, r=self.r)
        sample = self.normalize(sample)
        return sample


if __name__ == "__main__":
    # Test the dataset
    dataset = SyntheticTrajectoryDataset(dataset_size=10, T=20, r=1.0)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Turn: {sample['turn']:.3f}")
    print(f"  Position shape: {sample['position'].shape}")
    print(f"  Velocity shape: {sample['velocity'].shape}")
    print(f"  Start position: {sample['position'][0]}")
    print(f"  End position: {sample['position'][-1]}")
    print(f"  First velocity: {sample['velocity'][0]}")
    print(f"  Velocity mean: {sample['velocity'].mean(dim=0)}")
    print(f"  Velocity std: {sample['velocity'].std(dim=0)}")
    
    # Test unnormalized sample
    unnorm_sample = dataset.generate_trajectory(dataset.turns[0], T=dataset.T, r=dataset.r)
    print(f"\nUnnormalized velocity mean: {unnorm_sample['velocity'].mean(dim=0)}")
    print(f"Unnormalized velocity std: {unnorm_sample['velocity'].std(dim=0)}")
