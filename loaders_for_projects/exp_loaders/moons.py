"""
Moon dataset dataloader for 2D distribution learning.

Generates samples from two interleaving half-moon shapes.
"""

import numpy as np
import torch


class MoonDataset:
    """
    Dataset that generates samples from two interleaving half-moon distributions.
    
    Each sample is a 2D point (x, y) from the moons distribution.
    """
    
    def __init__(self, n_samples, noise=0.1, random_state=42, *args, **kwargs):
        """
        Args:
            n_samples: Number of samples in the dataset
            noise: Standard deviation of Gaussian noise added to the data
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.noise = noise
        self.rng = np.random.RandomState(random_state)
        
        # Pre-generate all samples for consistency
        self.samples = self._generate_moons()
    
    def _generate_moons(self):
        """
        Generate samples from two interleaving half-moon distributions.
        
        Returns:
            np.ndarray: Array of shape (n_samples, 2) with 2D points
        """
        n_samples_per_moon = self.n_samples // 2
        
        # First moon (upper)
        angles1 = self.rng.uniform(0, np.pi, n_samples_per_moon)
        x1 = np.cos(angles1)
        y1 = np.sin(angles1)
        
        # Second moon (lower, shifted and rotated)
        angles2 = self.rng.uniform(0, np.pi, n_samples_per_moon)
        x2 = 1 - np.cos(angles2)
        y2 = 0.5 - np.sin(angles2)
        
        # Combine both moons
        X = np.vstack([
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2])
        ])
        
        # Add Gaussian noise
        if self.noise > 0:
            X += self.rng.normal(0, self.noise, X.shape)
        
        return X.astype(np.float32)
    
    def __getitem__(self, idx):
        """
        Get a single 2D sample from the moons distribution.
        
        Args:
            idx: Index of the sample
            
        Returns:
            dict: Dictionary with 'value' key containing 2D point (x, y)
        """
        sample = dict()
        sample['value'] = self.samples[idx % len(self.samples)]
        return sample
    
    def __len__(self):
        return self.n_samples
