"""Online (EMA) velocity normalization for NAVSIM trajectories.

Separate file, as requested. TrajectoryNorm tracks the running mean/variance of
the per-channel (dx, dy) velocity with an exponential moving average and stores
them as (non-trainable) nn parameters so they travel with the checkpoint. This
mirrors the precomputed dataset normalization `(velocity - mean) / std`, but the
statistics live inside the model and are updated online while training.
"""

import torch
import torch.nn as nn


class TrajectoryNorm(nn.Module):
    """Per-channel (dx, dy) velocity normalizer with EMA-tracked statistics."""

    def __init__(self, decay=0.999, eps=1e-8):
        super().__init__()
        self.decay = decay
        self.eps = eps

        # requires_grad=False: part of the state_dict, never touched by the optimizer
        self.mean = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.var = nn.Parameter(torch.ones(2), requires_grad=False)
        self.initialized = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)

    @torch.no_grad()
    def update(self, velocity):
        """Update running mean/variance from a batch of velocities (B, T, 2)."""
        # reduce all but the channel dim
        batch_mean = velocity.mean(dim=(0, 1))
        batch_var = velocity.var(dim=(0, 1), unbiased=False)

        if bool(self.initialized):
            # Exponential moving average
            new_mean = self.decay * self.mean + (1.0 - self.decay) * batch_mean
            new_var = self.decay * self.var + (1.0 - self.decay) * batch_var
        else:
            # First batch seeds the statistics directly.
            new_mean = batch_mean
            new_var = batch_var
            self.initialized.fill_(True)

        # Write back into the persistent (checkpointed) tensors in place.
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)

    @property
    def std(self):
        return torch.sqrt(self.var) + self.eps

    def normalize(self, velocity):
        """(velocity - mean) / std, updating the EMA only while training."""
        if self.training:
            self.update(velocity)
        return (velocity - self.mean) / self.std

    def denormalize(self, velocity):
        """Inverse of normalize: velocity * std + mean."""
        return velocity * self.std + self.mean

    def forward(self, velocity):
        return self.normalize(velocity)
