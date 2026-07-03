"""
Online velocity normalization.

TrajectoryNorm tracks the running mean and variance of the trajectory velocity
(per dx, dy channel) with an exponential moving average, mirroring the
normalization used by NuplanHDF (`(velocity - mean) / std`) but with the
statistics maintained inside the model as torch tensors instead of a
precomputed numpy cache.

test:
python -m networks.norm
"""

print("import started")
import torch
import torch.nn as nn
print("import finished")


class TrajectoryNorm(nn.Module):
    """Per-channel (dx, dy) velocity normalizer with EMA-tracked statistics.

    The mean and variance are stored as (non-trainable) nn parameters so they
    travel with the model checkpoint. During training the statistics are
    updated from each batch's velocity via an exponential moving average; at
    eval time they are frozen and only used to normalize / denormalize.
    """

    def __init__(self, decay=0.99, eps=1e-8):
        super().__init__()
        self.decay = decay
        self.eps = eps

        # Stored as parameters (requires_grad=False) so they are part of the
        # state_dict / checkpoint but never touched by the optimizer.
        self.mean = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.var = nn.Parameter(torch.ones(2), requires_grad=False)
        # Whether the EMA has been seeded from real data yet.
        self.initialized = nn.Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )

    @torch.no_grad()
    def update(self, velocity):
        """Update the running mean/variance from a batch of velocities.

        Args:
            velocity: (B, T, 2) trajectory deltas.
        """
        # Reduce over every dim except the channel (dx, dy) dim.
        reduce_dims = tuple(range(velocity.dim() - 1))
        batch_mean = velocity.mean(dim=reduce_dims)
        batch_var = velocity.var(dim=reduce_dims, unbiased=False)

        if not bool(self.initialized):
            self.mean.copy_(batch_mean)
            self.var.copy_(batch_var)
            self.initialized.fill_(True)
        else:
            self.mean.mul_(self.decay).add_(batch_mean, alpha=1.0 - self.decay)
            self.var.mul_(self.decay).add_(batch_var, alpha=1.0 - self.decay)

    @property
    def std(self):
        return torch.sqrt(self.var) + self.eps

    def normalize(self, velocity):
        """(velocity - mean) / std, updating the EMA while training."""
        if self.training:
            self.update(velocity)
        return (velocity - self.mean) / self.std

    def denormalize(self, velocity):
        """Inverse of normalize: velocity * std + mean."""
        return velocity * self.std + self.mean

    def forward(self, velocity):
        return self.normalize(velocity)


if __name__ == "__main__":
    # Self-test: EMA converges to the data stats, eval() freezes them, and
    # normalize/denormalize round-trip is exact.
    torch.manual_seed(0)
    true_mean = torch.tensor([2.0, -1.0])
    true_std = torch.tensor([3.0, 0.5])

    def sample_batch(b=8, t=10):
        return torch.randn(b, t, 2) * true_std + true_mean

    norm = TrajectoryNorm(decay=0.9)

    # First batch seeds the statistics directly.
    norm.train()
    norm.normalize(sample_batch())
    assert bool(norm.initialized)

    # EMA converges towards the true data statistics.
    for _ in range(2000):
        norm.normalize(sample_batch())
    assert torch.allclose(norm.mean.data, true_mean, atol=0.1), norm.mean.data
    assert torch.allclose(norm.std.data, true_std, atol=0.1), norm.std.data

    # eval() must not update the running statistics.
    norm.eval()
    frozen_mean = norm.mean.clone()
    frozen_var = norm.var.clone()
    norm.normalize(sample_batch())
    assert torch.allclose(norm.mean, frozen_mean)
    assert torch.allclose(norm.var, frozen_var)

    # normalize / denormalize round-trip.
    v = sample_batch()
    rec = norm.denormalize(norm.normalize(v))
    assert torch.allclose(rec, v, atol=1e-4), (rec - v).abs().max()

    # Stats are part of the state_dict (checkpointed with the model).
    keys = set(norm.state_dict())
    assert {"mean", "var", "initialized"} <= keys, keys

    print("TrajectoryNorm self-test passed.")
    print("  mean:", [round(x, 3) for x in norm.mean.data.tolist()])
    print("  std: ", [round(x, 3) for x in norm.std.data.tolist()])
