"""Trajectory representations: what the flow-matching model generates.

A future is 8 poses (x, y, heading) at 2 Hz in the current ego frame. A codec maps
it to a vector z of size latent_dim and back:

    raw    the 24 numbers themselves
    poly   a least-squares polynomial in time per channel, without a constant term
           (every channel is 0 at the current frame): x(tau) = c_1 tau + ... + c_K tau^K,
           tau = t / 8 for t = 1..8. z holds 3 K coefficients. K < 8 removes the
           high-frequency part of the trajectory from the target, so the model
           cannot spend capacity on it.

Both standardize z per entry with statistics of the training futures (fit), stored
as buffers so they travel with the checkpoint.
"""

import torch
import torch.nn as nn

NUM_POSES = 8


class Codec(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("std", torch.ones(latent_dim))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    def to_z(self, future):
        raise NotImplementedError

    def from_z(self, z):
        raise NotImplementedError

    @torch.no_grad()
    def fit(self, futures):
        """futures (N, 8, 3) of the training set."""
        z = self.to_z(futures.float())
        self.mean.copy_(z.mean(0))
        self.std.copy_(z.std(0).clamp(min=1e-4))
        self.fitted.fill_(True)
        rec = (self.from_z(z)[..., :2] - futures[..., :2]).norm(dim=-1)        # (N, 8)
        return {"codec_ade": rec.mean().item(), "codec_fde": rec[:, -1].mean().item()}

    def encode(self, future):
        return (self.to_z(future.float()) - self.mean) / self.std

    def decode(self, z):
        return self.from_z(z.float() * self.std + self.mean)


class RawCodec(Codec):
    def __init__(self):
        super().__init__(NUM_POSES * 3)

    def to_z(self, future):
        return future.flatten(1)

    def from_z(self, z):
        return z.unflatten(1, (NUM_POSES, 3))


class PolyCodec(Codec):
    def __init__(self, order=3):
        super().__init__(3 * order)
        self.order = order
        tau = torch.arange(1, NUM_POSES + 1, dtype=torch.float64) / NUM_POSES
        basis = tau[:, None] ** torch.arange(1, order + 1)               # (8, K)
        self.register_buffer("basis", basis.float(), persistent=False)
        self.register_buffer("pinv", torch.linalg.pinv(basis).float(), persistent=False)   # (K, 8)

    def to_z(self, future):
        return (self.pinv @ future).flatten(1)                          # (B, K * 3)

    def from_z(self, z):
        return self.basis @ z.unflatten(1, (self.order, 3))             # (B, 8, 3)


def build_codec(kind, order=3):
    if kind == "raw":
        return RawCodec()
    if kind == "poly":
        return PolyCodec(order)
    raise ValueError(f"codec must be raw or poly, got {kind!r}")
