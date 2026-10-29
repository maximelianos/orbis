"""Fixed trajectory representations, alternatives to the VAE in trajectory_ae.py.

Both have the same interface as TrajectoryAE, so the model can switch between them:
    latent_dim                          size of z
    forward(v, frame_mask, scored)  ->  (loss, logs, z, z_mask)
    decode(z, T)                    ->  (B, T, 2) normalized velocity
z_mask is (B, latent_dim) bool or None (every entry is real). Neither has
trainable weights, so their loss is 0.

PolyTrajectory: the positions P_t = v_1 + ... + v_t (P_0 = 0) of each channel are
    fitted by c_1 tau + ... + c_K tau^K, tau = t / time_scale. z holds the K x 2
    coefficients, scaled to unit variance by an EMA. v_0 is not a real step
    (velocity is 0 at frame 0), so it is left out of the fit and decoded as 0.
RawTrajectory: z is the normalized velocity itself, zero-padded to max_frames.
"""

import torch
import torch.nn as nn

from exp_navsim.episode.norm import TrajectoryNorm
from exp_navsim.episode.metrics import masked_episode_mean


class PolyTrajectory(nn.Module):
    def __init__(self, order=2, time_scale=10.0, ridge=1e-6):
        super().__init__()
        self.order = order
        self.time_scale = time_scale                    # frames per unit of tau
        self.ridge = ridge                              # keeps the normal equations invertible
        self.latent_dim = 2 * order
        self.coef_norm = TrajectoryNorm(dim=self.latent_dim)

    def basis(self, T, device):
        """(T, K): tau^1 .. tau^K for frames 0 .. T-1."""
        tau = torch.arange(T, device=device, dtype=torch.float32) / self.time_scale
        return tau[:, None] ** torch.arange(1, self.order + 1, device=device)

    def encode(self, v, scored):
        """Least-squares fit over the scored frames. v: (B, T, 2) -> coefficients (B, K, 2)."""
        T = v.shape[1]
        with torch.autocast(v.device.type, enabled=False):     # linalg.solve needs fp32
            v = v.float()
            P = torch.cat([torch.zeros_like(v[:, :1]), v[:, 1:].cumsum(1)], dim=1)
            phi = self.basis(T, v.device)
            w_phi = (phi[None] * scored.float()[..., None]).transpose(1, 2)    # (B, K, T)
            A = w_phi @ phi + self.ridge * torch.eye(self.order, device=v.device)
            return torch.linalg.solve(A, w_phi @ P)

    def velocity(self, coeffs, T):
        """Coefficients (B, K, 2) -> velocity (B, T, 2), v_0 = 0."""
        P = self.basis(T, coeffs.device) @ coeffs.float()
        return torch.diff(P, dim=1, prepend=torch.zeros_like(P[:, :1]))

    def decode(self, z, T):
        coeffs = self.coef_norm.denormalize(z).reshape(z.shape[0], self.order, 2)
        return self.velocity(coeffs, T)

    def forward(self, v, frame_mask, scored):
        coeffs = self.encode(v, scored)
        z = self.coef_norm.normalize(coeffs.flatten(1))         # updates the EMA while training
        # Fit error, only logged.
        err = self.velocity(coeffs, v.shape[1]) - v
        rec = masked_episode_mean((err ** 2).mean(-1), scored)
        pos = masked_episode_mean((err[:, 1:].cumsum(1) ** 2).mean(-1), scored[:, 1:])
        return v.new_zeros(()), {"poly_rec": rec, "poly_pos": pos}, z, None


class RawTrajectory(nn.Module):
    def __init__(self, max_frames=64):
        super().__init__()
        self.max_frames = max_frames
        self.latent_dim = 2 * max_frames

    def forward(self, v, frame_mask, scored):
        B, T, _ = v.shape
        assert T <= self.max_frames, f"episode of {T} frames, traj_pure.max_frames is {self.max_frames}"
        z = v.new_zeros(B, self.max_frames, 2)
        z[:, :T] = v
        # Only scored frames enter the flow-matching loss.
        z_mask = scored.new_zeros(B, self.max_frames)
        z_mask[:, :T] = scored
        return v.new_zeros(()), {}, z.flatten(1), z_mask[..., None].expand(-1, -1, 2).flatten(1)

    def decode(self, z, T):
        return z.reshape(z.shape[0], self.max_frames, 2)[:, :T]
