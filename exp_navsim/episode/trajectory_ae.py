"""Trajectory autoencoder: normalized velocity (B, T, 2) -> z (B, latent_dim) -> (B, T, 2).

Encoder: transformer over the frames with a [CLS] token, padded frames masked.
Decoder: neural field f(z, frame index), so it decodes any length T and a shorter
T gives a prefix of a longer one. z is a small VAE latent (KL to N(0, 1)) so that
nearby values decode to similar trajectories and the denoiser gets a unit-scale
target.
"""

import numpy as np
import torch
import torch.nn as nn

from networks.DiT.dit import get_1d_sincos_pos_embed_from_grid
from exp_navsim.episode.metrics import masked_episode_mean


class FiLMBlock(nn.Module):
    """Residual MLP on the frame tokens, modulated by the latent code."""

    def __init__(self, D):
        super().__init__()
        self.norm = nn.LayerNorm(D, elementwise_affine=False)
        self.film = nn.Linear(D, 2 * D)
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))

    def forward(self, h, c):
        """h: (B, T, D) frame tokens, c: (B, D) latent code."""
        scale, shift = self.film(c)[:, None].chunk(2, dim=-1)
        return h + self.mlp(self.norm(h) * (1 + scale) + shift)


class TrajectoryAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim=1, enc_layers=2, dec_blocks=3,
                 kl_weight=1e-3, pos_weight=1.0, max_frames=1000):
        super().__init__()
        D = hidden_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.pos_weight = pos_weight
        # Sin-cos embedding of the absolute frame index, (max_frames, D).
        frame_emb = get_1d_sincos_pos_embed_from_grid(D, np.arange(max_frames, dtype=np.float32))
        self.register_buffer("frame_emb", torch.from_numpy(frame_emb).float(), persistent=False)

        self.in_proj = nn.Linear(2, D)
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        layer = nn.TransformerEncoderLayer(D, 4, 2 * D, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, enc_layers)
        self.to_stats = nn.Linear(D, 2 * latent_dim)           # (mu, logvar)

        self.z_mlp = nn.Sequential(nn.Linear(latent_dim, D), nn.GELU(), nn.Linear(D, D))
        self.blocks = nn.ModuleList([FiLMBlock(D) for _ in range(dec_blocks)])
        self.out = nn.Linear(D, 2)

    def encode(self, v, frame_mask):
        """v: (B, T, 2), frame_mask: (B, T) True = real. Returns mu, logvar (B, latent_dim)."""
        B, T, _ = v.shape
        x = self.in_proj(v) + self.frame_emb[:T]
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)  # (B, 1 + T, D)
        pad = torch.cat([frame_mask.new_zeros(B, 1), ~frame_mask], dim=1)
        # fp32: under CPU (bf16) autocast the encoder's fused fast path gets mixed dtypes.
        with torch.autocast(v.device.type, enabled=False):
            h = self.encoder(x.float(), src_key_padding_mask=pad)[:, 0]    # CLS token
        return self.to_stats(h).chunk(2, dim=-1)

    def decode(self, z, T):
        """z: (B, latent_dim) -> (B, T, 2) normalized velocity."""
        c = self.z_mlp(z)
        h = self.frame_emb[:T].expand(z.shape[0], -1, -1)
        for block in self.blocks:
            h = block(h, c)
        return self.out(h)

    def forward(self, v, frame_mask, scored):
        """Returns (loss, logs, mu, None). `scored` selects the frames that enter the loss.

        The None is the latent mask of the shared interface (trajectory_repr.py):
        every entry of mu is real.
        """
        mu, logvar = self.encode(v, frame_mask)
        logvar = logvar.clamp(-10.0, 10.0)                     # exp() stays finite in fp16
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        err = self.decode(z, v.shape[1]) - v
        rec = masked_episode_mean((err ** 2).mean(-1), scored)
        # Position error: velocity errors accumulate along the trajectory.
        pos = masked_episode_mean((err.cumsum(1) ** 2).mean(-1), scored)
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(-1).mean()
        loss = rec + self.pos_weight * pos + self.kl_weight * kl
        return loss, {"ae_rec": rec, "ae_pos": pos, "ae_kl": kl}, mu, None
