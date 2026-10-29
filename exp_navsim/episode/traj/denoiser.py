"""Trajectory-only denoiser: predicts the flow-matching velocity of the latent z of the future.

context velocities (B, K - 1, 2) --Linear--> + learned frame embedding --> (B, K - 1, D)
noisy z (B, latent_dim) + diffusion-time embedding --MLP--> one query token (B, 1, D)
depth x ContextTransformer: the query attends to [context tokens, query], AdaLN-Zero on t,
    context frames missing from short episodes are masked out
DiT FinalLayer --> (B, latent_dim)

Same layout as exp_navsim/episode/denoiser.py, with the past velocities in place of the image tokens.
"""

import torch
import torch.nn as nn

from networks.blocks import AdaLNSingle, ContextTransformer
from networks.DiT.dit import FinalLayer, TimestepEmbedder


class TrajectoryContextDenoiser(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_context, ff_dim, num_heads, depth=4, dropout=0.):
        """num_context: number of context velocities, context_traj - 1 (v_0 is always 0)."""
        super().__init__()
        D = self.hidden_dim = hidden_dim
        self.num_context = num_context

        self.context_proj = nn.Linear(2, D)
        self.frame_emb = nn.Parameter(torch.zeros(num_context, D))
        nn.init.trunc_normal_(self.frame_emb, std=0.02)

        self.t_embedder = TimestepEmbedder(D)
        self.query_mlp = nn.Sequential(nn.Linear(latent_dim + D, D), nn.GELU(), nn.Linear(D, D))
        self.adaLN = AdaLNSingle(D, D)
        self.blocks = nn.ModuleList([ContextTransformer(D, ff_dim, num_heads, dropout)
                                     for _ in range(depth)])

        # Zero-init as in DiT: the model starts by predicting a zero flow.
        self.final_layer = FinalLayer(D, patch_size=1, out_channels=latent_dim)
        for layer in (self.final_layer.adaLN_modulation[-1], self.final_layer.linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, context, context_mask, z_t, t):
        """
        Args:
            context:      (B, num_context, 2) normalized velocities v_1 .. v_{K-1}.
            context_mask: (B, num_context) bool, True = real frame.
            z_t:          (B, latent_dim) noisy latent.
            t:            (B,) diffusion time in [0, 1].
        Returns:
            (B, latent_dim) predicted flow.
        """
        tokens = self.context_proj(context) + self.frame_emb[:context.shape[1]]    # (B, K - 1, D)

        # DiT's timestep embedding expects t on the 0..1000 scale.
        time = self.t_embedder(t * 1000)                                   # (B, D)
        query = self.query_mlp(torch.cat([z_t, time], dim=-1))[:, None]    # (B, 1, D)
        # The query itself is always attended to, so a fully padded context still works.
        pad = torch.cat([~context_mask, context_mask.new_zeros(z_t.shape[0], 1)], dim=1)
        modulation = self.adaLN(time)
        for block in self.blocks:
            query = block(torch.cat([tokens, query], dim=1), query, modulation, key_padding_mask=pad)
        return self.final_layer(query, time)[:, 0]
