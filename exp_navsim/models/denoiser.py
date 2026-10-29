"""Probe denoiser: predicts the flow-matching velocity of the trajectory code z.

feature grids (B, T, C, h, w) --Linear per cell--> SpatialBlock --> N tokens per frame
    + learned frame embedding                                    --> (B, T * N, D)
optional ego status (velocity, acceleration) --MLP-->              one token
optional driving command (one-hot)           --Linear-->           one token
noisy z (B, latent_dim) + diffusion-time embedding --MLP--> one query token (B, 1, D)
depth x ContextTransformer: the query attends to [condition tokens, query], AdaLN-Zero on t
DiT FinalLayer --> (B, latent_dim)

With C = 0 there are no image tokens, which gives the status / command baselines.
"""

import torch
import torch.nn as nn

from networks.blocks import AdaLNSingle, ContextTransformer, SpatialBlock
from networks.DiT.dit import FinalLayer, TimestepEmbedder


class ProbeDenoiser(nn.Module):
    def __init__(self, latent_dim, in_channels, num_frames, hidden_dim=16, image_tokens=16, ff_dim=32,
                 num_heads=2, depth=4, use_status=False, use_command=False, dropout=0.):
        super().__init__()
        D = self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        if in_channels:
            self.image_proj = nn.Linear(in_channels, D)
            self.spatial_block = SpatialBlock(D, ff_dim, image_tokens, num_heads, dropout)
            # The pooling is unconditioned, so this marks which frame a token comes from.
            self.frame_emb = nn.Parameter(torch.zeros(num_frames, D))
            nn.init.trunc_normal_(self.frame_emb, std=0.02)
        self.status_mlp = nn.Sequential(nn.Linear(4, D), nn.GELU(), nn.Linear(D, D)) if use_status else None
        self.command_proj = nn.Linear(4, D) if use_command else None

        self.t_embedder = TimestepEmbedder(D)
        self.query_mlp = nn.Sequential(nn.Linear(latent_dim + D, D), nn.GELU(), nn.Linear(D, D))
        self.adaLN = AdaLNSingle(D, D)
        self.blocks = nn.ModuleList([ContextTransformer(D, ff_dim, num_heads, dropout) for _ in range(depth)])

        # Zero-init as in DiT: the model starts by predicting a zero flow.
        self.final_layer = FinalLayer(D, patch_size=1, out_channels=latent_dim)
        for layer in (self.final_layer.adaLN_modulation[-1], self.final_layer.linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def condition(self, batch):
        """Condition tokens (B, L, D); computed once per sampled trajectory batch."""
        tokens = []
        if self.in_channels:
            x = batch["features"]                                          # (B, T, C, h, w)
            B, T = x.shape[:2]
            x = self.image_proj(x.permute(0, 1, 3, 4, 2)).flatten(0, 1).permute(0, 3, 1, 2)
            tokens.append((self.spatial_block(x).unflatten(0, (B, T)) + self.frame_emb[:T, None]).flatten(1, 2))
        if self.status_mlp is not None:
            tokens.append(self.status_mlp(batch["status"])[:, None])
        if self.command_proj is not None:
            tokens.append(self.command_proj(batch["command"])[:, None])
        if not tokens:
            return batch["future"].new_zeros(batch["future"].shape[0], 0, self.hidden_dim)
        return torch.cat(tokens, dim=1)

    def forward(self, cond, z_t, t):
        """cond (B, L, D) from condition(), z_t (B, latent_dim), t (B,) in [0, 1] -> (B, latent_dim)."""
        # DiT's timestep embedding expects t on the 0..1000 scale.
        time = self.t_embedder(t * 1000)
        query = self.query_mlp(torch.cat([z_t, time], dim=-1))[:, None]
        modulation = self.adaLN(time)
        for block in self.blocks:
            query = block(torch.cat([cond, query], dim=1), query, modulation)
        return self.final_layer(query, time)[:, 0]
