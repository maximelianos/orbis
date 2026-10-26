"""Latent denoiser: predicts the flow-matching velocity of the trajectory latent z.

image latents (B, T_img, 2C, H, W) --Linear per cell--> SpatialBlock --> N tokens per frame,
    + learned frame embedding                                       --> (B, T_img * N, D)
noisy z (B, latent_dim) + diffusion-time embedding --MLP--> one query token (B, 1, D)
depth x ContextTransformer: the query attends to [image tokens, query], AdaLN-Zero on t
DiT FinalLayer --> (B, latent_dim)
"""

import torch
import torch.nn as nn

from networks.blocks import AdaLNSingle, ContextTransformer, SpatialBlock
from networks.DiT.dit import FinalLayer, TimestepEmbedder


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim, hidden_dim, image_encoder_channels, context_images,
                 image_tokens, ff_dim, num_heads, depth=4, dropout=0.):
        """image_encoder_channels: channels of ONE tokenizer latent; q_sem and q_rec come concatenated."""
        super().__init__()
        D = self.hidden_dim = hidden_dim
        self.context_images = context_images

        self.image_proj = nn.Linear(2 * image_encoder_channels, D)
        self.spatial_block = SpatialBlock(D, ff_dim, image_tokens, num_heads, dropout)
        # The pooling is unconditioned, so this marks which frame a token comes from.
        self.frame_emb = nn.Parameter(torch.zeros(context_images, D))
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

    def forward(self, encoded_q, z_t, t):
        """
        Args:
            encoded_q: (B, T_img, 2C, H, W) image latents; only the first context_images are used.
            z_t:       (B, latent_dim) noisy latent.
            t:         (B,) diffusion time in [0, 1].
        Returns:
            (B, latent_dim) predicted flow.
        """
        B, D = z_t.shape[0], self.hidden_dim
        encoded_q = encoded_q[:, :self.context_images]
        T_img = encoded_q.shape[1]

        x = self.image_proj(encoded_q.permute(0, 1, 3, 4, 2))              # (B, T_img, H, W, D)
        x = x.flatten(0, 1).permute(0, 3, 1, 2)                            # (B*T_img, D, H, W)
        image_tokens = self.spatial_block(x).reshape(B, T_img, -1, D) + self.frame_emb[:T_img, None]
        image_tokens = image_tokens.flatten(1, 2)                          # (B, T_img * N, D)

        # DiT's timestep embedding expects t on the 0..1000 scale.
        time = self.t_embedder(t * 1000)                                   # (B, D)
        query = self.query_mlp(torch.cat([z_t, time], dim=-1))[:, None]    # (B, 1, D)
        modulation = self.adaLN(time)
        for block in self.blocks:
            query = block(torch.cat([image_tokens, query], dim=1), query, modulation)
        return self.final_layer(query, time)[:, 0]
