"""Trajectory denoiser network (transformer), separated from flow matching.

This file holds only the neural network that maps a noisy trajectory (plus image
context and the diffusion time) to a predicted flow/velocity field. The
flow-matching schedule, noising, sampling and training loop live in
exp_navsim/model.py, so the architecture can be read and edited on its own.

The episode length T is not fixed: the first `context_traj` frames are the given
trajectory context, and every remaining frame (T - context_traj) is predicted.
`context_images` (independent) sets how many leading frames provide images. The
network reads T from the input, so it works for any episode length.

Architecture:
    image latents (B, T, C, H, W) --ImageAggregator--> per-frame tokens (B, T, D)
    noisy deltas (B, T, 2) + diffusion-time emb + frame emb + image tokens
        --> context tokens (first `context_traj` frames)  +  pred tokens (rest)
        --> stack of TemporalBlock cross-attention denoisers
        --> (B, T - context_traj, 2) predicted velocity field
"""

import torch
import torch.nn as nn

from networks.pos_emb import PositionEmbedding1d
from networks.blocks import TemporalBlock, SpatialBlock


class ImageAggregator(nn.Module):
    """Aggregate one encoded image latent per frame into a single token."""

    def __init__(self, image_encoder_channels, hidden_dim, context):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context = context
        # Project the encoder channel dim to the model hidden dim.
        self.image_enc_mlp = nn.Linear(image_encoder_channels, hidden_dim)
        # Spatially pool the (H, W) latent into one token, conditioned on the frame.
        self.spatial_block = SpatialBlock(
            embed_dim=hidden_dim,
            ff_dim=hidden_dim * 2,
            num_predictions=1,
        )

    def forward(self, encoded_q, frame_index):
        """
        Args:
            encoded_q:   (B, T, C, H, W) per-frame image latents.
            frame_index: (B, T, D) frame-position embedding.
        Returns:
            (B, T, D) per-frame image tokens; frames beyond the image context are
            zeroed so the model only sees images for the context frames.
        """
        B, T, enc_dim, H, W = encoded_q.shape
        D = self.hidden_dim

        encoded_batched = encoded_q.reshape(B * T, enc_dim, H, W)
        enc = encoded_batched.permute(0, 2, 3, 1)          # (B*T, H, W, C)
        enc = self.image_enc_mlp(enc)                      # (B*T, H, W, D)
        encoded_batched = enc.permute(0, 3, 1, 2)          # (B*T, D, H, W)

        condition = frame_index.reshape(B * T, D)
        output = self.spatial_block(encoded_batched, condition)
        output = output.squeeze(1).reshape(B, T, D)

        # Only the context frames provide image information.
        output[:, self.context:] = 0.0
        return output


class TrajectoryDenoiser(nn.Module):
    """Predicts the flow/velocity field that denoises a whole trajectory."""

    def __init__(self, hidden_dim, image_encoder_channels, context_traj, context_images):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_traj = context_traj      # known-trajectory frames = context
        self.context_images = context_images  # image-observed frames (independent)
        D = hidden_dim

        # Embedding of the diffusion timestep t.
        self.fm_time_mlp = nn.Sequential(
            PositionEmbedding1d(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # Embedding of the (normalized) frame index.
        self.frame_mlp = nn.Sequential(
            PositionEmbedding1d(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # Image latents -> per-frame tokens.
        self.image_aggregator = ImageAggregator(
            image_encoder_channels=image_encoder_channels,
            hidden_dim=D,
            context=context_images,
        )

        # Per-point embedding: [noisy dx, dy] (2) + time emb (D) + frame emb (D) + image token (D).
        self.context_mlp = nn.Sequential(
            nn.Linear(2 + D * 3, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.pred_mlp = nn.Sequential(
            nn.Linear(2 + D * 3, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # Cross-attention denoiser stack: the predicted tokens attend to the full
        # (context + pred) sequence at every layer.
        self.temporal_stack = nn.ModuleList([
            TemporalBlock(D, D * 2),
            TemporalBlock(D, D * 2),
            TemporalBlock(D, D * 2),
            TemporalBlock(D, D * 2),

            # TemporalBlock(D, D * 2),
            # TemporalBlock(D, D * 2),
            # TemporalBlock(D, D * 2),
            # TemporalBlock(D, D * 2),
        ])

        self.output_norm = nn.LayerNorm(D)
        self.output_mlp = nn.Linear(D, 2)

    def forward(self, encoded_q, true_deltas, noisy_deltas, t):
        """
        Args:
            encoded_q:    (B, T, C, H, W) image latents.
            true_deltas:  (B, T, 2) ground-truth velocity (used to fill context).
            noisy_deltas: (B, T, 2) noisy velocity to be denoised.
            t:            (B,) diffusion timestep.
        Returns:
            (B, T - context_traj, 2) predicted velocity field for the predicted frames.
        """
        B, T = noisy_deltas.shape[:2]
        D = self.hidden_dim
        device = noisy_deltas.device
        # Everything after the known-trajectory context is predicted; T is free.
        context_length = self.context_traj

        # Diffusion-time embedding, broadcast to every frame.
        flow_time = self.fm_time_mlp(t)                            # (B, D)
        flow_time_expand = flow_time.view(B, 1, -1).expand(B, T, -1)

        # Frame-position embedding.
        frame_index = torch.arange(0, T, dtype=noisy_deltas.dtype, device=device) / T
        frame_index = self.frame_mlp(frame_index)                 # (T, D)
        frame_index = frame_index.view(1, T, -1).expand(B, T, -1)

        # Per-frame image tokens (zero when there is no image context).
        if self.context_images > 0:
            encoded_agg = self.image_aggregator(encoded_q, frame_index)
        else:
            encoded_agg = torch.zeros((B, T, D), device=device)

        # Seed the trajectory context with ground-truth deltas, if requested.
        deltas = noisy_deltas.clone()
        if self.context_traj > 0:
            deltas[:, :self.context_traj] = true_deltas[:, :self.context_traj]

        # Assemble the per-point feature vector.
        x = torch.cat([deltas, flow_time_expand, frame_index, encoded_agg], dim=-1)
        assert x.shape == (B, T, 2 + D * 3)

        context = self.context_mlp(x[:, :context_length])          # (B, context_length, D)
        pred = self.pred_mlp(x[:, context_length:])                # (B, T - context_length, D)

        for transformer_block in self.temporal_stack:
            inp = torch.cat([context, pred], dim=1)
            pred = transformer_block(inp, pred, flow_time)

        pred_flow = self.output_mlp(self.output_norm(pred))        # (B, T - context_length, 2)
        assert pred_flow.shape == (B, T - context_length, 2)
        return pred_flow
