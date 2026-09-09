"""Trajectory denoiser network (transformer), separated from flow matching.

This file holds only the neural network that maps a noisy trajectory (plus image
context and the diffusion time) to a predicted flow/velocity field. The
flow-matching schedule, noising, sampling and training loop live in
exp_navsim/model.py, so the architecture can be read and edited on its own.

The episode length T is not fixed: the first `context_traj` frames are the given
trajectory context, and every remaining frame (T - context_traj) is predicted.
`context_images` (independent) sets how many leading frames provide images. The
network reads T from the input, so it works for any episode length.

Only the `context_images` observed frames are ever encoded, so the latents come
in as (B, T_img, C, H, W) with T_img <= context_images — the dataset reads no more
than that from disk.

Architecture:
    image latents (B, T_img, C, H, W) --ImageAggregator--> (B, T_img, N, D) tokens
        (N = image_tokens per frame, with a 1D position embedding over N)
    noisy deltas (B, T, 2) + diffusion-time emb + frame emb
        --> context tokens (first `context_traj` frames)  +  pred tokens (rest)
    stack of TemporalBlock cross-attention denoisers, where the pred tokens
    attend to [image tokens, context tokens, pred tokens]
        --> (B, T - context_traj, 2) predicted velocity field
"""

import torch
import torch.nn as nn

from networks.pos_emb import PositionEmbedding1d
from networks.blocks import TemporalBlock, SpatialBlock


class ImageAggregator(nn.Module):
    """Aggregate the encoded image latent of a frame into `num_tokens` tokens."""

    def __init__(self, image_encoder_channels, hidden_dim, context, num_tokens):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context = context
        self.num_tokens = num_tokens
        # Project the encoder channel dim to the model hidden dim.
        self.image_enc_mlp = nn.Linear(image_encoder_channels, hidden_dim)
        # Spatially pool the (H, W) latent into `num_tokens` tokens, conditioned
        # on the frame.
        self.spatial_block = SpatialBlock(
            embed_dim=hidden_dim,
            ff_dim=hidden_dim * 2,
            num_predictions=num_tokens,
        )
        # Distinguishes the N tokens of one frame from each other.
        self.token_pos_emb = PositionEmbedding1d(hidden_dim)

    def forward(self, encoded_q, frame_index):
        """
        Args:
            encoded_q:   (B, T_img, C, H, W) per-frame image latents. The dataset
                         reads only the image-context frames from disk, so T_img
                         is normally `context` and never has to reach T; anything
                         beyond the context is dropped here.
            frame_index: (B, T, D) frame-position embedding.
        Returns:
            (B, T_img, N, D) image tokens, N = num_tokens.
        """
        B, T, D = frame_index.shape
        N = self.num_tokens
        # Only the leading `context` frames provide image information.
        T_img = min(encoded_q.shape[1], self.context)
        encoded_q = encoded_q[:, :T_img]
        enc_dim, H, W = encoded_q.shape[2:]

        encoded_batched = encoded_q.reshape(B * T_img, enc_dim, H, W)
        enc = encoded_batched.permute(0, 2, 3, 1)          # (B*T_img, H, W, C)
        enc = self.image_enc_mlp(enc)                      # (B*T_img, H, W, D)
        encoded_batched = enc.permute(0, 3, 1, 2)          # (B*T_img, D, H, W)

        # Frame embedding of the observed frames conditions the spatial pooling.
        condition = frame_index[:, :T_img].reshape(B * T_img, D)
        tokens = self.spatial_block(encoded_batched, condition)     # (B*T_img, N, D)
        tokens = tokens.reshape(B, T_img, N, D)

        # Position embedding over the N tokens of a frame.
        token_index = torch.arange(0, N, dtype=tokens.dtype, device=tokens.device)
        token_emb = self.token_pos_emb(token_index).to(tokens.dtype)   # (N, D)
        return tokens + token_emb.view(1, 1, N, D)


class TrajectoryDenoiser(nn.Module):
    """Predicts the flow/velocity field that denoises a whole trajectory."""

    def __init__(self, hidden_dim, image_encoder_channels, context_traj, context_images,
                 image_tokens):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_traj = context_traj      # known-trajectory frames = context
        self.context_images = context_images  # image-observed frames (independent)
        self.image_tokens = image_tokens      # tokens produced per observed frame
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

        # Image latents -> `image_tokens` tokens per observed frame.
        self.image_aggregator = ImageAggregator(
            image_encoder_channels=image_encoder_channels,
            hidden_dim=D,
            context=context_images,
            num_tokens=image_tokens,
        )

        # Per-point embedding: [noisy dx, dy] (2) + time emb (D) + frame emb (D).
        # The image tokens are not concatenated per point: they enter the
        # attention stack as tokens of their own (see forward).
        self.context_mlp = nn.Sequential(
            nn.Linear(2 + D * 2, D),
            nn.GELU(),
            nn.Linear(D, D),
        )
        self.pred_mlp = nn.Sequential(
            nn.Linear(2 + D * 2, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # Cross-attention denoiser stack: the predicted tokens attend to the full
        # (image + context + pred) sequence at every layer.
        self.temporal_stack = nn.ModuleList([
            TemporalBlock(D, D * 2),
            TemporalBlock(D, D * 2),
            # TemporalBlock(D, D * 2),
            # TemporalBlock(D, D * 2),

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
            encoded_q:    (B, T_img, C, H, W) image latents of the observed frames.
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

        # Image tokens: N per observed frame, flattened into one token sequence
        # (empty when there is no image context).
        if self.context_images > 0:
            image_tokens = self.image_aggregator(encoded_q, frame_index)   # (B, T_img, N, D)
            image_tokens = image_tokens.reshape(B, -1, D)                  # (B, T_img * N, D)
        else:
            image_tokens = torch.zeros((B, 0, D), dtype=noisy_deltas.dtype, device=device)

        # Seed the trajectory context with ground-truth deltas, if requested.
        deltas = noisy_deltas.clone()
        if self.context_traj > 0:
            deltas[:, :self.context_traj] = true_deltas[:, :self.context_traj]

        # Assemble the per-point feature vector.
        x = torch.cat([deltas, flow_time_expand, frame_index], dim=-1)
        assert x.shape == (B, T, 2 + D * 2)

        context = self.context_mlp(x[:, :context_length])          # (B, context_length, D)
        pred = self.pred_mlp(x[:, context_length:])                # (B, T - context_length, D)

        # The predicted tokens attend to the images, the trajectory context and
        # themselves at every layer.
        for transformer_block in self.temporal_stack:
            inp = torch.cat([image_tokens, context, pred], dim=1)
            pred = transformer_block(inp, pred, flow_time)

        pred_flow = self.output_mlp(self.output_norm(pred))        # (B, T - context_length, 2)
        assert pred_flow.shape == (B, T - context_length, 2)
        return pred_flow
