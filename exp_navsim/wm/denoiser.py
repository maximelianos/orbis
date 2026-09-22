"""Trajectory denoiser conditioned on world-model latents.

Training command (repo root):
    PYTHONPATH=. python train_nuplan.py -c exp_navsim/wm/config.yaml --max_steps 20000 --logdir logs_navsim_wm

Network only — the flow-matching schedule, noising and sampling stay in
exp_navsim/model.py, and the world model that produces the latents stays in
exp_navsim/wm/wm_runner.py. This file sees latents and noisy trajectories.

Architecture
------------
    latents (B, F, C, H, W)           F = observed frames + imagined frames,
        |                                 C = the world model's latent channels
        |  LatentAggregator (ONE set of weights for every frame)
        v
    latent tokens (B, F*N, D)
                                       noisy velocities (B, T, 2)
                                          + diffusion time + frame index
                                          |  one shared MLP + a role vector
                                          v
                                       trajectory tokens (B, T, D)
        \______________________________ ___/
                                       v
    num_layers applications of num_unique_layers TemporalBlocks, in which the
    predicted tokens cross-attend to [latent tokens, context tokens, predicted]
        |
        v
    (B, T - context_traj, 2) velocity field

Why it is built this way
------------------------
Fewer weights:
  * The aggregator is shared by every frame; frames are told apart by a single
    learned vector each, so widening the world-model context costs D floats,
    not another aggregator.
  * One MLP embeds both the known-context and the predicted trajectory frames;
    a two-row `role` embedding replaces the second MLP.
  * Depth is decoupled from parameters: `num_layers` applications of
    `num_unique_layers` distinct TemporalBlocks (ALBERT-style weight tying), so
    a 4-deep model can carry 2 blocks' worth of weights.
  * Nothing in the model scales with the episode length T — the frame index
    enters through a closed-form sinusoid, so any T works and none can be
    memorised as a parameter.

Less overfitting:
  * Weight tying is itself a strong regulariser (one block must work at several
    depths).
  * Dropout inside every attention block.
  * `cond_dropout` blanks a sample's whole latent conditioning during training,
    so the head cannot collapse onto scene appearance and ignore the diffusion
    prior — the same trick as classifier-free guidance dropout.
  * The output layer is zero-initialised, so training starts from a zero
    velocity field instead of a random one.

Speed note: the aggregator's AdaLN condition is the frame's own slot vector,
*not* the diffusion time. The latent tokens are therefore constant along the
sampling ODE and `encode_latents` is called once per sampled trajectory rather
than once per diffusion step (see exp_navsim/wm/model.py).
"""

import torch
import torch.nn as nn

from networks.blocks import TemporalBlock, SpatialBlock
from exp_navsim.embeddings import sincos_embedding, FrameIndexEmbedding


class LatentAggregator(nn.Module):
    """One latent grid -> a few tokens, with one set of weights for all frames."""

    def __init__(self, latent_channels, hidden_dim, num_tokens, num_frames, dropout):
        """
        Args:
            latent_channels: channels of the world model's latent space.
            hidden_dim:      model width D.
            num_tokens:      tokens N extracted per frame. This is the width of
                             the whole conditioning path, so it trades capacity
                             against overfitting more directly than D does.
            num_frames:      how many distinct frame slots exist (observed +
                             imagined). Sets the size of the slot embedding and
                             is the only frame-count-dependent parameter.
            dropout:         dropout inside the spatial attention block.
        """
        super().__init__()
        self.num_tokens = num_tokens
        # Single linear from latent channels to model width, applied per grid cell.
        self.proj = nn.Linear(latent_channels, hidden_dim)
        # Cross-attention pooling of the (H, W) grid into N tokens. Its N query
        # vectors already carry a 1D position embedding, so the output tokens of
        # one frame are distinguishable without any extra embedding here.
        self.spatial = SpatialBlock(
            embed_dim=hidden_dim,
            ff_dim=hidden_dim * 2,
            num_predictions=num_tokens,
            dropout=dropout,
        )
        # Identity of a frame: "second observed frame", "imagined frame", ... It
        # both conditions the pooling (AdaLN) and marks the resulting tokens.
        self.slot_emb = nn.Parameter(torch.zeros(num_frames, hidden_dim))
        nn.init.trunc_normal_(self.slot_emb, std=0.02)

    def forward(self, latents):
        """
        Args:
            latents: (B, F, C, H, W), F <= num_frames.
        Returns:
            (B, F*N, D) conditioning tokens, frames concatenated in order.
        """
        B, F, C, H, W = latents.shape
        assert F <= self.slot_emb.shape[0], f"{F} frames > {self.slot_emb.shape[0]} slots"

        # (B, F, C, H, W) -> (B*F, D, H, W): project channels, keep the grid.
        x = self.proj(latents.permute(0, 1, 3, 4, 2))               # (B, F, H, W, D)
        x = x.permute(0, 1, 4, 2, 3).reshape(B * F, -1, H, W)       # (B*F, D, H, W)

        slots = self.slot_emb[:F]                                   # (F, D)
        condition = slots.unsqueeze(0).expand(B, F, -1).reshape(B * F, -1)
        tokens = self.spatial(x, condition)                         # (B*F, N, D)

        tokens = tokens.reshape(B, F, self.num_tokens, -1) + slots.view(1, F, 1, -1)
        return tokens.reshape(B, F * self.num_tokens, -1)


class WmTrajectoryDenoiser(nn.Module):
    """Flow field of a whole trajectory, conditioned on world-model latents."""

    def __init__(self, hidden_dim, latent_channels, latent_frames, latent_tokens,
                 context_traj, num_layers, num_unique_layers, dropout, cond_dropout,
                 max_frames=1000):
        """
        Args:
            hidden_dim:        model width D.
            latent_channels:   channels of the world model's latent space; must
                               match `WorldModelRunner.latent_channels`.
            latent_frames:     observed frames + imagined frames.
            latent_tokens:     tokens extracted per latent frame (N).
            context_traj:      leading frames whose velocity is given rather
                               than predicted (0 = predict the whole episode).
            num_layers:        how many cross-attention passes are applied.
            num_unique_layers: how many distinct blocks those passes cycle over.
                               Equal to num_layers = an ordinary stack; smaller
                               = weight tying, fewer parameters, stronger
                               regularisation.
            dropout:           dropout inside every attention block.
            cond_dropout:      probability of blanking a sample's entire latent
                               conditioning during training.
        """
        super().__init__()
        D = hidden_dim
        self.hidden_dim = D
        self.context_traj = context_traj
        self.num_layers = num_layers
        self.cond_dropout = cond_dropout

        self.aggregator = LatentAggregator(
            latent_channels=latent_channels,
            hidden_dim=D,
            num_tokens=latent_tokens,
            num_frames=latent_frames,
            dropout=dropout,
        )

        # One embedder for both scalars the trajectory path needs — diffusion
        # time and frame index. Both reach it as a sin-cos vector of the same
        # basis, so a second copy of these weights would buy nothing.
        self.scalar_mlp = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

        # Absolute frame index, tabulated once. Indexing by the absolute frame
        # number (not arange(T)/T) keeps the embedding of frame i independent of
        # the episode length, and so of which episodes padded the batch.
        self.frame_pos_emb = FrameIndexEmbedding(D, max_frames=max_frames)

        # Context frames and predicted frames share one MLP and are separated by
        # a two-row embedding instead of a second MLP.
        self.traj_mlp = nn.Sequential(nn.Linear(2 + D * 2, D), nn.GELU(), nn.Linear(D, D))
        self.role_emb = nn.Parameter(torch.zeros(2, D))
        nn.init.trunc_normal_(self.role_emb, std=0.02)

        # Weight-tied depth: num_layers passes over num_unique_layers blocks.
        self.blocks = nn.ModuleList([
            TemporalBlock(embed_dim=D, ff_dim=D * 2, dropout=dropout)
            for _ in range(num_unique_layers)
        ])

        self.output_norm = nn.LayerNorm(D)
        self.output_mlp = nn.Linear(D, 2)
        # Zero-init: the model starts by predicting a zero flow field, which is
        # the stable DiT convention and keeps early training from thrashing.
        nn.init.zeros_(self.output_mlp.weight)
        nn.init.zeros_(self.output_mlp.bias)

    # ------------------------------------------------------------------ #
    # Conditioning
    # ------------------------------------------------------------------ #
    def encode_latents(self, latents):
        """World-model latents (B, F, C, H, W) -> conditioning tokens (B, F*N, D).

        Kept separate from `forward` because it does not depend on the diffusion
        time: the caller computes it once and reuses it across the sampling ODE.
        """
        return self.aggregator(latents)

    def drop_conditioning(self, tokens):
        """Blank the whole conditioning of a random subset of samples (training).

        Regularisation, not augmentation: the trajectory head must stay able to
        produce a plausible trajectory from the diffusion prior alone, so it
        cannot reduce to "look up this scene's appearance". A no-op in eval.
        """
        if not self.training or self.cond_dropout <= 0.0:
            return tokens
        keep = torch.rand(tokens.shape[0], 1, 1, device=tokens.device) >= self.cond_dropout
        return tokens * keep.to(tokens.dtype)

    # ------------------------------------------------------------------ #
    # Trajectory tokens
    # ------------------------------------------------------------------ #
    def trajectory_tokens(self, true_deltas, noisy_deltas, time_emb):
        """One token per episode frame.

        Args:
            true_deltas:  (B, T, 2) ground-truth velocities; only the first
                          `context_traj` entries are read, to overwrite the
                          noise on the frames that are given rather than predicted.
            noisy_deltas: (B, T, 2) noisy velocities being denoised.
            time_emb:     (B, D) embedded diffusion timestep.
        Returns:
            (B, T, D)
        """
        B, T, _ = noisy_deltas.shape
        D = self.hidden_dim
        ctx = self.context_traj

        deltas = noisy_deltas.clone()
        if ctx > 0:
            deltas[:, :ctx] = true_deltas[:, :ctx]

        # Absolute frame index: closed-form, so no parameter depends on T.
        frame_emb = self.scalar_mlp(self.frame_pos_emb(T))               # (T, D)

        x = torch.cat([deltas,
                       time_emb.view(B, 1, D).expand(B, T, D),
                       frame_emb.view(1, T, D).expand(B, T, D)], dim=-1)
        tokens = self.traj_mlp(x)                                        # (B, T, D)

        # Two roles: given context frame vs predicted frame.
        role = torch.cat([self.role_emb[0].view(1, D).expand(ctx, D),
                          self.role_emb[1].view(1, D).expand(T - ctx, D)], dim=0)
        return tokens + role.view(1, T, D)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, latent_tokens, true_deltas, noisy_deltas, t, frame_mask=None):
        """
        Args:
            latent_tokens: (B, F*N, D) from `encode_latents`. Passed in rather
                           than computed here so the sampler pays for the
                           aggregation once instead of once per diffusion step.
            true_deltas:   (B, T, 2) ground-truth velocities.
            noisy_deltas:  (B, T, 2) noisy velocities.
            t:             (B,) diffusion timestep.
            frame_mask:    (B, T) bool, True where the frame is real. Batches are
                           padded to their longest episode, and a real frame must
                           not read a padded one. None attends to everything.
        Returns:
            (B, T - context_traj, 2) predicted velocity field.
        """
        ctx = self.context_traj
        time_emb = self.scalar_mlp(sincos_embedding(t, self.hidden_dim))   # (B, D)

        tokens = self.trajectory_tokens(true_deltas, noisy_deltas, time_emb)
        cond = self.drop_conditioning(latent_tokens)
        context, pred = tokens[:, :ctx], tokens[:, ctx:]

        # The latent tokens are always real (the world model produces every one
        # of them); only the trajectory half of the sequence can be padded.
        key_padding_mask = None
        if frame_mask is not None:
            latent_valid = torch.ones(cond.shape[:2], dtype=torch.bool,
                                      device=frame_mask.device)
            key_padding_mask = ~torch.cat([latent_valid, frame_mask], dim=1)

        # Every pass, the predicted tokens re-read the latents, the given
        # trajectory context, and their own current state.
        for i in range(self.num_layers):
            sequence = torch.cat([cond, context, pred], dim=1)
            pred = self.blocks[i % len(self.blocks)](sequence, pred, time_emb,
                                                     key_padding_mask=key_padding_mask)

        return self.output_mlp(self.output_norm(pred))
