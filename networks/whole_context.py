"""
Whole trajectory prediction model using flow matching diffusion.

Predicts the whole trajectory position deltas (velocity) with diffusion:
1. Takes turn parameter as context
2. Uses flow matching to denoise and predict the whole trajectory deltas
"""

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .pos_emb import PositionEmbedding1d
from .blocks import TemporalBlock, SpatialBlock
from .norm import TrajectoryNorm


class ImageAggregator(nn.Module):
    """Frame-wise latent aggregation from encoded image features."""

    def __init__(self, image_encoder_channels, hidden_dim, context_images):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_images = context_images
        self.image_enc_mlp = nn.Linear(image_encoder_channels, hidden_dim)
        self.spatial_block = SpatialBlock(
            embed_dim=hidden_dim,
            ff_dim=hidden_dim * 2,
            num_predictions=1,
        )

    def forward(self, encoded_q, frame_index):
        """
        Args:
            encoded_q: (B, T, C, H, W)
            frame_index: (B, T, D)

        Returns:
            output: (B, T, D)
        """
        B, T, enc_dim, H, W = encoded_q.shape
        D = self.hidden_dim

        encoded_batched = encoded_q.reshape(B * T, enc_dim, H, W)
        enc = encoded_batched.permute(0, 2, 3, 1)
        enc = self.image_enc_mlp(enc)
        encoded_batched = enc.permute(0, 3, 1, 2)

        condition = frame_index.reshape(B * T, D)
        output = self.spatial_block(encoded_batched, condition)
        output = output.squeeze(1).reshape(B, T, D)

        output[:, self.context_images:] = 0.0
        return output


class DiffusionModel(pl.LightningModule):
    """
    Minimal flow matching model for whole trajectory delta/velocity prediction.
    
    Architecture:
        turn: (B,) - turn parameter [0, 1] (global conditioning)
        noisy_deltas: (B, T, 2) - noisy trajectory deltas (velocity)
        t: (B,) - diffusion timestep
            ↓ Process each point with global turn conditioning
        For each trajectory point i:
            input: (B, 2 + D + 1) - [noisy_dx_i, noisy_dy_i, t_emb, turn]
            ↓ FC Module with residual blocks
        predicted_flow: (B, T, 2) - predicted velocity field for trajectory deltas
    """
    
    def __init__(
        self,
        hidden_dim,         # Hidden dimension for FC
        image_encoder_channels,
        sigma_min,        # Minimum noise level
        warmup_steps,     # Warmup steps
        min_lr_multiplier, # Minimum LR multiplier
        num_frames,
        context_traj,
        context_images,
        pred_steps,           # number of predicted time steps
        encode_images=False,  # True: encode raw batch["images"] on-the-fly with the frozen
                              #       tokenizer. False: expect precomputed batch["encoded_q_sem"].
        tokenizer_exp_dir="logs_tk/tokenizer_288x512",  # frozen image tokenizer
        tokenizer_ckpt="checkpoints/last.ckpt",
        tokenizer_config="config.yaml",
    ):
        super().__init__()
        self.save_hyperparameters()

        # On-the-fly image encoding. The tokenizer is created lazily (on first use)
        # so it lands on the correct device, and only when encode_images is set.
        self.encode_images = encode_images
        self.tokenizer_exp_dir = tokenizer_exp_dir
        self.tokenizer_ckpt = tokenizer_ckpt
        self.tokenizer_config = tokenizer_config
        self._encoder = None

        self.sigma_min = sigma_min
        self.warmup_steps = warmup_steps
        self.min_lr_multiplier = min_lr_multiplier
        self.num_frames = num_frames
        
        self.context_traj = context_traj
        self.context_images = context_images
        self.pred_steps = pred_steps
        
        self.hidden_dim = hidden_dim
        self.image_encoder_channels = image_encoder_channels
        D = hidden_dim
        
        # Encode diffusion time
        self.fm_time_mlp = nn.Sequential(
            PositionEmbedding1d(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D)
        )
        
        # Encode frame number
        self.frame_mlp = nn.Sequential(
            PositionEmbedding1d(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D)
        )
        
        self.image_aggregator = ImageAggregator(
            image_encoder_channels=self.image_encoder_channels,
            hidden_dim=D,
            context_images=self.context_images,
        )
        
        # FC module for denoising whole trajectory
        self.context_mlp = nn.Sequential(
            nn.Linear(2 + D * 3, D),  # noisy_point (2), diffusion time + frame (D)
            nn.GELU(),
            nn.Linear(D, D),
        )
        
        self.pred_mlp = nn.Sequential(
            nn.Linear(2 + D * 3, D),  # noisy_point (2), diffusion time + frame (D)
            nn.GELU(),
            nn.Linear(D, D),
        )
        
        self.temporal_stack = nn.ModuleList([
            TemporalBlock(D, D*2, self.pred_steps),
            TemporalBlock(D, D*2, self.pred_steps),
            TemporalBlock(D, D*2, self.pred_steps),
            TemporalBlock(D, D*2, self.pred_steps),
            
            TemporalBlock(D, D*2, self.pred_steps),
            TemporalBlock(D, D*2, self.pred_steps),
            TemporalBlock(D, D*2, self.pred_steps),
            TemporalBlock(D, D*2, self.pred_steps),
        ])
        
        self.output_norm = nn.LayerNorm(D)
        self.output_mlp = nn.Linear(D, 2)

        # Online (EMA) velocity normalization, replacing the precomputed
        # dataset normalization stats. Tracked in normalized (dx, dy) space.
        self.traj_norm = TrajectoryNorm()
        
    @property
    def encoder(self):
        """Lazily-constructed frozen image tokenizer (only when encode_images)."""
        if self._encoder is None:
            from loaders_for_projects.encoder import Encoder
            self._encoder = Encoder(
                exp_dir=self.tokenizer_exp_dir,
                ckpt=self.tokenizer_ckpt,
                config=self.tokenizer_config,
                device=self.device,
            )
        return self._encoder

    def get_encoded_q(self, batch):
        """Return semantic latents (B, T, C, H, W).

        If encode_images is set, encode raw batch["images"] with the frozen
        tokenizer; otherwise use precomputed batch["encoded_q_sem"].
        """
        if not self.encode_images:
            return batch["encoded_q_sem"]

        images = batch["images"]  # (B, T, C, H, W)
        B, T = images.shape[:2]
        flat = images.reshape(B * T, *images.shape[2:]).to(self.encoder.device)
        _, q_sem = self.encoder.encode(flat)
        q_sem = q_sem.reshape(B, T, *q_sem.shape[1:])
        # clone() to leave inference-mode; detached since the tokenizer is frozen
        return q_sem.detach().clone().to(device=images.device, dtype=torch.float32)

    def alpha(self, t):
        """Flow matching alpha schedule."""
        return 1.0 - t
    
    def sigma(self, t):
        """Flow matching sigma schedule."""
        return self.sigma_min + t * (1.0 - self.sigma_min)
    
    def A(self, t):
        """Flow matching A coefficient."""
        return 1.0
    
    def B(self, t):
        """Flow matching B coefficient."""
        return -(1.0 - self.sigma_min)

    def add_noise(self, x, t, noise=None):
        """
        Add noise according to flow matching schedule.
        
        Args:
            x: (B, ...) - target data
            t: (B,) - diffusion timesteps
            noise: (B, ...) - optional noise tensor
        """
        noise = torch.randn_like(x) if noise is None else noise
        s = [x.shape[0]] + [1] * (x.dim() - 1)
        x_t = self.alpha(t).view(*s) * x + self.sigma(t).view(*s) * noise
        return x_t, noise
    
    def forward(self, turn, encoded_q, true_deltas, noisy_deltas, t):
        """
        Predict flow/velocity field for denoising whole trajectory deltas.
        
        Args:
            turn: (B,) turn parameter [0, 1]
            encoded_q: (B, T, C, H, W)
            true_deltas: (B, T, 2)
            noisy_deltas: (B, T, 2) noisy trajectory deltas (velocity)
            t: (B,) diffusion timestep
        
        Returns:
            pred_flow: (B, T, 2) predicted velocity field for trajectory deltas
        """
        B, T = noisy_deltas.shape[:2]
        D = self.hidden_dim
        device = noisy_deltas.device
        context_length = T - self.pred_steps
        
        # Flow matching time embedding
        flow_time = self.fm_time_mlp(t)  # (B, D)
        flow_time_expand = flow_time.view(B, 1, -1).expand(B, T, -1)  # (B, T, D)
        
        # Frame index embedding
        frame_index = torch.arange(0, T, dtype=noisy_deltas.dtype, device=device) / T
        frame_index = self.frame_mlp(frame_index)  # (T, D)
        frame_index = frame_index.view(1, T, -1).expand(B, T, -1)  # (B, T, D)
        
        # === Image input
        if self.context_images > 0:
            encoded_agg = self.image_aggregator(encoded_q, frame_index)
        else:
            encoded_agg = torch.zeros((B, T, D), device=device)
        
        # Set trajectory context
        deltas = noisy_deltas.clone()
        if self.context_traj > 0:
            deltas[:, :self.context_traj] = true_deltas[:, :self.context_traj]
        
        # Concatenate deltas, time embedding
        x = torch.cat([deltas, flow_time_expand, frame_index, encoded_agg], dim=-1)
        assert x.shape == (B, T, 2 + D*3)
        
        # Embed context (needed?)
        context = self.context_mlp(x[:, :context_length])
        assert context.shape == (B, context_length, D)
        
        # Embed noised deltas (to be denoised)
        pred = self.pred_mlp(x[:, context_length:])
        assert pred.shape == (B, self.pred_steps, D)
        
        # Cross-attention denoiser
        for transformer_block in self.temporal_stack:
            inp = torch.cat([context, pred], dim=1)
            pred = transformer_block(inp, pred, flow_time)
        pred_flow = pred  # (B, T_pred, hidden_dim)
        
        # hidden_dim -> (x, y)
        pred_flow = self.output_mlp(self.output_norm(pred_flow))
        
        expected_shape = (B, self.pred_steps, 2)
        assert pred_flow.shape == expected_shape, f"Expected {expected_shape}, got {pred_flow.shape}"
        return pred_flow

    def compute_loss(self, batch):
        """
        Compute flow matching loss for whole trajectory delta prediction.
        
        Diffuses the entire trajectory deltas at once instead of autoregressive training.
        """
        deltas = self.traj_norm.normalize(batch["velocity"])  # (B, T, 2) - normalized ground truth deltas
        encoded_q = self.get_encoded_q(batch)  # (B, T, C, H, W)

        B, T = deltas.shape[:2]
        turn = batch["turn"] if "turn" in batch else torch.zeros(B, device=deltas.device)  # (B,) - turn parameter, fake if absent
        context_length = T - self.pred_steps
        
        # Sample diffusion timestep from [0, 1) for the whole trajectory
        t = torch.rand(B, device=deltas.device)  # (B,)
        
        # Add noise to whole trajectory deltas
        noisy_deltas, noise = self.add_noise(deltas, t)  # (B, T, 2)
        
        # Predict velocity field for the whole trajectory
        pred_v = self.forward(turn, encoded_q, deltas, noisy_deltas, t)  # (B, T_pred, 2)
        
        # Compute target for flow matching: v = A(t) * deltas + B(t) * noise
        t_expanded = t.view(B, 1, 1)  # (B, 1, 1)
        target = self.A(t_expanded) * deltas + self.B(t_expanded) * noise  # (B, T, 2)
        target = target[:, context_length:, :]  # Remove context
        
        # MSE loss over predicted trajectory points
        loss = ((pred_v - target) ** 2).mean()
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step with flow matching loss."""
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and LR scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Cosine annealing with warmup
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / (self.trainer.max_steps - self.warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return (1 - self.min_lr_multiplier) * cosine_decay + self.min_lr_multiplier
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    @torch.no_grad()
    def sample(self, batch, num_diffusion_steps, dataset=None, denormalize=False):
        """
        Sample whole trajectory deltas using flow matching diffusion and reconstruct positions.
        
        Args:
            batch: dict - batch size is inferred from batch
            num_diffusion_steps: number of diffusion steps
            dataset: SyntheticTrajectoryDataset for denormalization
        
        Returns:
            trajectory: (B, T, 2) predicted trajectory
        """
        self.eval()
        
        # Get batch size, turn parameter, and device
        gt_trajectory = batch["trajectory"]  # (B, T, 2)
        deltas = batch["velocity"]
        deltas = self.traj_norm.normalize(deltas)  # work in normalized space
        encoded_q = self.get_encoded_q(batch)

        B = deltas.shape[0]
        device = deltas.device
        turn = batch["turn"] if "turn" in batch else torch.zeros(B, device=device)  # (B,), fake if absent
        T = deltas.shape[1]  # number of trajectory timesteps to generate
        context_length = T - self.pred_steps
        assert context_length + self.pred_steps == T
        
        sampled_deltas = torch.randn(B, T, 2, device=device) # Start from pure noise for the whole trajectory deltas at t=1
        sampled_deltas[:, :context_length] = deltas[:, :context_length]

        t_steps = torch.linspace(1, 0, num_diffusion_steps + 1, device=device)  # Diffusion timesteps from t=1 to t=0
        
        # Denoise whole trajectory deltas via flow matching ODE
        for i in range(num_diffusion_steps):
            t = t_steps[i].repeat(B)  # (B,)
            
            # Predict velocity field for whole trajectory deltas
            pred_v = self.forward(turn, encoded_q, deltas, sampled_deltas, t)  # (B, T_pred, 2)
            assert pred_v.shape[1] == self.pred_steps
            
            # Update: integrate ODE backward from t=1 to t=0
            dt = t_steps[i] - t_steps[i + 1]
            sampled_deltas[:, context_length:] = sampled_deltas[:, context_length:] + pred_v * dt
        
        if denormalize:
            # Map back from normalized (dx, dy) space to raw velocities.
            sampled_deltas = self.traj_norm.denormalize(sampled_deltas)

        # Reconstruct trajectory from deltas (start from 0)
        trajectory = torch.cumsum(sampled_deltas, dim=1)  # v_t = x_t - x_{t-1}
        trajectory = trajectory - trajectory[:, :1, :]
        
        return trajectory
