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

# Add parent directory to path
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from networks.pos_emb import PositionEmbedding1d
from networks.blocks import TemporalBlock, SpatialBlock
from .cnn import CNN

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
        sigma_min,        # Minimum noise level
        warmup_steps,     # Warmup steps
        min_lr_multiplier, # Minimum LR multiplier
        n_diffusion_samples, # Number of diffusion time samples per trajectory point
        pred_steps,         # number of predicted time steps
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.sigma_min = sigma_min
        self.warmup_steps = warmup_steps
        self.min_lr_multiplier = min_lr_multiplier
        self.n_diffusion_samples = n_diffusion_samples
        self.pred_steps = pred_steps
        
        self.hidden_dim = hidden_dim
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
        
        # Image encoder
        self.image_encoder = CNN(3, D)
        
        # FC module for denoising whole trajectory
        self.mlp = nn.Sequential(
            nn.Linear(2 + D * 3 + 1, D),  # noisy_point (2), diffusion time + frame (D), turn (1)
            nn.GELU(),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D),  # output per-point features
        )
        
        self.temporal_denoiser = TemporalBlock(
            embed_dim=D,
            ff_dim=D*2,
            num_predictions=self.pred_steps
        )
        self.spatial_block = SpatialBlock(
            embed_dim=D,
            ff_dim=D*2,
            num_predictions=1
        )
        self.output_mlp = nn.Linear(D, 2)
        
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
    
    def forward(self, turn, images, noisy_deltas, t):
        """
        Predict flow/velocity field for denoising whole trajectory deltas.
        
        Args:
            turn: (B,) turn parameter [0, 1]
            images: (B, H, W, 3)
            noisy_deltas: (B, T, 2) noisy trajectory deltas (velocity)
            t: (B,) diffusion timestep
        
        Returns:
            pred_flow: (B, T, 2) predicted velocity field for trajectory deltas
        """
        B, T = noisy_deltas.shape[:2]
        D = self.hidden_dim
        device = noisy_deltas.device
        
        
        
        # Flow matching time embedding
        flow_time = self.fm_time_mlp(t)  # (B, D)
        flow_time = flow_time.view(B, 1, -1).expand(B, T, -1)  # (B, T, D)
        
        
        
        # Normalized frame index embedding
        frames = torch.arange(0, T, dtype=turn.dtype, device=device) / T
        frames_emb = self.frame_mlp(frames)  # (T, D)
        frames_emb_expanded = frames_emb.view(1, T, -1).expand(B, T, -1)  # (B, T, D)
        
        # Turn embedding
        turn = torch.zeros_like(turn)  # TODO remove conditioning
        turn_expanded = turn.view(B, 1, 1).expand(B, T, 1)  # (B, T, 1) - same turn for all time points
        
        # ==== Image encoder
        # permute images
        B, H, W, _ = images.shape
        images = images.permute((0, 3, 1, 2)) # (B, 3, H, W)
        
        images_emb = self.image_encoder(images)  # (B, hidden_dim, H, W)
        _, _, H, W = images_emb.shape
        
        # Spatial block
        condition = torch.zeros((B, D), device=device)
        
        # query shape = (B*T, D)
        images_emb_expanded = images_emb.view(B, 1, D, H, W).expand(B, T, D, H, W).reshape(B*T, D, H, W)
        condition = frames_emb_expanded.reshape(B*T, D)
        
        #images_vec = self.spatial_block(images_emb, )  # (B, 1, D)
        #images_vec = images_vec.expand(B, T, D)
        
        images_vec = self.spatial_block(images_emb_expanded, condition).squeeze() # (B*T, D)
        images_vec = images_vec.reshape(B, T, D)
        
        images_vec = torch.zeros((B, T, D), device=device)
        
        # Concatenate noisy trajectory deltas, time embedding, and global turn
        x = torch.cat([noisy_deltas, flow_time, frames_emb_expanded, images_vec, turn_expanded], dim=-1)
        x = self.mlp(x)
        assert x.shape == (B, T, D)
        
        # Cross-attention cross denoiser
        pred_flow = self.temporal_denoiser(x, flow_time[:, 0, :])  # (B, T, hidden_dim)
        
        # hidden_dim -> (x, y)
        pred_flow = self.output_mlp(pred_flow)
        
        expected_shape = (B, T, 2)
        assert pred_flow.shape == expected_shape, f"Expected output shape {expected_shape}, got {pred_flow.shape}"
        
        return pred_flow

    def compute_loss(self, batch):
        """
        Compute flow matching loss for whole trajectory delta prediction.
        
        Diffuses the entire trajectory deltas at once instead of autoregressive training.
        """
        turn = batch["turn"]  # (B,) - turn parameter
        deltas = batch["velocity"]  # (B, T, 2) - ground truth trajectory deltas
        images = batch["images"]  # (B, H, W, 3)
        
        B, T = deltas.shape[:2]
        
        # Sample diffusion timestep from [0, 1) for the whole trajectory
        t = torch.rand(B, device=deltas.device)  # (B,)
        
        # Add noise to whole trajectory deltas
        noisy_deltas, noise = self.add_noise(deltas, t)  # (B, T, 2)
        
        # Predict velocity field for whole trajectory deltas
        pred_v = self.forward(turn, images, noisy_deltas, t)  # (B, T, 2)
        
        # Compute target for flow matching: v = A(t) * deltas + B(t) * noise
        t_expanded = t.view(B, 1, 1)  # (B, 1, 1)
        target = self.A(t_expanded) * deltas + self.B(t_expanded) * noise  # (B, T, 2)
        
        # MSE loss over all trajectory points
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
    def sample(self, batch, context_length, num_diffusion_steps, dataset=None, denormalize=False):
        """
        Sample whole trajectory deltas using flow matching diffusion and reconstruct positions.
        
        Args:
            batch: dict - batch size is inferred from batch
            context_length: int - number of initial positions to condition on
            num_diffusion_steps: number of diffusion steps
            dataset: SyntheticTrajectoryDataset for denormalization
        
        Returns:
            trajectory: (B, T, 2) predicted trajectory
        """
        self.eval()
        
        
        # Get batch size, turn parameter, and device
        turn = batch["turn"]  # (B,)
        gt_trajectory = batch["trajectory"]  # (B, T, 2)
        images = batch["images"]  # (B, H, W, 3)
        
        B = turn.shape[0]
        device = turn.device
        T = self.pred_steps  # number of trajectory timesteps to generate
        
        # Start from pure noise for the whole trajectory deltas at t=1
        sampled_deltas = torch.randn(B, T, 2, device=device)
        
        # Diffusion timesteps from t=1 to t=0
        t_steps = torch.linspace(1, 0, num_diffusion_steps + 1, device=device)
        
        # Denoise whole trajectory deltas via flow matching ODE
        for i in range(num_diffusion_steps):
            t = t_steps[i].repeat(B)  # (B,)
            
            # Predict velocity field for whole trajectory deltas
            pred_v = self.forward(turn, images, sampled_deltas, t)  # (B, T, 2)
            
            # Update: integrate ODE backward from t=1 to t=0
            dt = t_steps[i] - t_steps[i + 1]
            sampled_deltas = sampled_deltas + pred_v * dt
        
        # Denormalize deltas if dataset provided
        if denormalize:
            deltas_batch = {'velocity': sampled_deltas}
            deltas_batch = dataset.denormalize(deltas_batch)
            sampled_deltas = deltas_batch['velocity']
        
        # Reconstruct trajectory from deltas (cumulative sum)
        # Initialize trajectory with context. Default at (0, 0)
        trajectory = torch.zeros(B, T, 2, device=device)
        trajectory[:, :context_length, :] = gt_trajectory[:, :context_length, :]
        
        for frame_idx in range(1, T):
            prev_point = trajectory[:, frame_idx-1, :]
            prev_delta = sampled_deltas[:, frame_idx-1, :]
            trajectory[:, frame_idx, :] = prev_point + prev_delta
        
        return trajectory
