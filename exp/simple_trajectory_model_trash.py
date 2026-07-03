"""
Simple trajectory prediction model using flow matching diffusion.

Predicts trajectory positions autoregressively:
1. Takes the previous position as context
2. Uses flow matching to denoise and predict the next position
3. Continues sampling to generate full trajectories
"""

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Add parent directory to path
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.pos_emb import PositionEmbedding1d

class SimpleDiffusionModel(pl.LightningModule):
    """
    Minimal flow matching model for trajectory prediction.
    
    Architecture:
        turn: (B,) - turn parameter [0, 1]
        prev_position: (B, 2) - previous (x, y) position (absolute)
        noisy_delta: (B, 2) - noisy position delta (x_t - x_{t-1})
        t: (B,) - diffusion timestep
            ↓ Concatenate
        input: (B, 5 + time_embed_dim) - [turn, prev_x, prev_y, noisy_dx, noisy_dy, t_emb]
            ↓ FC Module with residual blocks
        predicted_flow: (B, 2) - predicted velocity field for delta
    """
    
    def __init__(
        self,
        hidden_dim,         # Hidden dimension for FC
        sigma_min,        # Minimum noise level
        warmup_steps,     # Warmup steps
        min_lr_multiplier, # Minimum LR multiplier
        n_diffusion_samples, # Number of diffusion time samples per trajectory point
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.sigma_min = sigma_min
        self.warmup_steps = warmup_steps
        self.min_lr_multiplier = min_lr_multiplier
        self.n_diffusion_samples = n_diffusion_samples
        
        time_embed_dim = hidden_dim * 2
        
        self.time_mlp = nn.Sequential(
            PositionEmbedding1d(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.Tanh(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # FC module for denoising position
        self.denoiser = nn.Sequential(
            nn.Linear(1 + 2 + 2 + time_embed_dim, hidden_dim),  # turn (1) + prev_point (2) + noisy_delta (2) + time (hidden_dim)
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),  # predicted position (2D)
        )
        
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
            x: (B, 2)
            t: (B,)
            noise: (B, 2)
        """
        noise = torch.randn_like(x) if noise is None else noise
        t_expanded = t.view(-1, 1)
        x_t = self.alpha(t_expanded) * x + self.sigma(t_expanded) * noise
        return x_t, noise
    
    def forward(self, turn, prev_position, noisy_position, t):
        """
        Predict flow/velocity field for denoising position delta.
        
        Args:
            turn: (B,) turn parameter [0, 1]
            prev_position: (B, 2) previous position (absolute)
            noisy_position: (B, 2) noisy position
            t: (B,) diffusion timestep
        
        Returns:
            pred_flow: (B, 2) predicted velocity field for delta
        """
        # Concatenate turn, previous position (absolute), noisy delta, and time embedding
        t_emb = self.time_mlp(t)
        turn = torch.zeros_like(turn) # TODO remove conditioning
        turn_expanded = turn.view(-1, 1)  # (B, 1)
        
        x = torch.cat([turn_expanded, prev_position, noisy_position, t_emb], dim=1)  # (B, 5 + time_embed_dim)
        
        # Predict velocity field for delta
        pred_flow = self.denoiser(x)  # (B, 2)
        
        return pred_flow

    def compute_loss(self, batch):
        """
        Compute flow matching loss for trajectory prediction with autoregressive training.
        
        Possible improvements:
        - rollout loss
        - train several times with different diffusion noise
        
        """
        turn = batch["turn"]  # (B,) - turn parameter
        trajectory = batch["position"]  # (B, T, 2) - ground truth positions
        velocity = batch["velocity"]
        
        B, T = trajectory.shape[:2]
        
        losses = []
        
        #steps = np.random.randint(0, T-1, size=5)
        
        for start_idx in range(0, T):
            current_position = trajectory[:, start_idx, :].clone()  # (B, 2)
            target_delta = velocity[:, start_idx, :]  # (B, 2)
            
            # Sample diffusion timestep from [0, 1)
            t = torch.rand(B, device=trajectory.device)  # (B,)
            
            # Add noise to target delta
            noisy_delta, noise = self.add_noise(target_delta, t)  # (B, 2)
            
            # Predict velocity field for delta
            pred_v = self.forward(turn, current_position.detach(), noisy_delta, t)  # (B, 2)
            
            # Compute target for flow matching: v = A(t) * delta + B(t) * noise
            t_expanded = t.view(-1, 1)
            target = self.A(t_expanded) * target_delta + self.B(t_expanded) * noise  # (B, 2)
            
            # MSE loss
            loss = ((pred_v - target) ** 2).mean()
            losses.append(loss)
        
        # Average loss over all predictions
        total_loss = torch.stack(losses).mean()
        
        return total_loss
    
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
    def sample(self, batch, context_length, T, num_diffusion_steps, dataset):
        """
        Sample trajectory autoregressively using flow matching.
        
        Args:
            batch: dict - batch size is inferred from batch
            context_length: int
            T: number of trajectory timesteps to generate
            num_diffusion_steps: number of diffusion steps
            dataset: SyntheticTrajectoryDataset for denormalization
        
        Returns:
            sampled_trajectory: (B, T, 2) predicted trajectory positions
        """
        self.eval()
        
        # Get batch size, turn parameter, and device
        turn = batch["turn"]  # (B,)
        position = batch["position"]  # (B, T, 2)
        
        B = turn.shape[0]
        device = turn.device
        
        # Initialize trajectory with context. Default at (0, 0)
        trajectory = torch.zeros(B, T, 2, device=device)
        trajectory[:, :context_length, :] = position[:, :context_length, :]
        
        # Diffusion timesteps from t=1 to t=0
        t_steps = torch.linspace(1, 0, num_diffusion_steps + 1, device=device)
        
        # Autoregressively sample each position delta (normalized)
        for step_idx in range(context_length, T):
            # Get previous position as context
            prev_position = trajectory[:, step_idx - 1, :]  # (B, 2) - absolute position
            
            # Start from pure noise at t=1 for the delta
            sampled_delta = torch.randn(B, 2, device=device)

            e = torch.randn(B, 2, device=device)
            sigma = 0.
            
            # Denoise via flow matching ODE
            for i in range(num_diffusion_steps):
                t = t_steps[i].repeat(B)  # (B,)
                
                # Predict velocity field for delta
                pred_v = self.forward(turn, prev_position, sampled_delta, t) + sigma * e  # (B, 2)
                
                # Update: integrate ODE backward from t=1 to t=0
                dt = t_steps[i] - t_steps[i + 1]
                sampled_delta = sampled_delta + pred_v * dt
            
            # Denormalize velocity if dataset provided
            velocity_batch = {'velocity': sampled_delta}
            velocity_batch = dataset.denormalize(velocity_batch)
            sampled_delta = velocity_batch['velocity']
            
            # Store the new position by adding velocity to previous position
            trajectory[:, step_idx, :] = prev_position + sampled_delta
        
        return trajectory
