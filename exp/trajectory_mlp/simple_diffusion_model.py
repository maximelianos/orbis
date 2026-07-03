"""
The example model is a minimal model for testing the diffusion.
It does not use the image encoder or any advanced NN layers.

Predicts the next trajectory velocity:
1. Take existing context trajectory, extract the last velocity
2. Apply an FC module to diffuse the next velocity in the (dx, dy) data space
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

class SimpleDiffusionModel(pl.LightningModule):
    """
    Minimal flow matching model for trajectory prediction.
    
    Architecture:
        velocity: (B, T, 2) - normalized (dx, dy)
            ↓ Extract last velocity
        last_velocity: (B, 2)
            ↓ Add noise
        noisy_velocity: (B, 2)
            ↓ FC Module
        predicted_velocity: (B, 2)
    """
    
    def __init__(
        self,
        data_dim,
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
        
        # FC module for denoising velocity
        self.denoiser = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),  # prev_value (D) + time (1)
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, data_dim),  # predicted value
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
            x: (B, D)
            t: (B,)
            noise: (B, D)
        """
        noise = torch.randn_like(x) if noise is None else noise
        t_expanded = t.view(-1, 1)
        x_t = self.alpha(t_expanded) * x + self.sigma(t_expanded) * noise
        return x_t, noise
    
    def forward(self, noisy_value, t):
        """
        Denoise velocity.
        
        Args:
            prev_velocity: (B, D) previous velocity
            noisy_velocity: (B, D) noisy velocity
            t: (B,) diffusion timestep
        
        Returns:
            pred_velocity: (B, 1) predicted value
        """
        # Concatenate previous velocity, noisy velocity, and time
        t_expanded = t.view(-1, 1)  # (B, 1)
        x = torch.cat([noisy_value, t_expanded], dim=1)  # (B, 2)
        
        # Predict velocity
        pred_velocity = self.denoiser(x)  # (B, 2)
        
        return pred_velocity

    def compute_loss(self, batch):
        """Compute flow matching loss."""
        value = batch['value']  # (B, D)
        
        B, D = value.shape[:2]
        
        losses = []
        
        # Train on same data sample with different diffusion timesteps
        for _ in range(self.n_diffusion_samples):
            # Sample diffusion timestep from [0, 1)
            t = torch.rand(B, device=value.device)  # (B,)
            
            # Add noise to target
            noisy_velocity, noise = self.add_noise(value, t)  # (B, 2)
            
            # Predict velocity
            pred_velocity = self.forward(noisy_velocity, t)  # (B, 1)
            
            # Compute target for flow matching: v = A(t) * x + B(t) * noise
            t_expanded = t.view(-1, 1)
            target = self.A(t_expanded) * value + self.B(t_expanded) * noise  # (B, 2)
            
            # MSE loss
            loss = ((pred_velocity - target) ** 2).mean()
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
    def sample_step(self, sampled_value, t, dt):
        """
        Perform a single denoising step in the sampling process.
        
        Args:
            sampled_value: (B, D) current noisy value
            t: (B,) current diffusion timestep
            dt: scalar, time step size
        
        Returns:
            sampled_value: (B, D) updated value after one denoising step
        """
        # Predict velocity field
        pred_v = self.forward(sampled_value, t)  # (B, D)
        
        # Update: integrate ODE backward from t=1 to t=0
        sampled_value = sampled_value + pred_v * dt
        
        return sampled_value
    
    @torch.no_grad()
    def sample(self, batch, num_steps=20):
        """
        Sample value using flow matching.
        
        Args:
            batch: dict with keys:
                - 'value': (B, D) - can be ignored, not used during sampling
            num_steps: number of diffusion steps
        
        Returns:
            sampled_value: (B, D) predicted value
        """
        self.eval()
        
        B, D = batch['value'].shape
        device = batch['value'].device
        
        # Start from noise at t=1
        sampled_value = torch.randn(B, D, device=device)
        
        # Diffusion steps from t=1 to t=0
        t_steps = torch.linspace(1, 0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t = t_steps[i].repeat(B)  # (B,)
            dt = t_steps[i] - t_steps[i + 1]
            
            # Perform one denoising step
            sampled_value = self.sample_step(sampled_value, t, dt)
        
        return sampled_value
