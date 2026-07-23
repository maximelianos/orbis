"""Flow-matching trajectory model (LightningModule) for NAVSIM.

This file holds the flow-matching schedule, noising, sampling, and the training /
validation loops. The transformer that actually denoises the trajectory lives in
exp_navsim/denoiser.py (TrajectoryDenoiser), keeping the network architecture
separate from the diffusion logic.

Design points from docs/idea.md:
  * raw vs encoded flag (`encode_images`):
        True  -> load the frozen tokenizer and encode batch["images"] on the fly.
        False -> consume precomputed batch["encoded_q_sem"]; the encoder is never
                 constructed.
  * EMA velocity normalization via exp_navsim.norm.TrajectoryNorm.
  * validation_step samples each context several times and logs val/mse and
    val/std (STD of total turn) to TensorBoard, while printing only val loss.
"""

import math

import torch
import pytorch_lightning as pl

from exp_navsim.denoiser import TrajectoryDenoiser
from exp_navsim.norm import TrajectoryNorm
from exp_navsim.metrics import trajectory_mse, turn_std_over_samples


class NavsimTrajectoryModel(pl.LightningModule):
    """Flow-matching model predicting whole-trajectory velocity (dx, dy)."""

    def __init__(
        self,
        hidden_dim,
        image_encoder_channels,
        sigma_min,
        warmup_steps,
        min_lr_multiplier,
        context_traj,
        context_images,
        num_diffusion_steps=50,
        # validation
        num_val_samples=5,           # how many times to sample the same context
        # raw vs encoded image input
        encode_images=False,
        tokenizer_exp_dir="logs_tk/tokenizer_288x512",
        tokenizer_ckpt="checkpoints/last.ckpt",
        tokenizer_config="config.yaml",
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- raw vs encoded image input -------------------------------------
        # The tokenizer is created lazily (on first use) so it lands on the right
        # device, and only when encode_images is set.
        self.encode_images = encode_images
        self.tokenizer_exp_dir = tokenizer_exp_dir
        self.tokenizer_ckpt = tokenizer_ckpt
        self.tokenizer_config = tokenizer_config
        self._encoder = None

        # --- flow-matching / schedule hyperparameters -----------------------
        self.sigma_min = sigma_min
        self.warmup_steps = warmup_steps
        self.min_lr_multiplier = min_lr_multiplier
        # context_traj: known-trajectory frames (= context, not predicted).
        # context_images: image-observed frames (independent). Episode length is
        # free; the model predicts every frame after context_traj.
        self.context_traj = context_traj
        self.context_images = context_images
        self.num_diffusion_steps = num_diffusion_steps
        self.num_val_samples = num_val_samples

        # --- denoiser network (separate file) -------------------------------
        self.denoiser = TrajectoryDenoiser(
            hidden_dim=hidden_dim,
            image_encoder_channels=image_encoder_channels,
            context_traj=context_traj,
            context_images=context_images,
        )

        # --- online (EMA) velocity normalization ----------------------------
        self.traj_norm = TrajectoryNorm()

    # ------------------------------------------------------------------ #
    # Image encoding (only when encode_images)
    # ------------------------------------------------------------------ #
    @property
    def encoder(self):
        """Lazily-constructed frozen image tokenizer (only when encode_images)."""
        if self._encoder is None:
            from exp_navsim.encoder_io import build_encoder
            self._encoder = build_encoder(
                exp_dir=self.tokenizer_exp_dir,
                ckpt=self.tokenizer_ckpt,
                config=self.tokenizer_config,
                device=self.device,
            )
        return self._encoder

    def get_encoded_q(self, batch):
        """Return semantic latents (B, T, C, H, W).

        In raw mode, encode batch["images"] with the frozen tokenizer; in encoded
        mode, use the precomputed batch["encoded_q_sem"].
        """
        if not self.encode_images:
            return batch["encoded_q_sem"]

        images = batch["images"]                              # (B, T, C, H, W)
        B, T = images.shape[:2]
        flat = images.reshape(B * T, *images.shape[2:]).to(self.encoder.device)
        _, q_sem = self.encoder.encode(flat)
        q_sem = q_sem.reshape(B, T, *q_sem.shape[1:])
        # clone() to leave inference-mode; detached because the tokenizer is frozen.
        return q_sem.detach().clone().to(device=images.device, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Flow-matching schedule
    # ------------------------------------------------------------------ #
    def alpha(self, t):
        """Flow-matching alpha schedule."""
        return 1.0 - t

    def sigma(self, t):
        """Flow-matching sigma schedule."""
        return self.sigma_min + t * (1.0 - self.sigma_min)

    def add_noise(self, x, t, noise=None):
        """x_t = alpha(t) * x + sigma(t) * noise, for x of shape (B, ...)."""
        noise = torch.randn_like(x) if noise is None else noise
        shape = [x.shape[0]] + [1] * (x.dim() - 1)
        x_t = self.alpha(t).view(*shape) * x + self.sigma(t).view(*shape) * noise
        return x_t, noise

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def compute_loss(self, batch):
        """Flow-matching loss over the whole trajectory (all frames at once)."""
        deltas = self.traj_norm.normalize(batch["velocity"])       # (B, T, 2)
        encoded_q = self.get_encoded_q(batch)                      # (B, T, C, H, W)

        B, T = deltas.shape[:2]
        context_length = self.context_traj      # predict every frame after the context

        # Sample one diffusion timestep per trajectory in [0, 1).
        t = torch.rand(B, device=deltas.device)

        # Noise the whole trajectory and predict the velocity field.
        noisy_deltas, noise = self.add_noise(deltas, t)
        pred_v = self.denoiser(encoded_q, deltas, noisy_deltas, t)  # (B, T - context_traj, 2)

        # Flow-matching target: v = A(t) * deltas + B(t) * noise, with A = 1,
        # B = -(1 - sigma_min).
        target = deltas - (1.0 - self.sigma_min) * noise
        target = target[:, context_length:, :]                     # only predicted frames

        return ((pred_v - target) ** 2).mean()

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, batch, num_diffusion_steps=None):
        """Denoise the whole trajectory via the flow-matching ODE.

        Returns the reconstructed positions (B, T, 2), starting at the origin.
        """
        self.eval()
        steps = num_diffusion_steps or self.num_diffusion_steps

        deltas = self.traj_norm.normalize(batch["velocity"])       # work in normalized space
        encoded_q = self.get_encoded_q(batch)

        B, T = deltas.shape[:2]
        device = deltas.device
        context_length = self.context_traj      # predict every frame after the context

        # Start from pure noise, but keep the (known) context deltas fixed.
        sampled = torch.randn(B, T, 2, device=device)
        sampled[:, :context_length] = deltas[:, :context_length]

        # Integrate the ODE backward from t=1 to t=0.
        t_steps = torch.linspace(1, 0, steps + 1, device=device)
        for i in range(steps):
            t = t_steps[i].repeat(B)
            pred_v = self.denoiser(encoded_q, deltas, sampled, t)
            dt = t_steps[i] - t_steps[i + 1]
            sampled[:, context_length:] = sampled[:, context_length:] + pred_v * dt

        sampled = self.traj_norm.denormalize(sampled)              # back to real velocities

        # Reconstruct positions from velocities; re-origin at the first frame.
        trajectory = torch.cumsum(sampled, dim=1)
        return trajectory - trajectory[:, :1, :]

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validation_step(self, batch, batch_idx):
        """Sample each context num_val_samples times; log MSE and turn STD.

        Only val loss (== MSE) is shown on the command line; val/mse and val/std
        are still written to TensorBoard.
        """
        gt = batch["trajectory"]                                   # (B, T, 2)
        ctx = self.context_traj

        # N independent samples of the same context.
        preds = torch.stack([self.sample(batch) for _ in range(self.num_val_samples)], dim=0)  # (N, B, T, 2)

        gt_pred = gt[None, :, ctx:].expand_as(preds[:, :, ctx:])
        mse = trajectory_mse(preds[:, :, ctx:], gt_pred)
        std = turn_std_over_samples(preds)                         # STD of total turn across N

        self.log("val/loss", mse, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/mse", mse, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val/std", std, prog_bar=False, on_epoch=True, sync_dist=True)
        return mse

    # ------------------------------------------------------------------ #
    # Optimizer / schedule
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)

        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            # Linear warmup, then cosine decay down to min_lr_multiplier.
            if step < self.warmup_steps:
                return step / max(self.warmup_steps, 1)
            total = max(self.trainer.max_steps - self.warmup_steps, 1)
            progress = (step - self.warmup_steps) / total
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return (1 - self.min_lr_multiplier) * cosine_decay + self.min_lr_multiplier

        scheduler = LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
