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
  * validation_step logs val/loss (the same flow-matching objective as
    train/loss, so the train/val gap is directly readable) and samples each
    context several times for val/mse and val/std (STD of total turn); only val
    loss is printed, the rest goes to TensorBoard.
"""

import math

import torch
import pytorch_lightning as pl

from exp_navsim.denoiser import TrajectoryDenoiser
from exp_navsim.norm import TrajectoryNorm
from exp_navsim.metrics import trajectory_mse, turn_std_over_samples, masked_episode_mean


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
        image_tokens,
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
            image_tokens=image_tokens,
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
        """Return the image latents (B, T, 2*C, H, W).

        Both cached tokenizer latents are handed to the denoiser, concatenated
        over the channel axis as [q_sem, q_rec]: the semantic latent carries what
        the scene means, the reconstruction latent where things are, and the
        trajectory head wants both. The aggregator is built for 2*C channels
        accordingly (exp_navsim/denoiser.py).

        In raw mode, encode batch["images"] with the frozen tokenizer; in encoded
        mode, use the precomputed batch["encoded_q_sem"] / ["encoded_q_rec"].
        """
        if not self.encode_images:
            return torch.cat([batch["encoded_q_sem"], batch["encoded_q_rec"]], dim=2)

        images = batch["images"]                              # (B, T, C, H, W)
        B, T = images.shape[:2]
        flat = images.reshape(B * T, *images.shape[2:]).to(self.encoder.device)
        q_rec, q_sem = self.encoder.encode(flat)
        encoded = torch.cat([q_sem, q_rec], dim=1)            # (B*T, 2*C, H, W)
        encoded = encoded.reshape(B, T, *encoded.shape[1:])
        # clone() to leave inference-mode; detached because the tokenizer is frozen.
        return encoded.detach().clone().to(device=images.device, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Padding mask (variable-length episodes)
    # ------------------------------------------------------------------ #
    @staticmethod
    def frame_mask(batch, T):
        """Which of the T frames of each padded episode are real.

        exp_navsim.data.collate.pad_collate pads variable-length episodes to the
        longest one in the batch and records the true lengths in
        batch["metadata"]["length"]. The model still predicts all T frames for
        every sample; this mask keeps the padded tail out of the loss/metrics.

        Returns:
            (B, T) bool tensor; all-True when the batch carries no lengths
            (fixed-length windows).
        """
        length = batch.get("metadata", {}).get("length")
        device = batch["velocity"].device
        if length is None:
            return torch.ones(batch["velocity"].shape[0], T, dtype=torch.bool, device=device)
        frames = torch.arange(T, device=device)
        return frames[None, :] < length.to(device)[:, None]

    @classmethod
    def scored_mask(cls, batch, T):
        """Frames that may enter a loss or a metric: real, and not frame 0.

        Velocity is a finite difference, so v_0 is defined as 0 for every episode
        (exp_navsim/data/navsim_base.py). Scoring it rewards predicting a
        constant, and feeding it to the EMA normalizer drags the velocity mean
        towards zero; the same holds for the trajectory metrics, where frame 0 is
        the origin of both the prediction and the ground truth by construction.
        So frame 0 is dropped here — it stays in the *input*, where it is a real
        observed frame the model should attend to.
        """
        mask = cls.frame_mask(batch, T).clone()
        mask[:, 0] = False
        return mask

    @staticmethod
    def episode_mean(values, mask):
        """Per-frame mean inside each episode, then the mean over episodes.

        `values` is (B, T) and `mask` (B, T) selects the frames that count. The
        pooling itself lives in exp_navsim.metrics so that the loss and val/mse
        weight an episode the same way; see `masked_episode_mean` for why the
        length must not decide how much an episode counts.
        """
        return masked_episode_mean(values, mask)

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
        """Flow-matching loss over the whole trajectory (all frames at once).

        Padded frames of variable-length episodes are excluded from the loss (and
        from the velocity statistics) via the mask, without any per-sample loop.
        The surviving frames are averaged inside each episode before the episodes
        are averaged together (`episode_mean`), so a 40-frame scene does not
        outvote a 5-frame one. The diffusion timestep is drawn logit-normally
        rather than uniformly; see the comment at the draw.
        """
        B, T = batch["velocity"].shape[:2]
        # Two different masks: `frame_mask` says which frames exist (the attention
        # may not read the others), `scored_mask` which frames count towards the
        # loss and the statistics (the same, minus the constant frame 0).
        frame_mask = self.frame_mask(batch, T)                     # (B, T)
        scored = self.scored_mask(batch, T)                        # (B, T)
        deltas = self.traj_norm.normalize(batch["velocity"], scored)  # (B, T, 2)
        encoded_q = self.get_encoded_q(batch)                      # (B, T_img, C, H, W)

        context_length = self.context_traj      # predict every frame after the context

        # One diffusion timestep per trajectory, drawn logit-normally: sample
        # u ~ N(0, 1) and push it through the sigmoid, so t = sigmoid(u) lands in
        # (0, 1) with a density that peaks at t = 0.5 and thins out towards both
        # ends. Uniform sampling spends the same effort everywhere, but the ends
        # are the easy part of the schedule: near t = 0 the input is almost the
        # clean trajectory and near t = 1 it is almost pure noise, and in both
        # cases the velocity field is close to something the model gets right
        # early. The hard, ambiguous middle is where the trajectory's shape is
        # actually decided, and that is where this puts the samples. No extra
        # loss weighting is needed, the sampling density *is* the weighting.
        # This is the schedule from the SD3 paper (Esser et al., 2024).
        t = torch.sigmoid(torch.randn(B, device=deltas.device))

        # Noise the whole trajectory and predict the velocity field.
        noisy_deltas, noise = self.add_noise(deltas, t)
        pred_v = self.denoiser(encoded_q, deltas, noisy_deltas, t,
                               frame_mask=frame_mask)              # (B, T - context_traj, 2)

        # Flow-matching target: v = A(t) * deltas + B(t) * noise, with A = 1,
        # B = -(1 - sigma_min).
        target = deltas - (1.0 - self.sigma_min) * noise
        target = target[:, context_length:, :]                     # only predicted frames

        # Mean over the predicted *and* scored frames only, one vote per episode.
        return self.episode_mean(((pred_v - target) ** 2).mean(-1), scored[:, context_length:])

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
        T is the padded batch length: the same number of frames is predicted for
        every sample, and the caller drops the padded tail (frame_mask).
        """
        self.eval()
        steps = num_diffusion_steps or self.num_diffusion_steps

        deltas = self.traj_norm.normalize(batch["velocity"])       # work in normalized space
        encoded_q = self.get_encoded_q(batch)

        B, T = deltas.shape[:2]
        device = deltas.device
        context_length = self.context_traj      # predict every frame after the context
        frame_mask = self.frame_mask(batch, T)  # padded frames stay out of the attention

        # Start from pure noise, but keep the (known) context deltas fixed.
        sampled = torch.randn(B, T, 2, device=device)
        sampled[:, :context_length] = deltas[:, :context_length]

        # Integrate the ODE backward from t=1 to t=0.
        t_steps = torch.linspace(1, 0, steps + 1, device=device)
        for i in range(steps):
            t = t_steps[i].repeat(B)
            pred_v = self.denoiser(encoded_q, deltas, sampled, t, frame_mask=frame_mask)
            dt = t_steps[i] - t_steps[i + 1]
            sampled[:, context_length:] = sampled[:, context_length:] + pred_v * dt

        sampled = self.traj_norm.denormalize(sampled)              # back to real velocities

        # Reconstruct positions from velocities; re-origin at the first frame.
        trajectory = torch.cumsum(sampled, dim=1)
        return trajectory - trajectory[:, :1, :]

    # ------------------------------------------------------------------ #
    # Sampling-based metrics (shared by train and validation)
    # ------------------------------------------------------------------ #
    def sampled_mse(self, batch, num_samples=1):
        """MSE between sampled trajectories and the GT of `batch`.

        Sampling puts the model in eval mode (see `sample`), so the caller must
        restore the previous mode when calling this during training.

        Returns:
            (mse, preds, mask) — preds is (num_samples, B, T, 2), mask is the
            (B, T) scored mask; only predicted (post-context) frames that are
            real and not frame 0 enter the MSE (see `scored_mask`).
        """
        gt = batch["trajectory"]                                   # (B, T, 2)
        ctx = self.context_traj
        mask = self.scored_mask(batch, gt.shape[1])                # (B, T), padding-aware

        preds = torch.stack([self.sample(batch) for _ in range(num_samples)], dim=0)
        gt_pred = gt[None, :, ctx:].expand_as(preds[:, :, ctx:])
        mse = trajectory_mse(preds[:, :, ctx:], gt_pred, mask=mask[:, ctx:])
        return mse, preds, mask

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validation_step(self, batch, batch_idx):
        """Log the flow-matching loss, the sampled-trajectory MSE and the turn STD.

        val/loss is the same quantity as train/loss (compute_loss: one random
        diffusion timestep, velocity-field regression), so the two curves are
        directly comparable. val/mse and val/std need full ODE rollouts and
        measure the sampled trajectories instead. Only val loss is shown on the
        command line; all three go to TensorBoard.
        """
        # Same objective as training, on val data. The EMA velocity statistics
        # are not updated here: normalize() only updates while self.training.
        loss = self.compute_loss(batch)

        # N independent samples of the same context.
        mse, preds, mask = self.sampled_mse(batch, self.num_val_samples)   # preds: (N, B, T, 2)
        std = turn_std_over_samples(preds, mask=mask)              # STD of total turn across N

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/mse", mse, prog_bar=False, on_epoch=True, sync_dist=True)
        self.log("val/std", std, prog_bar=False, on_epoch=True, sync_dist=True)
        return loss

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
