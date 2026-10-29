"""Latent flow-matching trajectory model for NAVSIM.

The flag `trajectory` picks what z, the flow-matching target, is:
    vae   the trajectory autoencoder (exp_navsim/episode/trajectory_ae.py), trained jointly
          with the denoiser to reconstruct the normalized velocities through z
    poly  the coefficients of a polynomial fit of the positions (trajectory_repr.py)
    pure  the normalized velocity itself, padded to a fixed length (trajectory_repr.py)
The latent denoiser (exp_navsim/episode/denoiser.py) learns z given the images. The
flow-matching target is detached, so it does not shape the autoencoder.

Images: with `encode_images` the frozen tokenizer encodes batch["images"] on the
fly, otherwise the cached batch["encoded_q_sem"] / ["encoded_q_rec"] are used.
"""

import math

import torch
import pytorch_lightning as pl

from exp_navsim.episode.denoiser import LatentDenoiser
from exp_navsim.episode.trajectory_ae import TrajectoryAE
from exp_navsim.episode.trajectory_repr import PolyTrajectory, RawTrajectory
from exp_navsim.episode.norm import TrajectoryNorm
from exp_navsim.episode.regularization import ImageRegularizer
from exp_navsim.episode.metrics import trajectory_mse, turn_std_over_samples, masked_episode_mean


class NavsimTrajectoryModel(pl.LightningModule):
    def __init__(self, hidden_dim, image_encoder_channels, context_images, image_tokens, ff_dim,
                 num_heads, traj_ae, sigma_min, warmup_steps, min_lr_multiplier, num_diffusion_steps,
                 num_val_samples=5, regularization=None, encode_images=False,
                 tokenizer_exp_dir="logs_tk/tokenizer_288x512",
                 tokenizer_ckpt="checkpoints/last.ckpt", tokenizer_config="config.yaml",
                 trajectory="vae", traj_poly=None, traj_pure=None, max_pred_len=None):
        super().__init__()
        # A DictConfig in hparams would not load back with torch.load(weights_only=True).
        traj_ae, traj_poly, traj_pure, regularization = (
            dict(c or {}) for c in (traj_ae, traj_poly, traj_pure, regularization))
        self.save_hyperparameters()

        # Read by test_model.py and the remote viewer.
        self.context_traj = 0                   # the whole trajectory is predicted
        self.context_images = context_images
        self.encode_images = encode_images
        self.num_val_samples = num_val_samples
        self._encoder = None                    # built on first use, only with encode_images

        self.traj_norm = TrajectoryNorm()       # EMA velocity normalization
        self.image_reg = ImageRegularizer(**regularization)
        # Named traj_ae for every kind, so older checkpoints and resume_traj_ae_only still match.
        if trajectory == "vae":
            self.traj_ae = TrajectoryAE(**traj_ae)
        elif trajectory == "poly":
            self.traj_ae = PolyTrajectory(**traj_poly)
        elif trajectory == "pure":
            self.traj_ae = RawTrajectory(**traj_pure)
        else:
            raise ValueError(f"trajectory must be vae, poly or pure, got {trajectory!r}")
        self.denoiser = LatentDenoiser(self.traj_ae.latent_dim, hidden_dim, image_encoder_channels,
                                       context_images, image_tokens, ff_dim, num_heads)

    def get_encoded_q(self, batch):
        """Image latents (B, T_img, 2C, H, W): q_sem and q_rec concatenated over channels."""
        if not self.encode_images:
            return torch.cat([batch["encoded_q_sem"], batch["encoded_q_rec"]], dim=2)
        if self._encoder is None:
            from exp_navsim.encoders.orbis_io import build_encoder
            hp = self.hparams
            self._encoder = build_encoder(exp_dir=hp.tokenizer_exp_dir, ckpt=hp.tokenizer_ckpt,
                                          config=hp.tokenizer_config, device=self.device)
        images = batch["images"][:, :self.context_images]                   # (B, T_img, C, H, W)
        B, T = images.shape[:2]
        q_rec, q_sem = self._encoder.encode(images.flatten(0, 1).to(self._encoder.device))
        encoded = torch.cat([q_sem, q_rec], dim=1).reshape(B, T, -1, *q_sem.shape[2:])
        # clone() leaves inference mode; detached because the tokenizer is frozen.
        return encoded.detach().clone().to(device=images.device, dtype=torch.float32)

    def horizon(self, T):
        """Frames the model works on: frame 0 plus at most max_pred_len future frames."""
        max_pred_len = self.hparams.get("max_pred_len")
        return T if max_pred_len is None else min(T, max_pred_len + 1)

    def masks(self, batch):
        """(frame_mask, scored), both (B, T) bool, T = horizon of the batch length.

        frame_mask: real frames; pad_collate pads episodes to the longest one and
            stores the lengths in batch["metadata"]["length"].
        scored: frames that enter losses and metrics, i.e. real and not frame 0,
            whose velocity is 0 by construction.
        """
        v = batch["velocity"]
        B, T = v.shape[0], self.horizon(v.shape[1])
        length = batch.get("metadata", {}).get("length")
        if length is None:
            frame_mask = torch.ones(B, T, dtype=torch.bool, device=v.device)
        else:
            frame_mask = torch.arange(T, device=v.device)[None] < length.to(v.device)[:, None]
        scored = frame_mask.clone()
        scored[:, 0] = False
        return frame_mask, scored

    def add_noise(self, x, t):
        """x_t = (1 - t) x + sigma(t) noise, sigma(t) = sigma_min + t (1 - sigma_min). x: (B, d)."""
        noise = torch.randn_like(x)
        s = self.hparams.sigma_min
        return (1 - t[:, None]) * x + (s + t[:, None] * (1 - s)) * noise, noise

    def compute_loss(self, batch):
        frame_mask, scored = self.masks(batch)
        v = batch["velocity"][:, :scored.shape[1]]                          # cut to the horizon
        v = self.traj_norm.normalize(v, scored)                             # (B, T, 2)
        encoded_q = self.image_reg(self.get_encoded_q(batch))               # no-op in eval mode

        ae_loss, logs, z, z_mask = self.traj_ae(v, frame_mask, scored)

        # Flow matching on the latent, t uniform on [0, 1).
        z = z.detach()                                                      # (B, latent_dim)
        t = torch.rand(z.shape[0], device=z.device)
        z_t, noise = self.add_noise(z, t)
        target = z - (1.0 - self.hparams.sigma_min) * noise
        err = (self.denoiser(encoded_q, z_t, t) - target) ** 2
        # z_mask drops the padded frames of the pure trajectory.
        fm_loss = err.mean() if z_mask is None else masked_episode_mean(err, z_mask)

        stage = "train" if self.training else "val"
        self.log_dict({f"{stage}/{name}": value for name, value in {**logs, "fm": fm_loss}.items()},
                      on_epoch=True, sync_dist=True)
        return fm_loss + ae_loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """val/loss is the training objective; val/mse and val/std come from sampled trajectories."""
        loss = self.compute_loss(batch)
        _, scored = self.masks(batch)
        preds = torch.stack([self.sample(batch) for _ in range(self.num_val_samples)])  # (N, B, T, 2)
        gt = batch["trajectory"][None, :, :preds.shape[2]].expand_as(preds)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/mse", trajectory_mse(preds, gt, mask=scored), on_epoch=True, sync_dist=True)
        self.log("val/std", turn_std_over_samples(preds, mask=scored), on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def sample(self, batch, num_diffusion_steps=None):
        """Sample z by the flow-matching ODE (Euler, t from 1 to 0) and decode it.

        Returns positions (B, T, 2) starting at the origin; T is the padded batch
        length cut to the horizon (max_pred_len + 1), and the caller drops the
        padded tail.
        """
        self.eval()
        steps = num_diffusion_steps or self.hparams.num_diffusion_steps
        encoded_q = self.get_encoded_q(batch)
        B, T = batch["velocity"].shape[0], self.horizon(batch["velocity"].shape[1])

        z = torch.randn(B, self.traj_ae.latent_dim, device=self.device)
        t_steps = torch.linspace(1, 0, steps + 1, device=self.device)
        for i in range(steps):
            z = z + self.denoiser(encoded_q, z, t_steps[i].repeat(B)) * (t_steps[i] - t_steps[i + 1])

        trajectory = torch.cumsum(self.traj_norm.denormalize(self.traj_ae.decode(z, T)), dim=1)
        return trajectory - trajectory[:, :1]

    def configure_optimizers(self):
        # self.learning_rate is set by the training script.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        warmup, min_mult = self.hparams.warmup_steps, self.hparams.min_lr_multiplier

        def lr_lambda(step):
            # Linear warmup, then cosine decay down to min_lr_multiplier.
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(self.trainer.max_steps - warmup, 1)
            return min_mult + (1 - min_mult) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
