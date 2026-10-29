"""Trajectory-only baseline: the future predicted from the previous trajectory, no images.

The first K = context_traj frames of each window are known, the rest is predicted:

    normalized velocities v_1 .. v_{K-1} of the context (v_0 is 0 by construction)
      -> TrajectoryContextDenoiser (traj/denoiser.py), flow matching on z
      -> z is the trajectory representation (vae | poly | pure) of the future segment,
         frames K-1 .. K-1 + max_pred_len, whose origin is the present frame K-1
      -> decoded future velocities appended to the known context, then integrated

The batch is the usual one: the context frames are the leading frames of
batch["velocity"], so neither the dataloader nor the scripts need a new key.
sample() returns the whole window, starting at the origin of its frame 0, with
the first K positions equal to the ground truth, which is the context_traj
convention test_model.py and tools/remote_viewer already follow. Losses and
metrics only count the predicted frames (see masks).

Everything else (trajectory representation, EMA norm, validation metrics,
optimizer) is inherited from NavsimTrajectoryModel.
"""

import torch

from exp_navsim.episode.model import NavsimTrajectoryModel
from exp_navsim.episode.metrics import masked_episode_mean
from exp_navsim.episode.traj.denoiser import TrajectoryContextDenoiser


class TrajContextModel(NavsimTrajectoryModel):
    def __init__(self, context_traj, depth=4, **kwargs):
        assert context_traj >= 2, "context_traj must be >= 2: frame 0 alone carries no velocity"
        # Plain dicts: hparams are collected from this frame too, and a DictConfig
        # would not load back with weights_only=True.
        kwargs.update({k: dict(kwargs.get(k) or {}) for k in ("traj_ae", "traj_poly", "traj_pure", "regularization")})
        # The parent builds an image denoiser that is replaced below; these only keep it small.
        kwargs.update(encode_images=False, image_encoder_channels=1, image_tokens=1)
        super().__init__(**kwargs)
        hp = self.hparams
        self.context_traj = context_traj
        self.denoiser = TrajectoryContextDenoiser(self.traj_ae.latent_dim, hp.hidden_dim, context_traj - 1,
                                                  hp.ff_dim, hp.num_heads, depth)

    def get_encoded_q(self, batch):
        raise RuntimeError("TrajContextModel reads no images")

    def horizon(self, T):
        """Frames the model works on: the context plus at most max_pred_len future frames."""
        max_pred_len = self.hparams.get("max_pred_len")
        return T if max_pred_len is None else min(T, self.context_traj + max_pred_len)

    def masks(self, batch):
        """As in the parent, but the context frames are given, so they are not scored."""
        frame_mask, scored = super().masks(batch)
        scored[:, :self.context_traj] = False
        return frame_mask, scored

    def context(self, v, frame_mask):
        """Denoiser input: normalized v_1 .. v_{K-1} (B, K - 1, 2) and their frame mask."""
        return v[:, 1:self.context_traj], frame_mask[:, 1:self.context_traj]

    def compute_loss(self, batch):
        frame_mask, scored = self.masks(batch)
        K = self.context_traj
        assert scored.shape[1] > K, f"batch of {scored.shape[1]} frames has nothing to predict after context_traj={K}"
        v = batch["velocity"][:, :scored.shape[1]]                          # cut to the horizon
        v = self.traj_norm.normalize(v, scored)                             # (B, T, 2)
        context, context_mask = self.context(v, frame_mask)

        # The future segment starts at the present frame K-1. Its first velocity is
        # not a step of the segment, so it is set to v_0, the normalized zero,
        # exactly like frame 0 of a window in the image model (it is not scored).
        future = v[:, K - 1:].clone()
        future[:, 0] = v[:, 0]
        ae_loss, logs, z, z_mask = self.traj_ae(future, frame_mask[:, K - 1:], scored[:, K - 1:])

        # Flow matching on the latent, t uniform on [0, 1).
        z = z.detach()                                                      # (B, latent_dim)
        t = torch.rand(z.shape[0], device=z.device)
        z_t, noise = self.add_noise(z, t)
        target = z - (1.0 - self.hparams.sigma_min) * noise
        err = (self.denoiser(context, context_mask, z_t, t) - target) ** 2
        # z_mask drops the padded frames of the pure trajectory.
        fm_loss = err.mean() if z_mask is None else masked_episode_mean(err, z_mask)

        stage = "train" if self.training else "val"
        self.log_dict({f"{stage}/{name}": value for name, value in {**logs, "fm": fm_loss}.items()},
                      on_epoch=True, sync_dist=True)
        return fm_loss + ae_loss

    @torch.no_grad()
    def sample(self, batch, num_diffusion_steps=None):
        """Sample the future by the flow-matching ODE (Euler, t from 1 to 0).

        Returns positions (B, T, 2) starting at the origin; T is the padded batch
        length cut to the horizon (context_traj + max_pred_len). The first
        context_traj frames are the ground truth, the rest is sampled.
        """
        self.eval()
        steps = num_diffusion_steps or self.hparams.num_diffusion_steps
        frame_mask, _ = self.masks(batch)
        B, T = frame_mask.shape
        K = min(self.context_traj, T)                   # a window shorter than the context is all known
        v_true = batch["velocity"][:, :T].float()
        context, context_mask = self.context(self.traj_norm.normalize(v_true), frame_mask)

        z = torch.randn(B, self.traj_ae.latent_dim, device=self.device)
        t_steps = torch.linspace(1, 0, steps + 1, device=self.device)
        for i in range(steps):
            z = z + self.denoiser(context, context_mask, z, t_steps[i].repeat(B)) * (t_steps[i] - t_steps[i + 1])

        # The segment's frame 0 is the present frame, already in the context.
        future = self.traj_norm.denormalize(self.traj_ae.decode(z, T - K + 1))[:, 1:]
        trajectory = torch.cumsum(torch.cat([v_true[:, :K], future.float()], dim=1), dim=1)
        return trajectory - trajectory[:, :1]
