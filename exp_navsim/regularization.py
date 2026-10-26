"""Training-time regularization of the image conditioning.

The denoiser overfits by memorizing: every training episode comes with one fixed
pair of image latents, and those latents are detailed enough to identify the
episode, so the network can learn "this scene -> this trajectory" instead of a
distribution. The regularizers here make the latents a worse lookup key while
keeping what is shared between scenes:

  * latent noise:    Gaussian noise, scaled to each channel's spread in the batch;
  * grid masking:    a random subset of the H x W cells of every frame is zeroed;
  * cond dropout:    the whole image input of a sample is zeroed, as in
                     classifier-free guidance training (and the `cond_dropout`
                     of exp_navsim/wm/denoiser.py).

A zeroed cell or frame does not reach the aggregator as "nothing": the input
Linear maps it to its bias and the 2D position embedding is still added, so it
acts as a learned mask token, like MAE's.

Everything is a no-op outside training mode. Validation (`compute_loss` in eval
mode) and sampling therefore see the clean latents, and val/loss stays the
unregularized objective. The module has no parameters, so it does not change
the checkpoint.
"""

import torch
import torch.nn as nn


class ImageRegularizer(nn.Module):
    """Perturbs image latents (B, T_img, C, H, W) during training only."""

    def __init__(self, cond_dropout=0.0, mask_ratio=0.0, noise_std=0.0):
        """
        Args:
            cond_dropout: probability of zeroing a sample's whole image input.
            mask_ratio:   fraction of the H x W cells zeroed, drawn independently
                          per sample and frame.
            noise_std:    std of the added noise, relative to the per-channel std
                          of the batch latents. 0.1 means 10% of the channel's
                          own spread.
        """
        super().__init__()
        assert 0.0 <= cond_dropout < 1.0, f"cond_dropout must be in [0, 1), got {cond_dropout}"
        assert 0.0 <= mask_ratio < 1.0, f"mask_ratio must be in [0, 1), got {mask_ratio}"
        assert noise_std >= 0.0, f"noise_std must be >= 0, got {noise_std}"
        self.cond_dropout = cond_dropout
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std

    def forward(self, latents):
        """
        Args:
            latents: (B, T_img, C, H, W) image latents.
        Returns:
            (B, T_img, C, H, W), perturbed in training mode, unchanged otherwise.
        """
        if not self.training:
            return latents
        B, T, C, H, W = latents.shape

        # Noise first, so the masked cells and dropped samples below end up
        # exactly zero rather than zero plus noise.
        if self.noise_std > 0:
            # Per-channel spread over batch, frames and grid; detached, it only
            # sets the scale.
            channel_std = latents.detach().float().std(dim=(0, 1, 3, 4), keepdim=True)
            noise = torch.randn_like(latents) * (self.noise_std * channel_std).to(latents.dtype)
            latents = latents + noise

        if self.mask_ratio > 0:
            # One mask per (sample, frame), shared by all channels of a cell.
            keep = torch.rand(B, T, 1, H, W, device=latents.device) >= self.mask_ratio
            latents = latents * keep.to(latents.dtype)

        if self.cond_dropout > 0:
            keep = torch.rand(B, 1, 1, 1, 1, device=latents.device) >= self.cond_dropout
            latents = latents * keep.to(latents.dtype)

        return latents

    def extra_repr(self):
        return (f"cond_dropout={self.cond_dropout}, mask_ratio={self.mask_ratio}, "
                f"noise_std={self.noise_std}")
