"""NavsimTrajectoryModel probing frozen world-model features instead of tokenizer latents.

    cached latents of the observed frames
      -> frozen Orbis backbone, one pass, outputs of blocks WM_BLOCKS (wm_runner.py)
      -> LatentDenoiser over the next-frame slot only, as the basic model otherwise;
         the observed frames' tokens still hold the stage 1 latents, so they are not passed on

Everything else (trajectory AE, flow matching, sampling, metrics) is inherited.
"""

import torch

from exp_navsim.denoiser import LatentDenoiser
from exp_navsim.model import NavsimTrajectoryModel
from exp_navsim.wm.wm_runner import WorldModelRunner, feature_dim


class WmTrajectoryModel(NavsimTrajectoryModel):
    def __init__(self, wm_exp_dir, wm_ckpt="checkpoints/last.ckpt", wm_config="config.yaml",
                 wm_use_ema=True, wm_frame_rate=None, **kwargs):
        # Plain dicts: hparams are collected from this frame too, and a DictConfig
        # would not load back with weights_only=True.
        kwargs.update({k: dict(kwargs.get(k) or {}) for k in ("traj_ae", "traj_poly", "traj_pure", "regularization")})
        kwargs.update(encode_images=False, image_encoder_channels=feature_dim(wm_exp_dir, wm_config) // 2)
        super().__init__(**kwargs)
        hp = self.hparams
        # LatentDenoiser doubles image_encoder_channels (q_sem + q_rec), hence the // 2 above.
        # One frame: the backbone's features at the next-frame slot.
        self.denoiser = LatentDenoiser(self.traj_ae.latent_dim, hp.hidden_dim, hp.image_encoder_channels,
                                       1, hp.image_tokens, hp.ff_dim, hp.num_heads)
        self._world_model = None

    @property
    def world_model(self):
        """Built on first use, on the right device; never part of the state dict."""
        if self._world_model is None:
            hp = self.hparams
            self._world_model = WorldModelRunner(hp.wm_exp_dir, hp.wm_ckpt, hp.wm_config, hp.wm_use_ema, self.device)
        return self._world_model

    def get_encoded_q(self, batch):
        """(B, 1, len(WM_BLOCKS) * D, H, W), memoized on the batch for val sampling."""
        if "wm_features" not in batch:
            n, B = self.context_images, batch["velocity"].shape[0]
            rate = self.hparams.wm_frame_rate
            rate = batch["frame_rate"].reshape(B).float() if rate is None else torch.full((B,), float(rate))
            batch["wm_features"] = self.world_model(batch["encoded_q_rec"][:, :n], batch["encoded_q_sem"][:, :n], rate)
        return batch["wm_features"]
