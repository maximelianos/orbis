"""Trajectory model conditioned on an imagined world-model latent.

Training command (repo root):
    PYTHONPATH=. python train_nuplan.py -c exp_navsim/wm/config.yaml \
        --max_steps 20000 --logdir logs_navsim_wm

The idea, end to end:

    cached tokenizer latents of the observed frames
        -> frozen Orbis world model denoises the NEXT latent (a few Euler steps)
        -> [observed latents, imagined latent] -> latent aggregator -> tokens
        -> flow-matching trajectory diffusion, exactly as in the basic model

Only the middle two arrows are new. Everything else — the flow-matching
schedule, noising, the sampling ODE, the EMA velocity normalisation and the
validation metrics — is inherited unchanged from
exp_navsim.model.NavsimTrajectoryModel, so the two models are directly
comparable: same loss, same metrics, same curves.

The split across files is deliberate:
    wm_runner.py  world-model execution (frozen, no trainable weight)
    denoiser.py   the trainable architecture
    model.py      this file: wiring, and nothing else
"""

import torch

from exp_navsim.model import NavsimTrajectoryModel
from exp_navsim.wm.denoiser import WmTrajectoryDenoiser
from exp_navsim.wm.wm_runner import WorldModelRunner


class WmTrajectoryModel(NavsimTrajectoryModel):
    """Flow-matching trajectory model conditioned on world-model latents."""

    def __init__(self,
                 # --- inherited flow-matching core -------------------------
                 hidden_dim, image_encoder_channels, sigma_min, warmup_steps,
                 min_lr_multiplier, context_traj, context_images, image_tokens,
                 num_diffusion_steps, num_val_samples,
                 # --- frozen world model -----------------------------------
                 wm_exp_dir, wm_ckpt, wm_config, wm_steps, wm_use_ema,
                 wm_imagined_frames, wm_frame_rate,
                 # --- denoiser architecture --------------------------------
                 num_layers, num_unique_layers, dropout, cond_dropout):
        """
        Args:
            hidden_dim:             model width D of the trajectory denoiser.
            image_encoder_channels: channels of the *world model's* latent space
                                    (16 for a single-latent world model, 32 for
                                    the factorized one: rec + sem). Must equal
                                    WorldModelRunner.latent_channels; it is
                                    checked on the first batch.
            sigma_min:              flow-matching sigma floor.
            warmup_steps:           linear LR warmup before the cosine decay.
            min_lr_multiplier:      floor of the cosine decay.
            context_traj:           leading frames whose velocity is given, not
                                    predicted. 0 = predict the whole episode.
            context_images:         observed frames read from the latent cache.
                                    Keep equal to the dataset's context_images.
            image_tokens:           tokens the aggregator extracts per latent
                                    frame.
            num_diffusion_steps:    Euler steps of the *trajectory* sampling ODE.
            num_val_samples:        trajectories sampled per validation context.
            wm_exp_dir:             stage-2 experiment dir (config + checkpoint).
            wm_ckpt:                checkpoint path relative to wm_exp_dir.
            wm_config:              config path relative to wm_exp_dir.
            wm_steps:               Euler steps of the *latent* ODE per imagined
                                    frame. Kept small: the imagined latent only
                                    has to be informative, not photographic, and
                                    each step is one backbone forward pass.
            wm_use_ema:             use the world model's EMA weights.
            wm_imagined_frames:     latents imagined past the observed frames.
            wm_frame_rate:          frame rate the world model conditions on;
                                    null = the dataset's own rate.
            num_layers:             cross-attention passes in the denoiser.
            num_unique_layers:      distinct blocks those passes cycle over
                                    (< num_layers ties weights).
            dropout:                dropout inside every attention block.
            cond_dropout:           probability of blanking a sample's latent
                                    conditioning during training.
        """
        super().__init__(
            hidden_dim=hidden_dim,
            image_encoder_channels=image_encoder_channels,
            sigma_min=sigma_min,
            warmup_steps=warmup_steps,
            min_lr_multiplier=min_lr_multiplier,
            context_traj=context_traj,
            context_images=context_images,
            image_tokens=image_tokens,
            num_diffusion_steps=num_diffusion_steps,
            num_val_samples=num_val_samples,
            # This model always reads cached latents: it needs *both* cached
            # tensors (q_rec and q_sem) to rebuild the world model's input, so
            # the on-the-fly encoding path does not apply and its tokenizer is
            # never constructed.
            encode_images=False,
        )
        # The base recorded its own arguments; record this signature too, so
        # load_from_checkpoint can rebuild the model (it merges, it does not
        # replace). Needed by exp_navsim/test_model.py.
        self.save_hyperparameters()

        # The base __init__ built its own denoiser; swap in the compact one.
        # Done here instead of through a hook so exp_navsim/model.py stays
        # untouched — it costs one discarded construction at startup and
        # nothing at run time, and the discarded module never reaches a
        # checkpoint because the replacement happens before any state is loaded.
        self.denoiser = WmTrajectoryDenoiser(
            hidden_dim=hidden_dim,
            latent_channels=image_encoder_channels,
            latent_frames=context_images + wm_imagined_frames,
            latent_tokens=image_tokens,
            context_traj=context_traj,
            num_layers=num_layers,
            num_unique_layers=num_unique_layers,
            dropout=dropout,
            cond_dropout=cond_dropout,
        )

        self.latent_channels = image_encoder_channels
        self.wm_exp_dir = wm_exp_dir
        self.wm_ckpt = wm_ckpt
        self.wm_config = wm_config
        self.wm_steps = wm_steps
        self.wm_use_ema = wm_use_ema
        self.wm_imagined_frames = wm_imagined_frames
        self.wm_frame_rate = wm_frame_rate
        self._world_model = None

    # ------------------------------------------------------------------ #
    # The frozen world model
    # ------------------------------------------------------------------ #
    @property
    def world_model(self):
        """Frozen world model, built on first use.

        Lazy for the same reason the base model's tokenizer is: it lands on the
        right device, and a checkpoint can be loaded (or the module constructed
        on a machine without the stage-2 weights) without paying for a 9 GB
        read. Held in a plain attribute of a non-Module object, so its ~700 M
        frozen parameters never join this model's parameters or state dict.
        """
        if self._world_model is None:
            self._world_model = WorldModelRunner(
                exp_dir=self.wm_exp_dir,
                ckpt=self.wm_ckpt,
                config=self.wm_config,
                num_steps=self.wm_steps,
                use_ema=self.wm_use_ema,
                device=self.device,
            )
            assert self._world_model.latent_channels == self.latent_channels, (
                f"image_encoder_channels={self.latent_channels} but the world model's "
                f"latent space has {self._world_model.latent_channels} channels")
        return self._world_model

    def world_model_frame_rate(self, batch):
        """(B,) frame rate the world model conditions on.

        `wm_frame_rate: null` uses the dataset's own rate (2 Hz for NAVSIM). Set
        it explicitly to run the world model nearer the rate it was trained at,
        should the imagined latent degenerate at the data's rate.
        """
        batch_size = batch["velocity"].shape[0]
        if self.wm_frame_rate is None:
            return batch["frame_rate"].reshape(batch_size).float()
        return torch.full((batch_size,), float(self.wm_frame_rate))

    # ------------------------------------------------------------------ #
    # Conditioning (the one method that differs from the basic model)
    # ------------------------------------------------------------------ #
    def get_encoded_q(self, batch):
        """Conditioning tokens for the denoiser — the whole idea of this model.

            cached latents of the observed frames
              -> packed into the world model's latent space
              -> the world model denoises `wm_imagined_frames` next latents
              -> [observed, imagined] -> aggregator -> (B, F*N, D) tokens

        The base class calls this once per loss evaluation and once per sampled
        trajectory, and hands the result straight to the denoiser, which is why
        overriding it is all that is needed to change the conditioning.

        The world-model rollout is frozen and costs one backbone forward per
        Euler step, so it is memoized on the batch dict: validation samples the
        same context `num_val_samples` times and would otherwise repeat an
        identical rollout every time. Sharing one imagined future across those
        samples is also the behaviour we want — the spread between them should
        come from the trajectory diffusion, not from the world model
        re-imagining the scene.
        """
        latents = batch.get("wm_latents")
        if latents is None:
            observed = slice(None, self.context_images)
            latents = self.world_model(
                q_rec=batch["encoded_q_rec"][:, observed],
                q_sem=batch["encoded_q_sem"][:, observed],
                frame_rate=self.world_model_frame_rate(batch),
                num_frames=self.wm_imagined_frames,
            )
            batch["wm_latents"] = latents
        # Aggregation is trainable, so it stays outside the memoization.
        return self.denoiser.encode_latents(latents.to(self.device))
