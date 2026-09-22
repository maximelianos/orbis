"""Frozen Orbis world model, executed in latent space only.

Training command (repo root):
    PYTHONPATH=. python train_nuplan.py -c exp_navsim/wm/config.yaml \
        --max_steps 20000 --logdir logs_navsim_wm

This file is the only place that knows about the stage-2 world model, and it is
kept out of the trajectory model on purpose: it holds no trainable weight and
must never end up in a checkpoint.

Three things make the rollout cheap enough to sit inside a training loop:

  * Only the denoising backbone (STDiT) of the stage-2 checkpoint is built and
    loaded. The ~3 GB tokenizer that the stage-2 LightningModule normally wraps
    is skipped entirely — we feed cached latents in and read a latent out, so
    no image is ever encoded or decoded.
  * WorldModelRunner is a plain object, *not* an nn.Module. PyTorch therefore
    never registers its ~700 M frozen parameters as submodules of the Lightning
    model, and every trajectory checkpoint stays a few MB.
  * The next latent is produced by a handful of Euler steps of the same
    flow-matching ODE the world model was trained with. The step count is small
    on purpose: the imagined latent is a conditioning signal for a trajectory
    head, not a picture anybody looks at, so a coarse solve is enough.
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from util import instantiate_from_config


# --------------------------------------------------------------------------- #
# Checkpoint reading
# --------------------------------------------------------------------------- #
def _load_state_dict(path):
    """`state_dict` of a Lightning checkpoint, memory-mapped when torch allows.

    The stage-2 checkpoint is ~9 GB but only the backbone (~1/10 of it) is ever
    copied out, so mmap avoids pulling the whole file into RAM. Lightning
    checkpoints also carry pickled non-tensor payloads, which `weights_only`
    refuses — hence the plain fallback.

    Args:
        path: path to the .ckpt file.
    """
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)["state_dict"]
    except Exception:
        return torch.load(path, map_location="cpu")["state_dict"]


def _submodule_state(state, prefix):
    """The keys of one submodule of a state dict, with `prefix` stripped off."""
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def _check_input(net, num_frames, height, width):
    """Refuse latents the backbone cannot process, with a fix in the message.

    The released backbone is Swin-based (SwinSTDiTNoExtraMLP): every block bakes
    in one latent resolution — the window partitioning, the shifted-window
    attention masks and the relative-position tables are all built for it — and
    the frame embeddings are built for one maximum frame count. There is no
    honest way to feed it another grid: rebuilding the position embedding alone
    leaves the window masks wrong, and the mismatch then surfaces much later as
    an opaque assert inside the Swin block. So it is caught here instead, where
    the message can say what to change.

    Args:
        net:           the backbone.
        num_frames:    frames a single forward pass will see (context + target).
        height, width: latent grid being fed in.
    """
    grid = tuple(net.x_embedder.grid_size)
    if grid != (height, width):
        patch = net.x_embedder.patch_size[0]
        raise ValueError(
            f"world model expects a {grid[0]}x{grid[1]} latent grid, the cache holds "
            f"{height}x{width}. Re-cache at the world model's own resolution: set "
            f"data_long.<dataset>.size to [{grid[0] * patch}, {grid[1] * patch}] and point "
            f"cache.dir at a fresh directory in exp_navsim/wm/config.yaml, then run "
            f"`python -m exp_navsim.data.cache_latents --config exp_navsim/wm/config.yaml`."
        )
    if num_frames > net.max_num_frames:
        raise ValueError(
            f"world model has {net.max_num_frames} frame embeddings but was given "
            f"{num_frames} frames (context_images + 1). Lower model.params.context_images."
        )


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
class WorldModelRunner:
    """Frozen stage-2 world model used as a next-latent predictor."""

    def __init__(self, exp_dir, ckpt, config, num_steps, use_ema, device):
        """
        Args:
            exp_dir:   stage-2 experiment directory (holds `config` and `ckpt`).
            ckpt:      checkpoint path relative to exp_dir.
            config:    stage-2 config path relative to exp_dir. Read for the
                       backbone architecture and the latent scales; it must be
                       the config the checkpoint was trained with.
            num_steps: Euler steps of the latent ODE per imagined frame. Small
                       (single digits) — this is a conditioning signal, not an
                       image, and the cost is one backbone forward per step.
            use_ema:   take the EMA copy of the backbone weights (what the
                       released sampler uses) instead of the raw ones.
            device:    device string the backbone is moved to.
        """
        cfg = OmegaConf.load(Path(exp_dir) / config)
        params = cfg.model.params

        self.device = torch.device(device)
        self.num_steps = num_steps
        # Stage-2 rescales the diffusion time before it reaches the backbone.
        self.timescale = float(params.get("timescale", 1.0))

        # Latent scaling used while the world model was trained. A "ModelIF"
        # world model consumes the *factorized* tokenizer: the two cached
        # latents, each with its own scale, concatenated over channels
        # (16 + 16 -> 32). A plain "Model" consumes the single reconstruction
        # latent. The presence of `enc_scale_dino` is what tells the two apart.
        self.enc_scale = float(params.get("enc_scale", 1.0))
        dino = params.get("enc_scale_dino", None)
        self.enc_scale_dino = None if dino is None else float(dino)

        self.net = instantiate_from_config(params.generator_config)
        state = _load_state_dict(str(Path(exp_dir) / ckpt))
        prefix = "ema_vit." if use_ema and any(k.startswith("ema_vit.") for k in state) else "vit."
        self.net.load_state_dict(_submodule_state(state, prefix), strict=True)
        self.net.eval().requires_grad_(False)
        self.net.to(self.device)
        print(f"[wm] loaded {prefix[:-1]} from {exp_dir} "
              f"({sum(p.numel() for p in self.net.parameters()) / 1e6:.0f} M frozen params)")

    @property
    def latent_channels(self):
        """Channel count of the world model's latent space (16 or 16+16)."""
        return self.net.in_channels

    # ------------------------------------------------------------------ #
    # Latent space
    # ------------------------------------------------------------------ #
    def pack(self, q_rec, q_sem):
        """Cached tokenizer latents -> world-model input space.

        Mirrors `ModelIF.encode_frames` / `Model.encode_frames` exactly, minus
        the tokenizer call: the caching script already stored both latents.

        Args:
            q_rec, q_sem: (B, F, 16, H, W) reconstruction / semantic latents.
        Returns:
            (B, F, C, H, W) with C = `latent_channels`.
        """
        q_rec = q_rec.to(self.device, torch.float32)
        if self.enc_scale_dino is None:
            return q_rec * self.enc_scale
        q_sem = q_sem.to(self.device, torch.float32)
        return torch.cat([q_rec * self.enc_scale, q_sem * self.enc_scale_dino], dim=2)

    # ------------------------------------------------------------------ #
    # Rollout
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def imagine(self, context, frame_rate, num_frames):
        """Denoise future latents after `context`, autoregressively.

        Args:
            context:    (B, F, C, H, W) observed latents, in world-model space.
            frame_rate: (B,) Hz of the sequence — the backbone conditions on it,
                        so it must match the spacing of the context frames.
            num_frames: how many latents to imagine. One is the interesting
                        case; the loop exists so rolling further is a config
                        change rather than a code change.
        Returns:
            (B, F + num_frames, C, H, W): context and imagination in one tensor,
            which is exactly what the trajectory aggregator consumes.
        """
        # The conditioning window keeps its length below, so one check covers
        # every iteration: the backbone always sees len(window) + 1 frames.
        _check_input(self.net, context.shape[1] + 1, *context.shape[-2:])
        frame_rate = frame_rate.to(self.device, torch.float32)

        out = context
        window = context
        for _ in range(num_frames):
            imagined = self._denoise_next(window, frame_rate)
            out = torch.cat([out, imagined], dim=1)
            # Slide the conditioning window, exactly as the world model's own
            # roll_out does: the backbone has a fixed number of frame embeddings
            # and must never be asked for more frames than it was trained with.
            window = torch.cat([window[:, 1:], imagined], dim=1)
        return out

    def _denoise_next(self, context, frame_rate):
        """One flow-matching ODE solve for the single frame after `context`.

        Same schedule as the world model's own sampler (Euler, t: 1 -> 0, the
        network predicting -dx_t/dt) minus the image decoding — we keep the
        latent. Deterministic (no stochastic term), so one batch yields one
        consistent imagined future and the trajectory head, not the world model,
        stays the only source of multimodality.

        Returns:
            (B, 1, C, H, W) the imagined next latent.
        """
        b, _, c, h, w = context.shape
        x = torch.randn(b, 1, c, h, w, device=self.device)
        t_steps = torch.linspace(1, 0, self.num_steps + 1, device=self.device)

        with torch.autocast("cuda", torch.float16, enabled=self.device.type == "cuda"):
            for i in range(self.num_steps):
                t = t_steps[i].repeat(b)
                v = self.net(x, context, t=t * self.timescale, frame_rate=frame_rate)
                x = x + v.float() * (t_steps[i] - t_steps[i + 1])
        return x.float()

    def __call__(self, q_rec, q_sem, frame_rate, num_frames):
        """Cached latents -> [observed frames, imagined frames] in WM space."""
        return self.imagine(self.pack(q_rec, q_sem), frame_rate, num_frames)
