"""Frozen Orbis world model as a feature extractor, run in latent space only.

One forward of the STDiT backbone over [observed frames, pure-noise next frame]
at t = 1, stopped after the last probed block. The outputs of the WM_BLOCKS
blocks are LayerNormed (deeper residual streams are larger, so they would
dominate the concat) and concatenated over channels.

Only the next-frame slot is returned. The observed frames enter clean, so their
tokens still carry the stage 1 latents, and the probe could read the images from
them instead of from the world model's prediction.

Plain object, not an nn.Module: its ~700 M frozen parameters never reach a checkpoint.
"""

from pathlib import Path

import torch
import torch.nn.functional as nnf
from omegaconf import OmegaConf

from util import instantiate_from_config

# Loophole: backbone blocks (0-based, out of `depth`) whose outputs are probed.
WM_BLOCKS = (18,) # 5, 10, 15, 18


def wm_params(exp_dir, config):
    return OmegaConf.load(Path(exp_dir) / config).model.params


def feature_dim(exp_dir, config):
    """Channels of the probed features: len(WM_BLOCKS) * backbone width."""
    return len(WM_BLOCKS) * wm_params(exp_dir, config).generator_config.params.hidden_size


class WorldModelRunner:
    def __init__(self, exp_dir, ckpt, config, use_ema, device):
        p = wm_params(exp_dir, config)
        self.device = torch.device(device)
        self.timescale = float(p.get("timescale", 1.0))
        # Factorized ("ModelIF") world models read rec and sem latents concatenated, each with its own scale.
        self.enc_scale, self.enc_scale_dino = float(p.get("enc_scale", 1.0)), p.get("enc_scale_dino")

        self.net = instantiate_from_config(p.generator_config)
        path = str(Path(exp_dir) / ckpt)
        try:
            state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)["state_dict"]
        except Exception:
            state = torch.load(path, map_location="cpu")["state_dict"]
        prefix = "ema_vit." if use_ema and any(k.startswith("ema_vit.") for k in state) else "vit."
        self.net.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
        # Only the probed blocks are ever run.
        self.net.blocks = self.net.blocks[:max(WM_BLOCKS) + 1]
        self.net.eval().requires_grad_(False).to(self.device)
        print(f"[wm] {prefix[:-1]} from {exp_dir}, blocks {WM_BLOCKS}, "
              f"{sum(p.numel() for p in self.net.parameters()) / 1e6:.0f} M frozen params")

    def pack(self, q_rec, q_sem):
        """Cached latents (B, F, 16, H, W) -> world-model input space (B, F, C, H, W)."""
        x = q_rec.to(self.device, torch.float32) * self.enc_scale
        if self.enc_scale_dino is None:
            return x
        return torch.cat([x, q_sem.to(self.device, torch.float32) * float(self.enc_scale_dino)], dim=2)

    @torch.no_grad()
    def __call__(self, q_rec, q_sem, frame_rate):
        """-> (B, 1, len(WM_BLOCKS) * D, H, W) features of the next frame only."""
        ctx = self.pack(q_rec, q_sem)
        B, F, C, H, W = ctx.shape
        net = self.net
        assert tuple(net.x_embedder.grid_size) == (H, W), \
            f"world model needs a {tuple(net.x_embedder.grid_size)} latent grid, got {(H, W)}: re-cache at its resolution"
        assert F + 1 <= net.max_num_frames, f"{F + 1} frames > {net.max_num_frames} frame embeddings"

        t = torch.full((B,), self.timescale, device=self.device)
        noise = torch.randn(B, 1, C, H, W, device=self.device)
        feats = []
        with torch.autocast("cuda", torch.float16, enabled=self.device.type == "cuda"):
            x = net.preprocess_inputs(noise, ctx, t, frame_rate.to(self.device, torch.float32))  # (B, F+1, HW, D)
            c = net.get_condition_embeddings(t)
            for i, block in enumerate(net.blocks):
                x = block(x, c)
                if i in WM_BLOCKS:
                    feats.append(nnf.layer_norm(x[:, F:].float(), x.shape[-1:]))
        # Channels-last view, no copy: the denoiser projects channels per cell anyway.
        return torch.cat(feats, -1).transpose(2, 3).unflatten(-1, (H, W))
