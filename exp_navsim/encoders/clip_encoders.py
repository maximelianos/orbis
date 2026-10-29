"""Encoders that see the 4 history frames of a scene together.

    enc = build_encoder(name, device)            (frame_encoders.build_encoder)
    feats = enc(clip)          clip (B, 4, 3, 288, 512) in [-1, 1], oldest frame first
                               -> {feature_name: (B, C, 18, 32) float}

vjepa2  V-JEPA 2 ViT-L (facebook/vjepa2-vitl-fpc64-256). Tubelets of 2 frames, so the
        4 frames give 2 temporal slots; the slot of the 2 newest frames is kept.
        RoPE positions let it run at 288 x 512, which gives the same 18 x 32 grid.
wm      the frozen Orbis world model. The tokenizer encodes the 4 frames, the
        backbone runs once over [4 observed frames, pure-noise next frame] at t = 1,
        and for every block in WM_BLOCKS two token sets are returned:
            wm<b>      the next-frame slot, i.e. the model's guess of the future
            wm<b>_ctx  the last observed frame, contextualized by the history
"""

import torch

from exp_navsim.encoders.frame_encoders import imagenet_normalize

# 0-based blocks of the 24-block backbone: the ends of six equal stages.
WM_BLOCKS = (3, 7, 11, 15, 19, 23)


class VJepa2:
    def __init__(self, device, repo="facebook/vjepa2-vitl-fpc64-256"):
        from transformers import VJEPA2Model
        self.model = VJEPA2Model.from_pretrained(repo, torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.names = ("vjepa2",)

    @torch.inference_mode()
    def __call__(self, clip):
        B, T, _, H, W = clip.shape
        x = imagenet_normalize(clip.to(self.device).flatten(0, 1)).unflatten(0, (B, T))
        out = self.model(pixel_values_videos=x.to(torch.bfloat16), skip_predictor=True)
        tokens = out.last_hidden_state.float().unflatten(1, (T // 2, H // 16, W // 16))
        return {"vjepa2": tokens[:, -1].permute(0, 3, 1, 2)}


class OrbisWorldModel:
    def __init__(self, device, blocks=WM_BLOCKS, exp_dir="logs_wm/orbis_288x512",
                 tokenizer_dir="logs_tk/tokenizer_288x512"):
        from exp_navsim.encoders.orbis_io import build_encoder
        from exp_navsim.encoders.wm_runner import WorldModelRunner
        self.tokenizer = build_encoder(exp_dir=tokenizer_dir, device=device)
        self.runner = WorldModelRunner(exp_dir, "checkpoints/last.ckpt", "config.yaml", True, device,
                                       blocks=blocks)
        self.blocks = blocks
        self.names = tuple(f"wm{b}{s}" for b in blocks for s in ("", "_ctx"))

    @torch.inference_mode()
    def __call__(self, clip, frame_rate=2.0, seed=0):
        B, T = clip.shape[:2]
        q_rec, q_sem = self.tokenizer.encode(clip.flatten(0, 1))
        q_rec, q_sem = (q.float().unflatten(0, (B, T)) for q in (q_rec, q_sem))
        # Fixed noise in the next-frame slot, so a scene always gets the same features.
        generator = torch.Generator(device=self.runner.device).manual_seed(seed)
        feats = self.runner(q_rec, q_sem, torch.full((B,), frame_rate), generator=generator)
        out = {}
        for b in self.blocks:
            out[f"wm{b}"] = feats[b][:, -1]
            out[f"wm{b}_ctx"] = feats[b][:, -2]
        return out


CLIP_ENCODERS = {"vjepa2": VJepa2, "wm": OrbisWorldModel}
