"""Frozen image encoders compared by the probe, behind one interface.

    enc = build_encoder(name, device)
    feats = enc(images)        images (N, 3, 288, 512) in [-1, 1]
                               -> {feature_name: (N, C, 18, 32) float}

Per-frame encoders see one image at a time:
    orbis   the Orbis stage-1 tokenizer, two 16-channel latents: q_sem (pushed towards
            DINOv2 features) and q_rec (the detail branch)
    dinov2  DINOv2 ViT-B/14, last-layer patch tokens after the final norm, 768 channels;
            the image is resized to 252 x 448 so the 14 px patches give the same
            18 x 32 grid as the tokenizer

Clip encoders see the 4 history frames of a scene at once, see clip_encoders.py.
"""

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(images):
    """[-1, 1] -> ImageNet-normalized."""
    mean = images.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = images.new_tensor(IMAGENET_STD)[:, None, None]
    return ((images + 1) / 2 - mean) / std


class OrbisTokenizer:
    names = ("q_sem", "q_rec")

    def __init__(self, device, exp_dir="logs_tk/tokenizer_288x512"):
        from exp_navsim.encoders.orbis_io import build_encoder
        self.encoder = build_encoder(exp_dir=exp_dir, device=device)

    def __call__(self, images):
        q_rec, q_sem = self.encoder.encode(images)
        return {"q_sem": q_sem.float(), "q_rec": q_rec.float()}


class DinoV2:
    names = ("dinov2",)

    def __init__(self, device, arch="dinov2_vitb14"):
        self.model = torch.hub.load("facebookresearch/dinov2", arch, source="github",
                                    skip_validation=True).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, images):
        x = F.interpolate(images.to(self.device), size=(252, 448), mode="bilinear", antialias=True)
        with torch.autocast("cuda", torch.bfloat16):
            tokens = self.model.forward_features(imagenet_normalize(x))["x_norm_patchtokens"]
        return {"dinov2": tokens.float().transpose(1, 2).unflatten(-1, (18, 32))}


FRAME_ENCODERS = {"orbis": OrbisTokenizer, "dinov2": DinoV2}


def build_encoder(name, device="cuda"):
    if name in FRAME_ENCODERS:
        return FRAME_ENCODERS[name](device)
    from exp_navsim.encoders.clip_encoders import CLIP_ENCODERS
    return CLIP_ENCODERS[name](device)
