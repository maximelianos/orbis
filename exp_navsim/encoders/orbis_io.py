"""Image encoder load / encode / decode, kept in one place (like encoder.py).

Thin wrapper around the project's frozen tokenizer (loaders_for_projects.Encoder).
Importing that class is allowed; this file exists so every consumer (caching
script, model, decode test) shares one construction path and the same
encode/decode contract:

    encode(images)  -> (q_rec, q_sem)   each (B, C_lat, H_lat, W_lat)
    decode((q_rec, q_sem)) -> images    (B, 3, H, W)
"""

import timm

from loaders_for_projects.encoder import Encoder

_create_model = timm.create_model


def _create_model_compat(name, *args, **kwargs):
    # The tokenizer config names its ViT "timm/vit_base_patch16_224.mae"; timm >= 1.0.20
    # reads that as a Hub id. The weights come from the Orbis checkpoint anyway.
    if isinstance(name, str) and name.startswith("timm/"):
        name = name[len("timm/"):]
    return _create_model(name, *args, **kwargs)


timm.create_model = _create_model_compat


def build_encoder(exp_dir, ckpt="checkpoints/last.ckpt", config="config.yaml", device="cuda"):
    """Construct the frozen image tokenizer used for latent caching / decoding."""
    return Encoder(exp_dir=exp_dir, ckpt=ckpt, config=config, device=device)
