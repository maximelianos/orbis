"""Image encoder load / encode / decode, kept in one place (like encoder.py).

Thin wrapper around the project's frozen tokenizer (loaders_for_projects.Encoder).
Importing that class is allowed; this file exists so every consumer (caching
script, model, decode test) shares one construction path and the same
encode/decode contract:

    encode(images)  -> (q_rec, q_sem)   each (B, C_lat, H_lat, W_lat)
    decode((q_rec, q_sem)) -> images    (B, 3, H, W)
"""

from loaders_for_projects.encoder import Encoder


def build_encoder(exp_dir, ckpt="checkpoints/last.ckpt", config="config.yaml", device="cuda"):
    """Construct the frozen image tokenizer used for latent caching / decoding."""
    return Encoder(exp_dir=exp_dir, ckpt=ckpt, config=config, device=device)
