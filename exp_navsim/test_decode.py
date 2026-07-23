"""Test the latent cache by decoding it back to images.

    python -m exp_navsim.test_decode --config exp_navsim/config.yaml --num 3

Loads the first few cached episodes, decodes the stored latents with the
tokenizer, and plots BEV + decoded camera observations + trajectory using the
shared episode visualization (exp_navsim.draw.draw_episode). One PDF page per
episode.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf

from exp_navsim.encoder_io import build_encoder
from exp_navsim.data.navsim_base import NavsimLongBase
from exp_navsim.draw import draw_episode


@torch.inference_mode()
def decode_episode(encoder, q_rec, q_sem, chunk=16):
    """q_rec/q_sem: (T, C, H, W) numpy -> decoded images (T, 3, H, W) numpy."""
    q_rec = torch.from_numpy(q_rec).float()
    q_sem = torch.from_numpy(q_sem).float()
    out = []
    for i in range(0, len(q_rec), chunk):
        dec = encoder.decode((q_rec[i:i + chunk].to(encoder.device),
                              q_sem[i:i + chunk].to(encoder.device)))
        out.append(dec.float().cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_navsim/config.yaml")
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--out", default="exp_navsim/navsim_decoded.pdf")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    encoder = build_encoder(**cfg.cache.encoder)
    cdir = NavsimLongBase.cache_dir(cfg)
    files = sorted(Path(cdir).glob("*.h5"))[:args.num]
    assert files, f"No cached latents in {cdir}; run cache_latents first."

    with PdfPages(args.out) as pdf:
        for i, path in enumerate(files):
            print(f"  decoding episode {i + 1}/{len(files)}: {path.name}")
            with h5py.File(path, "r") as f:
                q_rec = f["encoded_q_rec"][:]
                q_sem = f["encoded_q_sem"][:]
                trajectory = f["trajectory"][:]
                token = f.attrs.get("token", path.stem)
            decoded = decode_episode(encoder, q_rec, q_sem, cfg.cache.get("chunk", 16))

            batch = {
                "decoded": np.expand_dims(decoded, 0),          # (1, T, 3, H, W)
                "trajectory": np.expand_dims(trajectory, 0),    # (1, T, 2)
            }
            fig = draw_episode(batch, 0, title=f"Decoded episode {i} — {token}")
            pdf.savefig(fig, dpi=120, bbox_inches="tight"); plt.close(fig)

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
