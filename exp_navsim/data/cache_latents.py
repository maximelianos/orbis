"""Run the image encoder over every episode and cache per-frame latents.

    python -m exp_navsim.data.cache_latents --config exp_navsim/config.yaml

Enumerates episodes from the long dataloader (deterministic order) and writes one
HDF5 file per episode under the cache dir (data/... by default). Each file stores
the per-frame latents plus the trajectory/velocity, so training can later read
latents from cache instead of re-encoding images:

    <episode_idx>.h5
        encoded_q_rec : (T, C, H, W)   reconstruction latents
        encoded_q_sem : (T, C, H, W)   semantic latents
        trajectory    : (T, 2)
        velocity      : (T, 2)
        attrs: frame_rate, token
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from exp_navsim.data.navsim_base import NavsimLongBase
from exp_navsim.encoder_io import build_encoder


@torch.inference_mode()
def encode_episode(encoder, images, chunk):
    """images: (T, 3, H, W) tensor -> (q_rec, q_sem) numpy (T, C, H, W)."""
    recs, sems = [], []
    for i in range(0, len(images), chunk):
        q_rec, q_sem = encoder.encode(images[i:i + chunk])
        recs.append(q_rec.float().cpu().numpy())
        sems.append(q_sem.float().cpu().numpy())
    return np.concatenate(recs, 0), np.concatenate(sems, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_navsim/config.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    # Caching needs raw front-camera images; surround cameras are not encoded.
    ds = NavsimLongBase.from_config(cfg, load_surround=False, split="all")

    out_dir = Path(NavsimLongBase.cache_dir(cfg))
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = build_encoder(**cfg.cache.encoder)

    print(f"Caching latents for {len(ds)} episodes -> {out_dir}")
    for idx in tqdm(range(len(ds))):
        out_path = out_dir / f"{idx:06d}.h5"
        if out_path.exists() and not args.overwrite:
            continue
        sample = ds[idx]
        images = torch.from_numpy(sample["images"]).float()
        q_rec, q_sem = encode_episode(encoder, images, cfg.cache["chunk"])

        with h5py.File(out_path, "w") as f:
            f.create_dataset("encoded_q_rec", data=q_rec, compression="lzf")
            f.create_dataset("encoded_q_sem", data=q_sem, compression="lzf")
            f.create_dataset("trajectory", data=sample["trajectory"], compression="lzf")
            f.create_dataset("velocity", data=sample["velocity"], compression="lzf")
            f.attrs["frame_rate"] = sample["frame_rate"]
            f.attrs["token"] = sample["metadata"]["token"]

    print("Done.")


if __name__ == "__main__":
    main()
