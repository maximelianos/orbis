"""Fixed-length windowed datasets for training.

The model consumes fixed-length windows (num_frames), so both the cached-latent
and the raw-image datasets sample a `num_frames` window from each (variable
length) episode. They share the trajectory/velocity helpers and the train/val
split with the long dataloader to write as little code as possible.

Two datasets, selected by the `data.mode` config flag:
  * NavsimLatentDataset     — reads cached HDF latents (encoded mode)
  * NavsimRawWindowDataset  — reads raw front-camera images (raw mode; the model
                              encodes on-the-fly)

Both return, per index:
    "velocity":   (num_frames, 2)      normalized inside the model
    "trajectory": (num_frames, 2)      window-local (starts at origin)
    "frame_rate": int
  encoded mode additionally: "encoded_q_sem"/"encoded_q_rec" (num_frames, C, H, W)
  raw mode additionally:     "images" (num_frames, 3, H, W)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from exp_navsim.data.navsim_long import NavsimLongDataset, traj_to_velocity, in_split


def _window_start(total, num_frames, rng, deterministic):
    """Choose a window start so [start, start+num_frames) fits in `total`."""
    if total <= num_frames:
        return 0
    return 0 if deterministic else int(rng.integers(0, total - num_frames + 1))


def _window_traj(trajectory, start, num_frames):
    """Slice a window and re-origin it; return (traj_local, velocity)."""
    traj = np.asarray(trajectory, dtype=np.float32)[start:start + num_frames]
    traj = traj - traj[0]                       # window-local coordinates
    return traj, traj_to_velocity(traj)


def _to_tensors(sample):
    return {k: (torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v)
            for k, v in sample.items()}


class NavsimLatentDataset(Dataset):
    """Windows of cached per-frame latents (encoded training mode)."""

    def __init__(self, cache_dir, num_frames, split="all", val_fraction=0.1,
                 latent_key="encoded_q_sem", seed=0, deterministic=False):
        super().__init__()
        from pathlib import Path
        self.num_frames = num_frames
        self.latent_key = latent_key
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

        files = sorted(Path(cache_dir).glob("*.h5"))
        print(f"loading data from {cache_dir}, detected {len(files)} h5 files, split {split}")
        self.files = []
        for p in files:
            with h5py.File(p, "r") as f:
                token = f.attrs.get("token", p.stem)
                length = f[latent_key].shape[0]
                #print(f"token {token}, length {length}")
            #print(f"length {length}, num_frames {num_frames}, in_split {in_split(token, split, val_fraction)}")
            if length >= num_frames and in_split(token, split, val_fraction):
                self.files.append(p)
        print(f"episodes {len(self.files)}")
        assert self.files, f"No cached episodes in {cache_dir} for split={split}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with h5py.File(self.files[idx % len(self.files)], "r") as f:
            total = f[self.latent_key].shape[0]
            start = _window_start(total, self.num_frames, self.rng, self.deterministic)
            sl = slice(start, start + self.num_frames)
            q_sem = f["encoded_q_sem"][sl]
            q_rec = f["encoded_q_rec"][sl]
            trajectory = f["trajectory"][:]
            frame_rate = int(f.attrs.get("frame_rate", 2))
        traj, vel = _window_traj(trajectory, start, self.num_frames)
        return _to_tensors({
            "encoded_q_sem": q_sem, "encoded_q_rec": q_rec,
            "trajectory": traj, "velocity": vel, "frame_rate": frame_rate,
        })


class NavsimRawWindowDataset(Dataset):
    """Windows of raw front-camera images (raw training mode)."""

    def __init__(self, num_frames, seed=0, deterministic=False, **long_kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        long_kwargs["load_surround"] = False
        self.source = NavsimLongDataset(**long_kwargs)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        sample = self.source[idx % len(self.source)]
        images = sample["images"]                      # (T, 3, H, W)
        total = len(images)
        start = _window_start(total, self.num_frames, self.rng, self.deterministic)
        sl = slice(start, start + self.num_frames)
        traj, vel = _window_traj(sample["trajectory"], start, self.num_frames)
        return _to_tensors({
            "images": images[sl],
            "trajectory": traj, "velocity": vel, "frame_rate": sample["frame_rate"],
        })
