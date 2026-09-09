"""Windowed datasets for training.

Both the cached-latent and the raw-image dataset sample a `num_frames` window
from each (variable length) episode. With `num_frames=0` the whole available
episode is used instead, so window lengths differ per sample and the batch has
to be assembled by exp_navsim.data.collate.pad_collate.

Only the leading `context_images` frames ever reach the image encoder (the
denoiser zeroes the rest), so that is all the latents/images that are read from
disk — the trajectory is always read in full, it is cheap.

Two datasets, selected by the `data.mode` config flag:
  * NavsimLatentDataset     — reads cached HDF latents (encoded mode)
  * NavsimRawWindowDataset  — reads raw front-camera images (raw mode; the model
                              encodes on-the-fly)

Both return, per index (T = window length, T_img = image-context length):
    "velocity":   (T, 2)      normalized inside the model
    "trajectory": (T, 2)      window-local (starts at origin)
    "frame_rate": int
    "metadata":   {"length": T}    valid (unpadded) window length
  encoded mode additionally: "encoded_q_sem"/"encoded_q_rec" (T_img, C, H, W)
  raw mode additionally:     "images" (T_img, 3, H, W)
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


class _WindowDatasetMixin:
    """Window-length bookkeeping shared by the latent and the raw dataset.

    num_frames=0 means "the whole episode", so the window length is only known
    per sample; `context_images=None` means "every frame carries an image".
    """

    def window_length(self, total):
        """Number of frames taken from an episode of `total` frames."""
        return total if self.num_frames == 0 else min(self.num_frames, total)

    def image_length(self, window):
        """Number of leading frames of a `window` whose images are read."""
        return window if self.context_images is None else min(self.context_images, window)

    @property
    def min_episode_frames(self):
        """Shortest episode this dataset can serve a window from."""
        return max(self.num_frames, 2)     # 2 frames = 1 velocity step


class NavsimLatentDataset(_WindowDatasetMixin, Dataset):
    """Windows of cached per-frame latents (encoded training mode)."""

    def __init__(self, cache_dir, num_frames, split="all", val_fraction=0.1,
                 latent_key="encoded_q_sem", seed=0, deterministic=False,
                 context_images=None):
        super().__init__()
        from pathlib import Path
        self.num_frames = num_frames
        self.context_images = context_images
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
            if length >= self.min_episode_frames and in_split(token, split, val_fraction):
                self.files.append(p)
        print(f"episodes {len(self.files)}")
        assert self.files, f"No cached episodes in {cache_dir} for split={split}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with h5py.File(self.files[idx % len(self.files)], "r") as f:
            total = f[self.latent_key].shape[0]
            num_frames = self.window_length(total)
            start = _window_start(total, num_frames, self.rng, self.deterministic)
            # Read only the latents the model actually looks at (h5py slices on disk).
            img_sl = slice(start, start + self.image_length(num_frames))
            q_sem = f["encoded_q_sem"][img_sl]
            q_rec = f["encoded_q_rec"][img_sl]
            trajectory = f["trajectory"][:]
            frame_rate = int(f.attrs.get("frame_rate", 2))
        traj, vel = _window_traj(trajectory, start, num_frames)
        return _to_tensors({
            "encoded_q_sem": q_sem, "encoded_q_rec": q_rec,
            "trajectory": traj, "velocity": vel, "frame_rate": frame_rate,
            "metadata": {"length": len(traj)},
        })


class NavsimRawWindowDataset(_WindowDatasetMixin, Dataset):
    """Windows of raw front-camera images (raw training mode)."""

    def __init__(self, num_frames, seed=0, deterministic=False, context_images=None,
                 **long_kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.context_images = context_images
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
        num_frames = self.window_length(total)
        start = _window_start(total, num_frames, self.rng, self.deterministic)
        img_sl = slice(start, start + self.image_length(num_frames))
        traj, vel = _window_traj(sample["trajectory"], start, num_frames)
        return _to_tensors({
            "images": images[img_sl],
            "trajectory": traj, "velocity": vel, "frame_rate": sample["frame_rate"],
            "metadata": {"length": len(traj)},
        })
