"""Windowed datasets for training.

Both the cached-latent and the raw-image dataset sample a `num_frames` window
from each (variable length) episode. With `num_frames=0` the whole available
episode is used instead, so window lengths differ per sample and the batch has
to be assembled by exp_navsim.episode.collate.pad_collate. In that mode
`random_start_min_frames` > 0 makes a training window start at a random frame
and run to the end of the episode (validation, being deterministic, always
starts at 0). A window that does not start at 0 is rotated into the ego frame
of its own first frame, see `_window_traj`.

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


def _random_start(total, min_frames):
    """Uniform start in [0, total - min_frames], so at least `min_frames` remain.

    Drawn with torch, not the dataset's numpy rng: DataLoader seeds torch
    differently in every worker, while each worker's copy of `self.rng` starts
    from the same seed and would hand every worker the same starts.
    """
    if total <= min_frames:
        return 0
    return int(torch.randint(0, total - min_frames + 1, ()).item())


def _heading_at(traj, s, min_disp=0.5):
    """Heading (rad) of the ego at frame `s`, estimated from the path itself.

    The cache holds (x, y) only, no yaw. A car moving forward points along its
    path, so the heading is the direction of the central difference
    traj[s + k] - traj[s - k], with k widened until the ego has moved at least
    `min_disp` metres (a stopped car has no direction to read). An ego that
    never moves gets 0, i.e. keeps the frame-0 heading, which is harmless since
    its whole path sits at the origin.
    """
    T = len(traj)
    for k in range(1, T):
        lo, hi = max(s - k, 0), min(s + k, T - 1)
        d = traj[hi] - traj[lo]
        if np.hypot(d[0], d[1]) >= min_disp:
            return float(np.arctan2(d[1], d[0]))
        if lo == 0 and hi == T - 1:
            break
    return 0.0


def _window_traj(trajectory, start, num_frames):
    """Slice a window into the ego frame of its first frame; return (traj_local, velocity).

    The cached trajectory is local to episode frame 0, origin and heading. A
    window starting there only needs the slice. A later window is re-origined
    at its first frame and rotated into that frame's heading, so x is forward
    in the image the window's first latent was taken from; a translation alone
    would leave it in frame 0's heading, which is wrong after any turn.
    """
    full = np.asarray(trajectory, dtype=np.float32)
    traj = full[start:start + num_frames] - full[start]
    if start > 0:
        theta = _heading_at(full, start)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, s], [-s, c]], dtype=np.float32)     # R(-theta)
        traj = traj @ rot.T
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

    def choose_window(self, total):
        """(start, num_frames) of the window taken from an episode of `total` frames.

        With num_frames=0 and `random_start_min_frames` > 0 (training only: a
        deterministic dataset, i.e. validation, always starts at 0), the window
        runs from a random start to the end of the episode, keeping at least
        `random_start_min_frames` frames. The images are read from that start, so
        one scene gives a different (image, trajectory) pair every time it is
        drawn instead of always the same one.
        """
        if (self.num_frames == 0 and self.random_start_min_frames > 0
                and not self.deterministic):
            start = _random_start(total, self.random_start_min_frames)
            return start, total - start
        num_frames = self.window_length(total)
        return _window_start(total, num_frames, self.rng, self.deterministic), num_frames

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
                 context_images=None, random_start_min_frames=0):
        super().__init__()
        from pathlib import Path
        self.num_frames = num_frames
        self.context_images = context_images
        self.random_start_min_frames = random_start_min_frames
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
            start, num_frames = self.choose_window(total)
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
                 random_start_min_frames=0, **long_kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.context_images = context_images
        self.random_start_min_frames = random_start_min_frames
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
        start, num_frames = self.choose_window(total)
        img_sl = slice(start, start + self.image_length(num_frames))
        traj, vel = _window_traj(sample["trajectory"], start, num_frames)
        return _to_tensors({
            "images": images[img_sl],
            "trajectory": traj, "velocity": vel, "frame_rate": sample["frame_rate"],
            "metadata": {"length": len(traj)},
        })
