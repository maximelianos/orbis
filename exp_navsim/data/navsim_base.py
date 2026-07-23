"""Shared core for the NAVSIM long (whole-episode) dataloaders.

The navhard and navtrain datasets differ *only* in how episodes are enumerated
and where their frame dicts come from — the per-frame format is identical
(same keys from navsim.common.dataclasses). Everything that is common lives here
so each concrete dataset stays a few lines:

    NavsimLongBase           — image transform, camera reading, frames -> sample
                               dict, subsampling, __len__/__getitem__, split.
    NavsimLongDataset        — navhard_two_stage (navsim_long.py)
    NavtrainLongDataset      — navtrain logs (navtrain_long.py)

A subclass only has to:
  * call super().__init__(sensor_root=..., ...)
  * populate self.episodes  (a list of opaque per-episode handles)
  * implement _episode_frames(handle) -> (frames_sorted, meta_dict)

The return dict (see navsim_long.py for the full contract) is:
    "images"     (T, 3, H, W)     front camera, [-1, 1]
    "cameras"    (T, n_cam, ...)   surround cameras (only if load_surround)
    "trajectory" (T, 2)           first-frame-local ego path
    "velocity"   (T, 2)           finite differences of trajectory
    "frame_rate" int
    "metadata"   {"path": ..., "token": ...}
"""

import hashlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Surround cameras used for the birds-eye-view visualization (front-centered order).
SURROUND_CAMERAS = ["CAM_L1", "CAM_L0", "CAM_F0", "CAM_R0", "CAM_R1"]
FRONT_CAMERA = "CAM_F0"


# --------------------------------------------------------------------------- #
# Reusable geometry / velocity helpers (also imported by the latent dataset)
# --------------------------------------------------------------------------- #
def quat_to_yaw(q):
    """Yaw (rotation about z) from a quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def poses_to_local_traj(global_poses):
    """Convert global (x, y, yaw) poses to (x, y) in the first frame's local frame.

    Args:
        global_poses: (T, 3) array of [x, y, yaw] in the global frame.
    Returns:
        (T, 2) array of local (x, y) positions, with frame 0 at the origin.
    """
    global_poses = np.asarray(global_poses, dtype=np.float64)
    x0, y0, th0 = global_poses[0]
    d = global_poses[:, :2] - np.array([x0, y0])
    c, s = np.cos(th0), np.sin(th0)
    # Rotate global displacement by -th0 into the first-frame heading.
    rot = np.array([[c, s], [-s, c]])
    return (d @ rot.T).astype(np.float32)


def traj_to_velocity(trajectory):
    """v_t = x_t - x_{t-1}, with v_0 = 0. Args/returns: (T, 2)."""
    traj = np.asarray(trajectory)
    v = np.zeros_like(traj)
    v[1:] = traj[1:] - traj[:-1]
    return v.astype(np.float32)


def build_image_transform(size):
    """Fixed resize + center-crop to `size`, ToTensor, then map [0,1] -> [-1,1]."""
    size = (size, size) if isinstance(size, int) else tuple(size)
    return transforms.Compose([
        transforms.Resize(min(size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),  # HWC uint8 -> CHW float [0,1]
    ])


def in_split(token, split, val_fraction):
    """Deterministic hash-based train/val membership check for a scene token."""
    if split == "all":
        return True
    bucket = int(hashlib.md5(str(token).encode()).hexdigest(), 16) % 1000  # int in range [0, 999]
    is_val = bucket < int(val_fraction * 1000)
    return is_val if split == "val" else (not is_val)


def poses_from_frames(frames):
    """(T, 3) [x, y, yaw] global ego poses from a list of frame dicts."""
    return np.array(
        [
            [
                fr["ego2global_translation"][0],
                fr["ego2global_translation"][1],
                quat_to_yaw(fr["ego2global_rotation"]),
            ]
            for fr in frames
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Base dataset
# --------------------------------------------------------------------------- #
class NavsimLongBase(Dataset):
    """One whole NAVSIM episode per index. Subclasses supply the episodes."""

    def __init__(
        self,
        sensor_root,                   # root that camera data_paths are relative to
        size=256,                      # square image size (H, W)
        stored_data_frame_rate=2,      # NAVSIM logs are ~2 Hz (0.5 s interval)
        frame_rate=2,                  # sampling rate; frame_interval = stored / rate
        front_camera=FRONT_CAMERA,
        load_surround=True,            # also load the 5 surround cameras (for vis)
    ):
        super().__init__()
        from pathlib import Path
        self.sensor_root = Path(sensor_root)
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.frame_interval = max(int(round(stored_data_frame_rate / frame_rate)), 1)
        self.frame_rate = frame_rate
        self.stored_data_frame_rate = stored_data_frame_rate
        self.front_camera = front_camera
        self.load_surround = load_surround
        self.transform = build_image_transform(self.size)
        self.episodes = []             # subclass must fill this

    # --- explicit dataset selection (navhard | navtrain) --------------------- #
    @staticmethod
    def from_config(cfg, **overrides):
        """Build the long dataset named by cfg.dataset from its cfg.data_long block.

        Explicit dispatch (no factory / no config rewriting): the flag maps to a
        concrete class, and that class is loaded with its own `data_long.<name>`
        params. `overrides` (e.g. load_surround=False for caching) win over config.
        """
        name = cfg.dataset
        params = dict(cfg.data_long[name])
        params.update(overrides)
        if name == "navhard":
            from exp_navsim.data.navsim_long import NavsimLongDataset
            return NavsimLongDataset(**params)
        if name == "navtrain":
            from exp_navsim.data.navtrain_long import NavtrainLongDataset
            return NavtrainLongDataset(**params)
        raise ValueError(f"unknown dataset {name!r} (expected 'navhard' or 'navtrain')")

    @staticmethod
    def cache_dir(cfg):
        """Latent-cache directory for the selected dataset (cfg.cache.dir[name])."""
        return cfg.cache.dir[cfg.dataset]

    # --- to be provided by the subclass -------------------------------------- #
    def _episode_frames(self, handle):
        """Return (frames_sorted, meta) for one episode handle. No subsampling."""
        raise NotImplementedError

    # --- reusable image / sample construction -------------------------------- #
    def _read_camera(self, frame, cam_name, required=True):
        """Load and transform one camera image; returns (3, H, W) in [-1, 1].

        If the image is missing (only expected for surround cameras of the
        partially-downloaded navtrain set) and `required` is False, a black
        frame is returned instead of raising.
        """
        cams = frame["cams"]
        # camera keys may be upper- or lower-case depending on the dump
        key = cam_name if cam_name in cams else cam_name.lower()
        path = self.sensor_root / cams[key]["data_path"]
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            return torch.full((3, *self.size), -1.0)
        img = Image.open(path).convert("RGB")
        return self.transform(img) * 2.0 - 1.0

    def _frames_to_sample(self, frames, meta):
        """Build the return dict from a list of (already-subsampled) frames."""
        trajectory = poses_to_local_traj(poses_from_frames(frames))   # (T, 2)
        velocity = traj_to_velocity(trajectory)                       # (T, 2)

        images = torch.stack(
            [self._read_camera(fr, self.front_camera) for fr in frames], dim=0
        )
        sample = {
            "images": images.numpy(),
            "trajectory": trajectory,
            "velocity": velocity,
            "frame_rate": self.frame_rate,
            "metadata": meta,
        }
        if self.load_surround:
            # (T, n_cam, 3, H, W): every camera at every frame, for the BEV panel.
            # Surround images may be absent in navtrain -> tolerate (required=False).
            surround = torch.stack(
                [torch.stack([self._read_camera(fr, c, required=False)
                              for c in SURROUND_CAMERAS], 0) for fr in frames],
                dim=0,
            )
            sample["cameras"] = surround.numpy()
        return sample

    # --- Dataset API --------------------------------------------------------- #
    def __len__(self):
        return len(self.episodes)

    def load_sample(self, idx):
        frames, meta = self._episode_frames(self.episodes[idx])
        frames = frames[:: self.frame_interval]
        return self._frames_to_sample(frames, meta)

    def __getitem__(self, idx):
        return self.load_sample(idx)

    def episode_length(self, idx):
        """Cheap (image-free) subsampled length of episode `idx`, for stats/vis."""
        frames, _ = self._episode_frames(self.episodes[idx])
        return len(frames[:: self.frame_interval])

    def episode_token(self, idx):
        """Unique token of episode `idx` (matches the `token` stored in the cache).

        Default reads it from the episode meta; subclasses override with a cheaper
        path that avoids loading frames.
        """
        return self._episode_frames(self.episodes[idx])[1]["token"]
