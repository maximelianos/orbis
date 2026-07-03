"""
Diffusion trajectory prediction.

Encodes context camera frames into latents (via loaders_for_projects.encoder.Encoder)
and runs the whole-trajectory flow-matching diffusion model
(networks.whole_context.DiffusionModel) to sample a *distribution* of future
trajectories. By diffusing ``n_trajectories`` noise samples in a single batch we
obtain many plausible futures for the same conditioning.

See evaluate/test_nuplan.py for how the batch is built and how predictions are
denormalized.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util import instantiate_from_config
from loaders_for_projects.encoder import Encoder
from evaluate.checkpoint import resolve_checkpoint, load_model_checkpoint


class VelocityNormalizer:
    """Velocity (de)normalization using the cached NuPlan stats.

    Mirrors NuplanHDF.normalize/denormalize so that model.sample() can call
    ``.denormalize`` without instantiating the full HDF5 dataset.
    """

    def __init__(self, cache_path):
        data = np.load(cache_path)
        self.mean = data["mean"]
        self.std = data["std"]

    def normalize_velocity(self, velocity):
        """Real-unit velocity -> normalized. Accepts numpy array or tensor."""
        if isinstance(velocity, torch.Tensor):
            mean = torch.as_tensor(self.mean).to(velocity)
            std = torch.as_tensor(self.std).to(velocity) + 1e-8
        else:
            velocity = np.asarray(velocity, dtype=np.float32)
            mean = self.mean
            std = self.std + 1e-8
        return (velocity - mean) / std

    def denormalize(self, sample):
        """Normalized -> real-unit velocity. ``sample`` is a dict with 'velocity'."""
        sample = dict(sample)
        velocity = sample["velocity"]
        if isinstance(velocity, torch.Tensor):
            mean = torch.as_tensor(self.mean).to(velocity)
            std = torch.as_tensor(self.std).to(velocity) + 1e-8
        else:
            mean = self.mean
            std = self.std + 1e-8
        sample["velocity"] = velocity * std + mean
        return sample


class TrajectoryPredictor:
    """Encodes context frames and samples trajectory futures with diffusion."""

    def __init__(
        self,
        diffusion_config="configs/nuplan.yaml",
        diffusion_ckpt=None,
        diffusion_logdir="logs_nuplan",
        last_ckpt=True,
        encoder_exp_dir="logs_tk/tokenizer_288x512",
        encoder_ckpt="checkpoints/last.ckpt",
        encoder_config="config.yaml",
        norm_cache="data/cache/nuplan_norm_stats.npz",
        image_size=(288, 512),
        n_trajectories=100,
        num_diffusion_steps=20,
        device="cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = tuple(image_size)  # (H, W)
        self.n_trajectories = n_trajectories
        self.num_diffusion_steps = num_diffusion_steps

        # --- image encoder (tokenizer) ---
        self.encoder = Encoder(
            exp_dir=encoder_exp_dir,
            ckpt=encoder_ckpt,
            config=encoder_config,
            device=str(self.device),
        )

        # --- diffusion trajectory model ---
        config = OmegaConf.load(str(PROJECT_ROOT / diffusion_config))
        self.model = instantiate_from_config(config.model).to(self.device)
        ckpt_path = resolve_checkpoint(
            ckpt=diffusion_ckpt, last_ckpt=last_ckpt, logdir=diffusion_logdir
        )
        self.model = load_model_checkpoint(
            self.model, ckpt_path, map_location=self.device, strict=False
        )
        self.model.eval()

        self.num_frames = self.model.num_frames
        self.context_images = self.model.context_images
        self.pred_steps = self.model.pred_steps
        self.context_length = self.num_frames - self.pred_steps

        # --- velocity normalization ---
        self.normalizer = VelocityNormalizer(str(PROJECT_ROOT / norm_cache))

        # Match the dataloader 'resize_center' preprocessing: resize short edge,
        # center crop to (H, W), to [0, 1], then map to [-1, 1].
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(min(self.image_size)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ]
        )

    def preprocess_images(self, images):
        """Convert a sequence of images to a model-ready tensor.

        Args:
            images: list/sequence of PIL.Image or HxWx3 uint8 numpy arrays.

        Returns:
            tensor (N, 3, H, W) in [-1, 1] on the predictor device.
        """
        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype(np.uint8))
            elif not isinstance(img, Image.Image):
                raise TypeError(f"Unsupported image type: {type(img)}")
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(self.image_transform(img))
        x = torch.stack(tensors, dim=0) * 2.0 - 1.0
        return x.to(self.device)

    @torch.no_grad()
    def encode_latents(self, images_tensor):
        """Encode (N, 3, H, W) images -> semantic latents (N, C, Hl, Wl)."""
        _, q_sem = self.encoder.encode(images_tensor)
        return q_sem.float()

    @torch.no_grad()
    def predict(self, images, history_velocity=None, n_trajectories=None):
        """Sample a distribution of future trajectories.

        Args:
            images: sequence of context frames (PIL or numpy). The first
                ``context_images`` frames are used as conditioning.
            history_velocity: optional (M, 2) real-unit velocity deltas for the
                history frames. Normalized internally. Missing entries are zero.
            n_trajectories: override the number of simultaneously diffused
                trajectories (defaults to ``self.n_trajectories``).

        Returns:
            numpy array (n_trajectories, num_frames, 2) of trajectory positions
            in the diffusion model's local frame (origin at the first frame).
        """
        n_traj = n_trajectories or self.n_trajectories
        T = self.num_frames

        latents = self.encode_latents(self.preprocess_images(images))
        n_ctx, C, Hl, Wl = latents.shape
        k = min(n_ctx, self.context_images)

        encoded = torch.zeros(n_traj, T, C, Hl, Wl, device=self.device, dtype=latents.dtype)
        encoded[:, :k] = latents[:k].unsqueeze(0)

        velocity = torch.zeros(n_traj, T, 2, device=self.device)
        if history_velocity is not None:
            hv = self.normalizer.normalize_velocity(
                torch.as_tensor(history_velocity, dtype=torch.float32, device=self.device)
            )
            m = min(hv.shape[0], self.context_length)
            velocity[:, :m] = hv[:m].unsqueeze(0)

        batch = {
            "turn": torch.zeros(n_traj, device=self.device),
            "trajectory": torch.zeros(n_traj, T, 2, device=self.device),
            "velocity": velocity,
            "encoded_q_sem": encoded,
        }

        trajectories = self.model.sample(
            batch=batch,
            num_diffusion_steps=self.num_diffusion_steps,
            dataset=self.normalizer,
            denormalize=True,
        )
        return trajectories.cpu().numpy()

    def future_positions(self, trajectories):
        """Slice predicted futures relative to the current (last context) frame.

        Args:
            trajectories: (B, num_frames, 2) output of ``predict``.

        Returns:
            (B, pred_steps, 2) future positions with origin at the current frame.
        """
        current = trajectories[:, self.context_length - 1 : self.context_length, :]
        return trajectories[:, self.context_length :, :] - current
