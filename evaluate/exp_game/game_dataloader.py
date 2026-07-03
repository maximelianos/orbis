"""
Game Trajectory DataLoader

Generates synthetic trajectories that follow Bezier curves and returns:
- turn parameter
- trajectory positions
- trajectory velocities
- rendered environment image
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


def bezier_curve(p0, p1, p2, p3, t):
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def generate_trajectory(turn, T, r):
    alpha_deg = 30 - 60 * turn
    alpha_rad = np.deg2rad(alpha_deg)

    x_end = r * np.cos(alpha_rad)
    y_end = r * np.sin(alpha_rad)

    p0 = np.array([0.0, 0.0], dtype=np.float32)
    p3 = np.array([x_end, y_end], dtype=np.float32)
    p1 = p0 + np.array([r, 0.0], dtype=np.float32) * 0.5
    p2 = p1

    t_values = np.linspace(0, 1, T)
    trajectory = np.array([bezier_curve(p0, p1, p2, p3, t) for t in t_values], dtype=np.float32)
    return trajectory


class GameTrajectoryDataset(Dataset):
    def __init__(
        self,
        dataset_size,
        T,
        r,
        image_height,
        image_width,
        turn_min=0.0,
        turn_max=1.0,
        output_normalization=True,
        normalization_cache_dir="data/cache",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dataset_size = dataset_size
        self.T = T
        self.r = r
        self.image_height = image_height
        self.image_width = image_width
        self.output_normalization = output_normalization

        self.normalization_cache_dir = Path(normalization_cache_dir)
        self.normalization_cache_dir.mkdir(parents=True, exist_ok=True)

        self.turns = np.random.uniform(turn_min, turn_max, size=dataset_size)

        self.precompute_normalization()

    def precompute_normalization(self):
        cache_filename = f"game_norm_stats_size{self.dataset_size}_T{self.T}_r{self.r}.npz"
        cache_path = self.normalization_cache_dir / cache_filename

        if cache_path.exists():
            print(f"Loading normalization stats from {cache_path}")
            data = np.load(cache_path)
            self.velocity_mean = data["mean"]
            self.velocity_std = data["std"]
            return

        print(f"Computing normalization stats for {self.dataset_size} samples...")
        velocities = []
        for idx in range(self.dataset_size):
            turn = self.turns[idx]
            sample = self.generate_trajectory(turn, T=self.T, r=self.r)
            velocities.append(sample["velocity"].numpy())

        velocities = np.concatenate(velocities, axis=0)
        self.velocity_mean = velocities.mean(axis=0)
        self.velocity_std = velocities.std(axis=0)

        np.savez(cache_path, mean=self.velocity_mean, std=self.velocity_std)
        print(f"Saved normalization stats to {cache_path}")
        print(f"  Velocity mean: {self.velocity_mean}")
        print(f"  Velocity std: {self.velocity_std}")

    # =================================================== Image render
    
    def _world_to_pixel(self, x, y):
        x_min, x_max = -0.2 * self.r, 1.1 * self.r
        y_min, y_max = -1.1 * self.r, 1.1 * self.r

        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)

        px = int(np.clip(x_norm, 0.0, 1.0) * (self.image_width - 1))
        py = int((1.0 - np.clip(y_norm, 0.0, 1.0)) * (self.image_height - 1))
        return px, py

    def _draw_square(self, image, center_xy, color, half_size):
        cx, cy = center_xy
        x0 = max(0, cx - half_size)
        x1 = min(self.image_width, cx + half_size + 1)
        y0 = max(0, cy - half_size)
        y1 = min(self.image_height, cy + half_size + 1)
        image[y0:y1, x0:x1, :] = color

    def render_image(self, turn, trajectory, draw_trajectory=False):
        # Return image (H, W, 3) in range [0, 1]
        
        image = np.ones((self.image_height, self.image_width, 3), dtype=np.float32)
        square_half_size = max(2, min(self.image_height, self.image_width) // 40)

        if draw_trajectory:
            line_color = np.array([0.2, 0.2, 0.2], dtype=np.float32)
            for i in range(len(trajectory) - 1):
                p0 = trajectory[i]
                p1 = trajectory[i + 1]
                steps = max(abs(float(p1[0] - p0[0])), abs(float(p1[1] - p0[1]))) * 200
                steps = max(2, int(steps))
                for j in range(steps + 1):
                    alpha = j / steps
                    x = (1.0 - alpha) * float(p0[0]) + alpha * float(p1[0])
                    y = (1.0 - alpha) * float(p0[1]) + alpha * float(p1[1])
                    px, py = self._world_to_pixel(x, y)
                    self._draw_square(image, (px, py), color=line_color, half_size=1)

        origin_px = self._world_to_pixel(0.0, 0.0)
        self._draw_square(image, origin_px, color=np.array([0.0, 0.0, 0.0], dtype=np.float32), half_size=square_half_size)

        destination = trajectory[-1]
        destination_px = self._world_to_pixel(float(destination[0]), float(destination[1]))

        # Main target: green
        main_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._draw_square(image, destination_px, color=main_color, half_size=square_half_size)

        # distractor targets from random turns: blue
        random_turns = np.random.uniform(0.0, 1.0, size=3)
        for random_turn in random_turns:
            random_traj = generate_trajectory(random_turn, T=self.T, r=self.r)
            random_dst = random_traj[-1]
            random_px = self._world_to_pixel(float(random_dst[0]), float(random_dst[1]))

            distractor_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            self._draw_square(image, random_px, color=distractor_color, half_size=square_half_size)

        return image

    def generate_trajectory(self, turn, T, r, draw_trajectory=False):
        trajectory = generate_trajectory(turn, T=T, r=r)

        velocity = np.zeros_like(trajectory, dtype=np.float32)
        velocity[:-1] = trajectory[1:] - trajectory[:-1]

        image = self.render_image(turn, trajectory, draw_trajectory=draw_trajectory)

        return {
            "turn": torch.tensor(turn, dtype=torch.float32),
            "position": torch.tensor(trajectory, dtype=torch.float32),
            "velocity": torch.tensor(velocity, dtype=torch.float32),
            "images": torch.tensor(image, dtype=torch.float32),
        }

    def normalize(self, sample):
        if not self.output_normalization:
            return sample

        sample = sample.copy()
        velocity = sample["velocity"]
        mu = torch.tensor(self.velocity_mean, dtype=velocity.dtype, device=velocity.device)
        sigma = torch.tensor(self.velocity_std, dtype=velocity.dtype, device=velocity.device) + 1e-8
        sample["velocity"] = (velocity - mu) / sigma
        return sample

    def denormalize(self, sample):
        if not self.output_normalization:
            return sample

        sample = sample.copy()
        velocity = sample["velocity"]
        mu = torch.tensor(self.velocity_mean, dtype=velocity.dtype, device=velocity.device)
        sigma = torch.tensor(self.velocity_std, dtype=velocity.dtype, device=velocity.device) + 1e-8
        sample["velocity"] = velocity * sigma + mu
        return sample

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        turn = self.turns[idx]
        sample = self.generate_trajectory(turn, T=self.T, r=self.r)
        sample = self.normalize(sample)
        return sample


if __name__ == "__main__":
    dataset = GameTrajectoryDataset(
        dataset_size=32,
        T=20,
        r=1.0,
        image_height=64,
        image_width=64,
    )

    pages = []
    num_examples = 12
    for _ in range(num_examples):
        turn = np.random.uniform(0.0, 1.0)
        sample = dataset.generate_trajectory(turn, T=dataset.T, r=dataset.r, draw_trajectory=False)
        frame = sample["images"].numpy()
        pages.append(Image.fromarray((frame * 255).astype(np.uint8)))

    out_path = "game_dataloader_examples.tiff"
    pages[0].save(out_path, save_all=True, append_images=pages[1:])
    print(f"Saved multipage TIFF: {out_path}")