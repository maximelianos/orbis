"""
python loaders_for_projects/nuplan_long.py

NuPlan dataloader that returns one *whole episode* per index instead of short
num_frames windows. Used to cache latents efficiently (see docs/idea.md): with
whole_episodes the index map holds a single window per video (starting at frame
0), so the samples are visited in a deterministic order and can be reused from
cache between runs without shuffling.
"""

import os
import sys

import torch
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders_for_projects.custom_multiframe_odo import MultiHDF5DatasetMultiFrameIdxMappingOdometry
from loaders_for_projects.nuplan_dataloader import NuPlanVelocityDataset


class NuPlanLongDataset(MultiHDF5DatasetMultiFrameIdxMappingOdometry):
    """
    NuPlan dataset that returns whole episodes and their velocities.

    Similar to NuPlanVelocityDataset (it converts the steering trajectory to
    velocities), but each index maps to a full video rather than a num_frames
    window. Inherits from MultiHDF5DatasetMultiFrameIdxMappingOdometry; only the
    functions that need whole-episode behaviour are overloaded.
    """

    def __init__(self, *args, whole_episodes=True, **kwargs):
        # whole_episodes comes from the config. We keep it out of the parent
        # custom_* constructors (they don't accept it as a forwarded kwarg) and
        # apply it ourselves in scan_h5_files, which runs during super().__init__.
        self._whole_episodes = whole_episodes
        super().__init__(*args, **kwargs)

        # sorted list of the HDF5 files containing the episodes (one per file)
        self.episode_paths = sorted(self.hdf5_paths)

    def scan_h5_files(self):
        # tell the base class to keep only the window starting at the first frame
        # (drop the sliding windows) before it builds the index mapping
        self.whole_episodes = self._whole_episodes
        super().scan_h5_files()

    def get_images_and_indices(self, idx):
        # like the base method, but read every frame_interval-th frame from
        # start_frame to the end of the video instead of just num_frames frames
        if idx >= len(self.index_to_starting_frame_map):
            raise IndexError(f'Index {idx} out of range for dataset of length {len(self.index_to_starting_frame_map)}')
        filename, key, start_frame = self.index_to_starting_frame_map[idx]
        file = self.get_h5_file(filename)
        frame_indices = range(start_frame, len(file[key]), self.frame_interval)
        images = [Image.fromarray(file[key][i]) for i in frame_indices]
        return images, (filename, key, start_frame)

    def get_odometry(self, filename, key, start_idx):
        # same whole-episode indexing as get_images_and_indices so images and
        # odometry stay aligned over the full episode
        odo_filename = filename.replace(self.frames_file_suffix, self.odo_files_suffix)
        odo_file = self.get_odo_file(odo_filename)
        frame_indices = range(start_idx, len(odo_file[key]), self.frame_interval)
        odo = [odo_file[key][i] for i in frame_indices]
        return torch.tensor(self.odo_transform(odo))

    def load_sample(self, idx):
        """
        Returns the same keys as NuPlanVelocityDataset (over the whole episode):
            "images":     (T, C, H, W)
            "steering":   (T, 2) == trajectory (x, y)
            "trajectory": (T, 2)
            "velocity":   (T, 2), d_t = p_t - p_{t-1}
            "frame_rate": int
        plus the episode source file:
            "metadata":   {"path": "/path/to/frames.h5"}
        """
        # parent __getitem__ uses our overloaded get_images_and_indices /
        # get_odometry, so it already returns the WHOLE episode here
        sample = super().__getitem__(idx)

        sample["images"] = np.array(sample["images"])
        sample["steering"] = np.array(sample["steering"])

        # convert trajectory to velocity, exactly like NuPlanVelocityDataset
        if sample["steering"] is not None:
            trajectory = sample["steering"][:, :2]  # [num_frames, 2] (x, y)
            assert trajectory.shape[1] == 2

            sample["velocity"] = NuPlanVelocityDataset.traj_to_velocity(trajectory)
            sample["steering"] = trajectory
            sample["trajectory"] = trajectory

        # metadata: which HDF5 file this episode was read from
        filename, _key, _start_frame = self.index_to_starting_frame_map[idx]
        sample["metadata"] = {"path": filename}

        return sample

    def __getitem__(self, idx):
        return self.load_sample(idx)

    def __len__(self):
        return super().__len__()


def main():
    """Visualize several whole episodes from NuPlanLongDataset, using draw_nuplan."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from omegaconf import OmegaConf

    from util import instantiate_from_config
    from loaders_for_projects.draw_nuplan import save_pictures

    config_path = "configs/long_data.yaml"
    num_episodes = 5
    pdf_output_path = "nuplan_long_episodes.pdf"

    # Build the whole-episode dataset from config
    config = OmegaConf.load(config_path)
    dataset = instantiate_from_config(config.data.params.train)
    print(f"Dataset created: {len(dataset)} episodes")

    # Inspect one sample
    sample = dataset[0]
    print(f"  images shape:     {sample['images'].shape}")
    print(f"  trajectory shape: {sample['trajectory'].shape}")
    print(f"  velocity shape:   {sample['velocity'].shape}")
    print(f"  metadata:         {sample['metadata']}")

    # Visualize several episodes, one page each
    indices = np.linspace(0, len(dataset) - 1, num_episodes).astype(int)
    with PdfPages(pdf_output_path) as pdf:
        for i, idx in enumerate(indices):
            print(f"  Processing episode {idx} ({i + 1}/{num_episodes})...")
            sample = dataset[int(idx)]
            title = f"Episode #{idx} ({sample['metadata']['path']})"
            fig = save_pictures(
                [sample["trajectory"]],
                title=title,
                images=sample["images"],
                output_path=None,
            )
            pdf.savefig(fig, dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved visualization to {pdf_output_path}")


if __name__ == "__main__":
    main()
