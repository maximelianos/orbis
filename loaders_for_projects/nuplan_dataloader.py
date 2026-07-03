"""
python loaders_for_projects/nuplan_dataloader.py
"""

import os
import sys
import torch
import numpy as np
import h5py
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders_for_projects.custom_multiframe_odo import MultiHDF5DatasetMultiFrameIdxMappingOdometry



class NuPlanVelocityDataset(MultiHDF5DatasetMultiFrameIdxMappingOdometry):
    """
    NuPlan dataset that returns trajectories and velocities.
    
    Inherits from MultiHDF5DatasetMultiFrameIdxMappingOdometry and converts
    the trajectory output to velocities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    def traj_to_velocity(trajectory):
        """
        Args:
            trajectory: numpy array [num_frames, 2], contains (x, y) positions
        
        Returns:
            velocity: numpy array [num_frames, 2], contains (dx, dy) velocities
        """
        traj = trajectory.copy()
        # Compute differences: v_t = x_t - x_{t-1}
        # First frame has zero velocity (no previous frame)
        v = np.zeros_like(traj)
        v[1:] = traj[1:] - traj[:-1]
        return v
    
    @staticmethod
    def velocity_to_traj(velocity, start_position=(0, 0)):
        """
        Convert velocities back to (x, y) trajectory coordinates.
        
        Args:
            velocity: numpy array [num_frames, 2], contains (dx, dy) velocities
            start_position: optional starting position [2] for (x, y)
        
        Returns:
            trajectory: numpy array [num_frames, 2], contains (x, y) positions
        """
        # Compute cumulative sum to get positions
        xy = np.cumsum(velocity, axis=0) + np.array(start_position, dtype=xy.dtype)
        return xy
    
    def load_sample(self, idx):
        """
        Existing keys:
        sample: {
            "images": (T, C, H, W) torch.float32,
            "steering": (T, 2) torch.float32
        }
        
        Convert all to numpy arrays.
        
        Returns new keys:
            "trajectory": equal to steering
            "velocity": (T, 2), d_t = p_t - p_{t-1}
        """
        # Get the sample from parent class
        sample = super().__getitem__(idx)
        
        sample["images"] = np.array(sample["images"])
        sample["steering"] = np.array(sample["steering"])
        
        # Convert trajectory to velocity if steering data exists
        if sample['steering'] is not None:
            trajectory = sample['steering'][:, :2]  # [num_frames, 2 or 3] (x, y) or (x, y, heading)
            assert trajectory.shape[1] == 2
            
            velocity = self.traj_to_velocity(trajectory)  # [num_frames, 2]
            
            # Store both for convenience
            sample['velocity'] = velocity
            sample['steering'] = trajectory
            sample['trajectory'] = trajectory
        
        return sample
    
    def __getitem__(self, idx):
        return self.load_sample(idx)

    def __len__(self):
        return super().__len__()


class NuplanHDF(torch.utils.data.Dataset):
    """
    NuPlan dataset that caches a subset of NuPlan into an HDF5 file.
    
    HDF5 File Structure:
    - /0
        - images: (T, C, H, W)
        - steering: (T, 2)
        - encoded_q_rec (Optional): (T, C_lat, H_lat, W_lat)
        - encoded_q_sem (Optional): (T, C_lat, H_lat, W_lat)
    - /1
        ...
    """
    def __init__(self, hdf5_path, source_dataset=None, return_image=False, encoder=None, 
                 output_normalization=True, normalization_cache_dir="data/cache", dataset_size=None):
        """There are two modes:
        source - loading samples from original dataset and saving them in a single HDF5,
        hdf - loading sample from HDF5 without original dataset.
        
        Args:
            hdf5_path: str
            source_dataset: NuplanHDF (source mode)
            encoder: image encoder model (source mode)
            
            output_normalization: bool (hdf mode)
            return_image: bool (hdf mode)
            dataset_size: cut the number of samples (hdf mode)
        """
        super().__init__()
        
        # "source": get samples from existing loader. "hdf": get from cached hdf
        self._mode = "source" if source_dataset is not None else "hdf"
        self.source_dataset = source_dataset
        self.return_image = return_image
        self.encoder = encoder
        
        self.hdf5_path = Path(hdf5_path)
        self.hdf5_path.parent.mkdir(parents=True, exist_ok=True)

        if self._mode == "source":
            # create the hdf file
            self.h5_file = h5py.File(self.hdf5_path, 'a')
            self.dataset_size = len(self.source_dataset)
        else:
            assert self.hdf5_path.exists(), f"HDF5 file {hdf5_path} does not exist."
            self.h5_file = h5py.File(self.hdf5_path, 'r')
            self.indices = sorted(int(key) for key in self.h5_file.keys())
            if dataset_size is not None:
                self.dataset_size = min(len(self.indices), dataset_size)
            else:
                self.dataset_size = len(self.indices)

        # normalization
        self.output_normalization = output_normalization
        self.normalization_cache_dir = Path(normalization_cache_dir)
        self.normalization_cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_filename = f"nuplan_norm_stats.npz"
        self.cache_path = self.normalization_cache_dir / cache_filename
        
        if self._mode == "hdf":
            self.load_normalization()
    
    # =========================== Source mode ===========================
    
    def precompute_normalization(self):
        print(f"Computing normalization stats for {self.dataset_size} samples...")
        velocities = []
        for idx in range(len(self)):
            hdf_idx = self._resolve_hdf_index(idx)
            sample = self.load_sample(hdf_idx)
            vel = sample["velocity"]
            velocities.append(vel)

        velocities = np.stack(velocities, axis=0) # [N, T, 2]
        self.velocity_mean = velocities.mean(axis=(0, 1))
        self.velocity_std = velocities.std(axis=(0, 1))

        np.savez(self.cache_path, mean=self.velocity_mean, std=self.velocity_std)
        print(f"Saved normalization stats to {self.cache_path}")
        print(f"  Velocity mean: {self.velocity_mean}")
        print(f"  Velocity std: {self.velocity_std}")
    
    def save_sample(self, idx):
        """
        Loads a sample from source dataset, encodes the images, 
        and writes the array data into the HDF5 group matching the sample index.
        
        No normalization here.
        """
        assert self._mode == "source", "Can save samples only in source mode"
        original_sample = self.source_dataset.load_sample(idx)
        
        images = original_sample["images"]  # Example: (T, 3, H, W)
        steering = original_sample["steering"]  # Example: (T, 2)
        
        assert images.ndim == 4, f"Expected images shape (T, C, H, W), got {images.shape}"
        assert steering.ndim == 2 and steering.shape[1] == 2, f"Expected steering shape (T, 2), got {steering.shape}"

        # Add batch dim, encode, and remove batch dim.
        # Shape goes from (B=1, T, C, H, W) to (1*T, C_lat, H_lat, W_lat)
        x = torch.tensor(images).unsqueeze(0)
        
        # Ensure proper encoding
        T = x.shape[1] 
        
        # Flaten batch and time for encoder: (B*T, C, H, W) where B=1
        x = x.view(-1, *x.shape[2:]) 
        
        q_rec, q_sem = self.encoder.encode(x)
        
        # Since B=1, we can just reshape back to (T, C_lat, H_lat, W_lat) implicitly or just use the B*T dim as T
        q_rec = q_rec.cpu().numpy()
        q_sem = q_sem.cpu().numpy()
        
        assert q_rec.ndim == 4 and q_rec.shape[0] == T, f"Expected q_rec shape (T, C_lat, H_lat, W_lat), got {q_rec.shape}"
        assert q_sem.ndim == 4 and q_sem.shape[0] == T, f"Expected q_sem shape (T, C_lat, H_lat, W_lat), got {q_sem.shape}"

        grp_name = str(idx)
        if grp_name in self.h5_file:
            del self.h5_file[grp_name]
        grp = self.h5_file.create_group(grp_name)
        
        grp.create_dataset("images", data=images, compression="lzf")
        grp.create_dataset("steering", data=steering, compression="lzf")
        
        if q_rec is not None:
            grp.create_dataset("encoded_q_rec", data=q_rec, compression="lzf")
            grp.create_dataset("encoded_q_sem", data=q_sem, compression="lzf")
    
    # =========================== HDF mode ===========================
    
    def load_normalization(self):
        assert self._mode == "hdf", "Load normalization only in hdf mode."
        
        print(f"Loading normalization stats from {self.cache_path}")
        if not self.cache_path.exists():
            print(f"Normalization cache doesn't exist: {self.cache_path}")
            self.precompute_normalization()
            
        data = np.load(self.cache_path)
        self.velocity_mean = data["mean"]
        self.velocity_std = data["std"]

    def normalize(self, sample):
        if not self.output_normalization:
            return sample

        sample = sample.copy()
        velocity = sample["velocity"]
        mean = self.velocity_mean
        std = self.velocity_std + 1e-8
        sample["velocity"] = (velocity - mean) / std
            
        return sample

    def denormalize(self, sample):
        """Sample has "velocity" key - a numpy array or torch tensor
        """
        if not self.output_normalization:
            return sample

        sample = sample.copy()
        velocity = sample["velocity"]
        
        mean = self.velocity_mean
        std = self.velocity_std + 1e-8
        if isinstance(velocity, torch.Tensor):
            mean = torch.tensor(mean).to(velocity)
            std = torch.tensor(std).to(velocity)

        sample["velocity"] = velocity * std + mean
            
        return sample

    def _resolve_hdf_index(self, idx):
        assert self._mode == "hdf", "Loading from hdf file only in hdf mode."
        assert 0 <= idx < len(self.indices), f"Index {idx} out of range for {len(self.indices)} saved samples"
        return self.indices[idx]

    def load_sample(self, idx):
        """
        Loads a sample directly from the generated HDF5 group.
        Dynamically calculates 'velocity' from cache 'steering', translating it.

        Returns:
            dict containing:
            - 'trajectory': numpy array of shape (T, 2) corresponding to cumulative steering.
            - 'velocity': numpy array, diff velocities of shape (T, 2) dx, dy
            - 'images' (Optional): numpy arrays of shape (T, C, H, W), for example (10, 3, 256, 256)
            - 'encoded_q_rec' (Optional): numpy arrays of shape (T, C_lat, H_lat, W_lat) representing reconstructed latents, for example (10, 16, 16, 16)
            - 'encoded_q_sem' (Optional): numpy arrays of shape (T, C_lat, H_lat, W_lat) representing spatial representations
        """
        assert self._mode == "hdf", "Loading from hdf file only in hdf mode."

        assert str(idx) in self.h5_file

        sample = {}
        grp = self.h5_file[str(idx)]
        trajectory = grp["steering"][:]
        velocity = NuPlanVelocityDataset.traj_to_velocity(trajectory)
        
        sample["turn"] = np.zeros(1) # TODO compatibility with model
        sample["trajectory"] = trajectory
        sample["velocity"] = velocity
        
        sample["images"] = np.zeros(1)
        
        if self.return_image:
            # sample["images"] = grp["images"][:]  # TODO remove
            if "encoded_q_rec" in grp and "encoded_q_sem" in grp:
                sample["encoded_q_rec"] = grp["encoded_q_rec"][:]
                sample["encoded_q_sem"] = grp["encoded_q_sem"][:]
        else:
            # dummy values
            sample["images"] = np.zeros(1)
            sample["encoded_q_rec"] = np.zeros(1)
            sample["encoded_q_sem"] = np.zeros(1)
                
        return sample

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        assert self._mode == "hdf", "Use operator [] only in hdf mode."
        idx = idx % len(self.indices)
        
        hdf_idx = self._resolve_hdf_index(idx)
        sample = self.load_sample(hdf_idx)
        sample = self.normalize(sample)
        
        # Convert all numpy arrays to PyTorch tensors
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                # PyTorch usually expects float32 rather than float64
                sample[k] = torch.from_numpy(v).float()
                    
        return sample

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()


try:
    from .buffer_dataset import BufferDataset
    from .encoder import Encoder
except ImportError:
    from loaders_for_projects.buffer_dataset import BufferDataset
    from loaders_for_projects.encoder import Encoder

class NuplanBufferedHDF(BufferDataset):
    def __init__(self, **kwargs):
        source_dataset = NuplanHDF(**kwargs)
        super().__init__(source_dataset=source_dataset, dataset_size=100000)


class NuPlanVelocityBuffered(BufferDataset):
    """NuPlanVelocityDataset (raw images) behind a BufferDataset ring buffer.

    Decompressing/augmenting nuPlan frames from HDF5 is the bottleneck; the
    buffer serves cached samples and only periodically draws fresh ones from
    disk, so the on-GPU encoder in the model is not starved. Buffer knobs are
    exposed explicitly; all other kwargs are forwarded to NuPlanVelocityDataset.
    """
    def __init__(
        self,
        initial_buffer_size: int,
        max_buffer_size: int,
        replace_rate: float,
        replace_count: int,
        dataset_size: int,
        seed: int = 0,
        **dataset_kwargs,
    ):
        source_dataset = NuPlanVelocityDataset(**dataset_kwargs)
        super().__init__(
            source_dataset=source_dataset,
            initial_buffer_size=initial_buffer_size,
            max_buffer_size=max_buffer_size,
            replace_rate=replace_rate,
            replace_count=replace_count,
            dataset_size=dataset_size,
            seed=seed,
        )
