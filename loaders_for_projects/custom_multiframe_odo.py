import os
import random
from tqdm import tqdm
from PIL import Image
import torch
import h5py
import numpy as np
import importlib
from scipy.spatial.transform import Rotation as R
from .custom_multiframe import MultiHDF5DatasetMultiFrameIdxMapping, MultiHDF5DatasetMultiFrameFromJSON
from .util import get_trajectory_from_speeds_and_yaw_rates
from omegaconf import OmegaConf


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class OdometryLoaderNuPlan:
    def __call__(self, odo_data):
        """
        odo_data: list of dicts, each dict contains IMU data for a frame
        """
        ret = np.stack([np.array([odo_frame['vx'], odo_frame['angular_rate_z']]) for odo_frame in odo_data], axis=0)
        assert ret.shape[1] == 2 and ret.shape[0] == len(odo_data), f"Unexpected odometry shape {ret.shape}, expected ({len(odo_data)}, 2)"
        return ret


class TrajectoryLoaderNuPlanFromSpeedYawRate:
    def __init__(self, frame_rate=5):
        self.frame_rate = frame_rate

    def __call__(self, odo_data):
        """
        odo_data: list of dicts, each dict contains IMU data for a frame
        """
        traj, headings = get_trajectory_from_speeds_and_yaw_rates(
            speeds=np.array([odo_frame['vx'] for odo_frame in odo_data]),
            yaw_rates=np.array([odo_frame['angular_rate_z'] for odo_frame in odo_data]),
            dt=1.0/self.frame_rate
        )

        assert traj.shape[1] == 2 and traj.shape[0] == len(odo_data), f"Unexpected odometry shape {traj.shape}, expected ({len(odo_data)}, 2)"
        steering = np.concatenate([traj, headings[:, None]], axis=-1)  # (num_frames, 3)
        return steering


class MultiHDF5DatasetMultiFrameIdxMappingOdometry(MultiHDF5DatasetMultiFrameIdxMapping):
    def __init__(self, size, hdf5_paths_file, num_frames, num_frames_odo=None, frames_file_suffix="frames.h5", odo_files_suffix="odometry.h5", stored_data_frame_rate=5, frame_rate=5, aug='resize_center', scale_min=0.15, scale_max=0.5, odo_transform_config=None):
        self.frames_file_suffix = frames_file_suffix
        self.odo_files_suffix = odo_files_suffix
        self.num_frames_odo = num_frames_odo if num_frames_odo is not None else num_frames

        # Cache for odometry files (lazy loading)
        self.odo_files_cache = {}

        # we use the maximum of num_frames and num_frames_odo to ensure we have enough frames to sample from
        super().__init__(size=size, hdf5_paths_file=hdf5_paths_file, num_frames=max(num_frames, self.num_frames_odo), stored_data_frame_rate=stored_data_frame_rate, frame_rate=frame_rate, aug=aug, scale_min=scale_min, scale_max=scale_max)
        self.num_frames = num_frames  # reset to original num_frames for image sampling

        if odo_transform_config is not None:
            self.odo_transform = instantiate_from_config(odo_transform_config)
        else:
            self.odo_transform = OdometryLoaderNuPlan()

    def get_odo_file(self, path):
        """Get odometry H5 file from cache, opening it if not already cached"""
        if path not in self.odo_files_cache:
            self.odo_files_cache[path] = h5py.File(path, 'r')
        return self.odo_files_cache[path]

    @staticmethod
    def frames_odo_matching_check(frames_h5, odo_h5):
        # check if the odometry file has the same keys and lengths as the frames file
        for key in frames_h5.keys():
            if 'meta_data' not in key:
                assert key in odo_h5.keys(), f'Odometry key {key} not found in frames data'
                assert len(odo_h5[key]) == len(frames_h5[key]), f'Odometry key {key} has different length than frames data'

    def scan_h5_files_odo(self):
        """Scan odometry files to verify matching (opens files temporarily)"""
        if self.odo_files_suffix is not None:
            for h5_path_frames in tqdm(self.hdf5_paths, desc='Scanning HDF5 files for odometry'):
                odo_h5_path = h5_path_frames.replace(self.frames_file_suffix, self.odo_files_suffix)
                # Open both files temporarily just to verify they match
                with h5py.File(h5_path_frames, 'r') as frames_h5, h5py.File(odo_h5_path, 'r') as odo_h5:
                    self.frames_odo_matching_check(frames_h5, odo_h5)

    def scan_h5_files(self):
        super().scan_h5_files()
        self.scan_h5_files_odo()

    def get_odometry(self, filename, key, start_idx):
        odo_filename = filename.replace(self.frames_file_suffix, self.odo_files_suffix)
        # Get odometry file from cache (opens if needed)
        odo_file = self.get_odo_file(odo_filename)
        odo = [odo_file[key][start_idx+i*self.frame_interval] for i in range(self.num_frames_odo)]
        return torch.tensor(self.odo_transform(odo))

    def __getitem__(self, idx):
        images, (filename, key, start_frame) = self.get_images_and_indices(idx)
        images = self.apply_transforms(images)
        odo = self.get_odometry(filename, key, start_frame) if self.odo_files_suffix is not None else None
        return {'images': images, 'steering': odo, 'frame_rate': self.frame_rate}

    def close(self):
        """Close all cached files (frames and odometry)"""
        super().close()
        for file in self.odo_files_cache.values():
            file.close()
        self.odo_files_cache.clear()



class MultiHDF5DatasetMultiFrameFromJSONOdometry(MultiHDF5DatasetMultiFrameFromJSON, MultiHDF5DatasetMultiFrameIdxMappingOdometry):
    def __init__(self, size, samples_json, num_frames, num_samples=800, frames_file_suffix="frames.h5", odo_files_suffix="odometry.h5", stored_data_frame_rate=5, frame_rate=5, odo_transform_config=None):
        frame_rate_multiplier = frame_rate/stored_data_frame_rate
        self.frame_rate = frame_rate
        self.stored_data_frame_rate = stored_data_frame_rate

        MultiHDF5DatasetMultiFrameFromJSON.__init__(self, size, samples_json, num_frames, frame_rate_multiplier=frame_rate_multiplier, num_samples=num_samples)

        self.frames_file_suffix = frames_file_suffix
        self.odo_files_suffix = odo_files_suffix
        # Cache for odometry files (lazy loading)
        self.odo_files_cache = {}

        if odo_files_suffix is not None:
            # Get unique h5 paths from index mapping
            unique_h5_paths = set([filename for filename, _, _ in self.index_to_starting_frame_map])

            for h5_path_frames in unique_h5_paths:
                odo_h5_path = h5_path_frames.replace(frames_file_suffix, odo_files_suffix)
                # Open both files temporarily just to verify they match
                with h5py.File(h5_path_frames, 'r') as frames_h5, h5py.File(odo_h5_path, 'r') as odo_h5:
                    # check if the odometry file has the same keys and lengths as the frames file
                    for key in odo_h5.keys():
                        if 'meta_data' not in key:
                            assert key in frames_h5.keys(), f'Odometry key {key} not found in frames data'
                            assert len(odo_h5[key]) == len(frames_h5[key]), f'Odometry key {key} has different length than frames data'

        if odo_transform_config is not None:
            self.odo_transform = instantiate_from_config(odo_transform_config)
        else:
            self.odo_transform = OdometryLoaderNuPlan()

    def get_odo_file(self, path):
        """Get odometry H5 file from cache, opening it if not already cached"""
        if path not in self.odo_files_cache:
            self.odo_files_cache[path] = h5py.File(path, 'r')
        return self.odo_files_cache[path]

    def __getitem__(self, idx):
        return MultiHDF5DatasetMultiFrameIdxMappingOdometry.__getitem__(self, idx)

    def close(self):
        """Close all cached files (frames and odometry)"""
        super().close()
        for file in self.odo_files_cache.values():
            file.close()
        self.odo_files_cache.clear()

