import os
import json
import math
import random
import importlib
import time
from collections import OrderedDict

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torchvision.transforms.v2.functional as FV2

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"



class RandomResizedCenterCrop(object):
    def __init__(self, size, scale=(0.5, 1.0), interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation
        self.size = size
        self.fixed_params = None

    def get_params(self, img):
        if self.fixed_params is None:
            width, height = img.size
            area = height * width
            aspect_ratio = width / height

            target_area = random.uniform(*self.scale) * area

            new_width = int(round((target_area * aspect_ratio) ** 0.5))
            new_height = int(round((target_area / aspect_ratio) ** 0.5))
            x1 = (new_width - self.size) // 2
            y1 = (new_height - self.size) // 2
            self.fixed_params = (new_width, new_height, x1, y1)
        return self.fixed_params

    def __call__(self, img):
        new_width, new_height, x1, y1 = self.get_params(img)
        img = img.resize((new_width, new_height), self.interpolation)
        return img.crop((x1, y1, x1 + self.size, y1 + self.size))

    def reset(self):
        self.fixed_params = None

class RandomHorizontalShiftCrop(object):
    def __init__(self, size, max_shift=60):
        """
        size: crop size (assumed square or tuple)
        max_shift: maximum horizontal shift in pixels (both directions)
        """
        self.size = size if isinstance(size, tuple) else (size, size)
        self.max_shift = max_shift
        self.fixed_params = None

    def get_params(self, img):
        if self.fixed_params is None:
            width, height = img.size
            crop_height, crop_width = self.size

            # vertical center
            top = (height - crop_height) // 2

            # horizontal center + random shift
            center_left = (width - crop_width) // 2
            shift = random.randint(-self.max_shift, self.max_shift)
            left = center_left + shift

            # clamp to image bounds
            left = max(0, min(left, width - crop_width))

            self.fixed_params = (left, top)
        return self.fixed_params

    def __call__(self, img):
        left, top = self.get_params(img)
        crop_height, crop_width = self.size
        return img.crop((left, top, left + crop_width, top + crop_height))

    def reset(self):
        self.fixed_params = None

class RandomShiftCrop(object):
    def __init__(self, size, max_shift_horizontal=60, max_shift_vertical=60):
        """
        size: Crop size. If an int is provided, the crop will be (size, size).
              If a tuple is provided, it should be (crop_width, crop_height).
        max_shift_horizontal: Maximum horizontal shift (in pixels) from the center of the crop.
        max_shift_vertical: Maximum vertical shift (in pixels) from the center of the crop.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.max_shift_horizontal = max_shift_horizontal
        self.max_shift_vertical = max_shift_vertical
        self.fixed_params = None

    def get_params(self, img):
        if self.fixed_params is None:
            width, height = img.size
            crop_height, crop_width = self.size

            # Calculate the center coordinates for the crop
            center_left = (width - crop_width) // 2
            center_top = (height - crop_height) // 2

            # Apply random horizontal and vertical shifts
            shift_horizontal = random.randint(-self.max_shift_horizontal, self.max_shift_horizontal)
            shift_vertical = random.randint(-self.max_shift_vertical, self.max_shift_vertical)

            left = center_left + shift_horizontal
            top = center_top + shift_vertical

            # Clamp the values to ensure the crop is entirely within the image boundaries
            left = max(0, min(left, width - crop_width))
            top = max(0, min(top, height - crop_height))

            self.fixed_params = (left, top)
        return self.fixed_params

    def __call__(self, img):
        left, top = self.get_params(img)
        crop_height, crop_width = self.size
        return img.crop((left, top, left + crop_width, top + crop_height))

    def reset(self):
        self.fixed_params = None


class NumpyToTensor:
    def __call__(self, x):
        assert isinstance(x, np.ndarray), f'input must be a numpy array, got {type(x)}'
        assert x.ndim == 3, 'input must be a 3D array'
        return torch.from_numpy(x).permute(2, 0, 1)

class VQGANPreprocess:
    def __call__(self, x):
        assert isinstance(x, torch.Tensor), 'input must be a tensor'
        return x / 127.5 - 1.0

    def inverse(self, x):
        assert isinstance(x, torch.Tensor), 'input must be a tensor'
        return (x + 1.0) * 127.5


class MultiHDF5DatasetMultiFrameIdxMapping(Dataset):
    '''
    This dataset maps each index to a specific frame in a specific video. Useful for validation and selecting subsets of frames.
    num_frames: number of frames to return for each index

    '''
    def __init__(self, size, hdf5_paths_file, num_frames, stored_data_frame_rate=5, frame_rate=5, aug='resize_center', scale_min=0.15, scale_max=0.5, whole_episodes=False):

        self.frame_interval = int(stored_data_frame_rate/frame_rate)
        self.frame_rate = frame_rate
        self.stored_data_frame_rate = stored_data_frame_rate

        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames
        # whole_episodes: keep only the subvideo starting at the first frame of each
        # video (drop all other sliding windows). See scan_h5_files below.
        self.whole_episodes = whole_episodes
        self.hdf5_paths_file = hdf5_paths_file
        # expand environment variables in path
        with open(os.path.expandvars(hdf5_paths_file), 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        # Cache for opened files (only store filenames, open on demand)
        self.hdf5_files_cache = {}

        # map each index to a specific (filename, key, starting_frame)
        self.scan_h5_files()

        self.aug = aug
        if self.aug == 'resize_center':
            self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                         transforms.CenterCrop(self.size),
                                         transforms.ToTensor(),
                                         ])
        elif self.aug == 'random_resize_center':
            self.custom_crop = RandomResizedCenterCrop(size=self.size, scale=(scale_min, scale_max))
            self.transform = transforms.Compose([
                                        self.custom_crop,
                                        transforms.ToTensor(),
                                        ])
        elif self.aug == 'random_shift':
            self.custom_crop = RandomShiftCrop(size=self.size, max_shift_horizontal=60, max_shift_vertical=30)
            self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                        self.custom_crop,
                                        transforms.ToTensor(),
                                        ])
        else:
            raise ValueError(f'Unknown augmentation type: {self.aug}')

    def get_h5_file(self, path):
        """Get H5 file from cache, opening it if not already cached"""
        if path not in self.hdf5_files_cache:
            self.hdf5_files_cache[path] = h5py.File(path, 'r')
        return self.hdf5_files_cache[path]

    def scan_h5_files(self):
        """Scan H5 files and build index mapping (opens files temporarily)"""
        self.index_to_starting_frame_map = []
        for path in tqdm(self.hdf5_paths, desc='Scanning HDF5 files'):
            # Open file temporarily just to scan
            with h5py.File(path, 'r') as file:
                keys = list(file.keys())
                for key in keys:
                    video_length = len(file[key])
                    # we take every nth frame, as long as we can get num_frames frames after that
                    max_frame_index = video_length - self.num_frames*self.frame_interval-1
                    for i in range(0, max_frame_index + 1):
                        # Store filename instead of file object
                        self.index_to_starting_frame_map.append((path, key, i))
                        # whole_episodes: only keep the subvideo starting at the first
                        # frame, drop all the other sliding windows of this video
                        if self.whole_episodes:
                            break

    def apply_same_transform_to_all(self, frames, transform):
        return [transform(frame) for frame in frames]

    def __len__(self):
        return len(self.index_to_starting_frame_map)

    def __str__(self):
        s = f'MultiHDF5DatasetMultiFrameIdxMapping({self.hdf5_paths_file}, num_samples={len(self)}, size={self.size}, num_frames={self.num_frames}, frame_interval={self.frame_interval})'
        return s

    def get_images_and_indices(self, idx):
        if idx >= len(self.index_to_starting_frame_map):
            raise IndexError(f'Index {idx} out of range for dataset of length {len(self.index_to_starting_frame_map)}')
        filename, key, start_frame = self.index_to_starting_frame_map[idx]
        # Get file from cache (opens if needed)
        file = self.get_h5_file(filename)
        images = [Image.fromarray(file[key][start_frame+i*self.frame_interval]) for i in range(self.num_frames)]
        return images, (filename, key, start_frame)

    def apply_transforms(self, images):
        if self.aug == 'random_resize_center' or self.aug == 'random_shift':
            self.custom_crop.reset()
        images = self.apply_same_transform_to_all(images, self.transform)
        return torch.stack(images, dim=0)*2-1

    def __getitem__(self, idx):
        images, _ = self.get_images_and_indices(idx)
        images = self.apply_transforms(images)
        return {'images': images, 'frame_rate': self.frame_rate}

    def close(self):
        """Close all cached files"""
        for file in self.hdf5_files_cache.values():
            file.close()
        self.hdf5_files_cache.clear()


class MultiHDF5DatasetMultiFrameFromJSON(MultiHDF5DatasetMultiFrameIdxMapping):
    """
    The structure of the JSON file should be as follows:
    [
        {
            "h5_path": <PATH TO THE H5 FILE CONTAINING THE VIDEO>,
            "video_key": <KEY/NAME OF THE VIDEO, e.g. 53773fdf-311fd624>
            "start_frame": <STARTING FRAME INDEX>
        }
    ]
    """
    def __init__(self, size, samples_json, num_frames, stored_data_frame_rate, frame_rate, num_samples=800):
        self.frame_interval = int(stored_data_frame_rate/frame_rate)
        self.frame_rate = frame_rate
        self.stored_data_frame_rate = stored_data_frame_rate

        self.samples_json = samples_json
        self.size = (size, size) if isinstance(size, int) else size
        self.num_frames = num_frames

        # read json
        with open(os.path.expandvars(samples_json), 'r') as f:
            self.samples = json.load(f)[:num_samples]

        # Cache for opened files (only filenames, open on demand)
        self.hdf5_files_cache = {}
        self.index_to_starting_frame_map = []

        for sample in self.samples:
            h5_path = sample['h5_path']
            key = sample['video_key']
            start_frame = sample['start_frame']
            # Store filename instead of file object
            self.index_to_starting_frame_map.append((h5_path, key, start_frame))

        self.aug = 'resize_center'
        self.transform = transforms.Compose([transforms.Resize(min(self.size)),
                                         transforms.CenterCrop(self.size),
                                         transforms.ToTensor(),
                                         ])

    def get_h5_file(self, path):
        """Get H5 file from cache, opening it if not already cached"""
        if path not in self.hdf5_files_cache:
            self.hdf5_files_cache[path] = h5py.File(path, 'r')
        return self.hdf5_files_cache[path]

    def get_images_and_indices(self, idx):
        filename, key, start_frame = self.index_to_starting_frame_map[idx]
        # Get file from cache (opens if needed)
        file = self.get_h5_file(filename)
        if len(file[key])<=start_frame+self.num_frames:
            start_frame = len(file[key]) - self.num_frames
        images = [Image.fromarray(file[key][start_frame+i*self.frame_interval]) for i in range(self.num_frames)]
        return images, (filename, key, start_frame)

    def close(self):
        """Close all cached files"""
        for file in self.hdf5_files_cache.values():
            file.close()
        self.hdf5_files_cache.clear()

    def __str__(self):
        s = f'MultiHDF5DatasetMultiFrameIdxMapping({self.samples_json}, num_samples={len(self)}, size={self.size}, num_frames={self.num_frames}, frame_interval={self.frame_interval})'
        return s

