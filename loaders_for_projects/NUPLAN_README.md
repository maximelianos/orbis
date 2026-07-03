# NuPlan Dataset Loader

Tools for loading and exploring NuPlan datasets stored in HDF5 format.

## Files

- **`nuplan_loader.py`**: Utility classes and functions for programmatic access
- **`../nuplan.py`**: Interactive CLI tool for dataset exploration

## Quick Start

### Interactive Exploration (Recommended for First Use)

Run the interactive script from the project root:

```bash
python nuplan.py
```

The script will guide you through:
1. Finding NuPlan HDF5 files in common locations
2. Listing all episodes in the dataset
3. Selecting an episode to explore
4. Extracting sample frames
5. Saving images to `./nuplan_samples/`
6. Checking for odometry data

### Programmatic Usage

```python
from loaders_for_projects.nuplan_loader import NuPlanHDF5Explorer

# Using context manager (recommended)
with NuPlanHDF5Explorer('/path/to/nuplan_frames.h5') as explorer:
    # List episodes
    print(f"Episodes: {explorer.episodes[:5]}")

    # Get episode info
    info = explorer.get_episode_info(explorer.episodes[0])
    print(f"Frames: {info['num_frames']}")
    print(f"Cameras: {info['camera_keys']}")

    # Sample frames from episode
    indices, frames = explorer.sample_episode_frames(
        episode_key=explorer.episodes[0],
        num_samples=5,
        camera_key='cam_F0'
    )
    print(f"Sampled frames shape: {frames.shape}")

    # Get single frame
    frame = explorer.get_frame(
        episode_key=explorer.episodes[0],
        frame_idx=0,
        camera_key='cam_F0'
    )

    # Get frame sequence
    sequence = explorer.get_frames_sequence(
        episode_key=explorer.episodes[0],
        start_idx=0,
        num_frames=10,
        camera_key='cam_F0',
        stride=1  # consecutive frames
    )

    # Get odometry data (if available)
    odo_data = explorer.get_odometry(
        episode_key=explorer.episodes[0],
        start_idx=0,
        num_frames=10
    )
    if odo_data:
        print(f"Odometry keys: {odo_data[0].keys()}")
```

## Dataset Structure

### HDF5 Files

NuPlan data is stored in paired HDF5 files:

```
nuplan_train_10Hz_640x360_frames.h5    # Image frames
nuplan_train_10Hz_640x360_odometry.h5  # Odometry data
```

### File Organization

```
frames.h5:
├── episode_0001/
│   ├── 0/
│   │   ├── cam_F0    # Front camera
│   │   ├── cam_B0    # Back camera
│   │   ├── cam_L0    # Left camera
│   │   └── cam_R0    # Right camera
│   ├── 1/
│   └── ...
├── episode_0002/
└── ...

odometry.h5:
├── episode_0001/
│   ├── 0/
│   │   ├── vx              # Forward velocity
│   │   ├── angular_rate_z  # Yaw rate
│   │   └── ...
│   ├── 1/
│   └── ...
└── ...
```

### Common Camera Keys

- `cam_F0`: Front camera (forward facing)
- `cam_B0`: Back camera (rear facing)
- `cam_L0`: Left camera
- `cam_R0`: Right camera

### Common Odometry Keys

- `vx`: Forward velocity (m/s)
- `angular_rate_z`: Yaw rate (rad/s)

## Integration with Orbis

### Using with Existing Data Loaders

The NuPlan loader utilities are compatible with the existing multiframe loaders:

```python
from loaders_for_projects.custom_multiframe_odo import (
    MultiHDF5DatasetMultiFrameIdxMappingOdometry
)

# Example configuration (see example_data_config.yaml)
dataset = MultiHDF5DatasetMultiFrameIdxMappingOdometry(
    hdf5_paths_file='/path/to/nuplan_train.txt',  # List of HDF5 files
    size=[256, 256],
    num_frames=10,
    stored_data_frame_rate=10,
    frame_rate=5,
    aug='random_shift',
    scale_min=0.15,
    scale_max=0.5,
)
```

### Creating Training Data Config

Create a `.txt` file listing your HDF5 files:

```bash
# nuplan_train.txt
/path/to/dataset1/nuplan_frames.h5
/path/to/dataset2/nuplan_frames.h5
/path/to/dataset3/nuplan_frames.h5
```

Then reference it in your config YAML:

```yaml
data:
  target: data.datamodule.DataModuleFromConfig
  params:
    batch_size: 16
    num_workers: 4
    train:
      - target: data.custom_multiframe_odo.MultiHDF5DatasetMultiFrameIdxMappingOdometry
        params:
          hdf5_paths_file: /path/to/nuplan_train.txt
          size: [256, 256]
          num_frames: 10
          stored_data_frame_rate: 10
          frame_rate: 5
          aug: random_shift
```

## Utility Functions

### Finding Files

```python
from loaders_for_projects.nuplan_loader import find_nuplan_files

# Find all NuPlan HDF5 files in a directory
files = find_nuplan_files('/path/to/nuplan/dataset')
print(f"Found {len(files)} files")
```

### Getting Available Cameras

```python
from loaders_for_projects.nuplan_loader import get_available_cameras
import h5py

with h5py.File('/path/to/frames.h5', 'r') as f:
    cameras = get_available_cameras(f, 'episode_0001')
    print(f"Available cameras: {cameras}")
```

## Tips

1. **Memory Management**: HDF5 files are memory-mapped by default, so you can work with large datasets efficiently

2. **Frame Rate Conversion**: The loaders support automatic frame rate conversion via the `stored_data_frame_rate` and `frame_rate` parameters

3. **Data Augmentation**: Use `aug='random_shift'` or `aug='random_resize_center'` for training data augmentation

4. **Odometry Integration**: Odometry data is automatically loaded when paired `.h5` files are available

5. **Batch Processing**: Use the `MultiHDF5Dataset` classes for efficient batch loading during training

## Troubleshooting

### File Locking Issues
If you encounter HDF5 file locking errors, the loaders automatically set:
```python
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
```

### Missing Odometry
If odometry data is not found, the loader will return `None` for steering information. This is normal if only frame data is available.

### Camera Key Not Found
Check available cameras using `get_available_cameras()` or inspect the first frame to see which cameras are present in your dataset.
