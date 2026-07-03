import torch
import numpy as np
import matplotlib.pyplot as plt

def save_pictures(trajectories, title, images=None, labels=None, output_path='nuplan_visualization.png'):
    """
    Plot 5 sample images and the x-y graph of the motion.
    
    Args:
        images: (optional) array of shape [num_frames, C, H, W]
        trajectories: list of arrays/tensors each with shape [num_frames, 2]
        title: plot title
        labels: optional list of labels for trajectories
        output_path: path to save the visualization (optional)

    Returns:
        fig: matplotlib figure
    """
    processed_trajectories = []
    for trajectory in trajectories:
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.cpu().numpy()
        trajectory = np.asarray(trajectory)
        assert trajectory.ndim == 2 and trajectory.shape[1] == 2, (
            f"Each trajectory must have shape (T, 2), got {trajectory.shape}"
        )
        processed_trajectories.append(trajectory)

    if labels is not None:
        assert len(labels) == len(processed_trajectories), (
            f"labels length ({len(labels)}) must match number of trajectories ({len(processed_trajectories)})"
        )
    
    # Select 5 frames evenly spaced
    num_frames = trajectories[0].shape[0]
    indices = np.linspace(0, num_frames - 1, 5, dtype=int)
    
    # Create figure with 2 rows: top for images, bottom for trajectory
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    
    # Plot 5 images
    if images is not None:
        # Convert tensors to numpy
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        for i, idx in enumerate(indices):
            ax = plt.subplot(2, 5, i + 1)
            img = images[idx]
            
            # Convert from CHW to HWC and normalize to [0, 1]
            if img.shape[0] == 3:  # RGB
                img = np.transpose(img, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            
            ax.imshow(img)
            ax.set_title(f'Frame {idx}')
            ax.axis('off')
    
    # Plot trajectory (x-y graph)
    ax_traj = plt.subplot(2, 1, 2)

    for i in range(len(processed_trajectories)):
        trajectory = processed_trajectories[i]
        x = trajectory[:, 0]
        y = trajectory[:, 1]

        if labels:
            ax_traj.plot(x, y, linewidth=2, label=labels[i])
        else:
            ax_traj.plot(x, y, linewidth=2)
        sample_indices = np.clip(indices, 0, len(trajectory) - 1)
        ax_traj.scatter(x[sample_indices], y[sample_indices], s=70, zorder=5)
        ax_traj.scatter(x[0], y[0], c='green', s=120, marker='o', zorder=6)
        ax_traj.scatter(x[-1], y[-1], c='orange', s=120, marker='s', zorder=6)
    
    ax_traj.set_xlabel('X position (m)', fontsize=12)
    ax_traj.set_ylabel('Y position (m)', fontsize=12)
    ax_traj.set_title('Vehicle Trajectory', fontsize=14, fontweight='bold')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.axis('equal')
    ax_traj.legend()
    
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
