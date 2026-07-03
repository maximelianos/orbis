"""
Visualization script for synthetic trajectories.

Creates a plot showing 10 different trajectories with varying turn parameters.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders_for_projects.synthetic_trajectory_dataloader import SyntheticTrajectoryDataset


def plot_trajectory(trajectory, ax=None, color='blue', label=None, linewidth=2, alpha=0.7, 
                   show_markers=True):
    """
    Plot a single trajectory.
    
    Args:
        trajectory: (T, 2) array of (x, y) positions (numpy or torch tensor)
        ax: Matplotlib axis (creates new if None)
        color: Line color
        label: Legend label
        linewidth: Line width
        alpha: Transparency
        show_markers: Whether to show start/end markers
        
    Returns:
        ax: Matplotlib axis with the plot
    """
    import torch
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convert to numpy if needed
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.cpu().numpy()
    
    x, y = trajectory[:, 0], trajectory[:, 1]
    
    # Plot trajectory line
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, label=label)
    
    # Show start and end markers
    if show_markers:
        ax.scatter(x[0], y[0], color=color, s=100, marker='o', 
                  edgecolors='black', linewidths=1.5, zorder=5)
        ax.scatter(x[-1], y[-1], color=color, s=100, marker='s', 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    return ax


def plot_trajectories(trajectories, labels=None, title='Trajectories', 
                      output_path=None, show_origin=True):
    """
    Plot multiple trajectories on the same axis.
    
    Args:
        trajectories: List of (T, 2) arrays or single (N, T, 2) array
        labels: Optional list of labels for each trajectory
        title: Plot title
        output_path: Path to save the plot (if None, displays instead)
        show_origin: Whether to mark the origin (0, 0)
        
    Returns:
        fig, ax: Matplotlib figure and axis
    """
    import torch
    
    # Handle different input formats
    if isinstance(trajectories, (torch.Tensor, np.ndarray)):
        if trajectories.ndim == 3:  # (N, T, 2)
            trajectories = [trajectories[i] for i in range(trajectories.shape[0])]
    
    num_traj = len(trajectories)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_traj))
    
    # Plot each trajectory
    for i, traj in enumerate(trajectories):
        label = labels[i] if labels is not None else None
        plot_trajectory(traj, ax=ax, color=colors[i], label=label)
    
    # Mark origin
    if show_origin:
        ax.scatter(0, 0, color='green', s=200, marker='*', 
                  edgecolors='black', linewidths=2, zorder=10, label='Start (0,0)')
    
    # Formatting
    ax.set_xlabel('X position (m)', fontsize=12)
    ax.set_ylabel('Y position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    if labels is not None or show_origin:
        ax.legend()
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        plt.close()
    else:
        plt.show()
    
    return fig, ax


def visualize_trajectories(num_trajectories=10, T=50, r=10.0, output_path='synthetic_trajectories.png'):
    """
    Visualize multiple synthetic trajectories with different turn parameters.
    
    Args:
        num_trajectories: Number of trajectories to plot
        T: Number of timesteps per trajectory
        r: Radius/distance to trajectory endpoint
        output_path: Path to save the output image
    """
    # Create dataset with fixed turn values uniformly distributed
    dataset = SyntheticTrajectoryDataset(size=num_trajectories, T=T, r=r)
    
    # Override with uniform turn values for better visualization
    dataset.turns = np.linspace(0, 1, num_trajectories)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map for trajectories
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_trajectories))
    
    # Plot each trajectory
    for i in range(num_trajectories):
        sample = dataset[i]
        turn = sample['turn'].item()
        position = sample['position'].numpy()  # (T, 2)
        
        x = position[:, 0]
        y = position[:, 1]
        
        # Plot trajectory
        ax.plot(x, y, color=colors[i], linewidth=2, alpha=0.7, 
                label=f'turn={turn:.2f}')
        
        # Mark start and end points
        ax.scatter(x[0], y[0], color=colors[i], s=100, marker='o', 
                  edgecolors='black', linewidths=1.5, zorder=5)
        ax.scatter(x[-1], y[-1], color=colors[i], s=100, marker='s', 
                  edgecolors='black', linewidths=1.5, zorder=5)
    
    # Add origin marker
    ax.scatter(0, 0, color='green', s=200, marker='*', 
              edgecolors='black', linewidths=2, zorder=10, label='Start (0,0)')
    
    # Set labels and title
    ax.set_xlabel('X position (m)', fontsize=12)
    ax.set_ylabel('Y position (m)', fontsize=12)
    ax.set_title(f'Synthetic Trajectories with Varying Turn Parameters\n'
                 f'(turn=0: left/30°, turn=1: right/-30°)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    # Add annotations
    ax.text(0.02, 0.98, f'○ = start point\n□ = end point', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()
    
    # Create a second plot showing turn parameter vs endpoint angle
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot endpoint positions
    turns_array = np.linspace(0, 1, num_trajectories)
    endpoints_x = []
    endpoints_y = []
    
    for i in range(num_trajectories):
        sample = dataset[i]
        position = sample['position'].numpy()
        endpoints_x.append(position[-1, 0])
        endpoints_y.append(position[-1, 1])
    
    ax1.scatter(turns_array, endpoints_x, color='blue', s=50, label='X coordinate')
    ax1.scatter(turns_array, endpoints_y, color='red', s=50, label='Y coordinate')
    ax1.set_xlabel('Turn parameter', fontsize=12)
    ax1.set_ylabel('Endpoint coordinate (m)', fontsize=12)
    ax1.set_title('Endpoint Coordinates vs Turn Parameter', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot angle vs turn
    angles = []
    for x, y in zip(endpoints_x, endpoints_y):
        angle_rad = np.arctan2(y, x)
        angle_deg = np.rad2deg(angle_rad)
        angles.append(angle_deg)
    
    ax2.plot(turns_array, angles, marker='o', color='green', linewidth=2)
    ax2.set_xlabel('Turn parameter', fontsize=12)
    ax2.set_ylabel('Endpoint angle (degrees)', fontsize=12)
    ax2.set_title('Endpoint Angle vs Turn Parameter', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(30, color='blue', linestyle='--', alpha=0.5, label='30° (turn=0)')
    ax2.axhline(-30, color='red', linestyle='--', alpha=0.5, label='-30° (turn=1)')
    ax2.legend()
    
    plt.tight_layout()
    output_path2 = output_path.replace('.png', '_analysis.png')
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved analysis to {output_path2}")
    plt.close()


if __name__ == "__main__":
    # Visualize 10 trajectories
    visualize_trajectories(num_trajectories=10, T=50, r=1.0, 
                          output_path='loaders_for_projects/synthetic_trajectories.png')
    
    print("\nVisualization complete!")
