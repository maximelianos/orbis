import numpy as np
import torch
import torch.nn as nn


def get_trajectory_from_speeds_and_yaw_rates(speeds, yaw_rates, dt):
    headings = np.cumsum(yaw_rates) * dt  # integrate yaw rates to get headings
    dx = speeds * np.cos(headings) * dt
    dy = speeds * np.sin(headings) * dt
    
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    traj = np.stack([x, y], axis=1)
    
    # Transform to local coordinates (first position is origin, first heading is along x-axis)
    traj -= traj[0]  # translate to origin
    initial_heading = headings[0]
    rotation_matrix = np.array([[np.cos(-initial_heading), -np.sin(-initial_heading)],
                                    [np.sin(-initial_heading),  np.cos(-initial_heading)]])
    local_traj = traj @ rotation_matrix.T  # rotate to align with initial heading
    
    return local_traj.astype(np.float32), headings.astype(np.float32)

def get_trajectory_from_speeds_and_yaw_rates_batch(speeds, yaw_rates, dt):
    """
    Args:
        speeds: Tensor of shape (B, N)
        yaw_rates: Tensor of shape (B, N)
        dt: Time step (scalar)
    Returns:
        local_traj: Tensor of shape (B, N, 2)
        headings: Tensor of shape (B, N)
    """
    assert speeds.shape == yaw_rates.shape, f"Speeds shape {speeds.shape} and yaw rates shape {yaw_rates.shape} do not match"
    B, N = speeds.shape
    
    # if dt is a scalar, ok, if dt is a tensor, make sure it has shape (B) and expand to (B, 1)
    if isinstance(dt, torch.Tensor):
        assert dt.shape == (B,), f"dt shape {dt.shape} does not match batch size {B}"
        dt = dt.view(B, 1)  # Shape: (B, 1)
    
    # Integrate yaw rates to get headings for each batch
    headings = torch.cumsum(yaw_rates, dim=1) * dt  # Shape: (B, N)

    # Calculate dx and dy for each batch
    dx = speeds * torch.cos(headings) * dt  # Shape: (B, N)
    dy = speeds * torch.sin(headings) * dt  # Shape: (B, N)

    # Calculate x and y for each batch
    x = torch.cumsum(dx, dim=1)  # Shape: (B, N)
    y = torch.cumsum(dy, dim=1)  # Shape: (B, N)

    # Stack x and y to form the trajectory for each batch
    traj = torch.stack([x, y], dim=2)  # Shape: (B, N, 2)

    # Transform to local coordinates for each batch
    traj = traj- traj[:, 0:1, :]  # Translate to origin for each batch
    initial_heading = headings[:, 0]  # Shape: (B,)

    # Create rotation matrices for each batch
    cos_theta = torch.cos(-initial_heading)  # Shape: (B,)
    sin_theta = torch.sin(-initial_heading)  # Shape: (B,)

    # Rotation matrix for each batch
    rotation_matrix = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=1),
        torch.stack([sin_theta,  cos_theta], dim=1)
    ], dim=1)  # Shape: (B, 2, 2)

    # Rotate to align with initial heading for each batch
    local_traj = torch.einsum('bni,bij->bnj', traj, rotation_matrix)  # Shape: (B, N, 2)

    return torch.cat([local_traj, headings.unsqueeze(-1)], dim=-1).float()  # Return (B, N, 3)

