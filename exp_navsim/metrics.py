"""Validation metrics for trajectory prediction.

Two quantities (see docs/idea.md):
  * MSE  — mean squared error between predicted and GT trajectories.
  * STD of total turn — sampling the same context several times, how much does the
    predicted heading change vary? "Total turn" is the summed turn angle along a
    trajectory; its STD over repeated samples measures multi-modal spread.
"""

import torch


def trajectory_mse(pred, gt):
    """Mean squared error. pred/gt: (..., T, 2)."""
    return ((pred - gt) ** 2).mean()


def total_turn(trajectory, eps=1e-6):
    """Sum of turn angles along a trajectory.

    Args:
        trajectory: (..., T, 2) positions.
    Returns:
        (...,) total absolute turn: sum over t of the angle between consecutive
        motion vectors.
    """
    seg = trajectory[..., 1:, :] - trajectory[..., :-1, :]      # (..., T-1, 2)
    heading = torch.atan2(seg[..., 1], seg[..., 0] + eps)        # (..., T-1)
    dtheta = heading[..., 1:] - heading[..., :-1]               # (..., T-2)
    # wrap to [-pi, pi]
    dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))
    return dtheta.sum(dim=-1)                                    # (...,)


def turn_std_over_samples(sampled_trajectories):
    """STD of total turn across repeated samples of the same context.

    Args:
        sampled_trajectories: (N, B, T, 2) — N runs of the same B contexts.
    Returns:
        scalar tensor: mean over the batch of the per-context STD of total turn.
    """
    turns = total_turn(sampled_trajectories)   # (N, B)
    return turns.std(dim=0).mean()             # STD over N runs, mean over batch
