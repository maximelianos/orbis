"""Validation metrics for trajectory prediction.

Two quantities (see docs/idea.md):
  * MSE  — mean squared error between predicted and GT trajectories.
  * STD of total turn — sampling the same context several times, how much does the
    predicted heading change vary? "Total turn" is the summed turn angle along a
    trajectory; its STD over repeated samples measures multi-modal spread.
"""

import torch


def masked_episode_mean(values, mask):
    """Mean over the frames of each episode, then over the episodes.

    Args:
        values: (..., T) per-frame quantity.
        mask:   (..., T) frame-validity mask, broadcastable against `values`.
    Returns:
        scalar tensor.

    Pooling every frame into one average instead would weight an episode by its
    length. NAVSIM scenes run from 5 to ~40 frames, so the long ones would count
    up to eight times as much as the short ones. Averaging within an episode
    first gives every episode one vote, in the loss and in the metrics alike.
    """
    mask = mask.to(values.dtype).expand_as(values)
    frames = mask.sum(dim=-1)                                    # (...)
    per_episode = (values * mask).sum(dim=-1) / frames.clamp(min=1.0)
    # An episode with no valid frame (all padding) must not pull the mean.
    counts = (frames > 0).to(values.dtype)
    return (per_episode * counts).sum() / counts.sum().clamp(min=1.0)


def trajectory_mse(pred, gt, mask=None):
    """Mean squared error. pred/gt: (..., T, 2).

    Averaged per episode before being averaged across episodes, so the value is
    comparable with the training loss and does not favour long episodes.

    Args:
        mask: optional (..., T) frame-validity mask (broadcastable against the
            leading dims of pred/gt), so padded frames of variable-length
            episodes do not enter the average.
    """
    sq = ((pred - gt) ** 2).mean(-1)                             # (..., T)
    if mask is None:
        # Equal lengths: the per-episode mean and the pooled mean coincide.
        return sq.mean()
    return masked_episode_mean(sq, mask)


def total_turn(trajectory, mask=None, eps=1e-6):
    """Sum of turn angles along a trajectory.

    Args:
        trajectory: (..., T, 2) positions.
        mask: optional (..., T) frame-validity mask. A turn angle at frame i is
            formed by the segments (i, i+1) and (i+1, i+2), so it is counted only
            when frame i+2 is valid — i.e. padded frames contribute nothing.
    Returns:
        (...,) total absolute turn: sum over t of the angle between consecutive
        motion vectors.
    """
    seg = trajectory[..., 1:, :] - trajectory[..., :-1, :]      # (..., T-1, 2)
    heading = torch.atan2(seg[..., 1], seg[..., 0] + eps)        # (..., T-1)
    dtheta = heading[..., 1:] - heading[..., :-1]               # (..., T-2)
    # wrap to [-pi, pi]
    dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))
    if mask is not None:
        dtheta = dtheta * mask[..., 2:].to(dtheta.dtype)
    return dtheta.sum(dim=-1)                                    # (...,)


def turn_std_over_samples(sampled_trajectories, mask=None):
    """STD of total turn across repeated samples of the same context.

    Args:
        sampled_trajectories: (N, B, T, 2) — N runs of the same B contexts.
        mask: optional (B, T) frame-validity mask, broadcast over the N runs.
    Returns:
        scalar tensor: mean over the batch of the per-context STD of total turn.
    """
    turns = total_turn(sampled_trajectories, mask=mask)   # (N, B)
    return turns.std(dim=0).mean()             # STD over N runs, mean over batch
