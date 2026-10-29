"""Metrics for a sampled distribution of futures against the one future that happened.

samples (K, B, 8, >=2) and gt (B, 8, >=2) positions in metres; only x, y are used.
Every function returns per-scene values (B,), so callers can average, stratify or
bootstrap them.

    ade_mean / fde_mean   average over samples of the displacement error
    ade_min / fde_min     best of K (minADE_K): does any sample cover the truth
    ade_of_mean           error of the sample mean, the conditional-mean answer
    energy                energy score, a proper scoring rule: lower is better and
                          it is minimized only by the true distribution
                          ES = E||X - y|| - 1/2 E||X - X'||   (trajectory = 16-vector)
    spread                E||X - X'|| over pairs of samples, as average displacement
    dispersion            spread / E ADE(X, y): about 1 for a calibrated sampler,
                          < 1 overconfident (collapsed), > 1 too wide
    lat_std / lon_std     std over samples of the final lateral / longitudinal position
"""

import torch


def pairwise_mean(x, dist):
    """Mean of dist(x_k, x_l) over k != l. x (K, B, ...) -> (B,)."""
    K = x.shape[0]
    d = dist(x[:, None], x[None])                            # (K, K, B)
    return d.sum((0, 1)) / max(K * (K - 1), 1)


def sample_metrics(samples, gt):
    xy, y = samples[..., :2].float(), gt[..., :2].float()
    disp = (xy - y).norm(dim=-1)                             # (K, B, 8)
    ade, fde = disp.mean(-1), disp[..., -1]                  # (K, B)
    flat, y_flat = xy.flatten(-2), y.flatten(-2)             # (K, B, 16)

    ade_pair = lambda a, b: (a - b).norm(dim=-1).mean(-1)    # trajectories (..., 8, 2)
    spread = pairwise_mean(xy, ade_pair)
    energy = (flat - y_flat).norm(dim=-1).mean(0) - 0.5 * pairwise_mean(flat, lambda a, b: (a - b).norm(dim=-1))
    return {
        "ade_mean": ade.mean(0),
        "fde_mean": fde.mean(0),
        "ade_min": ade.min(0).values,
        "fde_min": fde.min(0).values,
        "ade_of_mean": (xy.mean(0) - y).norm(dim=-1).mean(-1),
        "energy": energy,
        "spread": spread,
        "dispersion": spread / ade.mean(0).clamp(min=1e-3),
        "lon_std": xy[..., -1, 0].std(0),
        "lat_std": xy[..., -1, 1].std(0),
    }
