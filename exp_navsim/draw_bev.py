"""Draw a predicted trajectory distribution on top of a NAVSIM birds-eye-view.

    # standalone smoke check (real BEV + GT, synthetic fan of "predictions")
    python -m exp_navsim.draw_bev --config exp_navsim/config.yaml --num 3 \
        --out exp_navsim/bev_smoke.pdf

For the real thing, `test_model.py` calls `draw_bev_distribution` through
`draw.draw_episode` with the model's samples:

    python -m exp_navsim.test_model --config exp_navsim/config.yaml \
        --ckpt logs_navsim/<run>/checkpoints/last.ckpt --num 6

Companion to `exp_navsim/data/bev_extract.py`: that module produces the BEV
context (map api + anchor frame + GT path), this one renders it onto a single
matplotlib ax and overlays the sampled trajectories.

Independent of `visualize_cv_agent.py` — it takes plain (T, 2) arrays instead of
navsim `Trajectory` dataclasses, so the model's raw output can be drawn without
having to fake a `TrajectorySampling`.

Axis convention (from `navsim.visualization`): the ax plots **(y, x)** — screen x
is the ego's lateral axis, screen y is forward — and `configure_bev_ax` inverts
the x axis so positive y (left) appears on the left. Every helper here follows
that, and `_fit_limits` re-applies the inversion after widening the view.
"""

import warnings
from contextlib import contextmanager

import numpy as np
import torch

# Colors chosen to match `draw._plot_bev`: GT blue, predictions red.
GT_COLOR = "tab:blue"
PRED_COLOR = "tab:red"


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _plot_path(ax, traj, **kw):
    """Plot a (T, 2) ego-frame path in BEV axis order, prefixed with the origin."""
    traj = _to_numpy(traj)[:, :2]
    traj = np.concatenate([np.zeros((1, 2), dtype=traj.dtype), traj])
    return ax.plot(traj[:, 1], traj[:, 0], **kw)


def _fit_limits(ax, paths, margin=8.0, min_half_extent=16.0):
    """Widen the BEV window so every path fits, keeping equal aspect + inverted x.

    `configure_bev_ax` clamps to a fixed +-32 m box; long episodes drive out of it,
    so the view is re-centred on the union of all paths (a square, so the equal
    aspect ratio does not letterbox differently per panel).
    """
    pts = [np.zeros((1, 2))] + [_to_numpy(p)[:, :2] for p in paths if p is not None]
    pts = np.concatenate(pts, axis=0)
    cx, cy = pts[:, 1].mean(), pts[:, 0].mean()          # screen x = ego y
    half = max(
        np.abs(pts[:, 1] - cx).max(),
        np.abs(pts[:, 0] - cy).max(),
    ) + margin
    half = max(half, min_half_extent)
    # x limits stay reversed: left (+y) on the left, as configure_bev_ax set up.
    ax.set_xlim(cx + half, cx - half)
    ax.set_ylim(cy - half, cy + half)


def draw_bev_distribution(
    ax,
    bev,
    gt_trajectory=None,
    pred_trajectories=None,
    title="BEV — predicted distribution",
    max_samples=None,
    fit_limits=True,
):
    """Render map + agents, then overlay the GT path and the sampled distribution.

    Args:
        ax: matplotlib ax to draw into.
        bev: dict from `bev_extract.extract_bev` (needs "map_api" and "frame").
        gt_trajectory: (T, 2) ego-frame ground-truth path. Falls back to
            `bev["trajectory"]`; pass the batch's own trajectory to plot exactly
            what the model was scored against.
        pred_trajectories: (N, T, 2) sampled paths, or a single (T, 2) path.
        max_samples: draw at most this many samples (thins dense distributions).
        fit_limits: widen the view to contain all paths instead of the fixed
            +-32 m NAVSIM box.

    Returns the ax. Safe to call with `pred_trajectories=None` (plain BEV + GT).
    """
    from navsim.visualization.bev import add_configured_bev_on_ax
    from navsim.visualization.plots import configure_ax, configure_bev_ax

    add_configured_bev_on_ax(ax, bev["map_api"], bev["frame"])
    configure_bev_ax(ax)
    configure_ax(ax)

    if gt_trajectory is None:
        gt_trajectory = bev.get("trajectory")

    preds = None
    if pred_trajectories is not None:
        preds = _to_numpy(pred_trajectories)
        if preds.ndim == 2:                       # a single path -> (1, T, 2)
            preds = preds[None]
        if max_samples is not None and len(preds) > max_samples:
            preds = preds[np.linspace(0, len(preds) - 1, max_samples, dtype=int)]

    # Samples first (thin + translucent, so overlap reads as density), GT on top.
    if preds is not None and len(preds):
        alpha = float(np.clip(3.0 / len(preds), 0.06, 0.8))
        for pred in preds:
            _plot_path(ax, pred, color=PRED_COLOR, lw=1.0, alpha=alpha, zorder=4)
        endpoints = preds[:, -1, :2]
        ax.scatter(endpoints[:, 1], endpoints[:, 0], c=PRED_COLOR, s=60,
                   alpha=0.3, edgecolors="none", zorder=5)
        # One opaque proxy line for the legend (the real ones are too faint).
        ax.plot([], [], color=PRED_COLOR, lw=1.5,
                label=f"predicted ({len(preds)})")

    if gt_trajectory is not None:
        _plot_path(ax, gt_trajectory, color=GT_COLOR, lw=2.0, marker="o",
                   markersize=3, zorder=6, label="GT")

    if fit_limits:
        _fit_limits(ax, ([gt_trajectory] if gt_trajectory is not None else [])
                    + (list(preds) if preds is not None else []))

    if title:
        subtitle = f"{bev.get('ego_speed', float('nan')):.1f} m/s"
        ax.set_title(f"{title}\n{subtitle}", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
    return ax


def _main():
    """Smoke check: one page per episode, BEV + GT + a fake spread of predictions."""
    import argparse

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from omegaconf import OmegaConf

    from exp_navsim.data.bev_extract import extract_bev
    from exp_navsim.data.navsim_base import NavsimLongBase

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_navsim/config.yaml")
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--out", default="exp_navsim/bev_smoke.pdf")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    ds = NavsimLongBase.from_config(cfg, load_surround=False)
    rng = np.random.default_rng(0)

    with PdfPages(args.out) as pdf:
        for i in range(min(args.num, len(ds))):
            bev = extract_bev(ds, i)
            if bev is None:
                continue
            gt = bev["trajectory"]
            # Fake "predictions": GT plus a growing lateral fan, just to check drawing.
            ramp = np.linspace(0, 1, len(gt))[None, :, None] ** 2      # (1, T, 1)
            lateral = rng.normal(0, 2.0, size=(20, 1, 1)) * ramp       # (N, T, 1)
            preds = gt[None] + np.concatenate(
                [np.zeros_like(lateral), lateral], axis=-1)            # (N, T, 2)
            fig, ax = plt.subplots(figsize=(6, 6))
            draw_bev_distribution(ax, bev, gt, preds,
                                  title=f"episode {i} — {bev['map_name']}")
            pdf.savefig(fig, dpi=120, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    _main()
