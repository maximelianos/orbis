"""Episode visualization for exp_navsim.

`draw_episode` takes a *batch* dict and plots ONE sample from it. It is
field-driven: it only draws the panels whose fields are present in the batch.

Panels (top -> bottom):
  * top row: context front views          (needs "context_views") + the map BEV
             with the predicted distribution (needs "bev", see draw_bev.py)
  * 5 evenly spaced camera observations   (needs "cameras" or "images")
  * decoded camera observations           (needs "decoded" or "decoded_images")
  * birds-eye-view (top-down path, equal aspect, heading + start/end markers)
  * trajectory coordinates x(t), y(t)     (needs "trajectory"; extra
                                           "pred_trajectory" / "pred_trajectories"
                                           are overlaid on the BEV)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _pick(batch, key, sample_idx):
    """Fetch batch[key][sample_idx] as numpy, or None if absent."""
    if key not in batch or batch[key] is None:
        return None
    v = _to_numpy(batch[key])
    return v[sample_idx] if v.ndim > 0 and len(v) > sample_idx else v


def _pick_object(batch, key, sample_idx):
    """Like `_pick` but for a plain per-sample list of python objects (no numpy cast)."""
    values = batch.get(key)
    if values is None or sample_idx >= len(values):
        return None
    return values[sample_idx]


def _bev_panel(bev, trajectory, pred_runs):
    """(label, draw_fn) for the map-BEV + predicted-distribution panel.

    Drawing lives in draw_bev.py and pulls in navsim/nuPlan, so the import is lazy
    and a failure degrades to an explanatory placeholder instead of losing the page.
    The panel sets no title of its own — `_named_row` applies the label below.
    """
    label = "BEV — predicted distribution"
    if bev.get("ego_speed") is not None:
        label += f"\n{bev['ego_speed']:.1f} m/s"

    def _draw(ax):
        try:
            from exp_navsim.draw_bev import draw_bev_distribution
            draw_bev_distribution(ax, bev, gt_trajectory=trajectory,
                                  pred_trajectories=pred_runs, title=None)
        except Exception as e:                       # missing maps / navsim / nuplan
            ax.text(0.5, 0.5, f"BEV unavailable:\n{e}", ha="center", va="center",
                    fontsize=7, wrap=True, transform=ax.transAxes)
            ax.axis("off")

    return label, _draw


def _show_image(ax, img):
    """img: (C, H, W) or stacked surround (n_cam, C, H, W) -> displayed HWC."""
    img = _to_numpy(img)
    if img.ndim == 4:  # (n_cam, C, H, W) -> concatenate horizontally
        img = np.concatenate([np.transpose(c, (1, 2, 0)) for c in img], axis=1)
    elif img.shape[0] in (1, 3):  # (C, H, W)
        img = np.transpose(img, (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    ax.imshow(img)
    ax.axis("off")


def _plot_bev(ax, trajectory, extras):
    """Top-down path with heading arrows and any predicted overlays."""
    traj = _to_numpy(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], "-", lw=2, color="tab:blue", label="GT", zorder=3)
    ax.scatter(*traj[0], c="green", s=120, marker="o", zorder=6)
    ax.scatter(*traj[-1], c="orange", s=120, marker="s", zorder=6)
    # heading arrows along the path
    endpoints = []
    for _, pred in extras:
        pred = _to_numpy(pred)
        ax.plot(pred[:, 0], pred[:, 1], "--", lw=1.5, alpha=0.8, zorder=4)
        endpoints.append(pred[-1])
    if endpoints:
        # endpoint distribution: overlapping endpoints stack up into darker spots
        endpoints = np.asarray(endpoints)
        ax.scatter(endpoints[:, 0], endpoints[:, 1], c="red", s=100, alpha=0.25,
                   edgecolors="none", label="pred endpoints", zorder=5)
    ax.set_title("Birds-eye-view", fontweight="bold")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.axis("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)


def draw_episode(batch, sample_idx=0, title="", output_path=None):
    """Plot one sample from `batch`. Returns the matplotlib figure."""
    cameras = _pick(batch, "cameras", sample_idx)
    if cameras is None:
        cameras = _pick(batch, "images", sample_idx)
    decoded = _pick(batch, "decoded", sample_idx)
    if decoded is None:
        decoded = _pick(batch, "decoded_images", sample_idx)
    # First/last context front views (list of named frames), drawn as the top row.
    context_views = _pick(batch, "context_views", sample_idx)
    context_view_labels = batch.get("context_view_labels")
    trajectory = _pick(batch, "trajectory", sample_idx)
    # Map BEV context (list of per-sample dicts from data/bev_extract.py, or None).
    bev = _pick_object(batch, "bev", sample_idx)

    # collect predicted-trajectory overlays (single or a list of runs)
    extras = []
    pred_runs = None                        # (N, T, 2), for the map-BEV panel
    if "pred_trajectory" in batch:
        pred = _pick(batch, "pred_trajectory", sample_idx)
        extras.append(("pred", pred))
        pred_runs = pred[None]
    if "pred_trajectories" in batch:  # (B, N, T, 2): several runs of the same context
        runs = _to_numpy(batch["pred_trajectories"])[sample_idx]
        extras += [(f"pred[{i}]", runs[i]) for i in range(len(runs))]
        pred_runs = runs

    # Top row: the context views plus (optionally) the map BEV as a third panel.
    # Each entry is (label, draw_fn) so images and the BEV ax share one layout.
    top_row = []
    if context_views is not None:
        for j, img in enumerate(context_views):
            label = (context_view_labels[j]
                     if context_view_labels is not None and j < len(context_view_labels)
                     else f"view {j}")
            top_row.append((label, lambda ax, img=img: _show_image(ax, img)))
    if bev is not None:
        top_row.append(_bev_panel(bev, trajectory, pred_runs))

    # figure layout: (optional) top row, camera rows, one row for BEV + traj
    n_cam_rows = (int(bool(top_row)) + int(cameras is not None)
                  + int(decoded is not None))
    n_rows = n_cam_rows + 1
    fig = plt.figure(figsize=(16, 3 * n_cam_rows + 4))
    if title:
        fig.suptitle(title, fontsize=15)

    # One GridSpec for every panel, 10 = lcm(5, 2) columns: the 5-up camera rows
    # span 2 columns each and the bottom pair spans 5 each. Mixing separate 5- and
    # 2-column grids instead makes tight_layout bail out ("number of columns in
    # subplot specifications must be multiples of one another").
    gs = fig.add_gridspec(n_rows, 10)

    def _cell(row, slot):
        """Ax for one of the 5 slots of `row`."""
        return fig.add_subplot(gs[row, 2 * slot:2 * slot + 2])

    def _camera_row(row, imgs, tag):
        T = len(imgs)
        idxs = np.linspace(0, T - 1, 5, dtype=int)
        for j, fi in enumerate(idxs):
            ax = _cell(row, j)
            _show_image(ax, imgs[fi])
            ax.set_title(f"{tag} frame {fi}", fontsize=9)

    def _named_row(row, items):
        """Draw (label, draw_fn) panels centered in the 5-slot row."""
        offset = (5 - len(items)) // 2
        for j, (label, draw_fn) in enumerate(items):
            ax = _cell(row, offset + j)
            draw_fn(ax)
            ax.set_title(label, fontsize=9)

    row = 0
    if top_row:
        _named_row(row, top_row); row += 1
    if cameras is not None:
        _camera_row(row, cameras, "cam"); row += 1
    if decoded is not None:
        _camera_row(row, decoded, "decoded"); row += 1

    if trajectory is not None:
        ax_bev = fig.add_subplot(gs[n_rows - 1, 0:5])
        _plot_bev(ax_bev, trajectory, extras)

        ax_xy = fig.add_subplot(gs[n_rows - 1, 5:10])
        traj = _to_numpy(trajectory)
        t = np.arange(len(traj))
        ax_xy.plot(t, traj[:, 0], label="x(t)")
        ax_xy.plot(t, traj[:, 1], label="y(t)")
        ax_xy.set_title("Trajectory coordinates", fontweight="bold")
        ax_xy.set_xlabel("frame"); ax_xy.set_ylabel("m")
        ax_xy.grid(True, alpha=0.3); ax_xy.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97] if title else None)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
