"""Visualize the NAVSIM long dataloader.

    python -m exp_navsim.visualize_dataset --config exp_navsim/config.yaml --num 5

Saves ONE pdf whose first page reports dataset statistics (number of episodes
and the distribution of episode lengths), a second page with the distributions
of driven distance and total steering angle, a third page with the same two
distributions restricted to the long / sharply-turning episodes, followed by one
page per visualized episode (context front views + map BEV + 5 surround-camera
observations + trajectory), laid out like exp_navsim/test_model.py.

The statistics pages always cover the whole dataset; --min-distance /
--min-angle / --min-frames filter only which episodes get drawn.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from omegaconf import OmegaConf

from exp_navsim.data.navsim_base import NavsimLongBase
from exp_navsim.draw import draw_episode

# Statistics-page settings (fixed on purpose — these shape the report, not the run).
MIN_DISPLACEMENT = 5.0     # episodes moving less than this (m) are dropped from the
                           # steering histograms, where their angle is pure noise
HIST_MIN_DISTANCE = 100.0  # lower bound of the zoomed distance histogram (m)
HIST_MIN_ANGLE = 30.0      # lower bound of the zoomed |steering angle| histogram (deg)


def _stats_page(pdf, lengths):
    fig = plt.figure(figsize=(11, 8))
    fig.suptitle("NAVSIM long dataloader — dataset statistics", fontsize=16)
    lengths = np.asarray(lengths)
    txt = (
        f"episodes: {len(lengths)}\n"
        f"episode length (frames)  min={lengths.min()}  max={lengths.max()}  "
        f"mean={lengths.mean():.1f}  median={np.median(lengths):.0f}"
    )
    fig.text(0.1, 0.9, txt, fontsize=12, va="top", family="monospace")
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.65])
    ax.hist(lengths, bins=min(30, max(len(set(lengths.tolist())), 1)),
            color="tab:blue", alpha=0.8)
    ax.set_xlabel("episode length (frames)"); ax.set_ylabel("count")
    ax.set_title("Distribution of episode lengths")
    ax.grid(True, alpha=0.3)
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def trajectory_stats(traj):
    """(distance driven [m], steering angle [deg], displacement [m]) of one episode.

    Steering is the signed angle between (1, 0) — the initial heading, since the
    trajectory is expressed in the first frame's local frame — and the
    start->end vector.
    """
    traj = np.asarray(traj, dtype=np.float64)
    distance = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    d = traj[-1] - traj[0]
    return distance, float(np.degrees(np.arctan2(d[1], d[0]))), float(np.linalg.norm(d))


def _motion_page(pdf, distances, steerings, min_displacement):
    """Histograms of driven distance and of total steering angle."""
    fig, (ax_d, ax_s) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Ego motion statistics", fontsize=16)

    ax_d.hist(distances, bins=30, color="tab:blue", alpha=0.8)
    ax_d.set_xlabel("distance driven (m)"); ax_d.set_ylabel("count")
    ax_d.set_title(f"Distance driven  (median={np.median(distances):.1f} m)")

    ax_s.hist(steerings, bins=30, color="tab:orange", alpha=0.8)
    ax_s.set_xlabel("steering angle (deg)"); ax_s.set_ylabel("count")
    ax_s.set_title(
        f"Steering angle, |disp| >= {min_displacement:g} m  "
        f"({len(steerings)} episodes)"
    )
    for ax in (ax_d, ax_s):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _tail_motion_page(pdf, distances, steerings, min_distance, min_angle):
    """Same two histograms as `_motion_page`, restricted to the tails.

    The full-range histograms are dominated by the short / straight episodes, so
    this page zooms on the interesting ones: episodes driving at least
    `min_distance` metres, and episodes turning by at least `min_angle` degrees.
    """
    distances = np.asarray(distances)
    long_d = distances[distances >= min_distance]
    sharp_s = steerings[np.abs(steerings) >= min_angle]

    fig, (ax_d, ax_s) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Ego motion statistics — long / sharply-turning episodes", fontsize=16)

    def _hist(ax, values, color, xlabel, title):
        if len(values):
            ax.hist(values, bins=30, color=color, alpha=0.8)
        else:
            ax.text(0.5, 0.5, "no episodes", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xlabel(xlabel); ax.set_ylabel("count")
        ax.set_title(title); ax.grid(True, alpha=0.3)

    _hist(ax_d, long_d, "tab:blue", "distance driven (m)",
          f"Distance >= {min_distance:g} m  ({len(long_d)}/{len(distances)} episodes)")
    _hist(ax_s, sharp_s, "tab:orange", "steering angle (deg)",
          f"|steering| >= {min_angle:g} deg  ({len(sharp_s)}/{len(steerings)} episodes)")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _attach_top_row(batch, ds, episode_idx, sample, cfg, with_bev):
    """Top row of the episode page: two context front views + the map BEV.

    Mirrors what test_model.py builds for its prediction pages, except the front
    views come straight out of the episode sample (no token lookup needed) and
    there is no predicted distribution to overlay on the BEV.
    """
    images = sample["images"]                                      # (T, 3, H, W)
    last_ctx = min(cfg.model.params.context_images - 1, len(images) - 1)
    ctx_frames = sorted({0, last_ctx})                             # first & last context frame
    batch["context_views"] = images[None, ctx_frames]              # (1, len(ctx), 3, H, W)
    batch["context_view_labels"] = [f"context frame {i}" for i in ctx_frames]

    if not with_bev:
        return
    # anchor=0: the episode trajectory is local to frame 0, so BEV and path share
    # one frame (see exp_navsim/data/bev_extract.py).
    from exp_navsim.data.bev_extract import extract_bev
    try:
        batch["bev"] = [extract_bev(ds, episode_idx, anchor=0)]
    except Exception as e:                  # missing navsim / nuPlan maps
        print(f"  skipping BEV panel: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_navsim/config.yaml")
    ap.add_argument("--num", type=int, default=5, help="episodes to visualize")
    ap.add_argument("--out", default="exp_navsim/navsim_episodes.pdf")
    # Filters on which episodes get a page (the statistics pages always cover
    # the whole dataset). Defaults of 0 draw simply the first --num episodes.
    ap.add_argument("--min-distance", type=float, default=0.0,
                    help="only draw episodes driving at least this far (m)")
    ap.add_argument("--min-angle", type=float, default=0.0,
                    help="only draw episodes turning by at least this much (deg)")
    ap.add_argument("--min-frames", type=int, default=0,
                    help="only draw episodes with at least this many frames")
    ap.add_argument("--no-bev", action="store_true",
                    help="skip the map-BEV panel (needs navsim + nuPlan maps)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    ds = NavsimLongBase.from_config(cfg)
    print(f"Dataset: {len(ds)} episodes")

    # Episode lengths / trajectories (cheap: only parse frames, no images).
    lengths, distances, steerings, displacements = [], [], [], []
    from tqdm import tqdm
    for i in tqdm(range(len(ds))):
        traj = ds.episode_trajectory(i)
        lengths.append(len(traj))
        dist, steer, disp = trajectory_stats(traj)
        distances.append(dist); steerings.append(steer); displacements.append(disp)
    lengths = np.asarray(lengths)
    distances = np.asarray(distances)
    steerings = np.asarray(steerings)
    moved = np.asarray(displacements) >= MIN_DISPLACEMENT

    # Which episodes get a page: the first --num passing all three filters.
    cand = np.flatnonzero(
        (distances >= args.min_distance)
        & (np.abs(steerings) >= args.min_angle)
        & (lengths >= args.min_frames)
    )
    indices = cand[: args.num].tolist()
    print(f"{len(cand)} episodes pass the filters (distance >= {args.min_distance:g} m, "
          f"|steering| >= {args.min_angle:g} deg, frames >= {args.min_frames}); "
          f"drawing {len(indices)}")

    with PdfPages(args.out) as pdf:
        _stats_page(pdf, lengths)
        _motion_page(pdf, distances, steerings[moved], MIN_DISPLACEMENT)
        _tail_motion_page(pdf, distances, steerings[moved],
                          HIST_MIN_DISTANCE, HIST_MIN_ANGLE)
        for n, i in enumerate(indices):
            print(f"  visualizing episode {i} ({n + 1}/{len(indices)}), "
                  f"steering {steerings[i]:.1f} deg")
            sample = ds[i]
            # draw_episode expects a batch -> add a batch dim of size 1
            batch = {k: [v] if not isinstance(v, dict) else [v] for k, v in sample.items()}
            batch = {k: (np.stack(v) if isinstance(v[0], np.ndarray) else v)
                     for k, v in batch.items()}
            _attach_top_row(batch, ds, i, sample, cfg, with_bev=not args.no_bev)
            fig = draw_episode(
                batch, 0,
                title=f"Episode {i} — {sample['metadata']['token']} — "
                      f"{distances[i]:.1f} m, steering {steerings[i]:.1f} deg",
            )
            pdf.savefig(fig, dpi=120, bbox_inches="tight"); plt.close(fig)

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
