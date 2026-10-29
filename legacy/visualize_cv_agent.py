#!/usr/bin/env python3
"""
Visualize ConstantVelocityAgent predictions on NAVSIM scenes.

Each page of the output PDF shows one scene with two panels:
  Left  – BEV of the current frame (map + surrounding agents)
  Right – same BEV overlaid with the CV-agent trajectory (red) vs. the human
          trajectory (green)

Usage:
    python visualize_cv_agent.py [--split mini] [--num-scenes 6] [--output cv_agent_plots.pdf]
    
    python visualize_cv_agent.py --split test --num-scenes 6 --output cv_agent_plots.pdf
    
    python visualize_cv_agent.py --split test --num-scenes 6 --agent diffusion --output diffusion_agent_plots.pdf

Required environment variables (same as NAVSIM devkit):
    OPENSCENE_DATA_ROOT   – root of the dataset (contains navsim_logs/, sensor_blobs/, maps/)
    NUPLAN_MAPS_ROOT      – path to nuPlan maps
    NUPLAN_MAP_VERSION    – map version string, e.g. "nuplan-maps-v1.0"
"""

print("imports started")

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from diffusion_agent import DiffusionAgent
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG
from navsim.visualization.plots import configure_ax, configure_bev_ax

print("imports finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="mini", help="Dataset split: mini | test | trainval (default: mini)")
    parser.add_argument("--num-scenes", type=int, default=6, help="Number of scenes to visualize (default: 6)")
    parser.add_argument("--output", default="cv_agent_plots.pdf", help="Output PDF path (default: cv_agent_plots.pdf)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for scene selection (default: 42)")
    
    # diffusion agent
    parser.add_argument("--agent", default="cv", choices=["cv", "diffusion"],
                        help="Trajectory agent: cv (constant velocity) | diffusion (default: cv)")
    parser.add_argument("--n-trajectories", type=int, default=100,
                        help="Diffusion: number of trajectories sampled simultaneously (default: 100)")
    parser.add_argument("--num-diffusion-steps", type=int, default=20,
                        help="Diffusion: number of denoising steps (default: 20)")
    parser.add_argument("--show-all-samples", action="store_true",
                        help="Diffusion: plot every sampled trajectory instead of just the mean")
    return parser.parse_args()


def build_scene_loader(openscene_root: Path, split: str, sensor_config: SensorConfig) -> SceneLoader:
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=8,
    )
    return SceneLoader(
        # data_path=openscene_root / f"navsim_logs/{split}",
        # original_sensor_path=openscene_root / f"sensor_blobs/{split}",

        data_path=openscene_root / f"test_navsim_logs/test",
        original_sensor_path=openscene_root / f"test_sensor_blobs/test",

        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )


def compute_agent_trajectories(agent, agent_input, show_all_samples):
    """Return a list of agent Trajectory objects to plot.

    By default a single trajectory is returned (the CV prediction, or the mean
    of the diffusion distribution). When ``show_all_samples`` is set and the
    agent exposes a distribution, every sampled trajectory is returned so the
    full spread can be drawn.
    """
    if show_all_samples and hasattr(agent, "compute_trajectory_samples"):
        return agent.compute_trajectory_samples(agent_input)
    return [agent.compute_trajectory(agent_input)]


def plot_scene(scene, agent, agent_label, show_all_samples) -> plt.Figure:
    frame_idx = scene.scene_metadata.num_history_frames - 1

    agent_input = scene.get_agent_input()
    ego_status = agent_input.ego_statuses[-1]
    ego_speed = float((ego_status.ego_velocity ** 2).sum() ** 0.5)

    human_traj = scene.get_future_trajectory()
    agent_trajs = compute_agent_trajectories(agent, agent_input, show_all_samples)

    fig, (ax_scene, ax_traj) = plt.subplots(1, 2, figsize=(12, 6))

    # --- left: plain BEV ---
    add_configured_bev_on_ax(ax_scene, scene.map_api, scene.frames[frame_idx])
    configure_bev_ax(ax_scene)
    configure_ax(ax_scene)
    ax_scene.set_title("BEV — current frame", fontsize=10)

    # --- right: BEV + trajectories ---
    add_configured_bev_on_ax(ax_traj, scene.map_api, scene.frames[frame_idx])
    add_trajectory_to_bev_ax(ax_traj, human_traj, TRAJECTORY_CONFIG["human"])
    # Loop supports plotting an arbitrary number of agent trajectories at once.
    for agent_traj in agent_trajs:
        add_trajectory_to_bev_ax(ax_traj, agent_traj, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax_traj)
    configure_ax(ax_traj)
    ax_traj.set_title(f"{agent_label} (red) vs. human (green)", fontsize=10)

    legend_handles = [
        Line2D([0], [0], color=TRAJECTORY_CONFIG["human"]["line_color"],
               lw=2, marker="o", markersize=5, label="Human"),
        Line2D([0], [0], color=TRAJECTORY_CONFIG["agent"]["line_color"],
               lw=2, marker="o", markersize=5, label=agent_label),
    ]
    ax_traj.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.8)

    token = scene.scene_metadata.initial_token
    fig.suptitle(
        f"token: {token}   |   ego speed: {ego_speed:.2f} m/s   |   map: {scene.scene_metadata.map_name}",
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()

    openscene_root_str = os.environ.get("OPENSCENE_DATA_ROOT")
    if not openscene_root_str:
        sys.exit("Error: OPENSCENE_DATA_ROOT environment variable is not set.")
    openscene_root = Path(openscene_root_str)

    if args.agent == "diffusion":
        agent = DiffusionAgent(
            n_trajectories=args.n_trajectories,
            num_diffusion_steps=args.num_diffusion_steps,
        )
        agent_label = "Diffusion agent"
    else:
        agent = ConstantVelocityAgent()
        agent_label = "CV agent"

    agent.initialize()

    print(f"Loading scenes from '{args.split}' split …  (agent: {args.agent})")
    scene_loader = build_scene_loader(openscene_root, args.split, agent.get_sensor_config())
    tokens = scene_loader.tokens

    if not tokens:
        sys.exit(f"Error: no scenes found for split '{args.split}'.")

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(tokens, size=min(args.num_scenes, len(tokens)), replace=False)
    print(f"Selected {len(chosen)} / {len(tokens)} scenes  →  {args.output}")

    output_path = Path(args.output)
    with PdfPages(output_path) as pdf:
        for i, token in enumerate(chosen):
            scene = scene_loader.get_scene_from_token(token)

            ego_speed = float(
                (scene.get_agent_input().ego_statuses[-1].ego_velocity ** 2).sum() ** 0.5
            )
            print(f"  [{i+1:02d}/{len(chosen)}] token={token}  speed={ego_speed:.2f} m/s")

            fig = plot_scene(scene, agent, agent_label, args.show_all_samples)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Done. Saved {len(chosen)} pages → {output_path.resolve()}")


if __name__ == "__main__":
    main()
