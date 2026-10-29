"""NAVSIM agent that replays trajectories predicted offline by the probe.

The probe runs on cached features (eval/evaluate.py --save-predictions), so the
scoring scripts only need a lookup: token -> 8 poses. The agent asks for the scene
(requires_scene) because that is where navsim keeps the token.

    sample = k        the k-th sampled future
    sample = "mean"   the mean of the samples (headings averaged on the circle)
"""

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory


def mean_trajectory(samples):
    """(K, ..., 8, 3) -> (..., 8, 3)."""
    xy = samples[..., :2].mean(0)
    h = np.arctan2(np.sin(samples[..., 2]).mean(0), np.cos(samples[..., 2]).mean(0))
    return np.concatenate([xy, h[..., None]], axis=-1)


class PrecomputedAgent(AbstractAgent):
    def __init__(self, predictions, sample=0,
                 trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5)):
        # The scene carries the token; AbstractAgent stores requires_scene per instance.
        super().__init__(trajectory_sampling, requires_scene=True)
        self.predictions_path = predictions
        self.sample = sample

    def name(self):
        return f"precomputed_{self.sample}"

    def initialize(self):
        data = np.load(self.predictions_path)
        samples = data["samples"].astype(np.float32)                    # (K, N, 8, 3)
        poses = mean_trajectory(samples) if self.sample == "mean" else samples[int(self.sample)]
        self.poses = dict(zip(data["token"].tolist(), poses))
        self.fallbacks = 0

    def get_sensor_config(self):
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input, scene=None):
        poses = self.poses.get(scene.scene_metadata.initial_token)
        if poses is None:
            # A scene without a prediction (its images were missing): constant velocity.
            speed = float(np.linalg.norm(agent_input.ego_statuses[-1].ego_velocity))
            t = 0.5 * np.arange(1, 9)
            poses = np.stack([speed * t, np.zeros(8), np.zeros(8)], 1).astype(np.float32)
            self.fallbacks += 1
        return Trajectory(poses, self._trajectory_sampling)
