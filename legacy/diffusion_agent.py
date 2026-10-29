"""
Diffusion trajectory-prediction agent for NAVSIM.

Wraps evaluate.predict_traj.TrajectoryPredictor as an AbstractAgent: it consumes
the front-camera history frames from the agent input, encodes them to latents,
and samples a distribution of futures with the flow-matching diffusion model.

``compute_trajectory`` returns the mean of the sampled distribution. The full
distribution is available via ``compute_trajectory_samples`` for visualization.
"""

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory

from evaluate.predict_traj import TrajectoryPredictor


def _positions_to_poses(positions):
    """Convert (N, 2) future xy positions to (N, 3) (x, y, heading) poses.

    Heading is the direction of motion between consecutive points; near-zero
    steps reuse the previous heading.
    """
    positions = np.asarray(positions, dtype=np.float32)
    poses = np.zeros((positions.shape[0], 3), dtype=np.float32)
    poses[:, :2] = positions

    prev = np.zeros(2, dtype=np.float32)  # origin = current ego pose
    heading = 0.0
    for i in range(positions.shape[0]):
        delta = positions[i] - prev
        if np.hypot(delta[0], delta[1]) > 1e-3:
            heading = float(np.arctan2(delta[1], delta[0]))
        poses[i, 2] = heading
        prev = positions[i]
    return poses


class DiffusionAgent(AbstractAgent):
    """Diffusion-based trajectory prediction agent."""

    requires_scene = False

    def __init__(
        self,
        predictor: TrajectoryPredictor = None,
        num_history_frames: int = 4,
        frame_interval: float = 0.2,
        n_trajectories: int = 100,
        num_diffusion_steps: int = 20,
        device: str = "cuda",
        predictor_kwargs: dict = None,
    ):
        # Sampling derived from the diffusion model's prediction horizon.
        self.num_history_frames = num_history_frames
        self.frame_interval = frame_interval
        self.n_trajectories = n_trajectories
        self.num_diffusion_steps = num_diffusion_steps
        self._device = device
        self._predictor_kwargs = predictor_kwargs or {}
        self._predictor = predictor

        pred_steps = self._get_predictor().pred_steps
        super().__init__(
            TrajectorySampling(
                time_horizon=pred_steps * frame_interval,
                interval_length=frame_interval,
            )
        )

    def _get_predictor(self) -> TrajectoryPredictor:
        if self._predictor is None:
            self._predictor = TrajectoryPredictor(
                n_trajectories=self.n_trajectories,
                num_diffusion_steps=self.num_diffusion_steps,
                device=self._device,
                **self._predictor_kwargs,
            )
        return self._predictor

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        self._get_predictor()

    def get_sensor_config(self) -> SensorConfig:
        """Load the front camera for the history frames only."""
        history = list(range(self.num_history_frames))
        return SensorConfig(
            cam_f0=history,
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def _history_velocity(self, agent_input: AgentInput) -> np.ndarray:
        """Real-unit frame-to-frame xy deltas from the ego history poses."""
        poses = np.array([status.ego_pose[:2] for status in agent_input.ego_statuses], dtype=np.float32)
        velocity = np.zeros_like(poses)
        velocity[1:] = poses[1:] - poses[:-1]
        return velocity

    def _front_images(self, agent_input: AgentInput):
        return [cameras.cam_f0.image for cameras in agent_input.cameras]

    def predict_distribution(self, agent_input: AgentInput) -> np.ndarray:
        """Sample futures: returns (n_trajectories, pred_steps, 2) positions."""
        predictor = self._get_predictor()
        images = self._front_images(agent_input)
        history_velocity = self._history_velocity(agent_input)
        trajectories = predictor.predict(images, history_velocity=history_velocity)
        return predictor.future_positions(trajectories)

    def compute_trajectory_samples(self, agent_input: AgentInput):
        """Return a list of Trajectory objects, one per sampled future."""
        futures = self.predict_distribution(agent_input)
        return [Trajectory(_positions_to_poses(f), self._trajectory_sampling) for f in futures]

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """Inherited, see superclass. Returns the distribution mean."""
        futures = self.predict_distribution(agent_input)
        future = futures[0] #futures.mean(axis=0)
        return Trajectory(_positions_to_poses(future), self._trajectory_sampling)
