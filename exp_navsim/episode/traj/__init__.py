"""Trajectory-only baseline: the future predicted from the previous trajectory, no images.

    denoiser.py   TrajectoryContextDenoiser, the query attends to the context velocities
    model.py      TrajContextModel, NavsimTrajectoryModel with that denoiser
    config.yaml   training config (commands on top)
"""
