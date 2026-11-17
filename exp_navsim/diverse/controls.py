"""Control-space representation of a future: acceleration and yaw rate, integrated by a unicycle.

A future is 8 poses at 2 Hz in the ego frame. Its controls are, per 0.5 s step i:

    u_i      longitudinal acceleration (v_i - v_{i-1}) / dt, v_i = |p_i - p_{i-1}| / dt, v_0 = current speed
    omega_i  yaw rate (h_i - h_{i-1}) / dt from the logged headings, h_0 = 0

and from_controls integrates them back: pose i moves v_i * dt along the mean heading
(h_{i-1} + h_i) / 2. The yaw rate comes from the vehicle heading, not from the direction
of the steps, because that direction is sensor noise when the car crawls or stands.
Speeds are clamped at 0, so a decoded future never reverses, and sideways motion is
v * omega * dt: a slow car cannot jump sideways, whatever controls a model outputs
(dynamically-extended unicycle of Trajectron++, arXiv 2001.03093; the bicycle roll-out
of Hydra-NeXt, arXiv 2503.12030).
"""

import math

import torch
import torch.nn as nn

DT = 0.5

# NAVSIM / nuPlan comfort bounds (pdm_comfort_metrics); at 2 Hz the finite differences
# smooth the peaks, so these flags are a lower bound on the violations.
LON_ACC = (-4.05, 2.40)
LAT_ACC = 4.89
YAW_RATE = 0.95
LON_JERK = 4.13


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def speed_of(status):
    return status[:, :2].norm(dim=-1)


def to_controls(future, v0):
    """future (..., 8, >=2), v0 (...) -> controls (..., 8, 2) = (u, omega)."""
    p = torch.cat([torch.zeros_like(future[..., :1, :2]), future[..., :2]], -2)
    h = torch.cat([torch.zeros_like(future[..., :1, 2]), future[..., 2]], -1)
    omega = wrap(torch.diff(h, dim=-1)) / DT                                   # (..., 8)
    v = torch.cat([v0[..., None], (p[..., 1:, :] - p[..., :-1, :]).norm(dim=-1) / DT], -1)
    return torch.stack([torch.diff(v, dim=-1) / DT, omega], -1)


def from_controls(ctrl, v0):
    """controls (..., 8, 2), v0 (...) -> future (..., 8, 3)."""
    v = (v0[..., None] + torch.cumsum(ctrl[..., 0], -1) * DT).clamp(min=0)
    h = torch.cumsum(ctrl[..., 1], -1) * DT
    mid = h - 0.5 * ctrl[..., 1] * DT
    xy = torch.cumsum(torch.stack([torch.cos(mid), torch.sin(mid)], -1) * (v * DT)[..., None], -2)
    return torch.cat([xy, h[..., None]], -1)


def comfort_flags(future, v0):
    """Per trajectory (...,): does it break a comfort bound (2 Hz finite differences)?
    -> dict of bool tensors: lon_acc, lat_acc, yaw_rate, lon_jerk, any."""
    ctrl = to_controls(future, v0)
    u, omega = ctrl[..., 0], ctrl[..., 1]
    p = torch.cat([torch.zeros_like(future[..., :1, :2]), future[..., :2]], -2)
    v = (p[..., 1:, :] - p[..., :-1, :]).norm(dim=-1) / DT
    # a trajectory that does not move along its heading (sideways or backwards) gets the
    # yaw rate of its direction of travel, so a sideways jump counts as a lateral acceleration
    d = p[..., 1:, :] - p[..., :-1, :]
    travel = torch.atan2(d[..., 1], d[..., 0])
    travel = torch.cat([torch.zeros_like(travel[..., :1]), travel], -1)
    omega = torch.where(v > 1.0, wrap(torch.diff(travel, dim=-1)) / DT, omega)
    flags = {"lon_acc": ((u < LON_ACC[0]) | (u > LON_ACC[1])).any(-1),
             "lat_acc": ((v * omega).abs() > LAT_ACC).any(-1),
             "yaw_rate": (omega.abs() > YAW_RATE).any(-1),
             "lon_jerk": (torch.diff(u, dim=-1).abs() / DT > LON_JERK).any(-1)}
    flags["any"] = torch.stack(list(flags.values())).any(0)
    return flags


class ControlCodec(nn.Module):
    """Controls (B, 8, 2) <-> standardized z. order < 8: least-squares polynomial in time
    per channel (with constant term), which makes the generated controls smooth."""

    def __init__(self, order=8):
        super().__init__()
        self.order = order
        self.latent_dim = 2 * order
        tau = torch.arange(8, dtype=torch.float64) / 7
        basis = tau[:, None] ** torch.arange(order) if order < 8 else torch.eye(8, dtype=torch.float64)
        self.register_buffer("basis", basis.float(), persistent=False)                # (8, K)
        self.register_buffer("pinv", torch.linalg.pinv(basis).float(), persistent=False)
        self.register_buffer("mean", torch.zeros(self.latent_dim))
        self.register_buffer("std", torch.ones(self.latent_dim))

    def to_z(self, ctrl):
        return (self.pinv @ ctrl).flatten(1)

    def from_z(self, z):
        return self.basis @ z.unflatten(1, (self.order, 2))

    @torch.no_grad()
    def fit(self, ctrl):
        z = self.to_z(ctrl)
        self.mean.copy_(z.mean(0))
        self.std.copy_(z.std(0).clamp(min=1e-4))
        return {"ctrl_rec_rmse": (self.from_z(z) - ctrl).pow(2).mean((0, 1)).sqrt().tolist()}

    def encode(self, ctrl):
        return (self.to_z(ctrl) - self.mean) / self.std

    def decode(self, z):
        return self.from_z(z * self.std + self.mean)
