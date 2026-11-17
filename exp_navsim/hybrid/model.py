"""Regression + flow matching on the fast-planner backbone (planner/model.py).

Inputs are those of the 16_fast_planner regression planner: wm11_ctx image tokens, ego
velocity and acceleration, the 4 past poses and the driving command. One scene encoder,
two heads:

    regress   one query cross-attends to the scene and outputs the 8 poses (L1), as in 16
    flow      flow matching on a 24-vector (poses with the heading in metres, normalized
              per coordinate): a query built from the noisy vector, the time and the ego
              token cross-attends to the scene and outputs the velocity

mode decides what the flow generates:
    regress   no flow (the 16 planner)
    flow      the future itself (plain trajectory diffusion)
    resid     future - reference, where the reference is the regression output of the
              same model (detached), as in ResAD (arXiv 2510.08562), whose reference is a
              constant-velocity roll-out. The residual is normalized per coordinate with
              its training std (ResAD's point-wise residual normalization). The reference
              is also a condition token of the flow.

Reference perturbation (irp, m/s): the reference gets a constant-velocity offset with a
random velocity (std irp along x, irp / 2 along y) and the flow learns the residual from
the perturbed reference back to the logged future (ResAD's inertial reference
perturbation). At inference a perturbation per sample gives one more source of spread.

Prediction target (pred) of the network, with z_t = (1 - t) x + t eps:
    v     the velocity x - eps (default, the rectified-flow target)
    x0    the clean vector x; velocity (x0_hat - z_t) / t
    eps   the noise; velocity (z_t - eps_hat) / (1 - t)
each trained with MSE on its own target.

Normalization (norm) of the 24-vector before the noise is added:
    std     per-coordinate mean and std (default)
    none    no mean, a fixed 10 m for every coordinate: the far poses dominate the noise
            scale while the near poses are drowned at every t
    raw     no mean, metres (scale 1): the data is much wider than the unit noise
    global  per-coordinate mean, one std for all coordinates (keeps the relative scale
            of near and far poses)
    minmax  per-coordinate 1st / 99th percentile to [-1, 1] (Diffusion Policy, DiffusionDrive)
    pca     whitening with the full covariance (decorrelates the poses of one trajectory)
noise_scale c multiplies the scale after the fit, so the data has std 1 / c against the
unit noise (c > 1: the noise wins earlier in t, c < 1: later).
"""

import math

import torch
import torch.nn as nn

from exp_navsim.planner.model import HEADING_SCALE, CrossBlock, Planner, scaled

DT = 0.5
STEPS = torch.arange(1, 9, dtype=torch.float32) * DT


def unscale(x):
    """(…, 24) -> (…, 8, 3) poses, inverse of planner.model.scaled."""
    x = x.unflatten(-1, (8, 3))
    return torch.cat([x[..., :2], x[..., 2:] / HEADING_SCALE], -1)


def cv_offset(dv):
    """(N, 2) velocity offsets -> (N, 8, 3) constant-velocity displacement, heading 0."""
    xy = STEPS.to(dv.device)[None, :, None] * dv[:, None]
    return torch.cat([xy, torch.zeros_like(xy[..., :1])], -1)


def time_embedding(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000.0 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class FlowPlanner(Planner):
    def __init__(self, in_channels, num_frames, mode="flow", flow_depth=3, sigma_min=1e-5, pred="v", norm="std",
                 noise_scale=1.0, **kw):
        super().__init__(in_channels, num_frames, vocab=None, **kw)
        D = self.norm.normalized_shape[0]
        self.mode, self.sigma_min, self.pred, self.normalize = mode, sigma_min, pred, norm
        self.noise_scale = noise_scale
        self.register_buffer("mu", torch.zeros(24))
        self.register_buffer("sd", torch.ones(24))
        if norm == "pca":
            self.register_buffer("rot", torch.eye(24))
        if mode == "regress":
            return
        self.z_in = nn.Sequential(nn.Linear(24, D), nn.GELU(), nn.Linear(D, D))
        self.t_in = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.ref_in = nn.Sequential(nn.Linear(24, D), nn.GELU(), nn.Linear(D, D)) if mode == "resid" else None
        self.flow_blocks = nn.ModuleList([CrossBlock(D, 4, 0.0) for _ in range(flow_depth)])
        self.flow_norm = nn.LayerNorm(D)
        self.flow_out = nn.Linear(D, 24)
        nn.init.zeros_(self.flow_out.weight)
        nn.init.zeros_(self.flow_out.bias)

    # ------------------------------------------------------------- pieces -- #
    def regress(self, mem, ego):
        q = self.query.expand(len(mem), -1, -1) + ego
        for blk in self.decoder:
            q = blk(q, mem)
        reg = self.out(self.norm(q))[:, 0].view(len(q), 8, 3)
        return torch.cat([reg[..., :2], reg[..., 2:] / HEADING_SCALE], -1)

    def velocity(self, mem, ego, ref, z, t):
        """mem (N, L, D), ego (N, 1, D), ref (N, 8, 3) or None, z (N, 24), t (N,) -> (N, 24)."""
        temb = self.t_in(time_embedding(t, ego.shape[-1]))[:, None]
        q = self.z_in(z)[:, None] + ego + temb
        if self.ref_in is not None:
            q = q + self.ref_in(scaled(ref) / 10.0)[:, None]
        for blk in self.flow_blocks:
            q = blk(q + temb, mem)
        return self.flow_out(self.flow_norm(q))[:, 0]

    def to_velocity(self, out, z, t):
        """Network output for the chosen target -> velocity x - eps."""
        if self.pred == "x0":
            return (out - z) / t[:, None].clamp(min=0.02)
        if self.pred == "eps":
            return (z - out) / (1 - t[:, None]).clamp(min=0.02)
        return out

    def to_z(self, future, ref):
        x = scaled(future) - (scaled(ref) if self.mode == "resid" else 0) - self.mu
        if self.normalize == "pca":
            x = (x[..., :, None] * self.rot).sum(-2)                # fp32 under autocast
        return x / self.sd

    def from_z(self, z, ref):
        x = z * self.sd
        if self.normalize == "pca":
            x = (x[..., None, :] * self.rot).sum(-1)
        return unscale(x + self.mu + (scaled(ref) if self.mode == "resid" else 0))

    def perturbed(self, ref, irp):
        if irp <= 0:
            return ref
        dv = torch.randn(len(ref), 2, device=ref.device) * torch.tensor([irp, irp / 2], device=ref.device)
        return ref + cv_offset(dv)

    # ---------------------------------------------------------------- fit -- #
    @torch.no_grad()
    def fit(self, future, ref=None):
        """Normalization of what the flow generates, from training futures (and references)."""
        x = scaled(future) - (scaled(ref) if self.mode == "resid" else 0)
        mean = x.mean(0) if self.mode == "flow" else torch.zeros_like(self.mu)
        if self.normalize in ("none", "raw"):
            self.mu.zero_()
            self.sd.fill_(10.0 if self.normalize == "none" else 1.0)
        elif self.normalize == "global":
            self.mu.copy_(mean)
            self.sd.fill_((x - mean).pow(2).mean().sqrt().item())
        elif self.normalize == "minmax":
            lo, hi = torch.quantile(x, 0.01, dim=0), torch.quantile(x, 0.99, dim=0)
            self.mu.copy_((hi + lo) / 2)
            self.sd.copy_(((hi - lo) / 2).clamp(min=1e-2))
        elif self.normalize == "pca":
            self.mu.copy_(mean)
            ev, vec = torch.linalg.eigh(torch.cov((x - mean).T.double()))
            self.rot.copy_(vec.float())
            self.sd.copy_(ev.clamp(min=1e-4).sqrt().float())
        else:
            self.mu.copy_(mean)
            self.sd.copy_(x.std(0).clamp(min=1e-2))
        self.sd.mul_(self.noise_scale)
        z = self.to_z(future, ref)
        return {"sd_xy_4s": self.sd[21:23].tolist(), "sd_mean": self.sd.mean().item(),
                "z_std_near_far": [z[:, 0].std().item(), z[:, 21].std().item()]}

    # --------------------------------------------------------------- loss -- #
    def loss(self, batch, flow_target=None, irp=0.0):
        """L1 of the regression head (modes regress, resid) + flow matching (modes flow, resid).

        flow_target: (B, 8, 3) futures the flow learns instead of the logged one (multi-modal
        targets, reward-weighted samples); the regression head always learns the logged one.
        -> loss, logs, per-scene flow loss (B,) or None."""
        mem, ego = self.scene(batch)
        gt = batch["future"]
        logs, loss = {}, 0.0
        ref = None
        if self.mode in ("regress", "resid"):
            reg = self.regress(mem, ego).float()
            logs["l1"] = (scaled(reg) - scaled(gt)).abs().mean()
            loss = logs["l1"]
            ref = self.perturbed(reg.detach(), irp)
        if self.mode == "regress":
            return loss, logs, None
        target = gt if flow_target is None else flow_target
        x = self.to_z(target, ref)
        t = torch.rand(len(x), device=x.device)
        noise = torch.randn_like(x)
        s = self.sigma_min
        z_t = (1 - t[:, None]) * x + (s + t[:, None] * (1 - s)) * noise
        out = self.velocity(mem, ego, ref, z_t, t).float()
        target = {"v": x - (1 - s) * noise, "x0": x, "eps": noise}[self.pred]
        fm = ((out - target) ** 2).mean(-1)
        logs["fm"] = fm.mean()
        return loss + logs["fm"], logs, fm

    # ----------------------------------------------------------- sampling -- #
    @torch.no_grad()
    def sample(self, batch, num_samples, steps=10, irp=0.0, generator=None):
        """-> samples (K, B, 8, 3), reference (B, 8, 3) or None (regression output)."""
        mem, ego = self.scene(batch)
        B = len(mem)
        ref = self.regress(mem, ego) if self.mode in ("regress", "resid") else None
        if self.mode == "regress":
            return ref[None].expand(num_samples, -1, -1, -1), ref
        K = num_samples
        mem_k, ego_k = mem.repeat(K, 1, 1), ego.repeat(K, 1, 1)
        ref_k = None
        if ref is not None:
            ref_k = ref.repeat(K, 1, 1)
            if irp > 0:
                dv = torch.randn(K * B, 2, device=mem.device, generator=generator) * torch.tensor(
                    [irp, irp / 2], device=mem.device)
                ref_k = ref_k + cv_offset(dv)
        z = torch.randn(K * B, 24, device=mem.device, generator=generator)
        ts = torch.linspace(1, 0, steps + 1, device=mem.device)
        for i in range(steps):
            t = ts[i].expand(K * B)
            z = z + self.to_velocity(self.velocity(mem_k, ego_k, ref_k, z, t), z, t) * (ts[i] - ts[i + 1])
        return self.from_z(z, ref_k).unflatten(0, (K, B)), ref
