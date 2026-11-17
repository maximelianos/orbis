"""Planner on frozen ORBIS features with the py123d_garage (TransFuser-K1) recipe.

The 16_fast_planner model (planner/model.py) is one regression query over a two-layer
encoder of the cached wm11_ctx grid. This version keeps that backbone — frozen features,
all data on the GPU, minutes of training — and adds what the py123d_garage research logs
report as adopted (github.com/kesai-labs/py123d_garage/blob/main/docs/research_logs.md):

* **Flow-matching planner instead of L1 regression**, with the garage's loss choices: L1 on
  the clean trajectory (x0 target), uniform noise levels, and time conditioning by adaptive
  layer norm (AdaLN-Zero).
* **Zero-noise inference.** Eight Euler steps started from z = 0, so the plan is
  deterministic and costs one forward pass per step. `temperature > 0` starts from real
  noise instead and gives diverse plans from the same weights — the switch the garage calls
  `noise_temperature` / `resample_noise`.
* **Joint planning representations.** The flow denoises three groups of tokens together:
  waypoints (the 8 poses that are driven), actions (bounded longitudinal and lateral
  acceleration, rolled out through a unicycle and supervised on the poses) and the path
  (positions spaced by arc length, which keeps a spatial horizon at low speed). Actions and
  path are auxiliary outputs.
* **Attention masking** between those groups: the path reads the scene and the driving
  command, the waypoints and actions read the scene and the path but not the command, so the
  path carries the geometry and the waypoints the speed profile along it.
* **Register tokens** in the encoder and **QK-norm** in every attention.

On top of the garage recipe, a **metric-training head** in the style of Hydra-MDP /
GTRS-Dense: V anchor trajectories (k-means over the logged futures) cross-attend to the
scene and predict the PDM sub-scores that the simulator gives them. The same head scores
arbitrary trajectories, so the flow samples can be reranked by predicted EPDMS.

Several cameras (`cams` > 1, train.py --cams F0,L0,R0,B0): the cached grid of every camera
is projected by the same layer; each camera token gets a learned camera embedding, an MLP of
the unit viewing ray of its cell in the ego frame (cameras.py) and the shared positional
embedding of the grid. A camera with no image (NaN features, e.g. the real navhard scenes
without side images) gets a learned per-camera "missing" token instead of its features, and
`cam_drop` trains that case by dropping side cameras at random.

BEV queries (`bev_queries` = n > 0): a grid of n longitudinal x m lateral learned queries
over the extent of the BEV targets (square cells), plus a 2D sinusoidal embedding of the
metric cell centre, read all camera tokens through 2 layers of cross-attention + self-
attention (QK-norm blocks) and join the scene tokens of the encoder, so the flow and the
anchor scorer read them too. The auxiliary BEV head then decodes every BEV token linearly to
its sub-cells of the target grid.

Inference modes (`plan`):
    flow      the zero-noise flow output (one trajectory)
    vocab     the anchor with the best imitation + predicted-metric score
    rerank    flow samples and anchors together, ranked by the predicted metrics
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADING_SCALE = 5.0        # metres per radian in the losses, as in planner/model.py
N_METRICS = 8              # PDM sub-scores, in the order of planner/fast_pdm.METRICS
DT = 0.5
HORIZON = 8
PATH_DIST = torch.arange(1, 9, dtype=torch.float32) * 4.0     # path points every 4 m up to 32 m
A_LON_MAX, A_LAT_MAX = 4.0, 5.0                               # action bounds (m/s^2)
# weights of ego progress, TTC, lane keeping and history comfort in the EPDMS; the two-frame
# comfort needs the previous frame and cannot be predicted from one scene
METRIC_WEIGHTS = (5.0, 5.0, 2.0, 2.0)

GROUPS = ("wp", "act", "path")
DIMS = {"wp": 3 * HORIZON, "act": 2 * HORIZON, "path": 2 * len(PATH_DIST)}


def scaled(traj):
    """(..., 8, 3) poses -> (..., 24) with the heading in metres."""
    return torch.cat([traj[..., :2], traj[..., 2:] * HEADING_SCALE], -1).flatten(-2)


def unscaled(vec):
    """(..., 24) -> (..., 8, 3) poses."""
    x = vec.unflatten(-1, (HORIZON, 3))
    return torch.cat([x[..., :2], x[..., 2:] / HEADING_SCALE], -1)


# ------------------------------------------------------------------- path and actions

def path_targets(future, dists=PATH_DIST):
    """Logged future (B, 8, 3) -> path points (B, P, 2) at arc lengths `dists` and a mask.

    The path is the future polyline resampled by distance travelled; points beyond its
    total length repeat the end point, and the mask marks the points the future reaches.
    """
    xy = torch.cat([torch.zeros_like(future[:, :1, :2]), future[..., :2]], 1)      # (B, 9, 2)
    seg = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1)                                    # (B, 8)
    s = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(-1)], 1)               # (B, 9)
    d = dists.to(future.device)[None, :, None]                                     # (1, P, 1)
    lo = (s[:, None, :-1] <= d) & (d <= s[:, None, 1:])                            # (B, P, 8)
    j = lo.float().argmax(-1)                                                      # first segment that covers d
    g = torch.gather
    s0 = g(s, 1, j); s1 = g(s, 1, j + 1)
    a = g(xy, 1, j[..., None].expand(-1, -1, 2)); b = g(xy, 1, (j + 1)[..., None].expand(-1, -1, 2))
    w = ((d[..., 0] - s0) / (s1 - s0).clamp(min=1e-3)).clamp(0, 1)[..., None]
    ok = lo.any(-1)
    # points the future never reaches keep the end of the path, so an unmasked path is still
    # a sensible polyline (for plots); the loss ignores them
    return torch.where(ok[..., None], a + w * (b - a), xy[:, -1:]), ok


def action_targets(future, speed, dt=DT):
    """Logged future (B, 8, 3) and current speed (B,) -> (B, 8, 2) longitudinal and lateral
    acceleration, by finite differences along the logged poses (the inverse of `rollout`)."""
    xy = torch.cat([torch.zeros_like(future[:, :1, :2]), future[..., :2]], 1)
    th = torch.cat([torch.zeros_like(future[:, :1, 2]), future[..., 2]], 1)
    v = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1) / dt                                # (B, 8) mean speed per step
    v = torch.cat([speed[:, None], v], 1)                                         # (B, 9)
    a_lon = (v[:, 1:] - v[:, :-1]) / dt
    omega = (th[:, 1:] - th[:, :-1]) / dt
    a_lat = v[:, :-1] * omega
    return torch.stack([a_lon.clamp(-A_LON_MAX, A_LON_MAX), a_lat.clamp(-A_LAT_MAX, A_LAT_MAX)], -1)


def rollout(actions, speed, dt=DT, sub=4):
    """Actions (B, 8, 2) in m/s^2 and current speed (B,) -> poses (B, 8, 3).

    A unicycle driven by longitudinal and lateral acceleration, both clamped to the
    controller limits; the curvature follows from the lateral acceleration and the speed, so
    the turn stays feasible at every speed.
    """
    a_lon = actions[..., 0].clamp(-A_LON_MAX, A_LON_MAX)
    a_lat = actions[..., 1].clamp(-A_LAT_MAX, A_LAT_MAX)
    x = torch.zeros_like(speed)
    y, th, v = torch.zeros_like(x), torch.zeros_like(x), speed.clamp(min=0.0)
    out, h = [], dt / sub
    for i in range(actions.shape[1]):
        for _ in range(sub):
            omega = a_lat[:, i] / v.clamp(min=2.0)             # yaw rate; capped at low speed
            th = th + omega * h
            x = x + v * torch.cos(th) * h
            y = y + v * torch.sin(th) * h
            v = (v + a_lon[:, i] * h).clamp(min=0.0)
        out.append(torch.stack([x, y, th], -1))
    return torch.stack(out, 1)


def epdms_of(s):
    """(..., 8) sub-scores in [0, 1] -> (...) EPDMS: NC * DAC * DDC * TLC * weighted mean of the rest."""
    w = torch.tensor(METRIC_WEIGHTS, device=s.device, dtype=s.dtype)
    return s[..., :4].prod(-1) * (s[..., 4:] * w).sum(-1) / w.sum()


# -------------------------------------------------------------------------- attention

class Attention(nn.Module):
    """Multi-head attention with QK-norm (garage: stops the attention-logit growth)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.qn, self.kn = nn.LayerNorm(dim // heads), nn.LayerNorm(dim // heads)

    def forward(self, q, kv, mask=None):
        B, Lq, D = q.shape
        split = lambda t, L: t.view(B, L, self.h, D // self.h).transpose(1, 2)
        qh = self.qn(split(self.q(q), Lq))
        kh = self.kn(split(self.k(kv), kv.shape[1]))
        vh = split(self.v(kv), kv.shape[1])
        x = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
        return self.o(x.transpose(1, 2).reshape(B, Lq, D))


class Block(nn.Module):
    """Pre-norm attention + MLP; `adaln` makes the norms and the residuals conditional."""

    def __init__(self, dim, heads, dropout=0.0, adaln=False):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim, elementwise_affine=not adaln), nn.LayerNorm(
            dim, elementwise_affine=not adaln)
        self.nkv = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim))
        self.mod = None
        if adaln:
            # AdaLN: the conditioning starts neutral (no scale, no shift) but the residual
            # gates start at one, so the block behaves like a plain pre-norm block at step 0
            # and the encoder gets gradient immediately.
            self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
            nn.init.zeros_(self.mod[1].weight)
            nn.init.zeros_(self.mod[1].bias)
            with torch.no_grad():
                self.mod[1].bias[2 * dim:3 * dim] = 1.0
                self.mod[1].bias[5 * dim:] = 1.0

    def forward(self, x, kv=None, mask=None, cond=None):
        kv_in = x if kv is None else kv
        if self.mod is None:
            x = x + self.attn(self.n1(x), self.nkv(kv_in), mask)
            return x + self.mlp(self.n2(x))
        s1, b1, g1, s2, b2, g2 = self.mod(cond).chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, self.nkv(kv_in) if kv is not None else h, mask)
        return x + g2 * self.mlp(self.n2(x) * (1 + s2) + b2)


def sincos_2d(xy, dim, min_wavelength=2.0, max_wavelength=256.0):
    """(N, 2) metric positions -> (N, dim) fixed sin / cos embedding, dim / 4 wavelengths
    per axis log-spaced from min_wavelength to max_wavelength metres."""
    n = dim // 4
    lam = min_wavelength * (max_wavelength / min_wavelength) ** (torch.arange(n) / max(n - 1, 1))
    a = xy[:, :, None] * (2 * math.pi / lam)                                    # (N, 2, n)
    return torch.cat([a[:, 0].sin(), a[:, 0].cos(), a[:, 1].sin(), a[:, 1].cos()], -1)


def time_embedding(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000.0 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


# ------------------------------------------------------------------------------ model

class GaragePlanner(nn.Module):
    def __init__(self, in_channels, num_frames, grid=(9, 16), dim=256, depth=4, heads=8,
                 flow_depth=4, registers=8, dropout=0.1, vocab=None, groups=GROUPS,
                 mask_path=True, score_head=False, use_image=True, use_status=True,
                 use_past=True, use_command=True, flow=True, bev_shape=None, goal=False,
                 anchor_depth=2, cams=1, rays=None, cam_drop=0.0, bev_queries=0,
                 bev_extent=(0.0, 64.0, 24.0), bev_depth=2):
        """cams: cameras in the feature tensor (their channels concatenated, in_channels per
        camera); rays (cams, h, w, 3) unit ego-frame rays of the cells, needed for cams > 1;
        bev_queries: BEV queries along x (0 = off) over bev_extent = (x0, x1, y), the ego
        frame region x in [x0, x1), y in [-y, y) of the BEV targets."""
        super().__init__()
        self.use_image = use_image and in_channels > 0
        self.use_status, self.use_past, self.use_command = use_status, use_past, use_command
        self.groups = tuple(g for g in GROUPS if g in groups)
        self.flow, self.mask_path, self.dim = flow, mask_path, dim
        self.z_dim = sum(DIMS[g] for g in self.groups)

        ego_in = 4 * use_status + 12 * use_past
        self.ego = nn.Sequential(nn.Linear(max(ego_in, 1), dim), nn.GELU(), nn.Linear(dim, dim))
        self.cmd = nn.Linear(4, dim) if use_command else None
        # goal input (py123d_garage "Goal"): the logged pose at the horizon end, ego frame.
        # Its embedding is added to the ego token before the encoder (the scene reads it) and
        # after it (the flow conditioning and the anchor / scorer queries read it directly).
        # A missing goal (NaN, or dropped in training) takes the learned null embedding.
        self.goal_in = None
        if goal:
            self.goal_in = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
            self.goal_out = nn.Linear(dim, dim)
            self.goal_null = nn.Parameter(torch.zeros(1, dim))
            nn.init.trunc_normal_(self.goal_null, std=0.02)
        if self.use_image:
            self.image_proj = nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, dim))
            self.pos = nn.Parameter(torch.zeros(num_frames * grid[0] * grid[1], dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        self.reg = nn.Parameter(torch.zeros(1, registers, dim)) if registers else None
        if registers:
            nn.init.trunc_normal_(self.reg, std=0.02)
        # several cameras: camera embedding, ray MLP, missing-camera tokens (cams = 1 adds nothing)
        self.n_cams, self.cam_drop = cams, cam_drop
        if self.use_image and cams > 1:
            assert rays is not None and tuple(rays.shape) == (cams, *grid, 3), "cams > 1 needs the rays"
            self.cam_emb = nn.Parameter(torch.zeros(cams, dim))
            self.cam_missing = nn.Parameter(torch.zeros(cams, dim))
            nn.init.trunc_normal_(self.cam_emb, std=0.02)
            nn.init.trunc_normal_(self.cam_missing, std=0.02)
            self.ray_mlp = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
            self.register_buffer("rays", torch.as_tensor(rays, dtype=torch.float32).flatten(1, 2))
        # BEV queries
        self.n_bev = 0
        if bev_queries:
            assert self.use_image, "BEV queries read the camera tokens"
            x0, x1, y = bev_extent
            cell = (x1 - x0) / bev_queries
            self.bev_grid = (round(2 * y / cell), bev_queries)                  # (lateral, longitudinal)
            self.n_bev = self.bev_grid[0] * self.bev_grid[1]
            i, j = torch.meshgrid(torch.arange(self.bev_grid[0]), torch.arange(self.bev_grid[1]), indexing="ij")
            # row 0 is the left edge (y = +y), as in bev_targets.py
            centre = torch.stack([x0 + (j.flatten() + 0.5) * cell, y - (i.flatten() + 0.5) * cell], -1)
            self.register_buffer("bev_pos", sincos_2d(centre.float(), dim))
            self.bev_query = nn.Parameter(torch.zeros(self.n_bev, dim))
            nn.init.trunc_normal_(self.bev_query, std=0.02)
            self.bev_blocks = nn.ModuleList([Block(dim, heads, dropout) for _ in range(2 * bev_depth)])
        self.encoder = nn.ModuleList([Block(dim, heads, dropout) for _ in range(depth)])
        self.enc_norm = nn.LayerNorm(dim)
        self.n_ctx = 1 + bool(use_command)            # ego token, command token
        self.grid, self.num_frames = grid, num_frames

        # auxiliary BEV occupancy head (TransFuser-style perception supervision on the
        # trainable trunk; the image encoder itself stays frozen)
        self.bev_shape = tuple(bev_shape) if bev_shape is not None else None
        if self.bev_shape is not None and self.n_bev:
            # per-cell linear decode of the BEV tokens to their sy x sx sub-cells
            c, h, w = self.bev_shape
            assert h % self.bev_grid[0] == 0 and w % self.bev_grid[1] == 0, \
                f"BEV query grid {self.bev_grid} does not tile the target grid {(h, w)}"
            self.bev_sub = (h // self.bev_grid[0], w // self.bev_grid[1])
            self.bev_decode = nn.Linear(dim, c * self.bev_sub[0] * self.bev_sub[1])
        elif self.bev_shape is not None:
            c = self.bev_shape[0]
            self.bev_head = nn.Sequential(nn.Conv2d(dim, dim // 2, 1), nn.GELU(),
                                          nn.Conv2d(dim // 2, dim // 2, 3, padding=1), nn.GELU())
            self.bev_out = nn.Conv2d(dim // 2, c, 1)

        # planner: one token per representation, denoised jointly
        self.tok = nn.ParameterDict({g: nn.Parameter(torch.zeros(1, 1, dim)) for g in self.groups})
        self.z_in = nn.ModuleDict({g: nn.Linear(DIMS[g], dim) for g in self.groups})
        self.head = nn.ModuleDict({g: nn.Linear(dim, DIMS[g]) for g in self.groups})
        for g in self.groups:
            nn.init.zeros_(self.head[g].weight)
            nn.init.zeros_(self.head[g].bias)
        self.t_in = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.flow_blocks = nn.ModuleList([Block(dim, heads, 0.0, adaln=True) for _ in range(flow_depth)])
        self.flow_norm = nn.LayerNorm(dim)
        self.register_buffer("mu", torch.zeros(self.z_dim))
        self.register_buffer("sd", torch.ones(self.z_dim))

        # metric-training head over a trajectory vocabulary
        if vocab is None:
            self.register_buffer("vocab", None)
            self.scorer = None
        else:
            self.register_buffer("vocab", torch.as_tensor(vocab, dtype=torch.float32))   # (V, 8, 3)
            self.anchor = nn.Sequential(nn.Linear(3 * HORIZON, dim), nn.GELU(), nn.Linear(dim, dim))
            self.anchor_blocks = nn.ModuleList([Block(dim, heads, 0.0) for _ in range(anchor_depth)])
            self.anchor_norm = nn.LayerNorm(dim)
            self.logit = nn.Linear(dim, 1)
            self.offset = nn.Linear(dim, 3 * HORIZON)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
            self.scorer = nn.Linear(dim, N_METRICS) if score_head else None
        self.register_buffer("eval_vocab", None)
        # predict(plan="vocab"): candidates from this index on keep their raw trajectory (no offset)
        self.eval_raw_from = None

    # ------------------------------------------------------------------ encoder
    def scene(self, batch):
        """-> memory (B, L, D), ego token (B, 1, D), command token (B, 1, D) or None."""
        parts = []
        if self.use_status:
            parts.append(batch["status"] / 5.0)
        if self.use_past:
            parts.append(scaled(batch["past"]) / 10.0)
        x = torch.cat(parts, -1) if parts else batch["status"].new_zeros(len(batch["status"]), 1)
        ego = self.ego(x)[:, None]
        g = None
        if self.goal_in is not None:
            g = self.goal_embedding(batch["goal"])[:, None]
            ego = ego + g
        tokens = [ego]
        cmd = None
        if self.cmd is not None:
            cmd = self.cmd(batch["command"])[:, None]
            tokens.append(cmd)
        if self.use_image:
            if self.n_cams == 1:
                f = batch["features"].permute(0, 1, 3, 4, 2).flatten(1, 3)          # (B, T h w, C)
                img = self.image_proj(f) + self.pos
            else:
                img = self.camera_tokens(batch["features"])
            tokens.append(img)
        if self.reg is not None:
            tokens.append(self.reg.expand(len(ego), -1, -1))
        if self.n_bev:
            tokens.append(self.bev_tokens(img))
        mem = torch.cat(tokens, 1)
        for blk in self.encoder:
            mem = blk(mem)
        mem = self.enc_norm(mem)
        n_cmd = 1 if cmd is not None else 0
        ego_out = mem[:, :1] if g is None else mem[:, :1] + self.goal_out(g)
        return mem, ego_out, (mem[:, 1:2] if n_cmd else None)

    def camera_tokens(self, features):
        """(B, T, cams * C, h, w), cameras concatenated over C -> (B, cams * T * h * w, D)."""
        B, T, CC, h, w = features.shape
        K = self.n_cams
        f = features.view(B, T, K, CC // K, h, w).permute(0, 2, 1, 4, 5, 3)      # (B, K, T, h, w, C)
        valid = torch.isfinite(f).flatten(2).all(-1)                               # (B, K)
        if self.training and self.cam_drop > 0:
            drop = torch.rand(B, K, device=f.device) < self.cam_drop
            drop[:, 0] = False                                                     # the front camera stays
            valid = valid & ~drop
        x = self.image_proj(torch.nan_to_num(f).flatten(2, 4))                     # (B, K, T h w, D)
        x = torch.where(valid[..., None, None], x, self.cam_missing[None, :, None].to(x.dtype))
        geo = self.cam_emb[:, None] + self.ray_mlp(self.rays).repeat(1, T, 1)      # (K, T h w, D)
        return (x + self.pos + geo).flatten(1, 2)

    def bev_tokens(self, img):
        """Camera tokens (B, L, D) -> BEV tokens (B, n_bev, D): cross-attention to the cameras,
        then self-attention among the BEV cells, bev_depth times."""
        q = (self.bev_query + self.bev_pos)[None].expand(len(img), -1, -1)
        for i in range(0, len(self.bev_blocks), 2):
            q = self.bev_blocks[i](q, img)
            q = self.bev_blocks[i + 1](q)
        return q

    def goal_embedding(self, goal):
        """(B, 3) goal pose, NaN = no goal -> (B, D); scaled like the past poses."""
        ok = torch.isfinite(goal).all(-1, keepdim=True)
        e = self.goal_in(scaled(torch.nan_to_num(goal.float())[:, None]) / 10.0)
        return torch.where(ok, e, self.goal_null.to(e.dtype))

    def bev_logits(self, mem):
        """-> (B, C, H, W) occupancy logits from the image tokens of the encoder output, or
        from the BEV tokens (the last n_bev tokens) when there are BEV queries."""
        if self.n_bev:
            (c, h, w), (gy, gx), (sy, sx) = self.bev_shape, self.bev_grid, self.bev_sub
            x = self.bev_decode(mem[:, -self.n_bev:]).view(len(mem), gy, gx, c, sy, sx)
            return x.permute(0, 3, 1, 4, 2, 5).reshape(len(mem), c, h, w)
        n = self.num_frames * self.grid[0] * self.grid[1]
        x = mem[:, self.n_ctx:self.n_ctx + n]
        x = x.unflatten(1, (self.num_frames, *self.grid)).mean(1).permute(0, 3, 1, 2)
        x = self.bev_head(x)
        x = F.interpolate(x, size=self.bev_shape[1:], mode="bilinear", align_corners=False)
        return self.bev_out(x)

    def _mask(self, mem_len, device):
        """(G, mem_len + G) boolean mask, True = may attend; the command is key index 1.

        The path reads the scene and the command but not the waypoints or actions; the
        waypoints and actions read the scene and the path but not the command. The geometry
        therefore carries the navigation intent and the waypoints the speed along it.
        """
        if not (self.mask_path and self.cmd is not None and "path" in self.groups):
            return None
        idx = {g: i for i, g in enumerate(self.groups)}
        m = torch.ones(len(self.groups), mem_len + len(self.groups), dtype=torch.bool, device=device)
        for g in self.groups:
            if g != "path":
                m[idx[g], 1] = False
                m[idx["path"], mem_len + idx[g]] = False
        return m

    # --------------------------------------------------------------------- flow
    def denoise(self, mem, ego, z, t):
        """z (B, z_dim) normalized, t (B,) -> {group: raw output} of the x0 prediction."""
        cond = self.t_in(time_embedding(t, self.dim)) + ego[:, 0]
        q, off = [], 0
        for g in self.groups:
            d = DIMS[g]
            q.append(self.tok[g].expand(len(z), -1, -1) + self.z_in[g](z[:, off:off + d])[:, None])
            off += d
        q = torch.cat(q, 1)
        mask = self._mask(mem.shape[1], z.device)
        for blk in self.flow_blocks:
            kv = torch.cat([mem, q], 1)
            q = blk(q, kv, mask[None, None] if mask is not None else None, cond[:, None])
        q = self.flow_norm(q)
        return {g: self.head[g](q[:, i]) for i, g in enumerate(self.groups)}

    def pack(self, parts):
        return torch.cat([parts[g] for g in self.groups], -1)

    def unpack(self, z):
        out, off = {}, 0
        for g in self.groups:
            out[g] = z[:, off:off + DIMS[g]]
            off += DIMS[g]
        return out

    def to_z(self, parts):
        return (self.pack(parts) - self.mu) / self.sd

    def from_z(self, z):
        return self.unpack(z * self.sd + self.mu)

    @torch.no_grad()
    def fit(self, batch):
        """Per-coordinate mean and std of the targets (outputs/20: the best normalization)."""
        parts = self.targets(batch)
        x = self.pack({g: parts[g] for g in self.groups})
        self.mu.copy_(x.mean(0))
        self.sd.copy_(x.std(0).clamp(min=1e-2))
        return {"sd_mean": self.sd.mean().item()}

    def targets(self, batch):
        """Supervision of every representation, in the packed coordinates."""
        out = {"wp": scaled(batch["future"])}
        if "act" in self.groups:
            out["act"] = action_targets(batch["future"], batch["status"][:, 0]).flatten(-2)
        if "path" in self.groups:
            p, mask = path_targets(batch["future"])
            out["path"] = p.flatten(-2)
            out["path_mask"] = mask
        return out

    # --------------------------------------------------------------------- loss
    def loss(self, batch, w_act=0.2, w_path=0.2, w_roll=0.2, tau=1.0, w_score=1.0,
             score_targets=None, w_bev=1.0, bev_targets=None, w_rank=0.0, rank_tau=0.1,
             refined_targets=None, w_refined=1.0, row_weight=None, dense_targets=None, w_dense=1.0):
        """One scene pass; flow matching plus, where present, the metric and BEV heads.

        row_weight (B,): per-scene weight of the imitation losses (flow, path, actions, anchor
        CE and offset L1), as a weighted mean; None = plain mean. score_targets = (sub-scores
        (B, V, 8), mask (B,)[, element weight (B, V, 8)]); the weight makes the BCE a weighted
        mean (train.py: SimScale rows, EP only on safe anchors).
        refined_targets = (traj (B, K, 8, 3), sub-scores (B, K, 8), mask (B,)): trajectories of
        the planner's own output distribution (score_refined.py) with their simulated
        sub-scores; the scorer is fit on them as on the raw anchors (BCE, weight w_refined).
        dense_targets = (traj (K, 8, 3), sub-scores (B, K, 8), mask (B,), element weight (B, K, 8)):
        one anchor set shared by the batch (GTRS-Dense subset), BCE as a weighted mean (w_dense)."""
        mem, ego, _ = self.scene(batch)
        loss, logs = self._flow_loss(mem, ego, batch, w_act, w_path, w_roll, row_weight)
        if self.vocab is not None:
            l2, g2 = self._vocab_loss(mem, ego, batch, tau, w_score, score_targets,
                                      w_rank=w_rank, rank_tau=rank_tau, row_weight=row_weight)
            loss = loss + l2
            logs.update(g2)
            if refined_targets is not None and self.scorer is not None:
                traj, tgt, mask = refined_targets
                if mask.any():
                    sc = self.anchors(mem[mask], ego[mask], traj[mask].float())["scores"]
                    bce = F.binary_cross_entropy_with_logits(sc.float(), tgt[mask].float())
                    loss, logs["bce_ref"] = loss + w_refined * bce, bce
            if dense_targets is not None and self.scorer is not None:
                traj, tgt, mask, ew = dense_targets
                if mask.any():
                    n = int(mask.sum())
                    sc = self.anchors(mem[mask], ego[mask], traj[None].expand(n, -1, -1, -1))["scores"]
                    e = F.binary_cross_entropy_with_logits(sc.float(), tgt[mask], reduction="none") * ew[mask]
                    bce = e.sum() / ew[mask].sum().clamp(min=1)
                    loss, logs["bce_dense"] = loss + w_dense * bce, bce
        if self.bev_shape is not None and bev_targets is not None:
            tgt, mask = bev_targets
            if mask.any():
                bev = F.binary_cross_entropy_with_logits(self.bev_logits(mem)[mask].float(), tgt[mask])
                loss, logs["bev"] = loss + w_bev * bev, bev
        return loss, logs

    @staticmethod
    def _mean(e, w=None):
        """Mean of e (B, ...), or its per-scene mean weighted by w (B,)."""
        if w is None:
            return e.mean()
        return (e.reshape(len(e), -1).mean(1) * w).sum() / w.sum().clamp(min=1e-6)

    def _flow_loss(self, mem, ego, batch, w_act=0.2, w_path=0.2, w_roll=0.2, row_weight=None):
        """Flow matching with the x0 target and an L1 loss in physical units (garage recipe)."""
        w = row_weight
        tgt = self.targets(batch)
        x = self.to_z({g: tgt[g] for g in self.groups})
        if self.flow:
            t = torch.rand(len(x), device=x.device)                              # uniform noise levels
            z = (1 - t[:, None]) * x + t[:, None] * torch.randn_like(x)
        else:
            t, z = torch.zeros(len(x), device=x.device), torch.zeros_like(x)      # plain regression
        out = self.denoise(mem, ego, z, t)
        pred = self.from_z(self.pack({g: v.float() for g, v in out.items()}))    # physical units

        logs = {"wp": self._mean((pred["wp"] - tgt["wp"]).abs(), w)}
        loss = logs["wp"]
        if "path" in self.groups:
            # every path point is supervised, also the ones past the end of the future, whose
            # target is the end point (path_targets). Masking them (first version) left those
            # outputs free; they are fed back into z at every Euler step and the zero-noise plan
            # drifted from L2 1.30 (one step) to 4.15 m (eight steps), see diag_flow.py
            logs["path"] = self._mean((pred["path"] - tgt["path"]).abs(), w)
            loss = loss + w_path * logs["path"]
        if "act" in self.groups:
            logs["act"] = self._mean((pred["act"] - tgt["act"]).abs(), w)
            poses = rollout(pred["act"].unflatten(-1, (HORIZON, 2)), batch["status"][:, 0])
            logs["roll"] = self._mean((scaled(poses) - tgt["wp"]).abs(), w)
            loss = loss + w_act * logs["act"] + w_roll * logs["roll"]
        return loss, logs

    def _vocab_loss(self, mem, ego, batch, tau=1.0, w_score=1.0, score_targets=None, w_offset=1.0,
                    w_rank=0.0, rank_tau=0.1, row_weight=None):
        """Imitation over the anchors, L1 on the nearest refined anchor, BCE on PDM sub-scores.

        w_rank adds a listwise loss on the scorer: the softmax over the anchors of the predicted
        EPDMS / rank_tau is pulled towards the softmax of the true EPDMS / rank_tau (the EPDMS
        of the simulated sub-scores, formula of rank). BCE fits every sub-score on its own,
        while the selection only needs the order of the anchors near the top."""
        out = self.anchors(mem, ego)
        gt = batch["future"]
        d = (scaled(self.vocab)[None] - scaled(gt)[:, None]).unflatten(-1, (HORIZON, 3)).norm(dim=-1).mean(-1)
        target = F.softmax(-d / tau, -1)
        ce = self._mean(-(target * out["logits"].log_softmax(-1)).sum(-1), row_weight)
        nearest = d.argmin(-1)
        ref = out["traj"][torch.arange(len(gt)), nearest]
        l1 = self._mean((scaled(ref) - scaled(gt)).abs(), row_weight)
        loss, logs = ce + w_offset * l1, {"ce": ce, "l1_anchor": l1}
        if score_targets is not None and out.get("scores") is not None:
            tgt, mask = score_targets[:2]
            if mask.any():
                if len(score_targets) > 2:                      # element weights, weighted mean
                    ew = score_targets[2][mask]
                    bce = (F.binary_cross_entropy_with_logits(out["scores"][mask].float(), tgt[mask],
                                                              reduction="none") * ew).sum() / ew.sum().clamp(min=1)
                else:
                    bce = F.binary_cross_entropy_with_logits(out["scores"][mask].float(), tgt[mask])
                loss, logs["bce"] = loss + w_score * bce, bce
                if w_rank:
                    rk = F.cross_entropy(epdms_of(torch.sigmoid(out["scores"][mask].float())) / rank_tau,
                                         F.softmax(epdms_of(tgt[mask]) / rank_tau, -1))
                    loss, logs["rank"] = loss + w_rank * rk, rk
        return loss, logs

    # ----------------------------------------------------------------- anchors
    def anchors(self, mem, ego, trajectories=None):
        """Score a set of trajectories (default: the vocabulary) against the scene."""
        traj = self.vocab[None].expand(len(mem), -1, -1, -1) if trajectories is None else trajectories
        q = self.anchor(scaled(traj) / 10.0) + ego
        for blk in self.anchor_blocks:
            q = blk(q, mem)
        q = self.anchor_norm(q)
        off = self.offset(q).unflatten(-1, (HORIZON, 3))
        refined = traj + torch.cat([off[..., :2], off[..., 2:] / HEADING_SCALE], -1)
        out = {"logits": self.logit(q)[..., 0], "traj": refined}
        if self.scorer is not None:
            out["scores"] = self.scorer(q)
        return out

    @staticmethod
    def rank(out, mode="imitation", beta=0.03, top=8):
        """Selection score per candidate (B, N).

        imitation  the log probability of the imitation head alone
        epdms      the log of the EPDMS assembled from the predicted sub-scores, which is
                   what the metric-trained head is for (GTRS / Hydra-MDP selection)
        both       their sum
        mix        log EPDMS + beta * log p_imitation; a small beta keeps the plan close to
                   the imitation mode between frames, which is what two-frame comfort needs
                   (sweep_rank.py: 37.2 vs 30.7 for epdms on the 48-group proxy)
        top        the `top` most likely anchors by imitation, of those the best EPDMS
        """
        imi = out["logits"].log_softmax(-1)
        if mode == "imitation" or "scores" not in out:
            return imi
        ep = epdms_of(torch.sigmoid(out["scores"].float())).clamp_min(1e-6).log()
        if mode == "mix":
            return ep + beta * imi
        if mode == "top":
            idx = imi.topk(min(top, imi.shape[-1]), -1).indices
            return torch.full_like(ep, -1e9).scatter(-1, idx, ep.gather(-1, idx))
        return ep + (imi if mode == "both" else 0.0)

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def _sample(self, mem, ego, num_samples=1, steps=8, temperature=0.0):
        """-> (K, B, z_dim) in the normalized coordinates, by Euler integration from t = 1."""
        B, K = len(mem), num_samples
        mem_k, ego_k = mem.repeat(K, 1, 1), ego.repeat(K, 1, 1)
        if not self.flow:
            z = torch.zeros(K * B, self.z_dim, device=mem.device)
            t = torch.zeros(K * B, device=mem.device)
            out = self.denoise(mem_k, ego_k, z, t)
            return self.pack({g: v.float() for g, v in out.items()}).unflatten(0, (K, B))
        z = torch.randn(K * B, self.z_dim, device=mem.device) * temperature
        ts = torch.linspace(1, 0, steps + 1, device=mem.device)
        for i in range(steps):
            t = ts[i].expand(K * B)
            x0 = self.pack({g: v.float() for g, v in self.denoise(mem_k, ego_k, z, t).items()})
            z = z + (x0 - z) / t[:, None].clamp(min=1e-3) * (ts[i] - ts[i + 1])
        return z.unflatten(0, (K, B))

    @torch.no_grad()
    def sample(self, batch, num_samples=1, steps=8, temperature=0.0):
        """-> (K, B, 8, 3) waypoint plans. temperature 0 starts the flow at z = 0, which makes
        the plan deterministic; > 0 draws real noise and gives diverse plans."""
        mem, ego, _ = self.scene(batch)
        z = self._sample(mem, ego, num_samples, steps, temperature)
        return unscaled(self.from_z(z.flatten(0, 1))["wp"]).unflatten(0, z.shape[:2])

    @torch.no_grad()
    def predict(self, batch, k=8, plan="flow", steps=8, temperature=0.0, rank="imitation",
                n_samples=16, beta=0.03):
        """-> (k, B, 8, 3), the plan that is driven first.

        flow    the zero-noise flow plan, repeated k times
        vocab   the anchors ranked by imitation + predicted metrics (Hydra-MDP / GTRS)
        rerank  the zero-noise plan and `n_samples` noisy flow samples, ranked by the head
        """
        mem, ego, _ = self.scene(batch)
        wp = lambda z: unscaled(self.from_z(z.flatten(0, 1))["wp"]).unflatten(0, z.shape[:2])
        if plan == "flow":
            return wp(self._sample(mem, ego, 1, steps, temperature)).expand(k, -1, -1, -1)
        if plan == "vocab":
            # a denser vocabulary may be ranked than the one the head was trained on: the head
            # reads the trajectory itself, so it generalizes between the anchors (GTRS-Dense)
            dense = None if self.eval_vocab is None else self.eval_vocab[None].expand(
                len(mem), -1, -1, -1)
            out = self.anchors(mem, ego, dense)
            traj = out["traj"]
            if dense is not None and self.eval_raw_from is not None:
                traj = torch.cat([traj[:, :self.eval_raw_from], dense[:, self.eval_raw_from:]], 1)
            top = self.rank(out, rank, beta).topk(k, -1).indices
            return torch.gather(traj, 1, top[..., None, None].expand(-1, -1, HORIZON, 3)).transpose(0, 1)
        cand = torch.cat([wp(self._sample(mem, ego, 1, steps, 0.0)),
                          wp(self._sample(mem, ego, n_samples, steps, temperature or 1.0))]).transpose(0, 1)
        out = self.anchors(mem, ego, cand)
        # the offset head was trained to refine the anchors, so flow samples are only ranked
        top = self.rank(out, rank, beta).topk(min(k, cand.shape[1]), -1).indices
        sel = torch.gather(cand, 1, top[..., None, None].expand(-1, -1, HORIZON, 3)).transpose(0, 1)
        return sel if sel.shape[0] >= k else torch.cat([sel, sel[-1:].expand(k - sel.shape[0], -1, -1, -1)])
