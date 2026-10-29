"""Prediction-only planner on frozen features: no diffusion, one forward pass.

    scene tokens = [ego token, image tokens]
        ego token     MLP of velocity, acceleration (4), the 4 past poses (12), command (4)
        image tokens  Linear per cell of the cached grid (T, C, 9, 16) + learned position
    scene encoder: `depth` transformer layers over the scene tokens

Two heads:
    regress (vocab=None)  one query attends to the scene and regresses the 8 poses
                          (the plain "prediction-only probe")
    vocab                 V anchor queries (k-means trajectories, vocab.py), each the
                          anchor embedding + ego token, cross-attend to the scene
                          (no self-attention, so V can be large); per anchor a logit,
                          a pose offset and optionally PDM sub-score logits
                          (Hydra-MDP distillation targets, score_vocab.py)
    hybrid                both: the regressed trajectory is the top-1 output, the
                          vocabulary gives the other directions

Inference ranks the anchors by
    log softmax(logit) + sum_m w_m log sigmoid(metric_m)
and returns the top K refined anchors (the possible directions), best first.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADING_SCALE = 5.0   # metres per radian in the losses, as in vocab.py
# sub-scores predicted by the distillation head, in the order of fast_pdm.METRICS
N_METRICS = 8


class CrossBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm_q, self.norm_kv, self.norm_ff = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))

    def forward(self, q, kv):
        kv = self.norm_kv(kv)
        q = q + self.attn(self.norm_q(q), kv, kv, need_weights=False)[0]
        return q + self.ff(self.norm_ff(q))


def scaled(traj):
    """(…, 8, 3) poses -> (…, 24) with the heading in metres."""
    return torch.cat([traj[..., :2], traj[..., 2:] * HEADING_SCALE], -1).flatten(-2)


class Planner(nn.Module):
    def __init__(self, in_channels, num_frames, grid=(9, 16), vocab=None, dim=128, depth=2, heads=4,
                 decoder_depth=2, dropout=0.1, use_image=True, use_status=True, use_past=True,
                 use_command=True, score_head=False, hybrid=False):
        super().__init__()
        self.use_image = use_image and in_channels > 0
        self.use_status, self.use_past, self.use_command = use_status, use_past, use_command
        ego_in = 4 * use_status + 12 * use_past + 4 * use_command
        self.ego = nn.Sequential(nn.Linear(max(ego_in, 1), dim), nn.GELU(), nn.Linear(dim, dim))
        if self.use_image:
            self.image_proj = nn.Sequential(nn.LayerNorm(in_channels), nn.Linear(in_channels, dim))
            self.pos = nn.Parameter(torch.zeros(num_frames * grid[0] * grid[1], dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.decoder = nn.ModuleList([CrossBlock(dim, heads, dropout) for _ in range(decoder_depth)])
        self.norm = nn.LayerNorm(dim)

        self.hybrid = hybrid and vocab is not None
        if vocab is None or hybrid:
            self.query = nn.Parameter(torch.zeros(1, 1, dim))
            self.out = nn.Linear(dim, 24)
        if vocab is None:
            self.register_buffer("vocab", None)
        else:
            self.register_buffer("vocab", torch.as_tensor(vocab, dtype=torch.float32))     # (V, 8, 3)
            self.anchor = nn.Sequential(nn.Linear(24, dim), nn.GELU(), nn.Linear(dim, dim))
            self.logit = nn.Linear(dim, 1)
            self.offset = nn.Linear(dim, 24)
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)
            self.scores = nn.Linear(dim, N_METRICS) if score_head else None

    def ego_input(self, batch):
        parts = []
        if self.use_status:
            parts.append(batch["status"] / 5.0)
        if self.use_past:
            parts.append(scaled(batch["past"]) / 10.0)
        if self.use_command:
            parts.append(batch["command"])
        if not parts:
            return batch["status"].new_zeros(batch["status"].shape[0], 1)
        return torch.cat(parts, -1)

    def scene(self, batch):
        ego = self.ego(self.ego_input(batch))[:, None]                           # (B, 1, D)
        tokens = [ego]
        if self.use_image:
            x = batch["features"]                                                # (B, T, C, h, w)
            x = x.permute(0, 1, 3, 4, 2).flatten(1, 3)                           # (B, T h w, C)
            tokens.append(self.image_proj(x) + self.pos)
        return self.encoder(torch.cat(tokens, 1)), ego

    def forward(self, batch):
        """-> dict: regress: traj (B, 8, 3); vocab: logits (B, V), traj (B, V, 8, 3), scores (B, V, M)."""
        mem, ego = self.scene(batch)
        B = mem.shape[0]
        if self.vocab is None or self.hybrid:
            q = self.query.expand(B, -1, -1) + ego
            for blk in self.decoder:
                q = blk(q, mem)
            reg = self.out(self.norm(q))[:, 0].view(B, 8, 3)
            reg = torch.cat([reg[..., :2], reg[..., 2:] / HEADING_SCALE], -1)
            if self.vocab is None:
                return {"traj": reg}
        anchors = scaled(self.vocab)                                              # (V, 24)
        q = self.anchor(anchors / 10.0)[None] + ego                               # (B, V, D)
        for blk in self.decoder:
            q = blk(q, mem)
        q = self.norm(q)
        off = self.offset(q).view(B, -1, 8, 3)
        traj = self.vocab[None] + torch.cat([off[..., :2], off[..., 2:] / HEADING_SCALE], -1)
        out = {"logits": self.logit(q)[..., 0], "traj": traj}
        if self.scores is not None:
            out["scores"] = self.scores(q)
        if self.hybrid:
            out["reg"] = reg
        return out

    @torch.no_grad()
    def rank(self, out, weights=None):
        """Selection score per anchor (B, V): imitation log-prob plus weighted log sub-scores."""
        s = out["logits"].log_softmax(-1)
        if weights is not None and "scores" in out:
            s = s + (F.logsigmoid(out["scores"]) * torch.as_tensor(weights, device=s.device)).sum(-1)
        return s

    @torch.no_grad()
    def predict(self, batch, k=8, weights=None):
        """-> (k, B, 8, 3) trajectories, best first (k copies of the one output for regress)."""
        out = self(batch)
        if self.vocab is None:
            return out["traj"][None].expand(k, -1, -1, -1)
        kv = k - 1 if self.hybrid else k
        top = self.rank(out, weights).topk(kv, dim=-1).indices                   # (B, kv)
        vocab = torch.gather(out["traj"], 1, top[..., None, None].expand(-1, -1, 8, 3)).transpose(0, 1)
        return torch.cat([out["reg"][None], vocab]) if self.hybrid else vocab


def planner_loss(model, out, batch, tau=1.0, w_offset=1.0, w_score=1.0, score_targets=None, offset_all=False):
    """Imitation (soft cross-entropy over anchors by distance to the logged future),
    L1 on the refined nearest anchor, BCE on the PDM sub-scores where targets exist."""
    gt = batch["future"]
    if model.vocab is None:
        l1 = (scaled(out["traj"]) - scaled(gt)).abs().mean()
        return l1, {"l1": l1}
    d = (scaled(model.vocab)[None] - scaled(gt)[:, None]).view(gt.shape[0], -1, 8, 3).norm(dim=-1).mean(-1)
    target = F.softmax(-d / tau, -1)                                              # (B, V)
    ce = -(target * out["logits"].log_softmax(-1)).sum(-1).mean()
    if offset_all:
        # every anchor refined towards the logged future, weighted like the imitation target
        err = (scaled(out["traj"]) - scaled(gt)[:, None]).abs().mean(-1)          # (B, V)
        l1 = (target * err).sum(-1).mean()
    else:
        nearest = d.argmin(-1)
        ref = out["traj"][torch.arange(len(gt)), nearest]
        l1 = (scaled(ref) - scaled(gt)).abs().mean()
    loss, logs = ce + w_offset * l1, {"ce": ce, "l1": l1}
    if "reg" in out:
        l1_reg = (scaled(out["reg"]) - scaled(gt)).abs().mean()
        loss, logs["l1_reg"] = loss + l1_reg, l1_reg
    if score_targets is not None and "scores" in out:
        tgt, mask = score_targets                                                 # (B, V, M), (B,)
        if mask.any():
            bce = F.binary_cross_entropy_with_logits(out["scores"][mask], tgt[mask])
            loss = loss + w_score * bce
            logs["bce"] = bce
    return loss, logs
