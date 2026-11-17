"""A small PDM sub-score head on the dream features of K candidates per scene.

    # train (dream tokens of block 11) and the no-imagination control
    python -m exp_navsim.orbis2.dream_scorer train --dream outputs/28_inputs/dream/navtrain_h2 --name ds_b11
    python -m exp_navsim.orbis2.dream_scorer train --dream outputs/28_inputs/dream/navtrain_h2 --name ds_b11_ctrl --control
    # predict the sub-scores of the navhard candidates
    python -m exp_navsim.orbis2.dream_scorer predict --run outputs/28_inputs/dream_scorer/ds_b11 \
        --dream outputs/28_inputs/dream/navhard_near_h2

Input per candidate k of a scene:
    dream tokens   (H, C, 9, 16) of the H imagined frames (dream.py, --blocks 11 / 23 / 11,23
                   concatenated over C). --control replaces them by the Orbis 2 feature of the
                   current frame o2l2_b<b>_ctx (Task 1, the same PCA space as the clean-mode
                   dream features) repeated H times: same architecture, no imagination.
    current frame  wm11_ctx (ORBIS-1, (64, 9, 16)) as one more token frame
    trajectory     the candidate's 8 poses (x, y, heading), scaled, through an MLP
    ego status     velocity, acceleration (ego frame) and the one-hot command
Model: tokens -> linear to d + 9x16 position + frame embedding; 4 learned queries plus the
(trajectory + status) embedding attend once over all (H + 1) x 144 tokens; an MLP on the
pooled queries and the condition gives 8 logits. BCE against the PDM sub-scores of the anchors
(soft targets for EP, TTC, LK, HC as in the score banks).

Val split: whole logs, log_in_split(log, "val", 0.1), the hash of v22/train.py.
Metrics on val: per sub-score BCE, MAE and AUC (target > 0.5); pairwise ranking accuracy of
the predicted EPDMS within a scene, over all candidate pairs and over pairs of the top-4 anchors
(the hard case: all four are good), counting pairs whose true EPDMS differ by > 0.01; top-1
regret (best true EPDMS minus that of the predicted best).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_navsim.data.scenes import log_in_split
from exp_navsim.orbis2.common import load_index
from exp_navsim.planner.fast_pdm import METRICS, epdms

OUT = Path("outputs/28_inputs/dream_scorer")
POSE_SCALE = torch.tensor([20.0, 5.0, 0.5])
WEIGHTS = torch.tensor([5.0, 5.0, 2.0, 2.0])           # EP, TTC, LK, HC (fast_pdm.WEIGHTS)


def epdms_torch(p):
    """(..., 8) probabilities -> (...) EPDMS, fast_pdm.epdms without two-frame comfort."""
    w = WEIGHTS.to(p)
    return p[..., :4].prod(-1) * (p[..., 4:] * w).sum(-1) / w.sum()


# ---------------------------------------------------------------- data

def load_data(dream_dir, split, blocks, control, feature_dir="data/features"):
    """-> dict of numpy arrays, one row per scene of the dream dir."""
    dream_dir = Path(dream_dir)
    meta = np.load(dream_dir / "meta.npz")
    rows = meta["rows"]
    index = load_index(split)
    if control:
        H = np.load(dream_dir / f"dream_b{blocks[0]}.npy", mmap_mode="r").shape[2]
        cur = [np.load(Path(feature_dir) / split / f"o2l2_b{b}_ctx.npy", mmap_mode="r")[rows] for b in blocks]
        cur = np.concatenate(cur, 1)                                         # (N, C, 9, 16)
        K = meta["anchors"].shape[1]
        dream = [np.broadcast_to(cur[:, None, None], (len(rows), K, H, *cur.shape[1:]))]
    else:
        # memory maps, gathered per split in to_gpu (two blocks of 10k scenes are 6 GB)
        dream = [np.load(dream_dir / f"dream_b{b}.npy", mmap_mode="r") for b in blocks]
    ctx = np.load(Path(feature_dir) / split / "wm11_ctx.npy", mmap_mode="r")[rows]
    status = np.concatenate([index["velocity"], index["acceleration"], index["command"]], 1)[rows]
    d = dict(rows=rows, tokens=meta["tokens"], anchors=meta["anchors"], poses=meta["poses"],
             dream=dream, ctx=np.asarray(ctx), status=status.astype(np.float32),
             log=np.asarray(index["log"])[rows])
    if "sub_scores" in meta.files:
        d["target"] = meta["sub_scores"].astype(np.float32)
    return d


def to_gpu(d, idx, dev):
    """Scenes idx -> tensors on dev (features in fp16)."""
    dream = torch.cat([torch.from_numpy(np.ascontiguousarray(a[idx])).to(dev) for a in d["dream"]], 3)
    out = {"dream": dream.half(),
           "ctx": torch.from_numpy(d["ctx"][idx]).half().to(dev),
           "poses": torch.from_numpy(d["poses"][idx]).float().to(dev),
           "status": torch.from_numpy(d["status"][idx]).to(dev)}
    if "target" in d:
        out["target"] = torch.from_numpy(d["target"][idx]).to(dev)
    return out


# ---------------------------------------------------------------- model

class DreamScorer(nn.Module):
    def __init__(self, c_in, horizon, dim=192, heads=4, queries=4, dropout=0.1):
        super().__init__()
        self.tok = nn.Linear(c_in, dim)
        self.ctx_tok = nn.Linear(64, dim)
        self.pos = nn.Parameter(torch.randn(144, dim) * 0.02)
        self.frame = nn.Parameter(torch.randn(horizon + 1, dim) * 0.02)       # 0 = current frame
        self.traj = nn.Sequential(nn.Linear(24, dim), nn.GELU(), nn.Linear(dim, dim))
        self.status = nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim))
        self.q = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.head = nn.Sequential(nn.Linear((queries + 1) * dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * dim, len(METRICS)))

    def forward(self, dream, ctx, poses, status):
        """dream (B, H, C, 9, 16), ctx (B, 64, 9, 16), poses (B, 8, 3), status (B, 8) -> logits (B, 8)."""
        B, H = dream.shape[:2]
        x = self.tok(dream.float().flatten(3).transpose(2, 3))                # (B, H, 144, d)
        x = x + self.pos + self.frame[1:, None]
        c = self.ctx_tok(ctx.float().flatten(2).transpose(1, 2)) + self.pos + self.frame[0]
        kv = self.norm_kv(torch.cat([c, x.flatten(1, 2)], 1))                 # (B, (H + 1) 144, d)
        cond = self.traj((poses / POSE_SCALE.to(poses)).flatten(1)) + self.status(status)
        q = self.norm_q(self.q[None] + cond[:, None])
        a, _ = self.attn(q, kv, kv, need_weights=False)
        a = a + self.mlp(a)
        return self.head(torch.cat([a.flatten(1), cond], 1))


def flat_batch(g, scene_idx, k_idx):
    return (g["dream"][scene_idx, k_idx], g["ctx"][scene_idx], g["poses"][scene_idx, k_idx],
            g["status"][scene_idx])


@torch.no_grad()
def predict(model, g, batch=1024):
    """-> probabilities (N, K, 8) for every candidate of the scenes in g."""
    model.eval()
    N, K = g["poses"].shape[:2]
    s, k = torch.meshgrid(torch.arange(N), torch.arange(K), indexing="ij")
    s, k = s.flatten().to(g["poses"].device), k.flatten().to(g["poses"].device)
    out = []
    for i in range(0, len(s), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=g["poses"].is_cuda):
            out.append(torch.sigmoid(model(*flat_batch(g, s[i:i + batch], k[i:i + batch])).float()))
    return torch.cat(out).view(N, K, -1)


# ---------------------------------------------------------------- metrics

def auc(score, label):
    """Rank AUC of a binary label; NaN with one class only."""
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    r = np.argsort(np.argsort(score, kind="stable"), kind="stable") + 1
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def pairwise(pred, true, mask=None, margin=0.01):
    """Within-scene pairwise ranking accuracy of pred vs true, (N, K) each."""
    dp = pred[:, :, None] - pred[:, None, :]
    dt = true[:, :, None] - true[:, None, :]
    use = np.abs(dt) > margin
    use &= np.triu(np.ones(use.shape[1:], bool), 1)[None]
    if mask is not None:
        use &= mask[:, :, None] & mask[:, None, :]
    return float((np.sign(dp) == np.sign(dt))[use].mean()), int(use.sum())


def evaluate(prob, target, k_top=4):
    prob, target = prob.astype(np.float64), target.astype(np.float64)
    m = {}
    for j, name in enumerate(METRICS):
        p, t = prob[..., j].ravel(), target[..., j].ravel()
        bce = -(t * np.log(np.clip(p, 1e-6, 1)) + (1 - t) * np.log(np.clip(1 - p, 1e-6, 1))).mean()
        m[name] = dict(bce=round(float(bce), 4), mae=round(float(np.abs(p - t).mean()), 4),
                       auc=round(auc(p, t > 0.5), 4), pos_rate=round(float((t > 0.5).mean()), 3))
    ep_pred, ep_true = epdms(prob), epdms(target)
    m["pair_acc_all"], m["pairs_all"] = pairwise(ep_pred, ep_true)
    top = np.zeros(ep_true.shape, bool)
    top[:, :k_top] = True                                  # dream.py: the first k_top are the top anchors
    m["pair_acc_top4"], m["pairs_top4"] = pairwise(ep_pred, ep_true, top)
    pick = ep_pred.argmax(1)
    m["top1_regret"] = round(float((ep_true.max(1) - ep_true[np.arange(len(pick)), pick]).mean()), 4)
    pick4 = np.where(top, ep_pred, -1).argmax(1)
    m["top1_regret_top4"] = round(float((ep_true[:, :k_top].max(1) - ep_true[np.arange(len(pick4)), pick4]).mean()), 4)
    m["epdms_corr"] = round(float(np.corrcoef(ep_pred.ravel(), ep_true.ravel())[0, 1]), 4)
    return m


# ---------------------------------------------------------------- main

def train(args):
    dev = args.device
    blocks = [int(b) for b in args.blocks.split(",")]
    d = load_data(args.dream, args.split, blocks, args.control)
    val = np.array([log_in_split(l, "val", args.val_fraction) for l in d["log"]])
    if not val.any():
        # the dreamed scenes come from rows with PDM targets, which exclude the v22 val logs;
        # hold out the next hash bucket of whole logs instead
        import hashlib
        b = np.array([int(hashlib.md5(l.encode()).hexdigest(), 16) % 1000 for l in d["log"]])
        lo = args.val_fraction * 1000
        val = (b >= lo) & (b < 2 * lo)
        print(f"[data] no v22 val logs among the dreamed scenes, holding out hash bucket "
              f"[{lo:.0f}, {2 * lo:.0f}) instead", flush=True)
    tr_idx, va_idx = np.flatnonzero(~val), np.flatnonzero(val)
    print(f"[data] {len(tr_idx)} train / {len(va_idx)} val scenes, dream {[a.shape for a in d['dream']]}, "
          f"control {args.control}", flush=True)
    gtr, gva = to_gpu(d, tr_idx, dev), to_gpu(d, va_idx, dev)
    N, K, H, C = gtr["dream"].shape[:4]
    torch.manual_seed(args.seed)
    model = DreamScorer(C, H, args.dim, args.heads, args.queries, args.dropout).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_par / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        s = torch.randint(0, N, (args.batch,), device=dev, generator=gen)
        k = torch.randint(0, K, (args.batch,), device=dev, generator=gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            logits = model(*flat_batch(gtr, s, k)).float()
        loss = F.binary_cross_entropy_with_logits(logits, gtr["target"][s, k])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps:
            pv = predict(model, gva).cpu().numpy()
            tv = gva["target"].cpu().numpy()
            m = evaluate(pv, tv)
            print(f"step {step:5d}  loss {loss.item():.4f}  val pair_acc {m['pair_acc_all']:.3f} "
                  f"top4 {m['pair_acc_top4']:.3f}  regret {m['top1_regret']:.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    run = OUT / args.name
    run.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "c_in": C, "horizon": H}, run / "model.pt")
    m.update(params=n_par, train_scenes=len(tr_idx), val_scenes=len(va_idx), seconds=round(time.time() - t0))
    (run / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


def run_predict(args):
    dev = args.device
    ck = torch.load(Path(args.run) / "model.pt", map_location=dev, weights_only=False)
    a = ck["args"]
    blocks = [int(b) for b in a["blocks"].split(",")]
    d = load_data(args.dream, args.split, blocks, a["control"])
    model = DreamScorer(ck["c_in"], ck["horizon"], a["dim"], a["heads"], a["queries"], a["dropout"]).to(dev)
    model.load_state_dict(ck["model"])
    prob = []
    for i in range(0, len(d["rows"]), 1000):                # chunks of scenes keep the GPU copy small
        prob.append(predict(model, to_gpu(d, np.arange(i, min(i + 1000, len(d["rows"]))), dev)).cpu().numpy())
    prob = np.concatenate(prob)
    logep = np.log(np.clip(epdms(prob.astype(np.float64)), 1e-6, None)).astype(np.float32)
    out = Path(args.dream) / f"dream_scores_{Path(args.run).name}.npz"
    np.savez(out, token=d["tokens"], rows=d["rows"], anchors=d["anchors"], sub=prob, logep=logep)
    print(f"{out}: {prob.shape}")
    if "target" in d:
        print(json.dumps(evaluate(prob, d["target"]), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--dream", required=True, help="dream.py output dir")
    ap.add_argument("--split", default=None, help="index split of the dream dir (default navtrain / navhard)")
    ap.add_argument("--run", help="predict: the trained run dir")
    ap.add_argument("--name", default="ds_b11")
    ap.add_argument("--blocks", default="11", help="dream blocks, e.g. 11 or 23 or 11,23")
    ap.add_argument("--control", action="store_true", help="current-frame o2l2 feature instead of the dream")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--queries", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.split is None:
        args.split = "navhard" if "navhard" in str(args.dream) else "navtrain"
    train(args) if args.cmd == "train" else run_predict(args)


if __name__ == "__main__":
    main()
