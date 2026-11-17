"""Learned trajectory scorer with the planner's inputs, for sample-then-rerank.

    python -m exp_navsim.hybrid.scorer --name 19_scorer --gen <npz>,<npz>,...

Hydra-MDP (arXiv 2406.06978) learns the PDM sub-scores of every vocabulary trajectory;
here the candidates are any trajectories (vocabulary and generator samples), as in the
image-only verifier of 17/18 (diverse/verifier.py), but the scene encoder is the one of
the fast planner (planner/model.py): image tokens plus the ego token (velocity,
acceleration, past poses, command). Each candidate is embedded by an MLP, added to the
ego token and cross-attends to the scene; one logit per fast_pdm.METRICS entry, BCE
against the true sub-scores. The predicted EPDMS is fast_pdm.epdms of the sigmoids.

Data: verified.train_verifier.build_bank of the --gen files (vocabulary scores of 5000
navtrain scenes plus every scored generator sample of the 10000 cached scenes); 10% of
the logs held out (Spearman of predicted vs true EPDMS, AUC for off-road, and how often
the top-1 by prediction is within 0.05 of the best candidate).
-> logs_exp/<name>/scorer.pt
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_navsim.diverse.verifier import auc, epdms_torch
from exp_navsim.planner.fast_pdm import METRICS, epdms
from exp_navsim.planner.model import CrossBlock, Planner, scaled


class Scorer(Planner):
    def __init__(self, in_channels, num_frames, dim=256, depth=4, cand_depth=2, dropout=0.1):
        super().__init__(in_channels, num_frames, vocab=None, dim=dim, depth=depth, decoder_depth=0,
                         dropout=dropout)
        del self.query, self.out
        self.cand = nn.Sequential(nn.Linear(24, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([CrossBlock(dim, 4, dropout) for _ in range(cand_depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, len(METRICS)))

    def forward(self, batch, candidates):
        """candidates (B, K, 8, 3) -> sub-score logits (B, K, len(METRICS))."""
        mem, ego = self.scene(batch)
        q = self.cand(scaled(candidates) / 10.0) + ego
        for blk in self.blocks:
            q = blk(q, mem)
        return self.head(q)

    @torch.no_grad()
    def predict_epdms(self, batch, candidates, chunk=64):
        """candidates (K, B, 8, 3) -> predicted EPDMS (K, B)."""
        out = []
        for i in range(0, candidates.shape[1], chunk):
            b = {k: v[i:i + chunk] for k, v in batch.items()}
            out.append(epdms_torch(torch.sigmoid(self(b, candidates[:, i:i + chunk].transpose(0, 1).float()))).T)
        return torch.cat(out, 1)


def load_scorer(path):
    ck = torch.load(path, map_location="cuda")
    a = ck["args"]
    s = Scorer(ck["in_channels"], ck["num_frames"], a["dim"], a["depth"], a["cand_depth"]).cuda()
    s.load_state_dict(ck["model"])
    return s.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--cand-depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--scenes-per-batch", type=int, default=64)
    ap.add_argument("--cands-per-scene", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()
    run = Path("logs_exp") / args.name
    if run.exists():
        sys.exit(f"{run} exists")
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=1))
    (run / "README.md").write_text(f"# {args.name}\n\nscorer with ego inputs (exp_navsim/hybrid/scorer.py)\n\n"
                                   f"```\npython -m exp_navsim.hybrid.scorer {' '.join(sys.argv[1:])}\n```\n")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True

    from exp_navsim.data.scenes import SceneFeatureDataset
    from exp_navsim.hybrid.train import to_gpu
    from exp_navsim.verified.train_verifier import build_bank

    rows, cand, score, weight = build_bank(args.gen.split(","))
    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=["wm11_ctx"], split="all",
                             feature_dir="data/features", history=4)
    logs = np.asarray(ds.index["log"])[rows]
    uniq = np.unique(logs)
    held = set(np.random.default_rng(0).permutation(uniq)[:len(uniq) // 10])
    is_val = np.array([l in held for l in logs])
    data = to_gpu(ds, rows)
    cand, score, weight = (torch.from_numpy(x).cuda() for x in (cand, score, weight))
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(f"[data] {len(rows)} scenes ({len(tr)} train / {len(va)} held-out logs), bank {cand.shape[1]} candidates",
          flush=True)

    T, C = data["features"].shape[1:3]
    model = Scorer(C, T, args.dim, args.depth, args.cand_depth).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 300) * 0.5 * (
        1 + math.cos(math.pi * min(s / args.steps, 1))))
    tr_t = torch.from_numpy(tr).cuda()

    def batch(idx):
        b = {k: v[idx] for k, v in data.items()}
        b["features"] = b["features"].float()
        return b

    @torch.no_grad()
    def held_out():
        model.eval()
        p, y, top_gap = [], [], []
        for i in range(0, len(va), 16):
            idx = torch.from_numpy(va[i:i + 16]).cuda()
            pr = torch.sigmoid(model(batch(idx), cand[idx]))
            m = weight[idx] > 0
            pe, ye = epdms_torch(pr), epdms_torch(score[idx].float())
            for j in range(len(idx)):
                mj = m[j]
                top_gap.append((ye[j][mj].max() - ye[j][mj][pe[j][mj].argmax()]).item())
            p.append(pr[m].cpu().numpy())
            y.append(score[idx][m].cpu().numpy())
        model.train()
        p, y = np.concatenate(p).astype(np.float64), np.concatenate(y).astype(np.float64)
        pe, ye = epdms(p), epdms(y)
        dac = y[:, METRICS.index("drivable_area_compliance")] == 0
        gap = np.array(top_gap)
        return {"val/spearman": pd.Series(pe).corr(pd.Series(ye), method="spearman"),
                "val/auc_offroad": auc(-pe, dac), "val/top1_gap": gap.mean(), "val/top1_within_0.05": (gap < 0.05).mean()}

    for step in range(1, args.steps + 1):
        idx = tr_t[torch.randint(len(tr_t), (args.scenes_per_batch,), device="cuda")]
        j = torch.multinomial(weight[idx], args.cands_per_scene, replacement=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch(idx), cand[idx[:, None], j])
        loss = F.binary_cross_entropy_with_logits(logits.float(), score[idx[:, None], j].float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2500 == 0:
            print(f"step {step}  loss {loss.item():.4f}  " + "  ".join(f"{k} {v:.3f}" for k, v in held_out().items()),
                  flush=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "in_channels": C, "num_frames": T},
               run / "scorer.pt")
    res = held_out()
    (run / "held_out.json").write_text(json.dumps(res, indent=1))
    print(f"-> {run / 'scorer.pt'}  {res}")


if __name__ == "__main__":
    main()
