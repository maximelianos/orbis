"""Verifier on several scene sets: vocabulary scores and generator-sample scores, pooled.

    python -m exp_navsim.verified.train_verifier --name 18_verifier_v3 \
        --gen outputs/17_diversity/verifier/gen_scores_navtrain_sub.npz,outputs/18_verified_sampling/gen_scores_navtrain_sub2.npz

Same model as diverse/verifier.py. Every scene gets a padded bank of scored candidates:
the 256 vocabulary trajectories where planner/score_vocab.py scored them (navtrain_sub)
and the generator samples of every --gen file that covers the scene. A step draws 64
scenes and 64 candidates per scene, half from the vocabulary and half from the
generator samples where a scene has both. 10% of the logs are held out (AUC off-road and
Spearman on all their candidates). Test: the navtest samples of 17_diversity
(diverse.verifier.test_on_samples).
-> logs_exp/<name>/verifier.pt, outputs/18_verified_sampling/verifier_<name>.csv
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.diverse.train import to_gpu
from exp_navsim.diverse.verifier import PLANNER, SAMPLES, Verifier, auc, test_on_samples
from exp_navsim.planner.fast_pdm import METRICS, epdms

OUT = Path("outputs/18_verified_sampling")


def build_bank(gen_files, vocab_size=256):
    """-> rows (N,), candidates (N, C, 8, 3), scores (N, C, M), weights (N, C) as numpy."""
    st = np.load(PLANNER / f"vocab{vocab_size}_scores_navtrain_sub.npz")
    vocab = np.load(PLANNER / f"vocab_{vocab_size}.npy").astype(np.float32)
    voc = {int(r): s for r, s in zip(st["rows"], st["scores"])}
    gen = {}
    for f in gen_files:
        d = np.load(f)
        for r, c, s in zip(d["rows"], d["samples"], d["scores"]):
            gen.setdefault(int(r), []).append((c.astype(np.float32), s.astype(np.float32)))
    rows = np.array(sorted(set(voc) | set(gen)))
    n_gen = {r: sum(len(c) for c, _ in gen.get(r, [])) for r in rows}
    C = max((vocab_size if r in voc else 0) + n_gen[r] for r in rows)
    cand = np.zeros((len(rows), C, 8, 3), np.float32)
    score = np.zeros((len(rows), C, len(METRICS)), np.float32)
    weight = np.zeros((len(rows), C), np.float32)
    for i, r in enumerate(rows):
        j = 0
        parts = []
        if r in voc:
            parts.append((vocab, voc[r].astype(np.float32), "voc"))
        parts += [(c, s, "gen") for c, s in gen.get(r, [])]
        has_both = r in voc and n_gen[r] > 0
        for c, s, kind in parts:
            cand[i, j:j + len(c)], score[i, j:j + len(c)] = c, s
            share = 0.5 if has_both else 1.0
            weight[i, j:j + len(c)] = share / (vocab_size if kind == "voc" else n_gen[r])
            j += len(c)
    return rows, cand, score, weight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gen", required=True, help="comma-separated verifier_data.py npz files")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--scenes-per-batch", type=int, default=64)
    ap.add_argument("--cands-per-scene", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--test", default="plain_92k,full16_92k,full64_92k,full64_10k")
    args = ap.parse_args()
    run = Path("logs_exp") / args.name
    if run.exists():
        sys.exit(f"{run} exists")
    run.mkdir(parents=True)
    (run / "README.md").write_text(f"# {args.name}\n\nverifier on pooled scene sets "
                                   f"(exp_navsim/verified/train_verifier.py)\n\n```\npython -m "
                                   f"exp_navsim.verified.train_verifier {' '.join(sys.argv[1:])}\n```\n")
    torch.manual_seed(0)

    rows, cand, score, weight = build_bank(args.gen.split(","))
    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=["wm11_ctx"], split="all",
                             feature_dir="data/features", history=4)
    logs = np.asarray(ds.index["log"])[rows]
    uniq = np.unique(logs)
    held = set(np.random.default_rng(0).permutation(uniq)[:len(uniq) // 10])
    is_val = np.array([l in held for l in logs])
    feats = to_gpu(ds, rows)["features"]
    cand, score, weight = (torch.from_numpy(x).cuda() for x in (cand, score, weight))
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(f"[data] {len(rows)} scenes ({len(tr)} train / {len(va)} held-out logs), bank {cand.shape[1]} candidates",
          flush=True)

    model = Verifier(feats.shape[2], feats.shape[1], args.dim, depth=args.depth).cuda()
    model.fit(cand[weight > 0])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 300) * 0.5 * (
        1 + math.cos(math.pi * min(s / args.steps, 1))))
    tr_t = torch.from_numpy(tr).cuda()

    @torch.no_grad()
    def held_out():
        model.eval()
        p, y = [], []
        for i in range(0, len(va), 16):
            idx = torch.from_numpy(va[i:i + 16]).cuda()
            pr = torch.sigmoid(model({"features": feats[idx].float()}, cand[idx]))
            m = weight[idx] > 0
            p.append(pr[m].cpu().numpy())
            y.append(score[idx][m].cpu().numpy())
        model.train()
        p, y = np.concatenate(p).astype(np.float64), np.concatenate(y).astype(np.float64)
        pe, ye = epdms(p), epdms(y)
        dac = y[:, METRICS.index("drivable_area_compliance")] == 0
        return {"val/spearman": pd.Series(pe).corr(pd.Series(ye), method="spearman"),
                "val/auc_offroad": auc(-pe, dac)}

    for step in range(1, args.steps + 1):
        idx = tr_t[torch.randint(len(tr_t), (args.scenes_per_batch,), device="cuda")]
        j = torch.multinomial(weight[idx], args.cands_per_scene, replacement=True)
        logits = model({"features": feats[idx].float()}, cand[idx[:, None], j])
        loss = F.binary_cross_entropy_with_logits(logits, score[idx[:, None], j])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0:
            print(f"step {step}  loss {loss.item():.4f}  " + "  ".join(f"{k} {v:.3f}" for k, v in held_out().items()),
                  flush=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "in_channels": feats.shape[2],
                "num_frames": feats.shape[1]}, run / "verifier.pt")

    model.eval()
    out = [{"samples": "held-out logs", **held_out()}]
    for label in args.test.split(","):
        if (SAMPLES / f"per_sample_{label}.npz").exists():
            out.append(test_on_samples(model, label, ["wm11_ctx"]))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[-1].items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(OUT / f"verifier_{args.name}.csv", index=False)
    print(f"-> {OUT / f'verifier_{args.name}.csv'}")


if __name__ == "__main__":
    main()
