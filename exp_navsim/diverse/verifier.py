"""A learned verifier for sampled futures: predicts the PDM sub-scores of a candidate from the image.

    python -m exp_navsim.diverse.verifier --name 17_verifier

Generate-then-verify: the generator (IntentFlow, CtrlFlow, ...) keeps the diversity, the
verifier removes the samples that leave the road or hit something. As in Hydra-MDP
(arXiv 2406.06978) the PDM sub-scores of candidate trajectories are the targets, but here
the verifier scores samples of a generator instead of choosing from a fixed vocabulary.

Inputs are image features only (wm11_ctx, like the generators); the candidate (8 poses)
is embedded by an MLP and is the query of two cross-attention blocks over the image
tokens. Output: one logit per fast_pdm.METRICS entry, binary cross-entropy against the
sub-scores in [0, 1]; the predicted EPDMS is fast_pdm.epdms of the sigmoids.

Training data: the 5000 cached navtrain scenes (planner/cache_navtrain.py, training logs)
with the sub-scores of all 256 vocabulary trajectories (planner/score_vocab.py). 10% of
the logs are held out. With --gen, also the generator samples of verifier_data.py on
the same scenes (half of the candidates of a step come from there). Test: the generated navtest samples of score_samples.py
(<out>/per_sample_<label>.npz hold their true sub-scores), which the verifier never saw
in any form: AUC for off-road / collision, Spearman of the predicted vs true EPDMS, and
how many bad samples survive when only the better half by predicted EPDMS is kept.
-> logs_exp/<name>/verifier.pt, outputs/17_diversity/verifier/<name>.csv
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

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.diverse.train import to_gpu
from exp_navsim.models.denoiser import ProbeDenoiser
from exp_navsim.planner.fast_pdm import METRICS, WEIGHTS, epdms
from networks.blocks import ContextTransformer

PLANNER = Path("outputs/16_fast_planner")
OUT = Path("outputs/17_diversity/verifier")
SAMPLES = Path("outputs/17_diversity/navtest_samples")


def epdms_torch(sub):
    """sub (..., len(METRICS)) in [0, 1] -> EPDMS (...), without two-frame comfort."""
    num = sum(WEIGHTS[m] * sub[..., METRICS.index(m)] for m in METRICS[4:])
    return sub[..., :4].prod(-1) * num / sum(WEIGHTS[m] for m in METRICS[4:])


class Verifier(nn.Module):
    def __init__(self, in_channels, num_frames, dim=128, ff=256, heads=4, depth=2):
        super().__init__()
        self.scene = ProbeDenoiser(1, in_channels, num_frames, dim, 16, ff, heads, 1, use_status=False,
                                   use_command=False)
        self.register_buffer("traj_mean", torch.zeros(24))
        self.register_buffer("traj_std", torch.ones(24))
        self.traj = nn.Sequential(nn.Linear(24, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(ContextTransformer(dim, ff, heads, 0.0, adaln=False) for _ in range(depth))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, len(METRICS)))

    def fit(self, trajectories):
        flat = trajectories.reshape(-1, 24)
        self.traj_mean.copy_(flat.mean(0))
        self.traj_std.copy_(flat.std(0).clamp(min=1e-3))

    def forward(self, batch, candidates):
        """batch with features (B, ...), candidates (B, K, 8, 3) -> sub-score logits (B, K, len(METRICS))."""
        ctx = self.scene.condition(batch)
        q = self.traj((candidates.flatten(2) - self.traj_mean) / self.traj_std)
        for blk in self.blocks:
            q = blk(ctx, q)
        return self.head(q)

    @torch.no_grad()
    def predict_epdms(self, batch, candidates, chunk=64):
        """candidates (K, B, 8, 3), the layout of model.sample -> predicted EPDMS (K, B)."""
        out = []
        for i in range(0, candidates.shape[1], chunk):
            b = {k: v[i:i + chunk] for k, v in batch.items()}
            out.append(epdms_torch(torch.sigmoid(self(b, candidates[:, i:i + chunk].transpose(0, 1)))).T)
        return torch.cat(out, 1)


@torch.no_grad()
def filtered_sample(model, verifier, batch, num_samples, steps=20, mode="prop", oversample=4, generator=None,
                    accept=None):
    """Draw oversample * num_samples futures and keep num_samples per scene -> (K, B, 8, 3).

    accept=None   the num_samples with the highest predicted EPDMS (top-K selection)
    accept=tau    rejection sampling: the first num_samples (in draw order, so a random
                  subset) whose predicted EPDMS is >= tau; if fewer pass, the best of the
                  rejected fill up. Samples the model drew in proportion to p(future |
                  scene) stay in that proportion among the accepted ones.
    Without a verifier: plain model.sample."""
    if verifier is None or oversample <= 1:
        return model.sample(batch, num_samples, steps, mode=mode, generator=generator)[0]
    fut = model.sample(batch, num_samples * oversample, steps, mode=mode, generator=generator)[0]
    pred = verifier.predict_epdms({"features": batch["features"]}, fut)                  # (N, B)
    if accept is None:
        key = pred
    else:
        order = torch.arange(len(pred), device=pred.device, dtype=pred.dtype)[:, None]
        key = torch.where(pred >= accept, 2 - order / len(pred), pred)                  # accepted first, in draw order
    top = key.topk(num_samples, dim=0).indices                                           # (K, B)
    return fut.gather(0, top[..., None, None].expand(-1, -1, *fut.shape[2:]))


def load_verifier(path):
    ck = torch.load(path, map_location="cuda")
    v = Verifier(ck["in_channels"], ck["num_frames"], ck["args"]["dim"], depth=ck["args"]["depth"]).cuda()
    v.load_state_dict(ck["model"])
    return v.eval()


def auc(score, positive):
    """ROC AUC of score for the binary labels positive (numpy, rank formula)."""
    pos, neg = positive.sum(), (~positive).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return (ranks[positive].sum() - pos * (pos + 1) / 2) / (pos * neg)


def test_on_samples(verifier, label, features):
    """Verifier on the generated navtest samples of score_samples.py -> dict."""
    d = np.load(SAMPLES / f"per_sample_{label}.npz")
    ds = SceneFeatureDataset("data/navsim_index/navtest.pkl", features=features, split="all", require_future=False,
                             feature_dir="data/features", history=4)
    pos = {t: r for r, t in zip(ds.rows, np.asarray(ds.index["token"])[ds.rows])}
    data = to_gpu(ds, np.array([pos[t] for t in d["token"]]))
    batch = {"features": data["features"].float()}
    cand = torch.from_numpy(d["samples"].astype(np.float32)).cuda()                     # (K, N, 8, 3)
    pred = verifier.predict_epdms(batch, cand).cpu().numpy()
    sub = d["sub"]
    true = epdms(sub.astype(np.float64))
    dac_fail = sub[..., METRICS.index("drivable_area_compliance")] == 0
    nc_fail = sub[..., METRICS.index("no_at_fault_collisions")] == 0
    keep = pred >= np.median(pred, 0, keepdims=True)                                     # better half per scene
    return {"samples": label, "auc_offroad": auc(-pred.ravel(), dac_fail.ravel()),
            "auc_collision": auc(-pred.ravel(), nc_fail.ravel()),
            "spearman": pd.Series(pred.ravel()).corr(pd.Series(true.ravel()), method="spearman"),
            "bad_all": (true < 0.5).mean(), "bad_kept": (true[keep] < 0.5).mean(),
            "offroad_all": dac_fail.mean(), "offroad_kept": dac_fail[keep].mean(),
            "epdms_all": true.mean(), "epdms_kept": true[keep].mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", default="wm11_ctx")
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--scenes-per-batch", type=int, default=64)
    ap.add_argument("--cands-per-scene", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gen", default=None, help="verifier_data.py npz of scored generator samples")
    ap.add_argument("--test", default="plain_92k,full16_92k,full64_92k,full64_10k")
    args = ap.parse_args()
    run = Path("logs_exp") / args.name
    if run.exists():
        sys.exit(f"{run} exists")
    run.mkdir(parents=True)
    (run / "README.md").write_text(f"# {args.name}\n\nverifier of PDM sub-scores (exp_navsim/diverse/verifier.py)\n\n"
                                   f"```\npython -m exp_navsim.diverse.verifier {' '.join(sys.argv[1:])}\n```\n")
    torch.manual_seed(0)
    features = args.features.split(",")

    st = np.load(PLANNER / f"vocab{args.vocab}_scores_navtrain_sub.npz")
    vocab = torch.from_numpy(np.load(PLANNER / f"vocab_{args.vocab}.npy")).float().cuda()        # (V, 8, 3)
    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=features, split="all",
                             feature_dir="data/features", history=4)
    rows = st["rows"]
    logs = np.asarray(ds.index["log"])[rows]
    uniq = np.unique(logs)
    held = set(np.random.default_rng(0).permutation(uniq)[:len(uniq) // 10])
    is_val = np.array([l in held for l in logs])
    data = to_gpu(ds, rows)
    scores = torch.from_numpy(st["scores"].astype(np.float32)).cuda()                           # (N, V, M)
    gen = None
    if args.gen:
        gd = np.load(args.gen)
        assert (gd["tokens"] == st["tokens"][:len(gd["tokens"])]).all()
        gen = (torch.from_numpy(gd["samples"].astype(np.float32)).cuda(),
               torch.from_numpy(gd["scores"].astype(np.float32)).cuda())
        is_val = is_val[:len(gd["tokens"])]
        print(f"[data] generator samples {tuple(gd['samples'].shape)}", flush=True)
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(f"[data] {len(tr)} train / {len(va)} held-out-log scenes, vocab {len(vocab)}", flush=True)

    T, C = data["features"].shape[1:3]
    model = Verifier(C, T, args.dim, depth=args.depth).cuda()
    model.fit(vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 300) * 0.5 * (
        1 + math.cos(math.pi * min(s / args.steps, 1))))
    tr_t = torch.from_numpy(tr).cuda()

    @torch.no_grad()
    def held_out():
        model.eval()
        preds = []
        for i in range(0, len(va), 32):
            idx = torch.from_numpy(va[i:i + 32]).cuda()
            preds.append(torch.sigmoid(model({"features": data["features"][idx].float()},
                                             vocab[None].expand(len(idx), -1, -1, -1))))
        model.train()
        p, y = torch.cat(preds).cpu().numpy(), scores[torch.from_numpy(va).cuda()].cpu().numpy()
        pe, ye = epdms(p.astype(np.float64)), epdms(y.astype(np.float64))
        dac = y[..., METRICS.index("drivable_area_compliance")] == 0
        return {"val/spearman": pd.Series(pe.ravel()).corr(pd.Series(ye.ravel()), method="spearman"),
                "val/auc_offroad": auc(-pe.ravel(), dac.ravel()),
                "val/top1_epdms": ye[np.arange(len(ye)), pe.argmax(1)].mean(), "val/oracle": ye.max(1).mean()}

    for step in range(1, args.steps + 1):
        idx = tr_t[torch.randint(len(tr_t), (args.scenes_per_batch,), device="cuda")]
        n_voc = args.cands_per_scene if gen is None else args.cands_per_scene // 2
        cand = torch.randint(len(vocab), (args.scenes_per_batch, n_voc), device="cuda")
        traj, target = vocab[cand], scores[idx[:, None], cand]
        if gen is not None:
            j = torch.randint(gen[0].shape[1], (args.scenes_per_batch, args.cands_per_scene - n_voc), device="cuda")
            traj = torch.cat([traj, gen[0][idx[:, None], j]], 1)
            target = torch.cat([target, gen[1][idx[:, None], j]], 1)
        logits = model({"features": data["features"][idx].float()}, traj)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"step {step}  loss {loss.item():.4f}  " + "  ".join(f"{k} {v:.3f}" for k, v in held_out().items()),
                  flush=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "in_channels": C, "num_frames": T},
               run / "verifier.pt")

    model.eval()
    rows_out = [{"samples": "vocab (held-out logs)", **held_out()}]
    for label in args.test.split(","):
        if (SAMPLES / f"per_sample_{label}.npz").exists():
            rows_out.append(test_on_samples(model, label, features))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows_out[-1].items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(OUT / f"{args.name}.csv", index=False)
    print(f"-> {OUT / f'{args.name}.csv'}")


if __name__ == "__main__":
    main()
