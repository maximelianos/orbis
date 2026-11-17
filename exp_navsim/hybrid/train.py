"""Train a FlowPlanner (model.py) with all data on the GPU.

    python -m exp_navsim.hybrid.train --name 19_regress --mode regress --steps 10000
    python -m exp_navsim.hybrid.train --name 19_flow --mode flow
    python -m exp_navsim.hybrid.train --name 19_resid --mode resid --init logs_exp/19_regress/model.pt

Options beyond the plain objective:
    --ema d          exponential moving average of the weights (decay d, 0 = off); the
                     checkpoint holds both the EMA weights (model) and the raw ones (raw)
    --irp s          reference perturbation of the residual flow (model.py), std s m/s
    --status-noise s Gaussian noise (std s m/s) on the ego velocity input, the targets stay
    --mmgt npz,...   multi-modal targets: score banks of planner/score_vocab.py and
                     verifier_data.py / hybrid.sample_scores (scored candidates of cached
                     training scenes). --mmgt-frac of every batch comes from these scenes,
                     and the flow learns, with probability --mmgt-p, a random candidate
                     whose EPDMS >= --mmgt-thr instead of the logged future. --mmgt-temp T > 0
                     draws it with probability ~ w exp(EPDMS / T) (the bank weight w balances
                     vocabulary and generator candidates) instead of uniformly; --mmgt-only
                     trains on the scored scenes alone (the whole batch comes from them)
    --rw npz,...     reward-weighted fine-tuning (start from --init): hybrid.sample_scores
                     files of this model's own samples with their PDM scores. Group-relative
                     advantage per scene, A = (r - mean) / (std + 0.05), negative advantages
                     truncated to 0 (DiffusionDriveV2), and --rw-frac * batch extra samples
                     drawn with probability proportional to A get the flow loss with weight
                     --rw-weight, next to the imitation loss on logged futures
Every --eval-every steps: 2000 held-out-log scenes, 16 samples, 10 Euler steps (energy,
minADE, spread, L2 of the sample mean and of the regression output) and the EPDMS of the
sample mean / regression output on the 100-scene navtest subset.
-> logs_exp/<name>/model.pt, evals.csv, tb/
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.planner.fast_pdm import FastPDM, epdms, load_subset

SUBSET = "outputs/16_fast_planner/subset_100.json"


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--mode", default="flow", choices=["regress", "flow", "resid"])
    ap.add_argument("--features", default="wm11_ctx")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--decoder-depth", type=int, default=2)
    ap.add_argument("--flow-depth", type=int, default=3)
    ap.add_argument("--pred", default="v", choices=["v", "x0", "eps"], help="prediction target of the flow head")
    ap.add_argument("--norm", default="std", choices=["std", "none", "raw", "global", "minmax", "pca"],
                    help="normalization of the flow's vector (model.py)")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="data std 1 / c against the unit noise")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--init", default=None, help="checkpoint to start from (matching weights are loaded)")
    ap.add_argument("--irp", type=float, default=0.0)
    ap.add_argument("--status-noise", type=float, default=0.0)
    ap.add_argument("--mmgt", default=None)
    ap.add_argument("--mmgt-frac", type=float, default=0.25)
    ap.add_argument("--mmgt-p", type=float, default=0.5)
    ap.add_argument("--mmgt-thr", type=float, default=0.9)
    ap.add_argument("--mmgt-temp", type=float, default=0.0)
    ap.add_argument("--mmgt-only", action="store_true")
    ap.add_argument("--rw", default=None)
    ap.add_argument("--rw-frac", type=float, default=0.5)
    ap.add_argument("--rw-weight", type=float, default=1.0)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-samples", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=10)
    ap.add_argument("--pdm-workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def to_gpu(ds, rows):
    import torch
    x = np.concatenate([ds.features_of(rows[i:i + 8192]) for i in range(0, len(rows), 8192)])
    return {"features": torch.from_numpy(x).cuda(),                                       # fp16
            "future": torch.from_numpy(np.nan_to_num(ds.future[rows])).float().cuda(),
            "past": torch.from_numpy(ds.past[rows]).float().cuda(),
            "status": torch.from_numpy(ds.status[rows]).float().cuda(),
            "command": torch.from_numpy(ds.command[rows]).float().cuda(),
            "row": torch.from_numpy(rows).cuda()}


def batch_of(data, idx):
    b = {k: v[idx] for k, v in data.items()}
    b["features"] = b["features"].float()
    return b


def build_model(a, in_channels, num_frames):
    from exp_navsim.hybrid.model import FlowPlanner
    return FlowPlanner(in_channels, num_frames, mode=a["mode"], flow_depth=a["flow_depth"], dim=a["dim"],
                       depth=a["depth"], decoder_depth=a["decoder_depth"], dropout=a["dropout"],
                       pred=a.get("pred", "v"), norm=a.get("norm", "std"),
                       noise_scale=a.get("noise_scale", 1.0))


def load_model(path, raw=False):
    import torch
    ck = torch.load(path, map_location="cuda")
    model = build_model(ck["args"], ck["in_channels"], ck["num_frames"]).cuda()
    model.load_state_dict(ck["raw" if raw and "raw" in ck else "model"])
    return model.eval(), ck["args"]


def advantages(scores):
    """(N, G, M) sub-scores of G samples per scene -> (N, G) truncated group-relative advantages."""
    r = epdms(scores.astype(np.float64))
    a = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 0.05)
    return np.clip(a, 0, None), r


def main():
    args = parse()
    run = Path("logs_exp") / args.name
    if run.exists():
        sys.exit(f"{run} exists")
    sub_tokens, sub_prev = load_subset(SUBSET)
    scorer = FastPDM("navtest", sorted(set(sub_tokens) | set(sub_prev.values())), workers=args.pdm_workers)

    import torch
    from torch.utils.tensorboard import SummaryWriter

    from exp_navsim.data.scenes import SceneFeatureDataset
    from exp_navsim.eval.metrics import sample_metrics

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=1))
    (run / "README.md").write_text(f"# {args.name}\n\n{args.note}\n\n```\npython -m exp_navsim.hybrid.train "
                                   f"{' '.join(sys.argv[1:])}\n```\n")
    tb = SummaryWriter(str(run / "tb"))

    common = dict(features=args.features.split(","), feature_dir="data/features", history=4)
    train_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="train", max_scenes=args.max_scenes,
                                   seed=args.seed, **common)
    val_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="val", **common)
    test_ds = SceneFeatureDataset("data/navsim_index/navtest.pkl", split="all", require_future=False, **common)
    train = to_gpu(train_ds, train_ds.rows)
    val = to_gpu(val_ds, np.sort(np.random.default_rng(0).permutation(val_ds.rows)[:2000]))
    test_tokens = np.asarray(test_ds.index["token"])[test_ds.rows]
    need = list(dict.fromkeys(list(sub_prev) + list(sub_prev.values())))
    tpos = {t: r for t, r in zip(test_tokens, test_ds.rows)}
    test = to_gpu(test_ds, np.array([tpos[t] for t in need]))
    n = len(train["row"])
    pos = torch.full((len(train_ds.index["token"]),), -1, dtype=torch.long, device="cuda")
    pos[train["row"]] = torch.arange(n, device="cuda")

    T, C = train["features"].shape[1:3]
    model = build_model(vars(args), C, T).cuda()
    if args.init:
        ck = torch.load(args.init, map_location="cuda")
        own = model.state_dict()
        sd = {k: v for k, v in ck["model"].items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(sd, strict=False)
        print(f"[init] {len(sd)} of {len(own)} tensors from {args.init}")
    if args.mode != "regress":
        if args.init and "mu" in ck["model"] and ck["args"]["mode"] == args.mode:
            print("[fit] normalization from the initial checkpoint")
        else:
            with torch.no_grad():
                model.eval()
                idx = torch.randperm(n, device="cuda")[:20000]
                ref = torch.cat([model.regress(*model.scene(batch_of(train, idx[i:i + 1000])))
                                 for i in range(0, len(idx), 1000)]) if args.mode == "resid" else None
                print(f"[fit] {model.fit(train['future'][idx], ref)}")
                model.train()
    ema = copy.deepcopy(model).eval() if args.ema > 0 else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:.2f}M parameters, train {n} scenes", flush=True)

    # ---------------------------------------------------- multi-modal targets -- #
    mm = None
    if args.mmgt:
        from exp_navsim.verified.train_verifier import build_bank
        rows, cand, score, weight = build_bank(args.mmgt.split(","))
        r = epdms(score.astype(np.float64))
        good = (weight > 0) & (r >= args.mmgt_thr)
        prob = good * (weight * np.exp((r - 1) / args.mmgt_temp) if args.mmgt_temp > 0 else 1.0)
        p = pos[torch.from_numpy(rows).cuda()]
        keep = (p >= 0) & torch.from_numpy(good.any(1)).cuda()
        mm = {"pos": p[keep], "cand": torch.from_numpy(cand).cuda()[keep],
              "good": torch.from_numpy(prob).cuda()[keep].float()}
        print(f"[mmgt] {int(keep.sum())} scenes, {good[keep.cpu().numpy()].sum(1).mean():.1f} good candidates "
              f"per scene (EPDMS >= {args.mmgt_thr})", flush=True)

    # ------------------------------------------------- reward-weighted data -- #
    rw = None
    if args.rw:
        P, S, W, R = [], [], [], []
        for f in args.rw.split(","):
            d = np.load(f)
            a, r = advantages(d["scores"])
            P.append(np.repeat(d["rows"], a.shape[1]))
            S.append(d["samples"].reshape(-1, 8, 3).astype(np.float32))
            W.append(a.reshape(-1))
            R.append(r)
        P, S, W = np.concatenate(P), np.concatenate(S), np.concatenate(W)
        p = pos[torch.from_numpy(P).cuda()]
        ok = (p >= 0) & (torch.from_numpy(W).cuda() > 0)
        rw = {"pos": p[ok], "samples": torch.from_numpy(S).cuda()[ok], "w": torch.from_numpy(W).cuda()[ok].float()}
        print(f"[rw] {int(ok.sum())} samples with positive advantage of {len(W)}; mean sample EPDMS "
              f"{np.concatenate(R).mean():.3f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm = min(500, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / max(warm, 1)) * (
        0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1)))))

    rows_out = []

    @torch.no_grad()
    def do_eval(step):
        net = ema if ema is not None else model
        net.eval()
        t1 = time.time()
        g = torch.Generator(device="cuda").manual_seed(0)
        m = {}
        acc = {}
        for i in range(0, len(val["row"]), 100):
            b = batch_of(val, torch.arange(i, min(i + 100, len(val["row"])), device="cuda"))
            fut, ref = net.sample(b, args.eval_samples, args.eval_steps, generator=g)
            for k, v in sample_metrics(fut, b["future"]).items():
                acc.setdefault(k, []).append(v)
            acc.setdefault("l2_4s_mean", []).append((fut.mean(0)[:, -1, :2] - b["future"][:, -1, :2]).norm(dim=-1))
            if ref is not None:
                acc.setdefault("l2_4s_ref", []).append((ref[:, -1, :2] - b["future"][:, -1, :2]).norm(dim=-1))
        m.update({f"val/{k}": torch.cat(v).mean().item() for k, v in acc.items()})
        plans = []
        for i in range(0, len(test["row"]), 100):
            b = batch_of(test, torch.arange(i, min(i + 100, len(test["row"])), device="cuda"))
            fut, ref = net.sample(b, args.eval_samples, args.eval_steps, generator=g)
            mean = fut.mean(0)
            plans.append(torch.stack([mean, ref if ref is not None else mean, fut[0]], 1).cpu())
        plans = torch.cat(plans).numpy()
        df = scorer.score({t: plans[j] for j, t in enumerate(need)}, sub_prev)
        for k, name in enumerate(("mean", "ref", "sample")):
            m[f"navtest100/epdms_{name}"] = df[df.k == k].score.mean()
        for k, v in m.items():
            tb.add_scalar(k, v, step)
        rows_out.append({"step": step, **m})
        pd.DataFrame(rows_out).to_csv(run / "evals.csv", index=False)
        print(f"eval @ {step} ({time.time() - t1:.0f}s): " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()),
              flush=True)
        if net is model:
            model.train()

    def save():
        torch.save({"model": (ema or model).state_dict(), "raw": model.state_dict(), "args": vars(args),
                    "in_channels": C, "num_frames": T}, run / "model.pt")

    t0 = t_start = time.time()
    eval_time = 0.0
    n_mm = (args.batch if args.mmgt_only else int(args.batch * args.mmgt_frac)) if mm is not None else 0
    n_rw = int(args.batch * args.rw_frac) if rw is not None else 0
    for step in range(1, args.steps + 1):
        idx = torch.randint(n, (args.batch,), device="cuda")
        target = None
        if n_mm:
            j = torch.randint(len(mm["pos"]), (n_mm,), device="cuda")
            idx[:n_mm] = mm["pos"][j]
            c = torch.multinomial(mm["good"][j], 1)[:, 0]
            alt = mm["cand"][j, c]
            target = train["future"][idx].clone()
            swap = torch.rand(n_mm, device="cuda") < args.mmgt_p
            target[:n_mm][swap] = alt[swap]
        b = batch_of(train, idx)
        if args.status_noise > 0:
            b["status"] = b["status"].clone()
            b["status"][:, :2] += torch.randn_like(b["status"][:, :2]) * args.status_noise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, logs, _ = model.loss(b, flow_target=target, irp=args.irp)
            if n_rw:
                j = torch.multinomial(rw["w"], n_rw, replacement=True)
                b_rw = batch_of(train, rw["pos"][j])
                _, _, fm_rw = model.loss(b_rw, flow_target=rw["samples"][j], irp=args.irp)
                logs["fm_rw"] = fm_rw.mean()
                loss = loss + args.rw_weight * logs["fm_rw"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            d = min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        if step % 100 == 0:
            for k, v in logs.items():
                tb.add_scalar(f"train/{k}", v.item(), step)
            tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
        if step % 1000 == 0:
            print(f"step {step:6d}/{args.steps}  loss {loss.item():.4f}  "
                  + "  ".join(f"{k} {v.item():.3f}" for k, v in logs.items())
                  + f"  {1000 / (time.time() - t0):.0f} it/s", flush=True)
            t0 = time.time()
        if step % args.eval_every == 0 or step == args.steps:
            te = time.time()
            do_eval(step)
            save()
            eval_time += time.time() - te
            t0 = time.time()
    (run / "train_min.txt").write_text(f"{(time.time() - t_start - eval_time) / 60:.2f}\n")
    scorer.close()
    tb.close()


if __name__ == "__main__":
    main()
