"""Train the prediction-only planner (model.py) with all data on the GPU, scored by EPDMS on a navtest subset.

    python -m exp_navsim.planner.train --name 16_vocab256_wm11ctx --features wm11_ctx --vocab 256

Training scenes, validation logs and navtest scenes are gathered once into GPU
tensors, so a step is one indexed read and one forward / backward pass (no data
loader). Every --eval-every steps:
    val     held-out navtrain logs: L2 at 4 s and ADE of the top-1 output, best ADE of the top-8
    navtest EPDMS of the top-1 output on --subset (fast_pdm.FastPDM, exact official numbers),
            and of the best of the top-8 per scene (an oracle over the proposed directions)
At the end the same on --final-subset, and the top-8 for every navtest scene is
saved to <run>/navtest_top8.npz (eval/pdm.py can score it on the full split).

Runs go to logs_exp/<name> (tb/, config.json, README.md); one summary row per run is
appended to outputs/16_fast_planner/results.csv.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.planner.fast_pdm import FastPDM, load_subset

OUT = Path("outputs/16_fast_planner")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--features", default="wm11_ctx", help="comma-separated cached features; '' = none")
    ap.add_argument("--vocab", type=int, default=256, help="vocabulary size; 0 = plain regression")
    ap.add_argument("--hybrid", action="store_true", help="regression output first, vocabulary for the rest")
    ap.add_argument("--offset-all", action="store_true", help="offset loss on all anchors, weighted by the target")
    ap.add_argument("--no-status", action="store_true")
    ap.add_argument("--no-past", action="store_true")
    ap.add_argument("--no-command", action="store_true")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--decoder-depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=1.0, help="temperature (m) of the soft imitation target")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--score-targets", default=None, help="npz from score_vocab.py: PDM sub-scores of the vocabulary")
    ap.add_argument("--w-score", type=float, default=1.0)
    ap.add_argument("--target-frac", type=float, default=0.0,
                    help="fraction of every batch drawn from the scenes that have score targets")
    ap.add_argument("--rank-weights", default=None,
                    help="8 comma-separated weights of the log sub-scores at inference (fast_pdm.METRICS order)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--subset", default=str(OUT / "subset_100.json"))
    ap.add_argument("--final-subset", default=str(OUT / "subset_500.json"))
    ap.add_argument("--pdm-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse()
    run = Path("logs_exp") / args.name
    if run.exists():
        sys.exit(f"{run} exists")

    # The scoring pool forks, so it starts before CUDA is touched.
    sub_tokens, sub_prev = load_subset(args.subset)
    fin_tokens, fin_prev = load_subset(args.final_subset)
    all_tokens = sorted(set(sub_tokens) | set(sub_prev.values()) | set(fin_tokens) | set(fin_prev.values()))
    scorer = FastPDM("navtest", all_tokens, workers=args.pdm_workers)

    import torch
    from torch.utils.tensorboard import SummaryWriter

    from exp_navsim.data.scenes import SceneFeatureDataset
    from exp_navsim.planner.model import Planner, planner_loss

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=1))
    (run / "README.md").write_text(f"# {args.name}\n\n{args.note}\n\n```\npython -m exp_navsim.planner.train "
                                   f"{' '.join(sys.argv[1:])}\n```\n")
    tb = SummaryWriter(str(run / "tb"))
    dev = "cuda"
    feats = [f for f in args.features.split(",") if f]

    def gpu(ds, rows=None):
        rows = ds.rows if rows is None else rows
        x = np.concatenate([ds.features_of(rows[i:i + 8192]) for i in range(0, len(rows), 8192)])
        return {"features": torch.from_numpy(x).to(dev),                                     # fp16
                "future": torch.from_numpy(np.nan_to_num(ds.future[rows])).float().to(dev),
                "past": torch.from_numpy(ds.past[rows]).float().to(dev),
                "status": torch.from_numpy(ds.status[rows]).float().to(dev),
                "command": torch.from_numpy(ds.command[rows]).float().to(dev),
                "row": torch.from_numpy(rows).to(dev)}

    t0 = time.time()
    common = dict(features=feats, feature_dir="data/features", history=4)
    train_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="train", max_scenes=args.max_scenes,
                                   seed=args.seed, **common)
    val_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="val", **common)
    test_ds = SceneFeatureDataset("data/navsim_index/navtest.pkl", split="all", require_future=False, **common)
    train = gpu(train_ds)
    val_rows = np.sort(np.random.default_rng(0).permutation(val_ds.rows)[:4000])
    val = gpu(val_ds, val_rows)
    test = gpu(test_ds)
    test_tokens = np.asarray(test_ds.index["token"])[test_ds.rows]
    tok_pos = {t: i for i, t in enumerate(test_tokens)}
    print(f"[data] train {len(train_ds)}  val {len(val_rows)}  navtest {len(test_tokens)}  "
          f"features {tuple(train['features'].shape[1:])}  {time.time() - t0:.0f}s", flush=True)

    score_tgt = None
    if args.score_targets:
        st = np.load(args.score_targets)
        pos = torch.full((len(train_ds.index["token"]),), -1, dtype=torch.long)
        pos[torch.from_numpy(st["rows"])] = torch.arange(len(st["rows"]))
        score_tgt = (pos.to(dev), torch.from_numpy(st["scores"].astype(np.float16)).to(dev))
        print(f"[data] PDM sub-score targets for {len(st['rows'])} scenes, vocab {st['scores'].shape[1]}")

    vocab = np.load(OUT / f"vocab_{args.vocab}.npy") if args.vocab else None
    T, C = train["features"].shape[1:3] if feats else (1, 0)
    model = Planner(C, T, vocab=vocab, dim=args.dim, depth=args.depth, decoder_depth=args.decoder_depth,
                    dropout=args.dropout, use_image=bool(feats), use_status=not args.no_status,
                    use_past=not args.no_past, use_command=not args.no_command,
                    score_head=score_tgt is not None, hybrid=args.hybrid).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:.2f}M parameters", flush=True)
    weights = [float(w) for w in args.rank_weights.split(",")] if args.rank_weights else None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm = min(300, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / max(warm, 1)) * 0.5 * (
        1 + math.cos(math.pi * min(s / args.steps, 1))))

    def batch_of(data, idx):
        b = {k: v[idx] for k, v in data.items()}
        b["features"] = b["features"].float()
        return b

    @torch.no_grad()
    def predict(data, k=8):
        model.eval()
        n = len(data["row"])
        out = torch.cat([model.predict(batch_of(data, torch.arange(i, min(i + 512, n), device=dev)), k, weights)
                         for i in range(0, n, 512)], 1)
        model.train()
        return out                                                           # (k, N, 8, 3)

    eval_time = [0.0]

    def evaluate(step, tokens, prev, tag):
        t_eval = time.time()
        p = predict(val)
        gt = val["future"]
        d = (p[..., :2] - gt[None, :, :, :2]).norm(dim=-1)                   # (k, N, 8)
        m = {"val/l2_4s": d[0, :, -1].mean().item(), "val/ade": d[0].mean().item(),
             "val/ade_min8": d.mean(-1).min(0).values.mean().item()}
        pt = predict(test).cpu().numpy()
        need = list(dict.fromkeys(list(prev) + list(prev.values())))
        t1 = time.time()
        df = scorer.score({t: pt[:, tok_pos[t]] for t in need}, prev)
        m[f"{tag}/epdms"] = df[df.k == 0].score.mean()
        m[f"{tag}/epdms_best8"] = df.groupby("token").score.max().mean()
        for c in ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                  "time_to_collision_within_bound", "two_frame_extended_comfort"):
            m[f"{tag}/{c}"] = df[df.k == 0][c].mean()
        for k, v in m.items():
            tb.add_scalar(k, v, step)
        eval_time[0] += time.time() - t_eval
        print(f"eval @ {step} ({time.time() - t1:.0f}s pdm): " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()),
              flush=True)
        return m, pt

    t0 = t_start = time.time()
    n = len(train["row"])
    n_tgt = int(args.batch * args.target_frac) if score_tgt is not None else 0
    if n_tgt:
        with_tgt = torch.nonzero(score_tgt[0][train["row"]] >= 0)[:, 0]
        print(f"[data] {n_tgt} of {args.batch} per batch from the {len(with_tgt)} scenes with targets", flush=True)
    for step in range(1, args.steps + 1):
        idx = torch.randint(n, (args.batch,), device=dev)
        if n_tgt:
            idx[:n_tgt] = with_tgt[torch.randint(len(with_tgt), (n_tgt,), device=dev)]
        b = batch_of(train, idx)
        tgt = None
        if score_tgt is not None:
            p = score_tgt[0][b["row"]]
            mask = p >= 0
            tgt = (score_tgt[1][p.clamp(min=0)].float(), mask)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        out = {k: v.float() for k, v in out.items()}
        loss, logs = planner_loss(model, out, b, tau=args.tau, w_score=args.w_score, score_targets=tgt,
                                  offset_all=args.offset_all)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            for k, v in logs.items():
                tb.add_scalar(f"train/{k}", v.item(), step)
            tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
        if step % 500 == 0:
            print(f"step {step:6d}/{args.steps}  loss {loss.item():.4f}  "
                  + "  ".join(f"{k} {v.item():.3f}" for k, v in logs.items())
                  + f"  {500 / (time.time() - t0):.0f} it/s", flush=True)
            t0 = time.time()
        if step % args.eval_every == 0 and step != args.steps:
            evaluate(step, sub_tokens, sub_prev, "navtest100")
            t0 = time.time()

    train_min = (time.time() - t_start - eval_time[0]) / 60       # without the evaluations
    m, pt = evaluate(args.steps, sub_tokens, sub_prev, "navtest100")
    mf, _ = evaluate(args.steps, fin_tokens, fin_prev, "navtest500")
    m.update(mf)
    torch.save({"model": model.state_dict(), "args": vars(args)}, run / "model.pt")
    np.savez_compressed(run / "navtest_top8.npz", token=test_tokens, samples=pt.astype(np.float16))
    row = {"name": args.name, "features": args.features, "vocab": args.vocab, "steps": args.steps,
           "train_scenes": len(train_ds), "params_M": round(n_params / 1e6, 3),
           "train_min": round(train_min, 2), "note": args.note,
           **{k: round(float(v), 4) for k, v in m.items()}}
    path = OUT / "results.csv"
    df = pd.concat([pd.read_csv(path), pd.DataFrame([row])]) if path.exists() else pd.DataFrame([row])
    df.to_csv(path, index=False)
    print(f"-> {path}")
    scorer.close()
    tb.close()


if __name__ == "__main__":
    main()
