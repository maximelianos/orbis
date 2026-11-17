"""Train an IntentFlow (model.py) with all data on the GPU, evaluating the sampled distribution during training.

    python -m exp_navsim.diverse.train --name 17_full_h64_10k --anchors 16 --max-scenes 10000
    python -m exp_navsim.diverse.train --name 17_plain_h64_10k --anchors 0 --no-kin --max-scenes 10000
    python -m exp_navsim.diverse.train --name 17_ctrl_h64 --model ctrl --anchor-file anchors_ctrlprod_8x4.npy
    python -m exp_navsim.diverse.train --name 17_vocab4096 --model vocab --anchor-file vocab_fps_4096.npy

Every --eval-every steps, on 2000 scenes of held-out logs (val) and 2000 training scenes
(train_sub), K samples with --eval-steps Euler steps, in two sampling modes:
    prop    anchor k ~ p(k | scene) per sample: the model's distribution
    strat   samples split over the top-8 anchors: diverse by construction
and the heads: speed R^2 of the kinematic head, intent accuracy / NLL of the nearest
anchor, entropy of p(k | scene), number of distinct anchors among the prop samples, and
the share of samples that break a NAVSIM comfort bound (controls.comfort_flags).
Checkpoints every --ckpt-every steps; one row per evaluation goes to <run>/evals.csv.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.diverse.controls import comfort_flags, speed_of
from exp_navsim.diverse.model import CtrlFlow, CtrlVocab, IntentFlow, VocabDist
from exp_navsim.eval.metrics import sample_metrics

ANCHORS = Path("outputs/17_diversity/anchors")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--features", default="wm11_ctx")
    ap.add_argument("--model", default="intent", choices=["intent", "ctrl", "vocab", "ctrlvocab"],
                    help="intent: IntentFlow in pose space; ctrl: CtrlFlow; vocab: VocabDist; ctrlvocab: CtrlVocab")
    ap.add_argument("--anchors", type=int, default=16, help="number of intent anchors; 0 = none (--model intent)")
    ap.add_argument("--anchor-file", default=None, help="anchors / vocabulary under outputs/17_diversity/anchors")
    ap.add_argument("--ctrl-order", type=int, default=8, help="CtrlFlow: polynomial order of the controls, 8 = raw")
    ap.add_argument("--no-kin", action="store_true", help="no kinematic head, no constant-velocity prior")
    ap.add_argument("--command", action="store_true", help="driving command as an input token")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--tau", type=float, default=1.0, help="temperature (m) of the soft intent target")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--eval-samples", type=int, default=32)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def build_model(a, in_channels, num_frames, anchors=None):
    """Model of a run from its config dict (args); anchors: array or None."""
    kw = dict(use_command=a.get("command", False), hidden_dim=a["dim"], ff_dim=a["ff"], num_heads=a["heads"],
              depth=a["depth"], tau=a["tau"])
    kind = a.get("model", "intent")
    if kind == "ctrl":
        return CtrlFlow(in_channels, num_frames, None if anchors is None else torch.as_tensor(anchors).float(),
                        ctrl_order=a.get("ctrl_order", 8), **kw)
    if kind == "ctrlvocab":
        return CtrlVocab(in_channels, num_frames, torch.as_tensor(anchors).float(), **kw)
    if kind == "vocab":
        return VocabDist(in_channels, num_frames, anchors, **kw)
    return IntentFlow(in_channels, num_frames, anchors, use_kin=not a["no_kin"], **kw)


def to_gpu(ds, rows):
    x = np.concatenate([ds.features_of(rows[i:i + 8192]) for i in range(0, len(rows), 8192)])
    return {"features": torch.from_numpy(x).cuda(),                                       # fp16
            "future": torch.from_numpy(ds.future[rows]).float().cuda(),
            "status": torch.from_numpy(ds.status[rows]).float().cuda(),
            "command": torch.from_numpy(ds.command[rows]).float().cuda(),
            "row": torch.from_numpy(rows).cuda()}


def batch_of(data, idx):
    b = {k: v[idx] for k, v in data.items()}
    b["features"] = b["features"].float()
    return b


@torch.no_grad()
def evaluate(model, data, num_samples, steps, seed=0, chunk=500):
    """Sample metrics (both modes) and head metrics over all scenes of data -> dict of means."""
    model.eval()
    g = torch.Generator(device="cuda").manual_seed(seed)
    acc = {}
    n = len(data["row"])
    for i in range(0, n, chunk):
        b = batch_of(data, torch.arange(i, min(i + chunk, n), device="cuda"))
        for mode in ("prop", "strat") if model.num_anchors > 1 else ("prop",):
            fut, ks, kin, logits = model.sample(b, num_samples, steps, mode=mode, generator=g)
            for k, v in sample_metrics(fut, b["future"]).items():
                acc.setdefault(f"{mode}/{k}", []).append(v)
            if mode == "prop":
                fut_prop = fut
                distinct = (F.one_hot(ks, model.num_anchors).sum(0) > 0).sum(1)        # (B,)
                acc.setdefault("intent/distinct", []).append(distinct.float())
        if kin is not None:
            acc.setdefault("kin/sq_err", []).append((kin[:, 0] - b["status"][:, 0]) ** 2)
            acc.setdefault("kin/speed", []).append(b["status"][:, 0])
        for k, v in comfort_flags(fut_prop, speed_of(b["status"]).expand(len(fut_prop), -1)).items():
            acc.setdefault(f"comfort/{k}", []).append(v.float().mean(0))
        if model.num_anchors > 1:
            k_true = model.nearest_anchor(b["future"], b["status"])[1]
            logp = F.log_softmax(logits, 1)
            acc.setdefault("intent/acc", []).append((logits.argmax(1) == k_true).float())
            acc.setdefault("intent/nll", []).append(-logp.gather(1, k_true[:, None])[:, 0])
            acc.setdefault("intent/entropy", []).append(-(logp.exp() * logp).sum(1))
            top8 = logits.topk(min(8, model.num_anchors), 1).indices
            acc.setdefault("intent/top8_cover", []).append((top8 == k_true[:, None]).any(1).float())
    out = {k: torch.cat(v).mean().item() for k, v in acc.items() if not k.startswith("kin/")}
    for mode in ("prop", "strat"):
        if f"{mode}/spread" in out:
            out[f"{mode}/dispersion"] = out[f"{mode}/spread"] / max(out[f"{mode}/ade_mean"], 1e-3)
    if "kin/sq_err" in acc:
        speed = torch.cat(acc["kin/speed"])
        out["kin/speed_r2"] = 1 - torch.cat(acc["kin/sq_err"]).mean().item() / speed.var().item()
        out["kin/speed_rmse"] = torch.cat(acc["kin/sq_err"]).mean().sqrt().item()
    model.train()
    return out


def main():
    args = parse()
    run = Path("logs_exp") / args.name
    if run.exists():
        sys.exit(f"{run} exists")
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=1))
    (run / "README.md").write_text(f"# {args.name}\n\n{args.note}\n\n```\npython -m exp_navsim.diverse.train "
                                   f"{' '.join(sys.argv[1:])}\n```\n")
    (run / "checkpoints").mkdir()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tb = SummaryWriter(str(run / "tb"))

    common = dict(features=args.features.split(","), feature_dir="data/features", history=4)
    train_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="train", max_scenes=args.max_scenes,
                                   seed=0, **common)
    val_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="val", **common)
    rng = np.random.default_rng(0)
    train = to_gpu(train_ds, train_ds.rows)
    evals = {"val": to_gpu(val_ds, np.sort(rng.permutation(val_ds.rows)[:2000])),
             "train_sub": to_gpu(train_ds, np.sort(rng.permutation(train_ds.rows)[:2000]))}

    anchors = None
    if args.anchor_file or (args.model == "intent" and args.anchors):
        path = ANCHORS / (args.anchor_file or f"anchors_{'kin' if not args.no_kin else 'raw'}_{args.anchors}.npy")
        anchors = np.load(path)
        print(f"[anchors] {path}: {anchors.shape}")
    T, C = train["features"].shape[1:3]
    model = build_model(vars(args), C, T, anchors).cuda()
    print(f"[codec] {model.fit(train['future'], train['status'])}")
    n_params = sum(p.numel() for p in model.parameters())
    n_flow = sum(p.numel() for p in model.denoiser.parameters())
    print(f"[model] {n_params} parameters, {n_flow} in the denoiser; train {len(train['row'])} scenes", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / warm) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1)))))

    rows = []

    def do_eval(step):
        t1 = time.time()
        row = {"step": step}
        for name, data in evals.items():
            for k, v in evaluate(model, data, args.eval_samples, args.eval_steps).items():
                row[f"{name}/{k}"] = v
                tb.add_scalar(f"{name}/{k}", v, step)
        rows.append(row)
        pd.DataFrame(rows).to_csv(run / "evals.csv", index=False)
        keys = ("prop/energy", "prop/ade_min", "prop/dispersion", "strat/ade_min", "intent/entropy", "kin/speed_r2",
                "comfort/any")
        print(f"eval @ {step} ({time.time() - t1:.0f}s): " + "  ".join(
            f"{n}/{k} {row[f'{n}/{k}']:.3f}" for n in evals for k in keys if f"{n}/{k}" in row), flush=True)

    n = len(train["row"])
    t0 = time.time()
    for step in range(1, args.steps + 1):
        b = batch_of(train, torch.randint(n, (args.batch,), device="cuda"))
        loss, logs = model.loss(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            for k, v in logs.items():
                tb.add_scalar(f"train/{k}", v.item(), step)
            tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
        if step % 1000 == 0:
            print(f"step {step:6d}/{args.steps}  loss {loss.item():.4f}  "
                  + "  ".join(f"{k} {v.item():.3f}" for k, v in logs.items())
                  + f"  {1000 / (time.time() - t0):.0f} it/s", flush=True)
            t0 = time.time()
        if step % args.eval_every == 0 or step == 1000:
            do_eval(step)
            t0 = time.time()
        if step % args.ckpt_every == 0 or step in (1000, 5000):
            torch.save({"model": model.state_dict(), "args": vars(args), "in_channels": C, "num_frames": T},
                       run / "checkpoints" / f"step{step:07d}.pt")
    tb.close()


if __name__ == "__main__":
    main()
