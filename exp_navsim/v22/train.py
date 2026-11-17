"""Train the v22 planner (model.py) against the navhard two-stage EPDMS proxy.

    python -m exp_navsim.v22.train --name 24_flow --features wm11_ctx --groups wp,act,path

As in planner/train.py the training scenes, the validation logs and all navhard / navtest
scenes are gathered once into GPU tensors, so a step is one indexed read and one forward /
backward pass. What is new is the evaluation: every --eval-every steps the plan of every
token of `--eval-groups` navhard groups is scored by v22/fast_epdms.FastEPDMS, which gives
the official NAVSIM v2.2 two-stage EPDMS of that group subset. At the end the same runs on
--final-groups (0 = all 225 groups, i.e. the whole split) and the top-8 plans of every
navhard and navtest token are saved for the official scripts.

SimScale co-training (--sim-split simscale_pdm0 --sim-targets <npz>): every batch takes a share
--sim-frac of scenes of the sim index (pseudo-expert future, rendered history). Sim rows get the
scorer BCE on their simulated sub-scores (EP per --sim-ep-mode: SimScale's EP is 0 for unsafe
anchors, ours never is), no BEV loss, and the imitation losses (flow, path, actions, anchor CE,
offset L1) only with weight --sim-imitation (0 = off). Their goal is the pseudo-expert end pose,
augmented and dropped like the real goal. Validation stays real navtrain val.
Several rounds: --sim-split simscale_pdm0,simscale_pdm1 --sim-targets a.npz,b.npz (one per split).

Dense vocabulary (GTRS-Dense style, --dense-vocab <GTRS index npy> --dense-sim-targets <dense rows
npz per sim split>, simscale/rewards.py align): sim rows also get the scorer BCE on --vocab-drop
random anchors of the dense set (a new subset every step). Real rows keep the 256 anchors with their
FastPDM targets; the imitation CE stays on the 256. --eval-dense ranks 256 + dense at evaluation.

Post-training (--init <run>/model.pt) and LoRA tuning of the ORBIS encoder (--lora-blocks K, v22/lora.py):
the wm11_ctx feature of every batch is computed online from the frames with LoRA adapters on the last K
world-model blocks; at the end navhard is cached as wm11_ctx_lora_<name> and config.json points to it.

Runs go to logs_exp/<name>; one summary row per run is appended to outputs/24_v22/results.csv.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.v22.fast_epdms import FastEPDMS, all_groups, group_tokens

OUT = Path("outputs/24_v22")
VOCAB = Path("outputs/16_fast_planner")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--features", default="wm11_ctx")
    ap.add_argument("--feature-dir", default="data/features")
    ap.add_argument("--cams", default="F0",
                    help="cameras, e.g. F0,L0,R0,B0; side cameras read <feature>_<cam> (cache_features --camera)")
    ap.add_argument("--calib", default="data/features/camera_calib.json", help="v22/cameras.py output")
    ap.add_argument("--cam-drop", type=float, default=0.0,
                    help="probability of dropping a side camera of a training scene (missing-camera token)")
    ap.add_argument("--bev-queries", type=int, default=0,
                    help="BEV queries along x over the BEV target extent (0 = off); they join the scene tokens")
    ap.add_argument("--groups", default="wp", help="planning representations: wp[,act][,path]")
    ap.add_argument("--no-flow", action="store_true", help="plain regression with the same network")
    ap.add_argument("--no-mask-path", action="store_true")
    ap.add_argument("--registers", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=0, help="anchor vocabulary size; 0 = no metric head")
    ap.add_argument("--eval-vocab", type=int, default=0,
                    help="rank a denser vocabulary at inference than the one trained on")
    ap.add_argument("--score-targets", default=None, help="npz of PDM sub-scores of the vocabulary")
    ap.add_argument("--w-score", type=float, default=1.0)
    ap.add_argument("--refined-targets", default=None,
                    help="npz of refined trajectories + PDM sub-scores (score_refined.py), comma-separated")
    ap.add_argument("--w-refined", type=float, default=1.0, help="weight of the refined-trajectory BCE")
    ap.add_argument("--w-rank", type=float, default=0.0,
                    help="weight of the listwise scorer loss (softmax CE towards softmax(true EPDMS / tau))")
    ap.add_argument("--rank-tau", type=float, default=0.1, help="temperature of --w-rank")
    ap.add_argument("--anchor-depth", type=int, default=2, help="blocks of the anchor / scorer decoder")
    ap.add_argument("--target-frac", type=float, default=0.0,
                    help="share of every batch drawn from the scenes that have score targets")
    ap.add_argument("--plan", default="flow", choices=("flow", "vocab", "rerank"))
    ap.add_argument("--rank", default="imitation", choices=("imitation", "epdms", "both", "mix", "top"),
                    help="how the vocab / rerank modes score a candidate")
    ap.add_argument("--rank-beta", type=float, default=0.03, help="weight of log p_imitation in --rank mix")
    ap.add_argument("--temperature", type=float, default=0.0, help="flow noise at inference")
    ap.add_argument("--euler-steps", type=int, default=8)
    ap.add_argument("--div-temperature", type=float, default=1.0,
                    help="temperature of the final diversity measurement")
    ap.add_argument("--div-samples", type=int, default=16)
    ap.add_argument("--no-status", action="store_true")
    ap.add_argument("--no-past", action="store_true")
    ap.add_argument("--no-command", action="store_true")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--flow-depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--w-act", type=float, default=0.2)
    ap.add_argument("--w-path", type=float, default=0.2)
    ap.add_argument("--w-roll", type=float, default=0.2)
    ap.add_argument("--bev-targets", default=None,
                    help="npz from v22/bev_targets.py; adds the auxiliary BEV occupancy head")
    ap.add_argument("--w-bev", type=float, default=1.0)
    ap.add_argument("--goal", default="none", choices=("none", "end"),
                    help="end: condition on the logged pose at the horizon end (py123d_garage Goal)")
    ap.add_argument("--goal-drop", type=float, default=0.0,
                    help="probability of replacing the goal by the learned null goal in training")
    ap.add_argument("--goal-aug", type=float, default=0.0,
                    help="probability of replacing the training goal by a perturbed one (augment_goal)")
    ap.add_argument("--goal-aug-time", default="2,6",
                    help="range in s of the logged pose the augmented goal is taken from")
    ap.add_argument("--goal-aug-along", type=float, default=3.0, help="along-track goal noise std, m")
    ap.add_argument("--goal-aug-lat", type=float, default=0.5, help="lateral goal noise std, m")
    ap.add_argument("--goal-file", default="outputs/25_terra/navhard_goals.npz",
                    help="navhard goals of both stages (v22/goals.py)")
    ap.add_argument("--select-targets", action="store_true",
                    help="imitate the best-scoring anchor (true PDM sub-scores) instead of the log")
    ap.add_argument("--select-frac", type=float, default=1.0,
                    help="share of the score-target scenes whose target may be replaced")
    ap.add_argument("--select-goal-weight", type=float, default=0.0,
                    help="penalty per metre of anchor end point distance to the logged end point")
    ap.add_argument("--select-margin", type=float, default=0.0,
                    help="an anchor replaces the log only if it beats the log estimate by this")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights; 0 = off")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--sim-split", default=None,
                    help="SimScale index (data/navsim_index/<split>.pkl), e.g. simscale_pdm0; comma-separated for several rounds")
    ap.add_argument("--sim-targets", default=None,
                    help="npz of anchor sub-scores of the sim index rows; comma-separated, one per --sim-split entry")
    ap.add_argument("--sim-frac", type=float, default=0.5, help="share of every batch from the sim split")
    ap.add_argument("--sim-imitation", type=float, default=0.0,
                    help="weight of the imitation losses on sim rows (pseudo-expert future); 0 = scorer only")
    ap.add_argument("--sim-ep-mode", default="mask_unsafe", choices=("mask_unsafe", "drop", "raw"),
                    help="EP BCE of sim rows: only where NC, DAC, DDC, TLC are 1 | never | always")
    ap.add_argument("--dense-vocab", default=None,
                    help="GTRS index npy of the dense scorer vocabulary (data/simscale/rewards/dense2048_gtrs_index.npy)")
    ap.add_argument("--dense-sim-targets", default=None,
                    help="dense rows npz per --sim-split entry (rewards.py align), comma-separated")
    ap.add_argument("--vocab-drop", type=int, default=512,
                    help="dense anchors scored per step (random subset, vocabulary dropout); 0 = all")
    ap.add_argument("--w-dense", type=float, default=1.0, help="weight of the dense-anchor BCE of sim rows")
    ap.add_argument("--dense-replace", action="store_true",
                    help="sim rows get only the dense BCE, not the BCE on the 256 anchors")
    ap.add_argument("--eval-dense", action="store_true",
                    help="rank the 256 anchors + the dense set at evaluation (needs --dense-vocab)")
    ap.add_argument("--dense-offset", default="raw", choices=("raw", "refine"),
                    help="--eval-dense: plan of a chosen dense anchor is the raw trajectory or the offset-refined one")
    ap.add_argument("--eval-every", type=int, default=3000)
    ap.add_argument("--eval-groups", type=int, default=32)
    ap.add_argument("--final-groups", type=int, default=112,
                    help="0 = all 225 groups; -1 = no final EPDMS (e.g. when sweep_rank follows)")
    ap.add_argument("--groups-file", default=None,
                    help="json of groups (v22/proxy.py select) used for the final evaluation")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--save-navtest", action="store_true", help="also save navtest predictions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    ap.add_argument("--no-preload", action="store_true",
                    help="memory-map the feature arrays and read only the needed rows (smoke tests)")
    ap.add_argument("--val-scenes", type=int, default=4000)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--results", default=str(OUT / "results.csv"), help="csv the summary row is appended to")
    ap.add_argument("--init", default=None, help="model.pt of a run to start the planner from (post-training)")
    ap.add_argument("--lora-blocks", type=int, default=0,
                    help="LoRA post-training of the ORBIS encoder (v22/lora.py): adapters on the last K blocks "
                         "up to 11, features computed online from the frames; 0 = off (cached features)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=None, help="default: = rank")
    ap.add_argument("--lora-mlp", action="store_true", help="adapters also on the MLP fc1 / fc2")
    ap.add_argument("--lora-proj", action="store_true", help="the PCA projection (mean, basis, scale) is trainable")
    ap.add_argument("--lora-lr", type=float, default=1e-4, help="lr of the adapters (and projection)")
    ap.add_argument("--lora-workers", type=int, default=8, help="JPEG decoding workers")
    return ap.parse_args()


def pick_groups(n, seed=0):
    """A nested random subset of the 225 navhard groups (n = 0: all of them)."""
    g = all_groups()
    order = np.random.default_rng(seed).permutation(len(g))
    return [g[i] for i in order[:n]] if n else g


def navhard_goals(path, tokens):
    """(N, 3) goal of every navhard token (v22/goals.py: stage 1 = logged future end, stage 2 =
    the log pose 4 s after the same-timestamp log frame, in the synthetic ego frame)."""
    import torch
    g = np.load(path)
    lookup = dict(zip(g["token"].tolist(), g["goal"]))
    nan = np.full(3, np.nan, np.float32)
    goal = np.stack([lookup.get(t, nan) for t in tokens]).astype(np.float32)
    print(f"[data] navhard goals for {np.isfinite(goal).all(1).sum()} / {len(goal)} tokens", flush=True)
    return torch.from_numpy(goal)


def long_path(index, extra=4):
    """(N, 8 + extra, 3) logged poses at 0.5 s steps up to 4 + 0.5 extra s, current ego frame.

    A scene only stores 4 s of future, but navtrain windows slide by one frame, so the scene
    m frames later (found by its history image ids) has its origin at future[m - 1] and its
    last future pose 0.5 m s after ours. Chains that break (end of a log or of the filter) are
    extrapolated with the last displacement and heading.
    """
    fut = np.asarray(index["future"], np.float64)
    ids = np.asarray(index["frame_ids"])
    after = {int(v): i for i, v in enumerate(ids[:, -2])}
    nxt = np.array([after.get(int(v), -1) for v in ids[:, -1]])           # scene one frame later
    out = np.concatenate([fut, np.zeros((len(fut), extra, 3))], 1)
    j = np.arange(len(fut))
    for m in range(1, extra + 1):
        j = np.where(j >= 0, nxt[np.maximum(j, 0)], -1)
        o, q = fut[:, m - 1], fut[np.maximum(j, 0), -1]
        c, s = np.cos(o[:, 2]), np.sin(o[:, 2])
        logged = np.stack([o[:, 0] + c * q[:, 0] - s * q[:, 1], o[:, 1] + s * q[:, 0] + c * q[:, 1],
                           o[:, 2] + q[:, 2]], 1)
        k = 7 + m
        guess = out[:, k - 1].copy()
        guess[:, :2] += out[:, k - 1, :2] - out[:, k - 2, :2]
        out[:, k] = np.where((j >= 0)[:, None], logged, guess)
        print(f"[goal-aug] logged pose at {4 + 0.5 * m:.1f} s for {(j >= 0).mean():.1%} of the scenes",
              flush=True)
    return out.astype(np.float32)


def augment_goal(b, args):
    """With prob --goal-aug the goal becomes the logged pose at a random time in --goal-aug-time
    (interpolated on long_path) plus along-track and lateral noise. The imitation targets stay
    the logged (or selected) 4 s plan, so a far or shifted goal is only a route hint."""
    import torch
    path = b["goal_path"]
    n, dev = len(path), path.device
    full = torch.cat([torch.zeros_like(path[:, :1]), path], 1)                # t = 0, 0.5, ... s
    lo, hi = (float(v) for v in args.goal_aug_time.split(","))
    k = (lo + (hi - lo) * torch.rand(n, device=dev)) / 0.5
    i0 = k.floor().long().clamp(max=full.shape[1] - 2)
    w = (k - i0)[:, None]
    ar = torch.arange(n, device=dev)
    g = full[ar, i0] * (1 - w) + full[ar, i0 + 1] * w
    along = args.goal_aug_along * torch.randn(n, device=dev)
    lat = args.goal_aug_lat * torch.randn(n, device=dev)
    c, s = torch.cos(g[:, 2]), torch.sin(g[:, 2])
    g = torch.stack([g[:, 0] + c * along - s * lat, g[:, 1] + s * along + c * lat, g[:, 2]], 1)
    aug = torch.rand(n, device=dev) < args.goal_aug
    return torch.where(aug[:, None], g, b["goal"])


def select_targets(train, score_tgt, vocab, args):
    """Teacher-selected imitation targets (TerraTransfer++ style, privileged PDM teacher).

    For the score-target scenes (a random --select-frac of them) every anchor gets its true
    EPDMS from the simulated sub-scores, NC * DAC * DDC * TLC * weighted mean of EP, TTC, LK
    and HC (the formula of GaragePlanner.rank), minus --select-goal-weight * the distance of
    its end point to the logged end point. The logged trajectory is a candidate too: it has
    no simulated scores, so it counts as passing the four multiplicative terms and takes the
    weighted terms of its nearest anchor. The best candidate becomes the target of every
    imitation loss (waypoints, actions, path, flow, the anchor CE and the offset L1); when
    it is the log, nothing changes. Returns the new (N, 8, 3) futures.
    """
    import torch
    from exp_navsim.v22.model import HORIZON, METRIC_WEIGHTS, scaled
    dev = train["future"].device
    pos = score_tgt[0][train["row"]]
    has = torch.nonzero(pos >= 0)[:, 0]
    gen = torch.Generator().manual_seed(args.seed)
    has = has[torch.randperm(len(has), generator=gen)[:round(args.select_frac * len(has))].to(dev)]
    s = score_tgt[1][pos[has]].float()                                       # (S, V, 8)
    w = torch.tensor(METRIC_WEIGHTS, device=dev)
    true = s[..., :4].prod(-1) * (s[..., 4:] * w).sum(-1) / w.sum()          # (S, V)
    fut = train["future"][has]
    V = torch.as_tensor(vocab, dtype=torch.float32, device=dev)
    near = torch.cat([(scaled(V)[None] - scaled(f)[:, None]).unflatten(-1, (HORIZON, 3)).norm(dim=-1)
                      .mean(-1).argmin(-1) for f in fut.split(2048)])
    sn = s[torch.arange(len(has), device=dev), near]
    log_score = (sn[:, 4:] * w).sum(-1) / w.sum()
    eff = true
    dist = (V[None, :, -1, :2] - fut[:, None, -1, :2]).norm(dim=-1)        # (S, V) end point gap
    if args.select_goal_weight:
        eff = eff - args.select_goal_weight * dist
    best = eff.argmax(-1)
    take = eff.max(-1).values > log_score + args.select_margin
    out = train["future"].clone()
    out[has[take]] = V[best[take]]
    ar = torch.arange(len(has), device=dev)
    ade = (V[best] - fut)[..., :2].norm(dim=-1).mean(-1)
    print(f"[select] {len(has)} scenes considered, {take.float().mean().item():.1%} take an anchor; "
          f"true EPDMS of the target {torch.where(take, true[ar, best], log_score).mean().item():.3f} "
          f"(log estimate {log_score.mean().item():.3f}, nearest anchor {true[ar, near].mean().item():.3f}); "
          f"replaced: ADE to log {ade[take].mean().item():.2f} m, end gap {dist[ar, best][take].mean().item():.2f} m",
          flush=True)
    return out


def sim_score_weight(s, mode):
    """Element weights of sim sub-scores s (..., 8): the EP column per --sim-ep-mode."""
    import torch
    w = torch.ones_like(s)
    if mode == "mask_unsafe":
        w[..., 4] = (s[..., :4] >= 0.999).all(-1).float()                    # EP only on safe anchors
    elif mode == "drop":
        w[..., 4] = 0.0
    return w


def dense_targets(sim_rows, dense_tgt, dense_traj, n_real, args):
    """Dense-anchor BCE targets of a batch: a random --vocab-drop subset of the dense anchors,
    sub-scores of the sim rows (real rows masked out) -> (traj (K, 8, 3), sub-scores (B, K, 8),
    mask (B,), element weight (B, K, 8))."""
    import torch
    dev = sim_rows.device
    sub = torch.randperm(len(dense_traj), device=dev)
    if args.vocab_drop:
        sub = sub[:args.vocab_drop]
    p = dense_tgt[0][sim_rows]
    s = dense_tgt[1][p.clamp(min=0)[:, None], sub[None]].float()            # (n_sim, K, 8)
    w = sim_score_weight(s, args.sim_ep_mode)
    pad = lambda x: torch.cat([torch.zeros(n_real, *x.shape[1:], dtype=x.dtype, device=dev), x])
    return dense_traj[sub], pad(s), pad(p >= 0), pad(w)


def with_sim_rows(sim_rows, tgt, ref, bev, sim_tgt, n_real, n_sim, n_anchors, args):
    """Append the targets of the n_sim sim rows (after the n_real real ones) to the real targets.

    -> score targets (sub-scores, mask, element weight): the sim sub-scores, with the EP column
    weighted per --sim-ep-mode; refined and BEV targets masked out for sim rows; row weights of
    the imitation losses, 1 for real rows and --sim-imitation for sim rows."""
    import torch
    dev = sim_rows.device
    p = sim_tgt[0][sim_rows]
    s = sim_tgt[1][p.clamp(min=0)].float()                                   # (n_sim, V, 8)
    w = sim_score_weight(s, args.sim_ep_mode)
    if tgt is None:                                                          # no real score targets
        tgt = (torch.zeros(n_real, n_anchors, s.shape[-1], device=dev), torch.zeros(n_real, dtype=torch.bool, device=dev))
    tgt = (torch.cat([tgt[0], s]), torch.cat([tgt[1], p >= 0]), torch.cat([torch.ones_like(tgt[0]), w]))
    pad = lambda x: torch.cat([x, torch.zeros(n_sim, *x.shape[1:], dtype=x.dtype, device=dev)])
    if ref is not None:
        ref = tuple(pad(x) for x in ref)
    if bev is not None:
        bev = tuple(pad(x) for x in bev)
    row_weight = torch.cat([torch.ones(n_real, device=dev), torch.full((n_sim,), args.sim_imitation, device=dev)])
    return tgt, ref, bev, row_weight


def main():
    args = parse()
    run = Path("logs_exp") / args.name
    if run.exists():
        sys.exit(f"{run} exists")

    # the scoring pools fork, so they start before CUDA is touched
    # nested prefixes of one permutation, so the larger set covers the smaller one
    eval_groups = pick_groups(args.eval_groups)
    if args.groups_file:
        from exp_navsim.v22.fast_epdms import load_groups
        final_groups = load_groups(args.groups_file)
        eval_groups = final_groups[:args.eval_groups] if args.eval_groups else final_groups
        pool = final_groups
    else:
        final_groups = pick_groups(args.final_groups) if args.final_groups >= 0 else []
        n_pool = 0 if 0 in (args.eval_groups, args.final_groups) else max(args.eval_groups,
                                                                         args.final_groups)
        pool = pick_groups(n_pool)
    scorer = FastEPDMS(pool, workers=args.workers)

    import torch
    from torch.utils.tensorboard import SummaryWriter

    from exp_navsim.data.scenes import SceneFeatureDataset
    from exp_navsim.v22.model import GaragePlanner

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=1))
    (run / "README.md").write_text(f"# {args.name}\n\n{args.note}\n\n```\npython -m exp_navsim.v22.train "
                                   f"{' '.join(sys.argv[1:])}\n```\n")
    tb = SummaryWriter(str(run / "tb"))
    dev = args.device
    base_feats = [f for f in args.features.split(",") if f]
    cams = args.cams.split(",")
    assert cams[0] == "F0", "the front camera comes first"
    assert len(cams) == 1 or not args.save_navtest, "navtest has only the front camera"
    # camera-major order: features_of concatenates over C, the model splits it back per camera
    feats = [f if cam == "F0" else f"{f}_{cam}" for cam in cams for f in base_feats]

    def gpu(ds, rows=None):
        rows = ds.rows if rows is None else rows
        x = np.concatenate([ds.features_of(rows[i:i + 8192]) for i in range(0, len(rows), 8192)])
        return {"features": torch.from_numpy(x).to(dev),
                "future": torch.from_numpy(np.nan_to_num(ds.future[rows])).float().to(dev),
                # logged end pose; NaN (synthetic scenes) is replaced from --goal-file below
                "goal": torch.from_numpy(ds.future[rows][:, -1]).float().to(dev),
                "past": torch.from_numpy(ds.past[rows]).float().to(dev),
                "status": torch.from_numpy(ds.status[rows]).float().to(dev),
                "command": torch.from_numpy(ds.command[rows]).float().to(dev),
                "row": torch.from_numpy(rows).to(dev)}

    t0 = time.time()
    common = dict(features=feats, feature_dir=args.feature_dir, history=4, preload=not args.no_preload)
    train_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="train",
                                   max_scenes=args.max_scenes, seed=args.seed, **common)
    val_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="val", **common)
    hard_ds = SceneFeatureDataset("data/navsim_index/navhard.pkl", split="all", require_future=False, **common)
    train = gpu(train_ds)
    if args.goal_aug:
        assert args.goal != "none", "--goal-aug needs --goal end"
        train["goal_path"] = torch.from_numpy(long_path(train_ds.index)[train_ds.rows]).to(dev)
    val_rows = np.sort(np.random.default_rng(0).permutation(val_ds.rows)[:args.val_scenes])
    val = gpu(val_ds, val_rows)
    hard = gpu(hard_ds)
    sim = None
    if args.sim_split:
        # training logs only: sim scenes of navtrain validation logs stay out (log_in_split on the real log).
        # Several splits (rounds) are concatenated; "row" is the row in the concatenation of their
        # indices (split k starts at sim_offset[k]). Their features are memory-mapped and moved
        # to the GPU in chunks, so the host never holds them all.
        sim_splits = args.sim_split.split(",")
        sim_common = common if len(sim_splits) == 1 else dict(common, preload=False)
        parts, sim_offset, sim_indices = [], [0], []
        for split in sim_splits:
            sim_ds = SceneFeatureDataset(f"data/navsim_index/{split}.pkl", split="train", **sim_common)
            sim_indices.append(sim_ds.index)
            g = gpu(sim_ds)
            g["row"] = g["row"] + sim_offset[-1]
            if args.goal_aug:
                g["goal_path"] = torch.from_numpy(long_path(sim_ds.index)[sim_ds.rows]).to(dev)
            parts.append(g)
            sim_offset.append(sim_offset[-1] + len(sim_ds.index["token"]))
        sim = parts[0] if len(parts) == 1 else {k: torch.cat([g[k] for g in parts]) for k in parts[0]}
        del parts, g
        if len(sim_splits) > 1:
            print(f"[data] sim splits {sim_splits}: {len(sim['row'])} training scenes, features "
                  f"{sim['features'].numel() * sim['features'].element_size() / 1e9:.1f} GB on {dev}", flush=True)
    hard_tokens = np.asarray(hard_ds.index["token"])[hard_ds.rows]
    hard_pos = {t: i for i, t in enumerate(hard_tokens)}
    if args.goal != "none":
        hard["goal"] = navhard_goals(args.goal_file, hard_tokens).to(dev)
    print(f"[data] train {len(train_ds)}  val {len(val_rows)}  navhard {len(hard_tokens)}  "
          f"features {tuple(train['features'].shape[1:])}  {time.time() - t0:.0f}s", flush=True)

    score_tgt = None
    if args.score_targets:
        # comma-separated target files (e.g. navtrain_sub + navtrain_sub2) are pooled
        parts = [np.load(f) for f in args.score_targets.split(",")]
        st = {"rows": np.concatenate([p["rows"] for p in parts]),
              "scores": np.concatenate([p["scores"] for p in parts])}
        _, first = np.unique(st["rows"], return_index=True)
        st = {k: v[np.sort(first)] for k, v in st.items()}
        pos = torch.full((len(train_ds.index["token"]),), -1, dtype=torch.long)
        pos[torch.from_numpy(st["rows"])] = torch.arange(len(st["rows"]))
        score_tgt = (pos.to(dev), torch.from_numpy(st["scores"].astype(np.float16)).to(dev))
        print(f"[data] PDM sub-scores for {len(st['rows'])} scenes, vocab {st['scores'].shape[1]}", flush=True)

    sim_tgt = None
    if sim is not None:
        assert args.sim_targets and args.vocab, "--sim-split needs --sim-targets and --vocab"
        files = args.sim_targets.split(",")
        assert len(files) == len(sim_splits), "one --sim-targets file per --sim-split entry"
        pos = torch.full((sim_offset[-1],), -1, dtype=torch.long)
        scores = []
        for k, f in enumerate(files):
            st = np.load(f)
            assert st["scores"].shape[1] == args.vocab, f"sim targets are for {st['scores'].shape[1]} anchors"
            n_before = sum(len(x) for x in scores)
            pos[sim_offset[k] + torch.from_numpy(st["rows"])] = n_before + torch.arange(len(st["rows"]))
            scores.append(st["scores"].astype(np.float16))
        sim_tgt = (pos.to(dev), torch.from_numpy(scores[0] if len(scores) == 1 else np.concatenate(scores)).to(dev))
        del scores
        sim_with_tgt = torch.nonzero(sim_tgt[0][sim["row"]] >= 0)[:, 0]
        safe = (sim_tgt[1][..., :4] >= 0.999).all(-1).float().mean().item()
        print(f"[data] sim {args.sim_split}: {len(sim['row'])} training scenes, {len(sim_with_tgt)} with sub-scores, "
              f"anchors with NC, DAC, DDC, TLC = 1: {safe:.1%}; {args.sim_frac:.0%} of every batch, "
              f"imitation weight {args.sim_imitation}, EP {args.sim_ep_mode}", flush=True)

    dense_tgt = dense_traj = None
    if args.dense_vocab:
        from exp_navsim.simscale.rewards import gtrs_vocab
        dense_idx = np.load(args.dense_vocab)
        dense_traj = torch.from_numpy(gtrs_vocab()[dense_idx][:, 4::5].astype(np.float32)).to(dev)   # (K, 8, 3)
    if args.dense_sim_targets:
        assert sim is not None and args.dense_vocab, "--dense-sim-targets needs --sim-split and --dense-vocab"
        files = args.dense_sim_targets.split(",")
        assert len(files) == len(sim_splits), "one --dense-sim-targets file per --sim-split entry"
        # only the training rows of the sim splits; scores stay float16 on the device
        train_rows = sim["row"].cpu().numpy()
        metas = []
        for k, f in enumerate(files):
            d = np.load(f)
            assert np.array_equal(np.load(str(d["gtrs_index"])), dense_idx), f"{f} is for another dense set"
            keep = np.isin(d["rows"] + sim_offset[k], train_rows)
            metas.append((sim_offset[k] + d["rows"][keep], d["src"][keep], str(d["scores_file"])))
        n_dense = sum(len(m[0]) for m in metas)
        pos = torch.full((sim_offset[-1],), -1, dtype=torch.long)
        scores = torch.empty((n_dense, len(dense_idx), 8), dtype=torch.float16, device=dev)
        n = 0
        for rows, src, path in metas:
            D = np.load(path, mmap_mode="r")
            assert D.shape[1] == len(dense_idx)
            pos[torch.from_numpy(rows)] = n + torch.arange(len(rows))
            for i in range(0, len(src), 4096):
                scores[n + i:n + i + len(src[i:i + 4096])] = torch.from_numpy(np.ascontiguousarray(D[src[i:i + 4096]])).to(dev)
            n += len(rows)
        dense_tgt = (pos.to(dev), scores)
        print(f"[data] dense sub-scores for {n_dense} sim training scenes x {len(dense_idx)} anchors "
              f"({scores.numel() * 2 / 1e9:.1f} GB on {dev}), {args.vocab_drop or len(dense_idx)} per step, "
              f"w {args.w_dense}{', no 256-anchor BCE on sim rows' if args.dense_replace else ''}", flush=True)

    ref_tgt = None
    if args.refined_targets:
        parts = [np.load(f) for f in args.refined_targets.split(",")]
        rows = np.concatenate([p["rows"] for p in parts])
        _, first = np.unique(rows, return_index=True)
        first = np.sort(first)
        pos = torch.full((len(train_ds.index["token"]),), -1, dtype=torch.long)
        pos[torch.from_numpy(rows[first])] = torch.arange(len(first))
        ref_tgt = (pos.to(dev),
                   torch.from_numpy(np.concatenate([p["traj"] for p in parts])[first].astype(np.float16)).to(dev),
                   torch.from_numpy(np.concatenate([p["scores"] for p in parts])[first].astype(np.float16)).to(dev))
        print(f"[data] refined-trajectory targets for {len(first)} scenes, {ref_tgt[1].shape[1]} per scene",
              flush=True)

    bev_tgt, bev_shape = None, None
    from exp_navsim.v22 import bev_targets as bev_mod
    bev_extent = (bev_mod.X0, bev_mod.X1, bev_mod.Y)
    if args.bev_targets:
        bt = np.load(args.bev_targets)
        bev_shape = tuple(int(v) for v in bt["shape"])
        bev_extent = (float(bt["x0"]), float(bt["x0"]) + float(bt["cell"]) * bev_shape[2], float(bt["y"]))
        n_cells = int(np.prod(bev_shape))
        grids = np.unpackbits(bt["bits"], axis=1)[:, :n_cells].reshape(-1, *bev_shape)
        pos = torch.full((len(train_ds.index["token"]),), -1, dtype=torch.long)
        pos[torch.from_numpy(bt["rows"])] = torch.arange(len(bt["rows"]))
        bev_tgt = (pos.to(dev), torch.from_numpy(grids).to(dev))      # uint8 on the GPU
        print(f"[data] BEV targets for {len(bt['rows'])} scenes, grid {bev_shape}", flush=True)

    vocab = np.load(VOCAB / f"vocab_{args.vocab}.npy") if args.vocab else None
    if args.select_targets:
        assert score_tgt is not None and vocab is not None, "--select-targets needs --score-targets"
        train["future"] = select_targets(train, score_tgt, vocab, args)
    T, C = train["features"].shape[1:3] if feats else (1, 0)
    C //= len(cams)
    grid = tuple(train["features"].shape[3:5]) if feats else (9, 16)     # (9, 16), or (18, 32) unpooled
    rays = None
    if len(cams) > 1:
        from exp_navsim.v22.cameras import load_calib, token_rays
        rays = token_rays(load_calib(args.calib), cams, grid)
        missing = {cam: int((~torch.isfinite(hard["features"][:, :, i * C:(i + 1) * C]).flatten(1).all(1)).sum())
                   for i, cam in enumerate(cams)}
        print(f"[data] navhard scenes without camera images: {missing}", flush=True)
    model = GaragePlanner(C, T, grid=grid, dim=args.dim, depth=args.depth, heads=args.heads,
                          flow_depth=args.flow_depth, registers=args.registers, dropout=args.dropout,
                          vocab=vocab, groups=args.groups.split(","), mask_path=not args.no_mask_path,
                          score_head=score_tgt is not None or sim_tgt is not None, use_image=bool(feats),
                          use_status=not args.no_status, use_past=not args.no_past,
                          use_command=not args.no_command, flow=not args.no_flow,
                          bev_shape=bev_shape, goal=args.goal != "none",
                          anchor_depth=args.anchor_depth, cams=len(cams), rays=rays,
                          cam_drop=args.cam_drop, bev_queries=args.bev_queries,
                          bev_extent=bev_extent).to(dev)
    if args.bev_queries:
        print(f"[model] BEV queries {model.bev_grid} (lateral x longitudinal) over x [{bev_extent[0]}, "
              f"{bev_extent[1]}) m, y [-{bev_extent[2]}, {bev_extent[2]}) m", flush=True)
    if args.eval_vocab:
        model.eval_vocab = torch.as_tensor(np.load(VOCAB / f"vocab_{args.eval_vocab}.npy"),
                                           dtype=torch.float32, device=dev)
    if args.eval_dense:
        assert dense_traj is not None and vocab is not None, "--eval-dense needs --dense-vocab and --vocab"
        model.eval_vocab = torch.cat([model.vocab, dense_traj])
        model.eval_raw_from = len(vocab) if args.dense_offset == "raw" else None
    n_params = sum(p.numel() for p in model.parameters())
    stat = model.fit({"future": train["future"], "status": train["status"]})
    print(f"[model] {n_params / 1e6:.2f}M parameters  z_dim {model.z_dim}  {stat}", flush=True)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["model"])
        print(f"[model] initialized from {args.init}", flush=True)

    enc = None
    if args.lora_blocks:
        from exp_navsim.v22 import lora
        assert args.features == "wm11_ctx" and cams == ["F0"], "LoRA reproduces the front wm11_ctx feature only"
        assert not args.target_frac and not args.ema and not args.save_navtest, \
            "--lora-blocks: no --target-frac, --ema, --save-navtest"
        enc = lora.OnlineWM(k=args.lora_blocks, rank=args.lora_rank, alpha=args.lora_alpha, mlp=args.lora_mlp,
                            train_proj=args.lora_proj, device=dev)
        # one table of scenes: the real training rows (positions 0..n-1 of `train`), then the sim rows
        # (positions n.. = n + position in `sim`); each knows its split for the frame paths
        tabs = [lora.table(train_ds.index, train["row"].cpu().numpy(), 0)]
        lora_splits = [train_ds.index["split"]]
        if sim is not None:
            sim_rows = sim["row"].cpu().numpy()
            split_k = np.searchsorted(np.asarray(sim_offset), sim_rows, side="right") - 1
            fids = np.zeros((len(sim_rows), 4), np.int64)
            slot = np.zeros(len(sim_rows), np.int64)
            for k, ix in enumerate(sim_indices):
                m = split_k == k
                _, fids[m], slot[m] = lora.table(ix, sim_rows[m] - sim_offset[k])
                lora_splits.append(ix["split"])
            tabs.append((split_k + 1, fids, slot))
        lora_ds = lora.Clips(lora_splits, *(np.concatenate(c) for c in zip(*tabs)))

    ema = None
    if args.ema:
        ema = torch.optim.swa_utils.AveragedModel(
            model, avg_fn=lambda a, b, n: args.ema * a + (1 - args.ema) * b)

    if enc is None:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW([{"params": list(model.parameters())},
                                 {"params": enc.trainable(), "lr": args.lora_lr, "weight_decay": 0.0}],
                                lr=args.lr, weight_decay=args.weight_decay)
    warm = min(300, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / max(warm, 1)) * 0.5 * (
        1 + math.cos(math.pi * min(s / args.steps, 1))))

    def batch_of(data, idx):
        b = {k: v[idx] for k, v in data.items()}
        b["features"] = b["features"].float()
        return b

    @torch.no_grad()
    def predict(data, k=8, net=None):
        net = net or model
        net.eval()
        n = len(data["row"])
        out = torch.cat([net.predict(batch_of(data, torch.arange(i, min(i + args.eval_batch, n), device=dev)), k,
                                     plan=args.plan, steps=args.euler_steps,
                                     temperature=args.temperature, rank=args.rank,
                                     beta=args.rank_beta)
                         for i in range(0, n, args.eval_batch)], 1)
        net.train()
        return out                                                           # (k, B, 8, 3)

    eval_time = [0.0]

    def evaluate(step, groups, tag, net=None):
        t_eval = time.time()
        p = predict(val, net=net)
        gt = val["future"]
        d = (p[..., :2] - gt[None, :, :, :2]).norm(dim=-1)
        m = {"val/l2_4s": d[0, :, -1].mean().item(), "val/ade": d[0].mean().item(),
             "val/ade_min8": d.mean(-1).min(0).values.mean().item()}
        ph = predict(hard, net=net).cpu().numpy()
        if not groups:                                                       # --final-groups -1
            eval_time[0] += time.time() - t_eval
            print(f"eval @ {step} (no EPDMS): " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()),
                  flush=True)
            return m, ph
        tokens = group_tokens(groups)
        t1 = time.time()
        df = scorer.score({t: ph[:, hard_pos[t]] for t in tokens}, groups=groups)
        s = scorer.summary(df)
        m[f"{tag}/epdms"] = 100 * s.loc[0, "score"]
        m[f"{tag}/epdms_best8"] = 100 * df.groupby(["group", "half", "k"]).mean(
            numeric_only=True).reset_index().groupby(["group", "k"]).score.mean().groupby("group").max().mean()
        for c in ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                  "time_to_collision_within_bound", "lane_keeping", "two_frame_extended_comfort"):
            m[f"{tag}/{c}_s1"] = 100 * s.loc[0, f"{c}_stage_one"]
            m[f"{tag}/{c}_s2"] = 100 * s.loc[0, f"{c}_stage_two"]
        for k, v in m.items():
            tb.add_scalar(k, v, step)
        eval_time[0] += time.time() - t_eval
        print(f"eval @ {step} ({time.time() - t1:.0f}s pdm, {len(groups)} groups): "
              + "  ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
        return m, ph

    t0 = t_start = time.time()
    n = len(train["row"])
    n_tgt = int(args.batch * args.target_frac) if score_tgt is not None else 0
    if n_tgt:
        with_tgt = torch.nonzero(score_tgt[0][train["row"]] >= 0)[:, 0]
        print(f"[data] {n_tgt} of {args.batch} per batch from the {len(with_tgt)} scenes with targets",
              flush=True)
    n_sim = round(args.batch * args.sim_frac) if sim is not None else 0
    n_real = args.batch - n_sim
    if enc is not None:
        def lora_batches():
            # same composition as below: n_real uniform real scenes, n_sim sim scenes with targets
            rng = np.random.default_rng(args.seed)
            sim_pos = sim_with_tgt.cpu().numpy() if n_sim else None
            for _ in range(args.steps):
                p = rng.integers(n, size=n_real)
                if n_sim:
                    p = np.concatenate([p, n + sim_pos[rng.integers(len(sim_pos), size=n_sim)]])
                yield p.tolist()
        lora_iter = iter(lora.loader(lora_ds, lora_batches(), args.lora_workers))
        t_lora = time.time()
    for step in range(1, args.steps + 1):
        if enc is not None:
            pos, clips, missing, slot = next(lora_iter)
            pos = pos.to(dev)
            b = batch_of(train, pos[:n_real])
            if n_sim:
                bs = batch_of(sim, pos[n_real:] - n)
                b = {k: torch.cat([b[k], bs[k]]) for k in b}
            cached = b["features"]
            b["features"] = enc(clips, missing, slot)
            if step in (1, 10) or step % 100 == 0:
                d = (b["features"].detach() - cached).abs()
                rate = args.batch * (step - 1) / (time.time() - t_lora) if step > 1 else 0.0
                print(f"[lora] step {step}: |online - cached| mean {d.mean().item():.5f} max {d.max().item():.4f} "
                      f"(|cached| mean {cached.abs().mean().item():.4f}), {rate:.1f} scenes/s, "
                      f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
                tb.add_scalar("lora/feature_drift", d.mean().item(), step)
        else:
            idx = torch.randint(n, (n_real,), device=dev)
            if n_tgt:
                idx[:n_tgt] = with_tgt[torch.randint(len(with_tgt), (n_tgt,), device=dev)]
            b = batch_of(train, idx)
            if n_sim:
                # sim rows go after the real ones; b["row"][n_real:] are rows of the sim index
                bs = batch_of(sim, sim_with_tgt[torch.randint(len(sim_with_tgt), (n_sim,), device=dev)])
                b = {k: torch.cat([b[k], bs[k]]) for k in b}
        if args.goal_aug:
            b["goal"] = augment_goal(b, args)
        if args.goal_drop:
            drop = torch.rand(len(b["row"]), device=dev) < args.goal_drop
            b["goal"] = torch.where(drop[:, None], torch.full_like(b["goal"], float("nan")), b["goal"])
        tgt = bev = None
        real_rows = b["row"][:n_real]
        if score_tgt is not None:
            p = score_tgt[0][real_rows]
            mask = p >= 0
            tgt = (score_tgt[1][p.clamp(min=0)].float(), mask)
        ref = None
        if ref_tgt is not None:
            p = ref_tgt[0][real_rows]
            ref = (ref_tgt[1][p.clamp(min=0)], ref_tgt[2][p.clamp(min=0)], p >= 0)
        if bev_tgt is not None:
            p = bev_tgt[0][real_rows]
            mask = p >= 0
            bev = (bev_tgt[1][p.clamp(min=0)].float(), mask)
        row_weight = None
        if n_sim:
            tgt, ref, bev, row_weight = with_sim_rows(b["row"][n_real:], tgt, ref, bev, sim_tgt, n_real,
                                                      n_sim, model.vocab.shape[0], args)
        dense = None
        if dense_tgt is not None:
            dense = dense_targets(b["row"][n_real:], dense_tgt, dense_traj, n_real, args)
            if args.dense_replace:
                tgt[1][n_real:] = False
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, logs = model.loss(b, w_act=args.w_act, w_path=args.w_path, w_roll=args.w_roll,
                                    w_score=args.w_score, score_targets=tgt,
                                    w_bev=args.w_bev, bev_targets=bev,
                                    w_rank=args.w_rank, rank_tau=args.rank_tau,
                                    refined_targets=ref, w_refined=args.w_refined, row_weight=row_weight,
                                    **({"dense_targets": dense, "w_dense": args.w_dense} if dense is not None else {}))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if enc is not None:
            torch.nn.utils.clip_grad_norm_(enc.trainable(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            ema.update_parameters(model)
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
            evaluate(step, eval_groups, f"hard{len(eval_groups)}")
            t0 = time.time()

    train_min = (time.time() - t_start - eval_time[0]) / 60
    if enc is not None:
        # tuned encoder: save the adapters, cache navhard under a new feature name (sweep_rank reads it
        # through config.json "features"), and re-encode val and navhard for the final evaluation
        enc.eval()
        feat_name = f"wm11_ctx_lora_{args.name}"
        torch.save({"lora": enc.lora_state(), "config": enc.config(), "feature": feat_name, "init": args.init},
                   run / "lora.pt")
        x = lora.write_cache(enc, "navhard", feat_name, args.name, workers=args.lora_workers)
        hard["features"] = torch.from_numpy(x[hard_ds.rows][:, None]).to(dev)
        val["features"] = torch.from_numpy(lora.encode_rows(enc, val_ds.index, val_rows, workers=args.lora_workers,
                                                            desc="val")[:, None]).to(dev)
        del x
        args.train_features, args.features = args.features, feat_name
        (run / "config.json").write_text(json.dumps(vars(args), indent=1))
    net = ema.module if ema is not None else model
    m, ph = evaluate(args.steps, final_groups, f"hard{len(final_groups)}", net=net)

    # the diversity switch: the same weights sampled at a positive temperature
    if not args.no_flow:
        from exp_navsim.eval.metrics import sample_metrics
        net.eval()
        with torch.no_grad():
            s = torch.cat([net.sample(batch_of(val, torch.arange(i, min(i + args.eval_batch, len(val["row"])),
                                                                 device=dev)),
                                      args.div_samples, steps=args.euler_steps,
                                      temperature=args.div_temperature)
                           for i in range(0, len(val["row"]), args.eval_batch)], 1)
        net.train()
        d = sample_metrics(s, val["future"])
        for k in ("energy", "spread", "dispersion", "ade_min", "ade_mean"):
            m[f"div/{k}"] = float(d[k].mean())
        print(f"diversity at temperature {args.div_temperature} ({args.div_samples} samples): "
              + "  ".join(f"{k} {m[f'div/{k}']:.3f}" for k in
                          ("energy", "spread", "dispersion", "ade_min", "ade_mean")), flush=True)
    state = net.state_dict()
    if args.eval_dense:
        state.pop("eval_vocab", None)          # set again from --dense-vocab when reloading
    torch.save({"model": state, "args": {**vars(args), "bev_shape": bev_shape, "grid": list(grid),
                                                    "bev_extent": bev_extent}},
               run / "model.pt")
    np.savez_compressed(run / "navhard_top8.npz", token=hard_tokens, samples=ph.astype(np.float16))
    if args.save_navtest:
        test_ds = SceneFeatureDataset("data/navsim_index/navtest.pkl", split="all",
                                      require_future=False, **common)
        test = gpu(test_ds)
        pt = predict(test, net=net).cpu().numpy()
        np.savez_compressed(run / "navtest_top8.npz",
                            token=np.asarray(test_ds.index["token"])[test_ds.rows],
                            samples=pt.astype(np.float16))
    row = {"name": args.name, "features": args.features, "repr": args.groups,
           "flow": not args.no_flow, "vocab": args.vocab, "plan": args.plan,
           "metric_training": bool(args.score_targets), "bev_aux": bool(args.bev_targets),
           "cams": args.cams, "bev_queries": args.bev_queries, "sim": args.sim_split or "",
           "rank": args.rank, "steps": args.steps,
           "train_scenes": len(train_ds), "params_M": round(n_params / 1e6, 3),
           "train_min": round(train_min, 2), "seed": args.seed,
           "euler_steps": args.euler_steps, "temperature": args.temperature, "note": args.note,
           **({"dense": f"{args.vocab_drop}/{len(dense_traj)}" + ("+eval" if args.eval_dense else "")}
              if dense_tgt is not None else {}),
           **{k: round(float(v), 4) for k, v in m.items()}}
    path = Path(args.results)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(path), pd.DataFrame([row])]) if path.exists() else pd.DataFrame([row])
    df.to_csv(path, index=False)
    print(f"-> {path}")
    scorer.close()
    tb.close()


if __name__ == "__main__":
    main()
