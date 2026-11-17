"""Inference-time sweep of a trained v22 planner on the navhard proxy, without retraining.

    python -m exp_navsim.v22.sweep_rank logs_exp/24g_metric_vocab --groups-file outputs/24_v22/groups_48.json

Every selection rule becomes one candidate index k of a single FastEPDMS call, so the whole
sweep costs about one scoring pass per rule on the same workers. Rules:

  imi            the imitation head alone
  epdms          the predicted EPDMS of the metric head (Hydra-MDP / GTRS)
  mix_b          log EPDMS + b * log p_imitation
  top{n}_epdms   the n most likely anchors by imitation, of those the best predicted EPDMS
  raw_*          the same rule on the unrefined anchors (no offset head)
  dense*         the same rules on the 4096-anchor vocabulary (GTRS-Dense)
  gd_*           (--dense-index) epdms and mix_b over the 256 anchors + the GTRS dense set the
                 scorer was trained on (train.py --dense-vocab), with the ensemble if given;
                 gd_ takes the offset-refined plan of every candidate, gdraw_ the raw dense
                 trajectory for the dense candidates (the one that was scored) and the refined
                 plan for the 256
  flow           the zero-noise flow plan (if the run has a flow head)
  flow_rerank    the flow plan and 16 flow samples, the best predicted EPDMS
  oracle_top{n}  (--oracle n) of the n best refined anchors by mix_0.03, the one with the best
                 TRUE per-token EPDMS (all n scored by FastEPDMS first): the teacher ceiling
                 of a selector that only reorders the top-n of this model

Runs trained with --goal end read the goal of every navhard token from --goal-file
(v22/goals.py); --no-goal feeds the null goal instead, and --goal-alpha a scales the goal
embedding towards the null one (null + a * (goal - null); 1 = as trained). Two rules use the
goal only through the metric head:

  gsc_epdms      the anchor with the best predicted EPDMS of the goal pass, but its refined
                 trajectory (offset head) from a second, goal-free pass
  gsc_mix_b      the same with log EPDMS (goal pass) + b * log p_imitation (goal-free pass)

--per-token also writes the per-token rows of every rule (sweeps/<run><tag>_tokens.csv).

--ensemble run2[,run3] averages the anchor scorer of several runs trained on the same
vocabulary (same goal input and features): for every anchor the predicted sub-score
probabilities (--ens-mode prob) or the log EPDMS (--ens-mode logep) are averaged, the
imitation log probabilities likewise (log of the mean softmax, or mean log softmax), and the
refined trajectory is the mean of the refined anchors. Only the main 256-anchor rules (imi,
epdms, mix_b, top{n}_epdms) use the ensemble. Two pessimistic modes (imitation as in prob):

  --ens-mode lcb      log EPDMS = mean_m - k * std_m of the per-member log EPDMS (--lcb-k)
  --ens-mode minmult  the multiplicative sub-scores (NC, DAC, DDC, TLC) take the minimum
                      probability over the members, the weighted ones (EP, TTC, LK, HC) the mean

--multi-goal file.npz (token, goal (N, K, 3), optional goal_score (N, K); v22/goal_head.py)
runs the planner (and the ensemble) once per goal hypothesis j < --multi-goal-k and pools
the n * 256 refined anchors of every token:

  mg_epdms        argmax over the pooled candidates of the log EPDMS of their own pass
  mg_mix_b        argmax of ep + b * (imi + log w_j), w_j = softmax over j of goal_score
                  (uniform without goal_score)
  mg_null_mix_b   the same pooled trajectories rescored by a null-goal pass (scorer and
                  imitation logits applied to the trajectories, imi = log softmax over the pool)

--mg-include-null adds the null-goal pass's own 256 refined anchors to the pool as one more
hypothesis with log w = log(1/n) (the mean hypothesis weight). --mg-flow n adds the zero-noise
flow plan and n flow samples at temperature 1 of the null pass (runs with a flow head only).
The flow candidates have no imitation logit of their own: in mg_mix_b their ep is the
null-goal rescoring and their imi + log w the minimum over the anchor candidates of the pool;
in mg_null_mix_b they are rescored like the rest and their imi is the minimum imi of the
anchor candidates (the flow plans enter on predicted EPDMS, never on imitation).

Writes outputs/24_v22/sweeps/<run>.csv.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.v22.fast_epdms import COLS, FastEPDMS, group_tokens, load_groups

OUT = Path("outputs/24_v22")
VOCAB = Path("outputs/16_fast_planner")


def load_planner(run, dev="cuda", features_shape=None):
    import torch
    from exp_navsim.v22.runs import build_planner
    ck = torch.load(Path(run) / "model.pt", map_location=dev, weights_only=False)
    a = ck["args"]
    vocab = np.load(VOCAB / f"vocab_{a['vocab']}.npy") if a["vocab"] else None
    C, T = features_shape
    # multi-camera runs: C is the channels of all cameras, build_planner splits it
    m = build_planner(a, C, T, vocab).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, a


def ens_combine(outs, mode="prob", lcb_k=1.0):
    """Anchor outputs of the ensemble members -> (imitation score, log EPDMS), both (B, V)."""
    import torch
    from exp_navsim.v22.model import GaragePlanner, epdms_of
    if mode == "logep":
        ep = torch.stack([GaragePlanner.rank(o, "epdms") for o in outs]).mean(0)
        imi = torch.stack([GaragePlanner.rank(o, "imitation") for o in outs]).mean(0)
        return imi, ep
    imi = torch.stack([o["logits"].float().softmax(-1) for o in outs]).mean(0).clamp_min(1e-12).log()
    p = torch.stack([torch.sigmoid(o["scores"].float()) for o in outs])          # (M, B, V, 8)
    if mode == "prob":
        ep = epdms_of(p.mean(0)).clamp_min(1e-6).log()
    elif mode == "lcb":
        lep = epdms_of(p).clamp_min(1e-6).log()                                 # (M, B, V)
        ep = lep.mean(0) - lcb_k * lep.std(0, correction=0)
    elif mode == "minmult":
        # columns 0-3 are the multiplicative NC, DAC, DDC, TLC (model.epdms_of)
        ep = epdms_of(torch.cat([p[..., :4].min(0).values, p[..., 4:].mean(0)], -1)).clamp_min(1e-6).log()
    else:
        raise ValueError(mode)
    return imi, ep


def refine(nets, b, raw_c, steps, lr=0.05, lam=1.0, mode="prob", lcb_k=1.0):
    """Test-time ensemble-guided refinement of (n, M, 8, 3) raw anchors (the opt_mix_b rules).

    An additive offset delta of the anchor query (zero at the start) is fitted by Adam to
    J = mean over members of log EPDMS(anchors(mem, ego, raw + delta)) - lam * mean(delta^2)
    (x, y in metres, heading in radians with weight 1). The query is perturbed, not the
    refined trajectory, so at delta = 0 the scores and plans are exactly those of the main
    rules (opt0_mix_b == mix_b). -> refined trajectories (mean over members) and log EPDMS
    (ens_combine of the members with `mode`), both at the final delta."""
    import torch
    from exp_navsim.v22.model import epdms_of
    ctx = [tuple(t.detach() for t in net.scene(b)[:2]) for net in nets]
    delta = torch.zeros_like(raw_c, requires_grad=True)
    if steps:
        opt = torch.optim.Adam([delta], lr=lr)
        with torch.enable_grad():
            for _ in range(steps):
                lep = torch.stack([epdms_of(torch.sigmoid(net.anchors(me, ee, raw_c + delta)["scores"].float()))
                                   .clamp_min(1e-6).log() for net, (me, ee) in zip(nets, ctx)]).mean(0)
                J = lep - lam * (delta ** 2).mean((-2, -1))                      # (n, M)
                delta.grad, = torch.autograd.grad(-J.sum(), [delta])            # no parameter grads
                opt.step()
    with torch.no_grad():
        outs = [net.anchors(me, ee, raw_c + delta.detach()) for net, (me, ee) in zip(nets, ctx)]
    traj = torch.stack([o["traj"] for o in outs]).mean(0)
    _, ep = ens_combine(outs, mode, lcb_k)
    return traj, ep


def pool_rule(ep, imi, log_w, beta):
    """Pooled multi-goal candidates: ep, imi (B, n, V), log_w (B, n) -> flat argmax index."""
    s = ep + beta * (imi + log_w[..., None])
    return s.flatten(1).argmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--groups-file", default="outputs/24_v22/groups_48.json")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--dense", type=int, default=4096)
    ap.add_argument("--dense-index", default="",
                    help="GTRS index npy of a dense set (e.g. data/simscale/rewards/dense2048_gtrs_index.npy): gd_* rules")
    ap.add_argument("--betas", default="0.03,0.1,0.3,1")
    ap.add_argument("--tops", default="8,32")
    ap.add_argument("--rules", default="", help="comma-separated subset of the rules to score")
    ap.add_argument("--tag", default="", help="suffix of the output csv")
    ap.add_argument("--out", default=str(OUT), help="directory of sweeps/ and per_group/")
    ap.add_argument("--oracle", type=int, default=0, help="n of the oracle_top{n} rule; 0 = off")
    ap.add_argument("--goal-file", default="outputs/25_terra/navhard_goals.npz")
    ap.add_argument("--no-goal", action="store_true", help="null goal for goal-conditioned runs")
    ap.add_argument("--goal-alpha", type=float, default=1.0, help="goal embedding weight vs null")
    ap.add_argument("--per-token", action="store_true", help="also save the per-token rows")
    ap.add_argument("--ensemble", default="", help="comma-separated runs whose scorers are averaged in")
    ap.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    ap.add_argument("--ens-mode", default="prob", choices=("prob", "logep", "lcb", "minmult"))
    ap.add_argument("--lcb-k", type=float, default=1.0, help="std weight of --ens-mode lcb")
    ap.add_argument("--multi-goal", default="", help="npz with goal (N, K, 3) [+ goal_score (N, K)]")
    ap.add_argument("--multi-goal-k", type=int, default=4, help="goal hypotheses pooled per token")
    ap.add_argument("--mg-include-null", action="store_true",
                    help="add the null-goal pass's own refined anchors to the multi-goal pool")
    ap.add_argument("--mg-flow", type=int, default=0,
                    help="add the null-pass zero-noise flow plan + n samples at temperature 1 to the pool")
    ap.add_argument("--opt-steps", type=int, default=0, help="Adam steps of the opt_mix_b rules; 0 = off")
    ap.add_argument("--opt-lr", type=float, default=0.05)
    ap.add_argument("--opt-lam", type=float, default=1.0, help="weight of the mean squared offset")
    ap.add_argument("--opt-topm", type=int, default=8, help="candidates refined per token")
    ap.add_argument("--rescore", action="store_true",
                    help="rank the refined anchors by the scorer applied to the refined trajectories "
                         "themselves (second anchor pass, every ensemble member), not to the raw anchors")
    args = ap.parse_args()
    out_dir = Path(args.out)

    from exp_navsim.v22.fast_epdms import all_groups
    groups = all_groups() if args.groups_file == "all" else load_groups(args.groups_file)
    scorer = FastEPDMS(groups, workers=args.workers)          # forks before CUDA

    import torch
    from exp_navsim.data.scenes import SceneFeatureDataset
    from exp_navsim.v22.model import HORIZON, GaragePlanner, epdms_of

    run = Path(args.run)
    import json
    cfg = json.loads((run / "config.json").read_text())
    from exp_navsim.v22.runs import feature_dir, feature_names
    feats = feature_names(cfg)                       # + <feature>_<cam> of multi-camera runs
    # members may read a prefix of the largest feature list (e.g. front only vs front + side cams):
    # the batch holds the largest list, each net gets its leading channels (features_of concats on C)
    run_feats = {str(run): feats}
    for r in [x for x in args.ensemble.split(",") if x]:
        run_feats[r] = feature_names(json.loads((Path(r) / "config.json").read_text()))
    load_cfg = cfg
    for r, f in run_feats.items():
        if len(f) > len(feats):
            feats, load_cfg = f, json.loads((Path(r) / "config.json").read_text())
    for r, f in run_feats.items():
        assert feats[:len(f)] == f, f"ensemble member {r} reads features {f}, not a prefix of {feats}"
    ds = SceneFeatureDataset("data/navsim_index/navhard.pkl", split="all", require_future=False,
                             features=feats, feature_dir=feature_dir(load_cfg), history=4)
    tokens = np.asarray(ds.index["token"])[ds.rows]
    need = set(group_tokens(groups))
    sel = np.flatnonzero([t in need for t in tokens])
    rows = ds.rows[sel]
    dev = args.device
    b = {"features": torch.from_numpy(ds.features_of(rows)).float().to(dev),
         "future": torch.zeros(len(rows), HORIZON, 3, device=dev),
         "past": torch.from_numpy(ds.past[rows]).float().to(dev),
         "status": torch.from_numpy(ds.status[rows]).float().to(dev),
         "command": torch.from_numpy(ds.command[rows]).float().to(dev)}
    if cfg.get("goal", "none") != "none":
        from exp_navsim.v22.train import navhard_goals
        b["goal"] = navhard_goals(args.goal_file, tokens[sel]).to(dev)
        if args.no_goal:
            b["goal"][:] = float("nan")
    if args.multi_goal:
        assert "goal" in b, "--multi-goal needs a goal-conditioned run"
        g = np.load(args.multi_goal)
        n_hyp = min(args.multi_goal_k, g["goal"].shape[1])
        pos = {t: i for i, t in enumerate(g["token"].tolist())}
        idx = np.array([pos.get(t, -1) for t in tokens[sel]])
        mg = np.full((len(idx), n_hyp, 3), np.nan, np.float32)
        mg[idx >= 0] = g["goal"][idx[idx >= 0], :n_hyp]
        score = np.zeros((len(idx), n_hyp), np.float32)
        if "goal_score" in g:
            score[idx >= 0] = g["goal_score"][idx[idx >= 0], :n_hyp]
        b["mg_goal"] = torch.from_numpy(mg).to(dev)
        b["mg_logw"] = torch.from_numpy(score).to(dev).log_softmax(-1)
        print(f"[sweep] {n_hyp} goal hypotheses for {(idx >= 0).sum()} / {len(idx)} tokens, "
              f"goal_score {'yes' if 'goal_score' in g else 'no (uniform)'}", flush=True)
    T, C = b["features"].shape[1:3]
    def load_sub(r):
        n = len(run_feats[str(r)])
        Cn = C * n // len(feats)
        net, a_ = load_planner(r, dev, (Cn, T))
        if n < len(feats):
            orig = net.scene
            net.scene = lambda batch, *xs, _o=orig, _c=Cn, **kw: _o(dict(batch, features=batch["features"][:, :, :_c]), *xs, **kw)
            print(f"[sweep] {r}: first {Cn} / {C} feature channels", flush=True)
        return net, a_
    m, a = load_sub(run)
    ens = [load_sub(r)[0] for r in args.ensemble.split(",") if r]
    for e in ens:
        assert torch.equal(e.vocab, m.vocab), "ensemble members need the same vocabulary"
    if m.goal_in is not None and args.goal_alpha != 1.0:
        full = m.goal_embedding

        def weak(goal):
            null = m.goal_null.to(torch.float32)
            return null + args.goal_alpha * (full(goal) - null)
        m.goal_embedding = weak

    gd_traj = None
    if args.dense_index:
        from exp_navsim.simscale.rewards import gtrs_vocab
        gd = torch.from_numpy(gtrs_vocab()[np.load(args.dense_index)][:, 4::5].astype(np.float32)).to(dev)
        gd_traj = torch.cat([m.vocab, gd])                                      # (256 + K, 8, 3)
        print(f"[sweep] gd_ rules over {len(gd_traj)} candidates ({len(gd)} dense)", flush=True)

    gather = lambda traj, idx: torch.gather(traj, 1, idx[:, None, None, None].expand(
        -1, 1, HORIZON, 3))[:, 0]

    def anchor_out(net, b):
        """The 256-anchor outputs of one model on a chunk of scenes."""
        mem, ego, _ = net.scene(b)
        out = {}
        for i in range(0, len(b["status"]), 256):
            for k, val in net.anchors(mem[i:i + 256], ego[i:i + 256]).items():
                out.setdefault(k, []).append(val)
        return {k: torch.cat(v) for k, v in out.items()}

    def ensemble(out, b):
        """-> (outputs with the mean refined anchors, imitation score, log EPDMS) of m + ens."""
        outs = [out] + [anchor_out(e, b) for e in ens]
        out = dict(out, traj=torch.stack([o["traj"] for o in outs]).mean(0))
        imi, ep = ens_combine(outs, args.ens_mode, args.lcb_k)
        return out, imi, ep

    def scored(b):
        """-> refined anchors (n, V, 8, 3), imitation score and log EPDMS (n, V) of m (+ ens)."""
        out = anchor_out(m, b)
        if ens:
            out, imi, ep = ensemble(out, b)
            return out["traj"], imi, ep
        return out["traj"], GaragePlanner.rank(out, "imitation"), GaragePlanner.rank(out, "epdms")

    def optimize(b, raw_c, steps):
        return refine([m] + ens, b, raw_c, steps, args.opt_lr, args.opt_lam,
                      args.ens_mode if ens else "prob", args.lcb_k)

    def score_traj(b, traj):
        """Null-goal style rescoring of arbitrary (n, K, 8, 3) trajectories by m (+ ens):
        -> imitation (log softmax over the K, ensemble: log of the mean softmax) and log EPDMS
        (ens_combine of the members)."""
        outs = []
        step = max(1, 65536 // traj.shape[1])                     # scenes per anchor pass
        for net in [m] + ens:
            mem_e, ego_e, _ = net.scene(b)
            o = {}
            for i in range(0, len(traj), step):
                for k, val in net.anchors(mem_e[i:i + step], ego_e[i:i + step], traj[i:i + step]).items():
                    o.setdefault(k, []).append(val)
            outs.append({k: torch.cat(v) for k, v in o.items() if k != "traj"})
        imi, ep = ens_combine(outs, args.ens_mode if ens else "prob", args.lcb_k)
        return imi, ep

    def rescore(traj, b):
        """log EPDMS of the given (n, K, 8, 3) trajectories, sub-score probabilities averaged
        over the model and its ensemble members."""
        probs = []
        for net in [m] + ens:
            mem_e, ego_e, _ = net.scene(b)
            sc = torch.cat([net.anchors(mem_e[i:i + 256], ego_e[i:i + 256], traj[i:i + 256])["scores"]
                            for i in range(0, len(traj), 256)])
            probs.append(torch.sigmoid(sc.float()))
        return epdms_of(torch.stack(probs).mean(0)).clamp_min(1e-6).log()

    def rules_of(b, need):
        """All selection rules on one chunk of scenes -> {rule: (n, 8, 3)}."""
        rows = b["status"]  # only its length is used below
        rules = {}
        mem, ego, _ = m.scene(b)
        if "wp" in m.groups:
            rules["flow"] = m.predict(b, 1, plan="flow")[0]
        if m.vocab is not None:
            sets = {"": None}
            if args.dense and m.scorer is not None and any(n.startswith("dense_") for n in need):
                v = torch.as_tensor(np.load(VOCAB / f"vocab_{args.dense}.npy"), dtype=torch.float32, device=dev)
                sets["dense_"] = v[None].expand(len(b["status"]), -1, -1, -1)
            for pre, traj in sets.items():
                out = {}
                bs = 256 if traj is None else 32
                for i in range(0, len(b["status"]), bs):
                    sl = slice(i, i + bs)
                    o = m.anchors(mem[sl], ego[sl], None if traj is None else traj[sl])
                    for k, val in o.items():
                        out.setdefault(k, []).append(val)
                out = {k: torch.cat(v) for k, v in out.items()}
                raw = (m.vocab[None].expand(len(b["status"]), -1, -1, -1) if traj is None else traj)
                imi = GaragePlanner.rank(out, "imitation")
                ens_ep = None
                if pre == "" and ens:
                    out, imi, ens_ep = ensemble(out, b)
                cand = {"imi": imi}
                if "scores" in out:
                    ep = GaragePlanner.rank(out, "epdms") if ens_ep is None else ens_ep
                    if pre == "" and args.rescore:
                        ep = rescore(out["traj"].float(), b)
                    cand["epdms"] = ep
                    for beta in [float(x) for x in args.betas.split(",") if x]:
                        cand[f"mix_{beta:g}"] = ep + beta * imi
                    if pre == "" and args.oracle:
                        top = (ep + 0.03 * imi).topk(args.oracle, -1).indices          # (n, K)
                        rules["_oracle"] = torch.gather(out["traj"], 1, top[..., None, None].expand(
                            -1, -1, HORIZON, 3))
                    for n in [int(x) for x in args.tops.split(",") if x]:
                        top = imi.topk(n, -1).indices
                        masked = torch.full_like(ep, -1e9).scatter(1, top, ep.gather(1, top))
                        cand[f"top{n}_epdms"] = masked
                for name, score in cand.items():
                    idx = score.argmax(-1)
                    rules[pre + name] = gather(out["traj"], idx)
                    if pre == "" and name in ("imi", "epdms"):
                        rules["raw_" + name] = gather(raw, idx)
                if pre == "" and "scores" in out and (args.opt_steps or any(n.startswith("opt0_") for n in need)):
                    M = min(args.opt_topm, ep.shape[1])
                    for beta in [float(x) for x in args.betas.split(",") if x]:
                        top = (ep + beta * imi).topk(M, -1).indices                     # (n, M)
                        raw_c = torch.gather(raw, 1, top[..., None, None].expand(-1, -1, HORIZON, 3))
                        imi_c = imi.gather(1, top)
                        for tag, steps in (("opt0_", 0), ("opt_", args.opt_steps)):
                            if tag == "opt_" and steps == 0:
                                continue
                            traj_c, ep_c = optimize(b, raw_c, steps)
                            rules[f"{tag}mix_{beta:g}"] = gather(traj_c, (ep_c + beta * imi_c).argmax(-1))
                if pre == "" and "goal" in b and "scores" in out and any(n.startswith("gsc_") for n in need):
                    # goal only through the metric head: plans and imitation of a goal-free pass
                    nb = dict(b, goal=torch.full_like(b["goal"], float("nan")))
                    mem0, ego0, _ = m.scene(nb)
                    o0 = {}
                    for i in range(0, len(b["status"]), 256):
                        for k, val in m.anchors(mem0[i:i + 256], ego0[i:i + 256]).items():
                            o0.setdefault(k, []).append(val)
                    o0 = {k: torch.cat(v) for k, v in o0.items()}
                    imi0 = GaragePlanner.rank(o0, "imitation")
                    ep = GaragePlanner.rank(out, "epdms")
                    rules["gsc_epdms"] = gather(o0["traj"], ep.argmax(-1))
                    for beta in [float(x) for x in args.betas.split(",") if x]:
                        rules[f"gsc_mix_{beta:g}"] = gather(o0["traj"], (ep + beta * imi0).argmax(-1))
            if gd_traj is not None and m.scorer is not None and any(n.startswith("gd") for n in need):
                n_b, V = len(b["status"]), m.vocab.shape[0]
                va = gd_traj[None].expand(n_b, -1, -1, -1)
                outs = []
                for net in [m] + ens:
                    mem_e, ego_e, _ = net.scene(b)
                    o = {}
                    for i in range(0, n_b, 32):
                        for k, val in net.anchors(mem_e[i:i + 32], ego_e[i:i + 32], va[i:i + 32]).items():
                            o.setdefault(k, []).append(val)
                    outs.append({k: torch.cat(v) for k, v in o.items()})
                traj = torch.stack([o["traj"] for o in outs]).mean(0)
                if ens:
                    imi, ep = ens_combine(outs, args.ens_mode, args.lcb_k)
                else:
                    imi, ep = GaragePlanner.rank(outs[0], "imitation"), GaragePlanner.rank(outs[0], "epdms")
                raw_d = torch.cat([traj[:, :V], va[:, V:]], 1)
                cand = {"epdms": ep, **{f"mix_{beta:g}": ep + beta * imi
                                        for beta in [float(x) for x in args.betas.split(",") if x]}}
                for name, score in cand.items():
                    idx = score.argmax(-1)
                    rules["gd_" + name] = gather(traj, idx)
                    rules["gdraw_" + name] = gather(raw_d, idx)
            if "mg_goal" in b and m.scorer is not None:
                # multi-goal pooling: one pass per goal hypothesis, n * V candidates per token
                n_hyp = b["mg_goal"].shape[1]
                trajs, imis, eps, logws = [], [], [], []
                for j in range(n_hyp):
                    t_j, i_j, e_j = scored(dict(b, goal=b["mg_goal"][:, j]))
                    trajs.append(t_j), imis.append(i_j), eps.append(e_j)
                    logws.append(b["mg_logw"][:, j:j + 1].expand_as(i_j))
                nb = dict(b, goal=torch.full_like(b["goal"], float("nan")))
                if args.mg_include_null:
                    # the null pass as one more hypothesis, log w = log of the mean weight = -log n
                    t_j, i_j, e_j = scored(nb)
                    trajs.append(t_j), imis.append(i_j), eps.append(e_j)
                    logws.append(torch.full_like(i_j, -math.log(n_hyp)))
                flat = torch.cat(trajs, 1)                                       # (B, N, 8, 3)
                prior = torch.cat(imis, 1) + torch.cat(logws, 1)                 # imi + log w_j
                ep_all = torch.cat(eps, 1)
                n_pool = flat.shape[1]
                if args.mg_flow and m.flow and "wp" in m.groups:
                    # zero-noise flow plan + n samples at temperature 1 of the null pass; their
                    # own-pass ep is the null-goal rescoring, their prior the min of the pool
                    from exp_navsim.v22.model import unscaled
                    torch.manual_seed(0)
                    mem0, ego0, _ = m.scene(nb)
                    wp = lambda z: m.from_z(z.flatten(0, 1))["wp"]
                    c0 = unscaled(wp(m._sample(mem0, ego0, 1, 8, 0.0))).unflatten(0, (1, len(b["status"])))
                    cs = unscaled(wp(m._sample(mem0, ego0, args.mg_flow, 8, 1.0))).unflatten(
                        0, (args.mg_flow, len(b["status"])))
                    fl = torch.cat([c0, cs]).transpose(0, 1).float()             # (B, n + 1, 8, 3)
                    _, ep_fl = score_traj(nb, fl)
                    flat = torch.cat([flat, fl], 1)
                    ep_all = torch.cat([ep_all, ep_fl], 1)
                    prior = torch.cat([prior, prior.min(-1, keepdim=True).values.expand(-1, fl.shape[1])], 1)
                betas = [0.0] + [float(x) for x in args.betas.split(",") if x]
                for beta in betas:
                    name = "mg_epdms" if beta == 0 else f"mg_mix_{beta:g}"
                    rules[name] = gather(flat, (ep_all + beta * prior).argmax(-1))
                if any(n.startswith("mg_null") for n in need) or not args.rules:
                    imi0, ep0 = score_traj(nb, flat.float())
                    if flat.shape[1] > n_pool:                                  # flow candidates
                        imi0 = torch.cat([imi0[:, :n_pool], imi0[:, :n_pool].min(-1, keepdim=True).values
                                          .expand(-1, flat.shape[1] - n_pool)], 1)
                    for beta in betas:
                        name = "mg_null_epdms" if beta == 0 else f"mg_null_mix_{beta:g}"
                        rules[name] = gather(flat, (ep0 + beta * imi0).argmax(-1))
            if m.scorer is not None and m.flow and "wp" in m.groups and any(n.startswith(("flow_", "union_")) for n in need):
                torch.manual_seed(0)
                wp = lambda z: m.from_z(z.flatten(0, 1))["wp"]
                from exp_navsim.v22.model import unscaled
                c0 = unscaled(wp(m._sample(mem, ego, 1, 8, 0.0))).unflatten(0, (1, len(b["status"])))
                cs = unscaled(wp(m._sample(mem, ego, 16, 8, 1.0))).unflatten(0, (16, len(b["status"])))
                cand = torch.cat([c0, cs]).transpose(0, 1)
                o = m.anchors(mem, ego, cand)
                idx = GaragePlanner.rank(o, "epdms").argmax(-1)
                rules["flow_rerank"] = gather(cand, idx)
                idx = GaragePlanner.rank(o, "mix", 0.03).argmax(-1)
                rules["flow_rerank_mix"] = gather(cand, idx)
                # union: the refined anchors and the flow samples in one candidate set
                va = m.vocab[None].expand(len(b["status"]), -1, -1, -1)
                o = m.anchors(mem, ego, torch.cat([va, cand], 1))
                both = torch.cat([o["traj"][:, :va.shape[1]], cand], 1)
                for beta in (0.0, 0.03):
                    idx = (GaragePlanner.rank(o, "mix", beta)).argmax(-1)
                    rules[f"union_mix_{beta:g}"] = gather(both, idx)
        return rules

    need = args.rules.split(",") if args.rules else ["dense_", "flow_", "union_"] + (["gd"] if gd_traj is not None else [])
    parts = {}
    with torch.no_grad():
        for i in range(0, len(rows), 512):
            bc = {k: v[i:i + 512] for k, v in b.items()}
            for k, v in rules_of(bc, need).items():
                parts.setdefault(k, []).append(v)
    rules = {k: torch.cat(v) for k, v in parts.items()}
    if args.oracle:
        cand = rules.pop("_oracle").cpu().numpy()                             # (N, n, 8, 3)
        t0 = time.time()
        _, tok = scorer.score({t: cand[i] for i, t in enumerate(tokens[sel])}, per_token=True)
        best = tok.loc[tok.groupby("token").score.idxmax(), ["token", "k"]]
        pick = dict(zip(best.token, best.k))
        # stage-two branches that no candidate reaches were not scored; they keep the mix plan
        idx = torch.tensor([int(pick.get(t, 0)) for t in tokens[sel]], device=dev)
        rules[f"oracle_top{args.oracle}"] = torch.from_numpy(cand).to(dev)[torch.arange(len(idx), device=dev), idx]
        print(f"[sweep] oracle over the top {args.oracle}: {len(pick)} tokens scored in "
              f"{time.time() - t0:.0f} s, mean chosen rank {best.k.mean():.1f}", flush=True)

    names = [n for n in rules if not args.rules or n in args.rules.split(",")]
    stack = torch.stack([rules[n] for n in names], 1).cpu().numpy()          # (N, K, 8, 3)
    print(f"[sweep] {len(names)} rules on {len(groups)} groups: {names}", flush=True)
    t0 = time.time()
    trajs = {t: stack[i] for i, t in enumerate(tokens[sel])}
    if args.per_token:
        df, tok = scorer.score(trajs, per_token=True)
        tok["rule"] = [names[k] for k in tok.k]
        (out_dir / "sweeps").mkdir(parents=True, exist_ok=True)
        tok.to_csv(out_dir / "sweeps" / f"{run.name}{args.tag}_tokens.csv", index=False)
    else:
        df = scorer.score(trajs)
    s = FastEPDMS.summary(df)
    res = []
    for k, n in enumerate(names):
        r = {"run": run.name, "rule": n, "groups": len(groups), "epdms": round(100 * s.loc[k, "score"], 2)}
        for c in COLS[:-1]:
            r[f"{c}_s1"] = round(100 * s.loc[k, f"{c}_stage_one"], 1)
            r[f"{c}_s2"] = round(100 * s.loc[k, f"{c}_stage_two"], 1)
        res.append(r)
    res = pd.DataFrame(res)
    # per-rule plans and per-group rows, so figures.py can draw run__rule like any run
    (out_dir / "per_group").mkdir(parents=True, exist_ok=True)
    (out_dir / "sweeps").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "sweeps" / f"{run.name}{args.tag}_plans.npz", token=tokens[sel],
                        rules=np.array(names), plans=stack.astype(np.float16))
    for k, n in enumerate(names):
        df[df.k == k].assign(k=0).to_csv(out_dir / "per_group" / f"{run.name}__{n}{args.tag}.csv", index=False)
    res.to_csv(out_dir / "sweeps" / f"{run.name}{args.tag}.csv", index=False)
    short = {c: c.split("_")[0][:4] + c[-3:] for c in res.columns if c.endswith(("_s1", "_s2"))}
    print(res.rename(columns=short).to_string(index=False))
    print(f"[sweep] scored in {time.time() - t0:.0f} s")
    scorer.close()


if __name__ == "__main__":
    main()
