"""Per navhard token: the 256 refined anchors and their scores from the 5-member v22 ensemble.

    # (a) near log goal, ens-mode prob      (sweep reference: mix_0.03 = 45.99)
    python -m exp_navsim.orbis2.navhard_candidates --setting near
    # (b) null goal, ens-mode minmult       (sweep reference: mix_0.03 = 40.03)
    python -m exp_navsim.orbis2.navhard_candidates --setting null

Same code path as v22/sweep_rank.py with --ensemble (main run 24aa_goal_aug + 24ab, 24ai, 24aj,
24ak): every member's anchor head on the 256-anchor vocabulary, refined trajectories averaged
over the members, imitation = log of the mean softmax, log EPDMS from ens_combine (prob or
minmult), mix = log EPDMS + 0.03 * imitation. The argmax of mix is the sweep's mix_0.03 plan.

Output outputs/28_inputs/candidates/navhard_<setting>.npz:
    token (N,), row (N,) navhard index rows, in the order of sweep_rank (tokens of all groups)
    mix, ep, imi (N, 256)     mix score, ensemble log EPDMS, imitation log probability
    sub (N, 256, 8) float32   ensemble sub-score probabilities (fast_pdm.METRICS order; for
                              minmult the first 4 are the member minimum, as in ens_combine)
    traj (N, 256, 8, 3)       refined anchors (mean over members), the plans that would be driven
    top (N, 8)                anchor ids of the 8 best by mix, best first
"""

import argparse
import json
from pathlib import Path

import numpy as np

MAIN = "logs_exp/24aa_goal_aug"
MEMBERS = ("logs_exp/24ab_goal_aug_select", "logs_exp/24ai_aug_s1", "logs_exp/24aj_aug_s2",
           "logs_exp/24ak_select_s1")
SETTINGS = {"near": dict(goal_file="outputs/25_terra/navhard_goals_near.npz", null=False, mode="prob"),
            "null": dict(goal_file="outputs/25_terra/navhard_goals_near.npz", null=True, mode="minmult")}
BETA = 0.03
OUT = Path("outputs/28_inputs/candidates")


def ens_subscores(outs, mode):
    """Member outputs -> the (B, V, 8) sub-score probabilities that ens_combine turns into EPDMS."""
    import torch
    p = torch.stack([torch.sigmoid(o["scores"].float()) for o in outs])
    if mode == "minmult":
        return torch.cat([p[..., :4].min(0).values, p[..., 4:].mean(0)], -1)
    return p.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=sorted(SETTINGS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="first N tokens only (smoke test)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--main", default=MAIN, help="main run (its config gives the features)")
    ap.add_argument("--members", default=",".join(MEMBERS), help="comma-separated ensemble members")
    args = ap.parse_args()
    st = SETTINGS[args.setting]

    import torch
    from exp_navsim.data.scenes import SceneFeatureDataset
    from exp_navsim.v22.fast_epdms import all_groups, group_tokens
    from exp_navsim.v22.model import HORIZON
    from exp_navsim.v22.runs import feature_dir, feature_names
    from exp_navsim.v22.sweep_rank import ens_combine, load_planner
    from exp_navsim.v22.train import navhard_goals

    cfg = json.loads((Path(args.main) / "config.json").read_text())
    feats = feature_names(cfg)
    ds = SceneFeatureDataset("data/navsim_index/navhard.pkl", split="all", require_future=False,
                             features=feats, feature_dir=feature_dir(cfg), history=4)
    tokens = np.asarray(ds.index["token"])[ds.rows]
    need = set(group_tokens(all_groups()))
    sel = np.flatnonzero([t in need for t in tokens])
    if args.limit:
        sel = sel[:args.limit]
    rows = ds.rows[sel]
    dev = args.device
    b = {"features": torch.from_numpy(ds.features_of(rows)).float().to(dev),
         "future": torch.zeros(len(rows), HORIZON, 3, device=dev),
         "past": torch.from_numpy(ds.past[rows]).float().to(dev),
         "status": torch.from_numpy(ds.status[rows]).float().to(dev),
         "command": torch.from_numpy(ds.command[rows]).float().to(dev)}
    assert cfg.get("goal", "none") != "none"
    b["goal"] = navhard_goals(st["goal_file"], tokens[sel]).to(dev)
    if st["null"]:
        b["goal"][:] = float("nan")
    T, C = b["features"].shape[1:3]
    nets = [load_planner(r, dev, (C, T))[0] for r in [args.main] + [m for m in args.members.split(",") if m]]

    def anchor_out(net, bc):
        mem, ego, _ = net.scene(bc)
        out = {}
        for i in range(0, len(bc["status"]), 256):
            for k, val in net.anchors(mem[i:i + 256], ego[i:i + 256]).items():
                out.setdefault(k, []).append(val)
        return {k: torch.cat(v) for k, v in out.items()}

    parts = {k: [] for k in ("mix", "ep", "imi", "sub", "traj")}
    with torch.no_grad():
        for i in range(0, len(rows), 512):                   # the chunking of sweep_rank
            bc = {k: v[i:i + 512] for k, v in b.items()}
            outs = [anchor_out(net, bc) for net in nets]
            imi, ep = ens_combine(outs, st["mode"])
            parts["mix"].append(ep + BETA * imi)
            parts["ep"].append(ep), parts["imi"].append(imi)
            parts["sub"].append(ens_subscores(outs, st["mode"]))
            parts["traj"].append(torch.stack([o["traj"] for o in outs]).mean(0))
    res = {k: torch.cat(v).float().cpu().numpy() for k, v in parts.items()}
    res["top"] = np.argsort(-res["mix"], axis=1, kind="stable")[:, :8]
    out = Path(args.out) if args.out else OUT / f"navhard_{args.setting}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, token=tokens[sel], row=rows, setting=args.setting, mode=st["mode"], **res)
    print(f"{out}: {len(rows)} tokens, mix {res['mix'].shape}, traj {res['traj'].shape}")


if __name__ == "__main__":
    main()
