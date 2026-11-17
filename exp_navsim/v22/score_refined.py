"""PDM sub-scores of a trained planner's refined anchors on the navtrain score-target scenes.

The metric head of the v22 planner is trained on the sub-scores of the 256 raw anchors
(planner/score_vocab.py), but the plan it selects is the refined anchor (anchor + offset
head). This script runs a trained planner on the scenes that have raw-anchor targets, takes
its 256 refined anchors per scene (logged end-pose goal, as in training without goal
augmentation) and scores them with FastPDM (non-reactive, same as score_vocab.py). The
result is a second training set of (trajectory, sub-scores) pairs for the scorer that lies
on the planner's own output distribution (train.py --refined-targets).

    python -m exp_navsim.v22.score_refined logs_exp/24aa_goal_aug \
        --targets outputs/16_fast_planner/vocab256_scores_navtrain_sub.npz:navtrain_sub,\
outputs/16_fast_planner/vocab256_scores_navtrain_sub2.npz:navtrain_sub2 \
        --out outputs/26_onpolicy/refined_24aa.npz

Writes tokens, rows, split, traj (N, 256, 8, 3) and scores (N, 256, 8) in METRICS order.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from exp_navsim.planner.fast_pdm import METRICS, FastPDM, epdms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--targets", required=True, help="comma-separated raw-target npz:metric-cache-split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--max-scenes", type=int, default=0, help="per split, for a smoke test")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    sets = []
    for item in args.targets.split(","):
        f, split = item.split(":")
        d = np.load(f)
        n = args.max_scenes or len(d["rows"])
        sets.append((split, list(d["tokens"][:n]), np.asarray(d["rows"][:n]), d["scores"][:n]))

    # scoring pools fork before CUDA is touched
    scorers = {split: FastPDM(split, toks, workers=args.workers // len(sets), preload=False)
               for split, toks, _, _ in sets}

    import torch
    from exp_navsim.data.scenes import SceneFeatureDataset
    from exp_navsim.v22.sweep_rank import load_planner

    cfg = json.loads((Path(args.run) / "config.json").read_text())
    feats = [f for f in cfg["features"].split(",") if f]
    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="train", features=feats,
                             feature_dir="data/features", history=4)
    dev = "cuda"

    def batch(rows):
        return {"features": torch.from_numpy(ds.features_of(rows)).float().to(dev),
                "future": torch.from_numpy(np.nan_to_num(ds.future[rows])).float().to(dev),
                "goal": torch.from_numpy(ds.future[rows][:, -1]).float().to(dev),
                "past": torch.from_numpy(ds.past[rows]).float().to(dev),
                "status": torch.from_numpy(ds.status[rows]).float().to(dev),
                "command": torch.from_numpy(ds.command[rows]).float().to(dev)}

    m = None
    all_tok, all_rows, all_split, all_traj, all_sc = [], [], [], [], []
    t0 = time.time()
    for split, toks, rows, raw in sets:
        trajs = []
        with torch.no_grad():
            for i in range(0, len(rows), 256):
                b = batch(rows[i:i + 256])
                if m is None:
                    T, C = b["features"].shape[1:3]
                    m, _ = load_planner(args.run, dev, (C, T))
                if m.goal_in is None:
                    b.pop("goal")
                mem, ego, _ = m.scene(b)
                trajs.append(m.anchors(mem, ego)["traj"].float().cpu().numpy())
        trajs = np.concatenate(trajs)                                            # (N, V, 8, 3)
        off = np.linalg.norm(trajs[..., :2] - m.vocab.cpu().numpy()[None, ..., :2], axis=-1)
        print(f"[{split}] {len(rows)} scenes refined; offset mean {off.mean():.2f} m, "
              f"at 4 s {off[..., -1].mean():.2f} m", flush=True)
        sc = []
        for i in range(0, len(toks), args.chunk):
            chunk = toks[i:i + args.chunk]
            res = scorers[split].candidate_scores({t: trajs[i + j] for j, t in enumerate(chunk)})
            sc.append(np.stack([res[t] for t in chunk]).astype(np.float16))
            done = i + len(chunk)
            rate = done / (time.time() - t0)
            print(f"[{split}] {done}/{len(toks)}  {rate:.1f} scenes/s", flush=True)
        sc = np.concatenate(sc)
        e_raw, e_ref = epdms(raw.astype(np.float64)), epdms(sc.astype(np.float64))
        print(f"[{split}] EPDMS without two-frame comfort, raw / refined: mean over anchors "
              f"{e_raw.mean():.4f} / {e_ref.mean():.4f}, best per scene {e_raw.max(1).mean():.4f} / "
              f"{e_ref.max(1).mean():.4f}; refined better than its raw anchor in "
              f"{(e_ref > e_raw + 1e-6).mean():.1%}, worse in {(e_ref < e_raw - 1e-6).mean():.1%}", flush=True)
        all_tok += toks
        all_rows.append(rows)
        all_split += [split] * len(toks)
        all_traj.append(trajs.astype(np.float16))
        all_sc.append(sc)
        scorers[split].close()
    np.savez(out, tokens=np.array(all_tok), rows=np.concatenate(all_rows), split=np.array(all_split),
             traj=np.concatenate(all_traj), scores=np.concatenate(all_sc))
    print(f"-> {out} ({time.time() - t0:.0f} s)")
    for j, name in enumerate(METRICS):
        print(f"  {name:32s} refined mean {np.concatenate(all_sc)[..., j].astype(np.float32).mean():.3f}")


if __name__ == "__main__":
    main()
