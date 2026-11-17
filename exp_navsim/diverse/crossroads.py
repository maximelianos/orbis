"""Diversity of sampled futures at crossroads (junctions.py), per model.

    python -m exp_navsim.diverse.crossroads eval --panels 17_plain_h64_10k/step0100000,17_full_h64_10k/step0100000
    python -m exp_navsim.diverse.crossroads bev --panels ... --out outputs/17_diversity/crossroads/bev.png

Scenes: validation scenes whose first junction ahead offers left, straight and right
(the map decides, not the logged future). "taken" is the subset where the logged future
reaches one of the branches within 4 s (otherwise it stops or crawls in front of it).

Every sample is assigned to a branch class with junctions.classify. Per scene:
    branches@1     number of the three turn classes with at least one of the K samples
    branches@10%   ... with at least 10% of the samples
    p_taken        share of samples on the class the driver took (taken subset)
    off            share of samples that are on no branch (off the lanes)
    before         share of samples that stop before the branches separate
plus the sample metrics of eval/metrics.py on the same scenes.
-> outputs/17_diversity/crossroads/crossroads.csv, bev_*.png
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.diverse.junctions import CLASSES, OUT as JOUT, classify, is_crossroad
from exp_navsim.diverse.plot_bev import draw, load_model, load_scenes
from exp_navsim.diverse.train import batch_of, to_gpu
from exp_navsim.eval.metrics import sample_metrics

OUT = Path("outputs/17_diversity/crossroads")
TURN3 = ("left", "straight", "right")
COLORS = {"left": "tab:green", "straight": "tab:blue", "right": "tab:orange", "uturn": "tab:purple"}


def crossroad_rows(records):
    rows = np.array(sorted(r for r, x in records.items() if is_crossroad(x)))
    taken = np.array([records[r]["gt"] in TURN3 for r in rows])
    return rows, taken


def sample_model(model, data, num_samples, mode, seed=0, chunk=500):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    n = len(data["row"])
    for i in range(0, n, chunk):
        b = batch_of(data, torch.arange(i, min(i + chunk, n), device="cuda"))
        out.append(model.sample(b, num_samples, 20, mode=mode, generator=g)[0])
    return torch.cat(out, 1)                                            # (K, N, 8, 3)


def branch_stats(samples, rows, records):
    """samples (K, N, 8, 3) numpy -> per-scene dict of arrays."""
    K, N = samples.shape[:2]
    counts = np.zeros((N, len(CLASSES)))
    for j, r in enumerate(rows):
        br = records[r]["branches"]
        for k in range(K):
            counts[j, CLASSES.index(classify(samples[k, j], br))] += 1
    share = counts / K
    turn = share[:, [CLASSES.index(c) for c in TURN3]]
    gt = np.array([records[r]["gt"] for r in rows])
    p_taken = np.array([share[j, CLASSES.index(g)] if g in TURN3 else np.nan for j, g in enumerate(gt)])
    return {"branches@1": (turn > 0).sum(1), "branches@10%": (turn >= 0.1).sum(1), "p_taken": p_taken,
            "off": share[:, CLASSES.index("off")], "before": share[:, CLASSES.index("before")],
            **{f"share_{c}": share[:, CLASSES.index(c)] for c in TURN3}}


def evaluate(args):
    with open(args.records, "rb") as f:
        records = pickle.load(f)
    rows, taken = crossroad_rows(records)
    print(f"{len(rows)} crossroad scenes, {taken.sum()} where the logged future takes a branch")
    first = Path("logs_exp") / args.panels.split(",")[0].split("/")[0]
    import json
    feats = json.loads((first / "config.json").read_text())["features"].split(",")
    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=feats, split="val")
    data = to_gpu(ds, rows)
    gt_stats = branch_stats(data["future"].cpu().numpy()[None], rows, records)

    out = []
    labels = args.labels.split(",") if args.labels else args.panels.split(",")
    for p, lab in zip(args.panels.split(","), labels):
        run, ck = p.split("/")
        model, _ = load_model(Path("logs_exp") / run / "checkpoints" / f"{ck}.pt")
        for mode in args.modes.split(",") if model.num_anchors > 1 else ["prop"]:
            s = sample_model(model, data, args.samples, mode)
            m = {k: v.cpu().numpy() for k, v in sample_metrics(s, data["future"]).items()}
            st = branch_stats(s.cpu().numpy(), rows, records)
            row = {"model": lab, "mode": mode}
            for sub, mask in (("all", np.ones_like(taken)), ("taken", taken)):
                row[f"{sub}/energy"] = m["energy"][mask].mean()
                row[f"{sub}/ade_min"] = m["ade_min"][mask].mean()
                row[f"{sub}/dispersion"] = m["spread"][mask].mean() / m["ade_mean"][mask].mean()
                for k in ("branches@1", "branches@10%", "off", "before"):
                    row[f"{sub}/{k}"] = st[k][mask].mean()
            row["taken/p_taken"] = np.nanmean(st["p_taken"][taken])
            out.append(row)
            print("  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
                  flush=True)
    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(OUT / "crossroads.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(f"logged futures: taken {taken.mean():.2f}, branch of taken: "
          + ", ".join(f"{c} {np.mean([records[r]['gt'] == c for r in rows[taken]]):.2f}" for c in TURN3))


def bev(args):
    with open(args.records, "rb") as f:
        records = pickle.load(f)
    rows, taken = crossroad_rows(records)
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(rows[taken], args.num, replace=False))
    import json
    first = Path("logs_exp") / args.panels.split(",")[0].split("/")[0]
    feats = json.loads((first / "config.json").read_text())["features"].split(",")
    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=feats, split="val")
    data = to_gpu(ds, rows)
    tokens = [ds.index["token"][r] for r in rows]
    scenes = load_scenes(tokens, [ds.index["log"][r] for r in rows])
    panels = args.panels.split(",")
    labels = args.labels.split(",") if args.labels else panels
    preds = []
    for p in panels:
        run, ck = p.split("/")
        model, _ = load_model(Path("logs_exp") / run / "checkpoints" / f"{ck}.pt")
        s = sample_model(model, data, args.samples, args.modes.split(",")[0])
        preds.append(s.cpu().numpy())
    gt = data["future"].cpu().numpy()
    fig, axes = plt.subplots(args.num, len(panels), figsize=(3.6 * len(panels), 3.6 * args.num), squeeze=False)
    for i, r in enumerate(rows):
        rec = records[r]
        for j, s in enumerate(preds):
            ax = axes[i, j]
            cls = [classify(s[k, i], rec["branches"]) for k in range(s.shape[0])]
            draw(ax, scenes[tokens[i]], s[:, i], np.zeros(s.shape[0], int), gt[i], 1)
            for turn, pts in rec["branches"]:
                ax.plot(pts[:, 1], pts[:, 0], color=COLORS[turn], lw=4, alpha=0.25, zorder=4)
            ax.set_xlim(35, -35)
            ax.set_ylim(-10, 60)
            hist = ", ".join(f"{c[0].upper()} {cls.count(c)}" for c in TURN3 + ("before", "off") if cls.count(c))
            ax.set_title((f"{labels[j]}\n" if i == 0 else "") + f"gt {rec['gt']} | {hist}", fontsize=8)
    fig.suptitle("crossroads: samples red, logged future black, branches shaded (green L, blue S, orange R)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"-> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "bev"])
    ap.add_argument("--panels", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--modes", default="prop,strat")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--records", default=str(JOUT / "junctions_val.pkl"))
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "bev.png"))
    args = ap.parse_args()
    evaluate(args) if args.cmd == "eval" else bev(args)


if __name__ == "__main__":
    main()
