"""Samples of several FlowPlanner runs on the NAVSIM bird's-eye view, same scenes for all.

    python -m exp_navsim.hybrid.plot_bev --specs 19_regress,19_flow,19_resid --out outputs/19_hybrid/bev.png \
        --scorer logs_exp/19_scorer/scorer.pt

One row per scene: front image, then one BEV per spec (evaluate.py spec syntax). Samples
are thin lines coloured by the scorer's predicted EPDMS (red = low, green = high), the
regression reference dashed blue, the scorer's pick thick green, the logged future black.
--split val: held-out navtrain logs with a turn or a stop (viz.plot_samples.pick_scenes).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from exp_navsim.data.resize_frames import frame_file
from exp_navsim.data.scenes import COMMANDS, SceneFeatureDataset
from exp_navsim.diverse.plot_bev import load_scenes
from exp_navsim.hybrid.evaluate import parse_spec
from exp_navsim.hybrid.scorer import load_scorer
from exp_navsim.hybrid.train import batch_of, load_model, to_gpu
from exp_navsim.viz.plot_samples import pick_scenes


def draw(ax, scene, samples, quality, ref, pick, gt):
    from navsim.visualization.bev import add_configured_bev_on_ax
    from navsim.visualization.plots import configure_ax, configure_bev_ax
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[3])
    configure_bev_ax(ax)
    configure_ax(ax)
    cmap = plt.get_cmap("RdYlGn")
    line = lambda s, **kw: ax.plot(np.r_[0, s[:, 1]], np.r_[0, s[:, 0]], **kw)
    for s, q in zip(samples, quality):
        line(s, color=cmap(q), lw=0.9, alpha=0.7, zorder=5)
    if ref is not None:
        line(ref, color="tab:blue", lw=1.6, ls="--", zorder=6)
    if pick is not None:
        line(pick, color="darkgreen", lw=2.2, zorder=7)
    line(gt, color="black", lw=2.0, zorder=6)
    pts = np.concatenate([samples[:, :, :2].reshape(-1, 2), gt[:, :2], np.zeros((1, 2))])
    lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
    half = max(12.0, (hi - lo).max() / 2 + 6)
    cx, cy = (lo + hi) / 2
    ax.set_xlim(cy + half, cy - half)
    ax.set_ylim(cx - half, cx + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    specs = args.specs.split(",")
    labels = args.labels.split(",") if args.labels else specs

    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=["wm11_ctx"], split="val")
    rows = ds.rows[pick_scenes(ds, args.num, args.seed)]
    data = to_gpu(ds, rows)
    batch = batch_of(data, torch.arange(len(rows), device="cuda"))
    tokens = [ds.index["token"][r] for r in rows]
    scenes = load_scenes(tokens, [ds.index["log"][r] for r in rows])
    scorer = load_scorer(args.scorer) if args.scorer else None

    preds = []
    for spec in specs:
        s = parse_spec(spec)
        model, a = load_model(Path("logs_exp") / s["run"] / "model.pt", raw=s["raw"])
        g = torch.Generator(device="cuda").manual_seed(0)
        fut, ref = model.sample(batch, 1 if a["mode"] == "regress" else s["k"], 10, irp=s["irp"], generator=g)
        q = scorer.predict_epdms(batch, fut) if scorer is not None else torch.ones(fut.shape[:2], device="cuda")
        pick = fut[q.argmax(0), torch.arange(fut.shape[1], device="cuda")] if scorer is not None else None
        preds.append((fut.cpu().numpy(), q.cpu().numpy(), None if ref is None else ref.cpu().numpy(),
                      None if pick is None else pick.cpu().numpy()))
    gt = data["future"].cpu().numpy()

    cols = 1 + len(specs)
    fig, axes = plt.subplots(args.num, cols, figsize=(3.4 * cols, 3.4 * args.num), squeeze=False)
    for i, r in enumerate(rows):
        ax = axes[i, 0]
        ax.imshow(Image.open(frame_file(ds.index["split"], ds.index["frame_ids"][r][-1])))
        ax.set_title(f"cmd {COMMANDS[int(ds.command[r].argmax())]}, v = {ds.status[r][0]:.1f} m/s", fontsize=9)
        ax.axis("off")
        for j, (fut, q, ref, pick) in enumerate(preds):
            ax = axes[i, j + 1]
            draw(ax, scenes[tokens[i]], fut[:, i], q[:, i], None if ref is None else ref[i],
                 None if pick is None else pick[i], gt[i])
            ax.set_title((f"{labels[j]}\n" if i == 0 else "") + f"pred. EPDMS {q[:, i].mean():.2f}", fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
