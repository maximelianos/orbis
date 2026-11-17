"""Sampled futures of IntentFlow runs on the NAVSIM bird's-eye view (map + agents).

    python -m exp_navsim.diverse.plot_bev --out outputs/17_diversity/bev_10k.png \
        --panels 17_plain_h64_10k/step0005000,17_plain_h64_10k/step0100000,17_full_h64_10k/step0100000

One row per validation scene: the current front image, then one BEV per panel
(run/checkpoint) with the K samples coloured by their intent anchor, the logged future
in black. Scenes are picked among those whose logged future turns or stops (seeded, the
same scenes for every call). The map and the boxes come from navsim's SceneLoader.

A panel is run/checkpoint[@mode[@tau]]: mode prop / strat / nucleus0.8 (default --mode),
tau a rejection threshold of the verifier (--verifier, verifier.filtered_sample): the
kept samples are coloured, the rejected ones of the same draw are thin grey dashes.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from exp_navsim.data.resize_frames import frame_file
from exp_navsim.data.scenes import COMMANDS, SceneFeatureDataset
from exp_navsim.diverse.train import batch_of, build_model, to_gpu
from exp_navsim.diverse.verifier import load_verifier
from exp_navsim.viz.plot_samples import pick_scenes

LOGS = Path("navsim/download/trainval_navsim_logs/trainval")
SENSORS = Path("navsim/download/trainval_sensor_blobs/trainval")


def load_model(path):
    ck = torch.load(path, map_location="cuda")
    a = ck["args"]
    anchors = ck["model"]["anchors"]
    model = build_model(a, ck["in_channels"], ck["num_frames"], anchors if len(anchors) > 1 else None).cuda()
    model.load_state_dict(ck["model"])
    return model.eval(), a


def load_scenes(tokens, logs):
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    loader = SceneLoader(data_path=LOGS, original_sensor_path=SENSORS,
                         scene_filter=SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                                                  log_names=sorted(set(logs)), tokens=list(tokens)),
                         sensor_config=SensorConfig.build_no_sensors())
    return {t: loader.get_scene_from_token(t) for t in tokens}


def draw(ax, scene, samples, anchor_ids, gt, num_anchors, rejected=None):
    from navsim.visualization.bev import add_configured_bev_on_ax
    from navsim.visualization.plots import configure_ax, configure_bev_ax
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[3])
    configure_bev_ax(ax)
    configure_ax(ax)
    cmap = plt.get_cmap("tab20" if num_anchors > 10 else "tab10")
    for s in [] if rejected is None else rejected:
        ax.plot(np.r_[0, s[:, 1]], np.r_[0, s[:, 0]], color="0.35", lw=0.7, ls="--", alpha=0.7, zorder=4)
    for s, k in zip(samples, anchor_ids):
        c = cmap(k % cmap.N) if num_anchors > 1 else "tab:red"
        ax.plot(np.r_[0, s[:, 1]], np.r_[0, s[:, 0]], color=c, lw=1.0, alpha=0.6, zorder=5)
        ax.scatter(s[-1, 1], s[-1, 0], color=c, s=8, zorder=5)
    ax.plot(np.r_[0, gt[:, 1]], np.r_[0, gt[:, 0]], color="black", lw=2.2, zorder=6)
    pts = np.concatenate([samples[:, :, :2].reshape(-1, 2), gt[:, :2], np.zeros((1, 2))])
    lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
    half = max(12.0, (hi - lo).max() / 2 + 6)
    cx, cy = (lo + hi) / 2
    ax.set_xlim(cy + half, cy - half)                 # screen x = ego y, left on the left
    ax.set_ylim(cx - half, cx + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", required=True, help="comma-separated run/checkpoint stems under logs_exp")
    ap.add_argument("--labels", default=None, help="comma-separated column titles")
    ap.add_argument("--mode", default="prop", choices=["prop", "strat"])
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verifier", default="logs_exp/17_verifier/verifier.pt")
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    panels = args.panels.split(",")
    labels = args.labels.split(",") if args.labels else panels
    first = json.loads((Path("logs_exp") / panels[0].split("/")[0] / "config.json").read_text())
    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=first["features"].split(","), split="val")
    pos = pick_scenes(ds, args.num, args.seed)
    rows = ds.rows[pos]
    data = to_gpu(ds, rows)
    batch = batch_of(data, torch.arange(len(rows), device="cuda"))
    tokens = [ds.index["token"][r] for r in rows]
    scenes = load_scenes(tokens, [ds.index["log"][r] for r in rows])

    preds = []
    verifier = None
    for p in panels:
        parts = p.split("@")
        stem, mode, tau = parts[0], parts[1] if len(parts) > 1 else args.mode, parts[2] if len(parts) > 2 else None
        model, _ = load_model(Path("logs_exp") / stem.split("/")[0] / "checkpoints" / f"{stem.split('/')[1]}.pt")
        g = torch.Generator(device="cuda").manual_seed(0)
        n = args.samples * (args.oversample if tau else 1)
        fut, ks, kin, _ = model.sample(batch, n, args.steps, mode=mode, generator=g)
        rejected = None
        if tau:
            verifier = verifier or load_verifier(args.verifier)
            ok = (verifier.predict_epdms({"features": batch["features"]}, fut) >= float(tau)).cpu().numpy()  # (N, B)
            f, k = fut.cpu().numpy(), ks.cpu().numpy()
            # per scene: the first K accepted in draw order (filled up with rejected ones), the rejected ones grey
            keep = [np.r_[np.flatnonzero(ok[:, i]), np.flatnonzero(~ok[:, i])][:args.samples] for i in range(f.shape[1])]
            rejected = [f[np.flatnonzero(~ok[:, i])[:args.samples], i] for i in range(f.shape[1])]
            fut = np.stack([f[keep[i], i] for i in range(f.shape[1])], 1)
            ks = np.stack([k[keep[i], i] for i in range(f.shape[1])], 1)
            preds.append((fut, ks, None if kin is None else kin.cpu().numpy(), model.num_anchors, rejected))
            continue
        preds.append((fut.cpu().numpy(), ks.cpu().numpy(), None if kin is None else kin.cpu().numpy(),
                      model.num_anchors, None))
    gt = data["future"].cpu().numpy()

    cols = 1 + len(panels)
    fig, axes = plt.subplots(args.num, cols, figsize=(3.4 * cols, 3.4 * args.num), squeeze=False)
    for i, r in enumerate(rows):
        ax = axes[i, 0]
        ax.imshow(Image.open(frame_file(ds.index["split"], ds.index["frame_ids"][r][-1])))
        cmd = COMMANDS[int(ds.command[r].argmax())]
        ax.set_title(f"cmd {cmd}, v = {ds.status[r][0]:.1f} m/s", fontsize=9)
        ax.axis("off")
        for j, (fut, ks, kin, M, rejected) in enumerate(preds):
            ax = axes[i, j + 1]
            draw(ax, scenes[tokens[i]], fut[:, i], ks[:, i], gt[i], M, None if rejected is None else rejected[i])
            sub = f"{labels[j]}\n" if i == 0 else ""
            sub += f"{len(np.unique(ks[:, i]))} intents" if M > 1 else ""
            sub += f"  v_hat {kin[i, 0]:.1f}" if kin is not None else ""
            sub += f"  {len(rejected[i])} of {args.samples * args.oversample} rejected" if rejected is not None else ""
            ax.set_title(sub, fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
