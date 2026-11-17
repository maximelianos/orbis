"""Top-k outputs of a trained planner (train.py) on a test split, for eval/pdm.py.

    python -m exp_navsim.planner.predict --run logs_exp/16_regress_wm11ctx_d256 --split navhard
    python -m exp_navsim.eval.pdm score --split navhard --predictions logs_exp/16_regress_wm11ctx_d256/navhard_top8.npz

Writes <run>/<split>_top8.npz (token, samples (8, N, 8, 3)), the same format as
navtest_top8.npz from train.py. navhard includes the synthetic second-stage scenes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.planner.model import Planner
from exp_navsim.planner.train import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="navhard")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    dev = "cuda"

    feats = [f for f in cfg["features"].split(",") if f]
    ds = SceneFeatureDataset(f"data/navsim_index/{args.split}.pkl", split="all", require_future=False,
                             features=feats, feature_dir="data/features", history=4)
    x = torch.from_numpy(np.concatenate([ds.features_of(ds.rows[i:i + 8192])
                                         for i in range(0, len(ds.rows), 8192)])).float()
    data = {"features": x,
            "past": torch.from_numpy(ds.past[ds.rows]).float(),
            "status": torch.from_numpy(ds.status[ds.rows]).float(),
            "command": torch.from_numpy(ds.command[ds.rows]).float()}

    vocab = np.load(OUT / f"vocab_{cfg['vocab']}.npy") if cfg["vocab"] else None
    T, C = x.shape[1:3] if feats else (1, 0)
    model = Planner(C, T, vocab=vocab, dim=cfg["dim"], depth=cfg["depth"], decoder_depth=cfg["decoder_depth"],
                    dropout=cfg["dropout"], use_image=bool(feats), use_status=not cfg["no_status"],
                    use_past=not cfg["no_past"], use_command=not cfg["no_command"],
                    score_head=bool(cfg.get("score_targets")), hybrid=cfg["hybrid"]).to(dev)
    model.load_state_dict(torch.load(run / "model.pt", map_location=dev)["model"])
    model.eval()
    weights = [float(w) for w in cfg["rank_weights"].split(",")] if cfg.get("rank_weights") else None

    out = []
    with torch.no_grad():
        for i in range(0, len(ds.rows), 512):
            batch = {k: v[i:i + 512].to(dev) for k, v in data.items()}
            out.append(model.predict(batch, args.k, weights).cpu())
    samples = torch.cat(out, 1).numpy()
    tokens = np.asarray(ds.index["token"])[ds.rows]
    path = run / f"{args.split}_top{args.k}.npz"
    np.savez_compressed(path, token=tokens, samples=samples.astype(np.float16))
    print(f"{len(tokens)} scenes -> {path}")


if __name__ == "__main__":
    main()
