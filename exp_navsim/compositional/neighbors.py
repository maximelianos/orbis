"""What the frozen features group by: city or maneuver? Nearest training neighbors of held-out scenes.

    python -m exp_navsim.compositional.neighbors --features wm11_ctx,vjepa2,dinov2,q_sem

For every (city, command) pair, up to 400 validation scenes (held-out logs) look up their
k = 20 nearest training scenes by cosine similarity of the last observed frame's feature
grid (C x 9 x 16, flattened, centered). Reported per pair: the share of neighbors from the
same city, with the same command, and from the same (city, command) pair, next to the
shares expected by chance (the training frequencies). An encoder that groups by maneuver
can transfer "left turn" across cities; one that groups by city cannot.
-> outputs/22_compositional/neighbors.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exp_navsim.compositional.sweep import CITY, COMMANDS, OUT
from exp_navsim.data.scenes import SceneFeatureDataset


def last_frame(ds, rows):
    x = np.concatenate([ds.features_of(rows[i:i + 8192]) for i in range(0, len(rows), 8192)])
    return torch.from_numpy(x[:, -1].reshape(len(rows), -1)).float()        # (N, C*9*16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="wm11_ctx,vjepa2,dinov2,q_sem")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--per-pair", type=int, default=400)
    args = ap.parse_args()
    rows_out = []
    for feat in args.features.split(","):
        common = dict(features=[feat], feature_dir="data/features", history=4)
        tr = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="train", **common)
        va = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="val", **common)
        xt = last_frame(tr, tr.rows)
        mu = xt.mean(0, keepdim=True)
        xt = torch.nn.functional.normalize(xt - mu, dim=1).half().cuda()
        city_t = np.asarray(tr.index["city"])[tr.rows]
        cmd_t = np.asarray(tr.index["command"])[tr.rows].argmax(1)
        city_v = np.asarray(va.index["city"])[va.rows]
        cmd_v = np.asarray(va.index["command"])[va.rows].argmax(1)
        rng = np.random.default_rng(0)
        for c in CITY:
            for m in (0, 1, 2):
                idx = np.flatnonzero((city_v == c) & (cmd_v == m))
                if len(idx) < 20:
                    continue
                idx = rng.permutation(idx)[:args.per_pair]
                q = last_frame(va, va.rows[idx])
                q = torch.nn.functional.normalize(q - mu, dim=1).half().cuda()
                nn_idx = (q @ xt.T).topk(args.k, dim=1).indices.cpu().numpy()       # (n, k)
                same_city = (city_t[nn_idx] == c).mean()
                same_cmd = (cmd_t[nn_idx] == m).mean()
                same_pair = ((city_t[nn_idx] == c) & (cmd_t[nn_idx] == m)).mean()
                rows_out.append({"features": feat, "pair": f"{CITY[c]}_{COMMANDS[m]}", "n": len(idx),
                                 "same_city": same_city, "same_command": same_cmd, "same_pair": same_pair,
                                 "chance_city": (city_t == c).mean(), "chance_command": (cmd_t == m).mean(),
                                 "chance_pair": ((city_t == c) & (cmd_t == m)).mean()})
                print(rows_out[-1], flush=True)
        del xt
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT / "neighbors.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
