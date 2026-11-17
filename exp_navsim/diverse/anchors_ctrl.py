"""Anchors in control space (controls.py) and the trajectory vocabulary of VocabDist.

    python -m exp_navsim.diverse.anchors_ctrl --joint 16,32 --product 8x4,8x8 --vocab 1024,4096

joint     k-means on the standardized (acceleration, yaw rate) sequences -> (M, 8, 2)
product   k-means on the yaw-rate sequences (path shape, M_p centres) and separately on the
          acceleration sequences (speed profile, M_s), every pair is an anchor
          -> (M_p * M_s, 8, 2). Directions and speed profiles vary independently, so the
          top anchors of a scene are not all longitudinal variants of one direction.
vocab     farthest-point sampling of logged training futures (VADv2) -> (V, 8, 3)
ctrlvocab farthest-point sampling of the standardized, clipped control sequences -> (V, 8, 2)

Controls are clipped to [-6, 4] m/s2 and +-1 rad/s before clustering. Only training logs; the validation logs give the oracle ADE to the nearest anchor (rolled
out from the logged speed). -> outputs/17_diversity/anchors/, figure anchors_ctrl.png
with the anchors rolled out at 3 and 10 m/s.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.diverse.anchors import OUT, kmeans
from exp_navsim.diverse.controls import from_controls, speed_of, to_controls
from exp_navsim.diverse.model import farthest_point_vocab


def farthest_points(x, num, seed=0):
    """Farthest-point sampling, Euclidean, x (N, d) -> indices (num,)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    idx = [torch.randint(len(x), (1,), generator=g, device=x.device).item()]
    d = torch.full((len(x),), float("inf"), device=x.device)
    for _ in range(num - 1):
        d = torch.minimum(d, (x - x[idx[-1]]).norm(dim=-1))
        idx.append(d.argmax().item())
    return torch.tensor(idx, device=x.device)


def oracle_ade(anchor_poses, future):
    """anchor_poses (B, M, 8, >=2), future (B, 8, >=2) -> ADE to the nearest anchor (B,)."""
    return (anchor_poses[..., :2] - future[:, None, :, :2]).norm(dim=-1).mean(-1).min(1).values


def ctrl_oracle(anchors, future, status):
    v0 = speed_of(status)
    out = [oracle_ade(from_controls(anchors[None].expand(len(f), -1, -1, -1), v[:, None].expand(-1, len(anchors))), f)
           for f, v in zip(future.split(1024), v0.split(1024))]
    return torch.cat(out).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", default="16,32")
    ap.add_argument("--product", default="8x4,8x8")
    ap.add_argument("--vocab", default="1024,4096")
    ap.add_argument("--ctrl-vocab", default="1024,4096")
    args = ap.parse_args()

    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=[], split="train")
    val = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=[], split="val")
    fut = lambda d: torch.from_numpy(d.future[d.rows]).float().cuda()
    stat = lambda d: torch.from_numpy(d.status[d.rows]).float().cuda()
    f, s, vf, vs = fut(ds), stat(ds), fut(val), stat(val)
    ctrl = to_controls(f, speed_of(s))
    # a few logs have acceleration spikes of 20 m/s2 (localization jumps); they would get centres of their own
    ctrl = torch.stack([ctrl[..., 0].clamp(-6, 4), ctrl[..., 1].clamp(-1, 1)], -1)
    std = ctrl.flatten(0, 1).std(0)                                  # per channel
    print(f"control std: acceleration {std[0]:.3f} m/s2, yaw rate {std[1]:.3f} rad/s")
    OUT.mkdir(parents=True, exist_ok=True)

    sets = {}
    for m in [int(x) for x in args.joint.split(",") if x]:
        c, _ = kmeans((ctrl / std).flatten(1), m)
        sets[f"ctrl_{m}"] = c.view(m, 8, 2) * std
    for pm in [x for x in args.product.split(",") if x]:
        mp, ms = (int(x) for x in pm.split("x"))
        cp, _ = kmeans(ctrl[..., 1], mp)
        cs, _ = kmeans(ctrl[..., 0], ms)
        sets[f"ctrlprod_{pm}"] = torch.stack([torch.stack([a, w], -1) for w in cp for a in cs])
    for name, a in sets.items():
        np.save(OUT / f"anchors_{name}.npy", a.cpu().numpy())
        print(f"{name}: {len(a)} anchors, val oracle ADE {ctrl_oracle(a, vf, vs):.3f} m", flush=True)
    for v in [int(x) for x in args.vocab.split(",") if x]:
        voc = farthest_point_vocab(f, v)
        np.save(OUT / f"vocab_fps_{v}.npy", voc.cpu().numpy())
        d = torch.cat([oracle_ade(voc[None, ..., :2].expand(len(x), -1, -1, -1), x) for x in vf.split(256)])
        print(f"vocab_fps_{v}: val oracle ADE {d.mean():.3f} m", flush=True)

    for v in [int(x) for x in args.ctrl_vocab.split(",") if x]:
        voc = ctrl[farthest_points((ctrl / std).flatten(1), v)]
        np.save(OUT / f"ctrlvocab_fps_{v}.npy", voc.cpu().numpy())
        print(f"ctrlvocab_fps_{v}: val oracle ADE {ctrl_oracle(voc, vf, vs):.3f} m", flush=True)

    names = list(sets)
    fig, axes = plt.subplots(2, len(names), figsize=(3.6 * len(names), 7), squeeze=False)
    for j, name in enumerate(names):
        a = sets[name]
        for i, v in enumerate((3.0, 10.0)):
            p = from_controls(a, torch.full((len(a),), v, device=a.device)).cpu().numpy()
            ax = axes[i, j]
            for t in p:
                ax.plot(np.r_[0, t[:, 1]], np.r_[0, t[:, 0]], lw=0.8)
            ax.set_aspect("equal")
            ax.invert_xaxis()
            ax.set_title(f"{name}, rolled out at {v:.0f} m/s", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "anchors_ctrl.png", dpi=120)


if __name__ == "__main__":
    main()
