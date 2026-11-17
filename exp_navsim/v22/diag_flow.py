"""Why is the v22 flow plan worse than the regression? Per-t denoising error and samplers.

    python -m exp_navsim.v22.diag_flow logs_exp/24c_joint
"""

import sys
from pathlib import Path

import numpy as np
import torch

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.v22.model import GaragePlanner, action_targets, path_targets, scaled, unscaled


def main():
    run = Path(sys.argv[1])
    ck = torch.load(run / "model.pt", map_location="cuda", weights_only=False)
    a = ck["args"]
    from exp_navsim.v22.runs import build_planner, feature_dir, feature_names
    common = dict(features=feature_names(a), feature_dir=feature_dir(a), history=4)
    ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="val", **common)
    rows = np.sort(np.random.default_rng(0).permutation(ds.rows)[:2000])
    dev = "cuda"
    b = {"features": torch.from_numpy(ds.features_of(rows)).float().to(dev),
         "future": torch.from_numpy(np.nan_to_num(ds.future[rows])).float().to(dev),
         "past": torch.from_numpy(ds.past[rows]).float().to(dev),
         "status": torch.from_numpy(ds.status[rows]).float().to(dev),
         "command": torch.from_numpy(ds.command[rows]).float().to(dev),
         "goal": torch.from_numpy(ds.future[rows][:, -1]).float().to(dev)}     # goal runs: logged end
    T, C = b["features"].shape[1:3]
    vocab = None
    if a["vocab"]:
        vocab = np.load(f"outputs/16_fast_planner/vocab_{a['vocab']}.npy")
    m = build_planner(a, C, T, vocab).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    gt = b["future"]
    l2 = lambda p: (p[..., -1, :2] - gt[..., -1, :2]).norm(dim=-1).mean().item()
    with torch.no_grad():
        mem, ego, _ = m.scene(b)
        tgt = m.targets(b)
        x = m.to_z({g: tgt[g] for g in m.groups})
        print("per-t x0 error of the waypoints (L2 at 4 s, GT-noised input):")
        for t in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            tt = torch.full((len(x),), t, device=dev)
            torch.manual_seed(0)
            z = (1 - t) * x + t * torch.randn_like(x)
            out = m.from_z(m.pack({g: v.float() for g, v in m.denoise(mem, ego, z, tt).items()}))
            print(f"  t {t:.1f}  {l2(unscaled(out['wp'])):.3f}")
            if t == 1.0:
                z0 = torch.zeros_like(x)
                out = m.from_z(m.pack({g: v.float() for g, v in m.denoise(mem, ego, z0, tt).items()}))
                print(f"  t 1.0, z = 0  {l2(unscaled(out['wp'])):.3f}")
        for steps in (1, 2, 4, 8, 16):
            for temp in (0.0, 1.0):
                torch.manual_seed(0)
                z = m._sample(mem, ego, 8 if temp else 1, steps, temp)
                wp = unscaled(m.from_z(z.flatten(0, 1))["wp"]).unflatten(0, z.shape[:2])
                print(f"  steps {steps:2d} temp {temp}  first {l2(wp[0]):.3f}  mean-of-samples {l2(wp.mean(0)):.3f}")
        # same Euler loop, but the path / action part of every x0 is re-derived from its waypoints
        speed = b["status"][:, 0]
        for fix in ("path", "act", "both", "unmasked-path"):
            for steps in (4, 8):
                z = torch.zeros_like(x)
                ts = torch.linspace(1, 0, steps + 1, device=dev)
                for i in range(steps):
                    t = ts[i].expand(len(z))
                    x0 = m.pack({g: v.float() for g, v in m.denoise(mem, ego, z, t).items()})
                    parts = m.from_z(x0)
                    fut = unscaled(parts["wp"])
                    if fix in ("path", "both") and "path" in m.groups:
                        parts["path"] = path_targets(fut)[0].flatten(-2)
                    if fix == "unmasked-path" and "path" in m.groups:
                        pt, ok = path_targets(fut)
                        keep = ok[..., None].expand(-1, -1, 2).flatten(1)
                        parts["path"] = torch.where(keep, parts["path"], pt.flatten(-2))
                    if fix in ("act", "both") and "act" in m.groups:
                        parts["act"] = action_targets(fut, speed).flatten(-2)
                    x0 = m.to_z(parts)
                    z = z + (x0 - z) / t[:, None].clamp(min=1e-3) * (ts[i] - ts[i + 1])
                print(f"  fix {fix:13s} steps {steps}  {l2(unscaled(m.from_z(z)['wp'])):.3f}")


if __name__ == "__main__":
    main()
