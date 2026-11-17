"""Samples and plans of FlowPlanner runs: EPDMS on the navtest 500 subset, distribution metrics on held-out logs.

    python -m exp_navsim.hybrid.evaluate --specs 19_flow,19_flow@raw,19_resid@irp1 --scorer logs_exp/19_scorer/scorer.pt

A spec is run[@raw][@irp<s>][@k<K>]: raw = the raw weights instead of the EMA, irp =
reference perturbation at sampling time (std s m/s), k = number of samples (default 32).
run = vocab<V> is the Hydra-MDP-style baseline: the candidates are the V k-means
trajectories of 16_fast_planner (vocab.py), the same for every scene, and the scorer
picks one of them.
For every scene of the navtest subset (and its previous frame, for two-frame comfort)
the run draws K samples (10 Euler steps) and one plan per rule:

    mean      mean of the samples
    ref       the regression output of the same model (modes regress, resid)
    medoid    the sample with the smallest mean ADE to the others
    scorer    the sample with the highest EPDMS predicted by --scorer (ego inputs, scorer.py)
    verifier  the same with the image-only verifier v3 of 18_verified_sampling
    oracle    the sample with the highest true EPDMS (privileged: map and agents; upper bound)

Rule plans and samples go through FastPDM in one call (official sub-scores and EPDMS,
two-frame comfort against the same rule / sample index of the previous frame). Per
sample: mean EPDMS, share of bad samples (EPDMS < 0.5), off-road (DAC = 0), at-fault
collision (NC = 0). On 2000 held-out-log scenes: energy, minADE, mean ADE, ADE of the
mean, spread, dispersion (eval/metrics.py).
-> outputs/19_hybrid/results.csv (one row per spec), per_scene/<label>.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.planner.fast_pdm import METRICS, FastPDM, load_subset

OUT = Path("outputs/19_hybrid")
SUBSET = "outputs/16_fast_planner/subset_500.json"
VERIFIER = "logs_exp/18_verifier_v3/verifier.pt"
RULES = ("mean", "ref", "medoid", "scorer", "verifier")


def parse_spec(spec):
    parts = spec.split("@")
    out = {"run": parts[0], "raw": False, "irp": 0.0, "k": 32}
    for p in parts[1:]:
        if p == "raw":
            out["raw"] = True
        elif p.startswith("irp"):
            out["irp"] = float(p[3:])
        elif p.startswith("k"):
            out["k"] = int(p[1:])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(OUT / "results.csv"))
    args = ap.parse_args()
    specs = args.specs.split(",")
    labels = args.labels.split(",") if args.labels else specs

    tokens, prev = load_subset(SUBSET)
    need = list(dict.fromkeys(tokens + list(prev.values())))
    scorer_pdm = FastPDM("navtest", need, workers=args.workers)

    import torch
    from exp_navsim.data.scenes import SceneFeatureDataset
    from exp_navsim.diverse.verifier import load_verifier
    from exp_navsim.eval.metrics import sample_metrics
    from exp_navsim.hybrid.scorer import load_scorer
    from exp_navsim.hybrid.train import batch_of, load_model, to_gpu

    common = dict(features=["wm11_ctx"], feature_dir="data/features", history=4)
    test_ds = SceneFeatureDataset("data/navsim_index/navtest.pkl", split="all", require_future=False, **common)
    pos = {t: r for r, t in zip(test_ds.rows, np.asarray(test_ds.index["token"])[test_ds.rows])}
    need = [t for t in need if t in pos]
    test = to_gpu(test_ds, np.array([pos[t] for t in need]))
    val_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="val", **common)
    val = to_gpu(val_ds, np.sort(np.random.default_rng(0).permutation(val_ds.rows)[:2000]))
    verifier = load_verifier(VERIFIER)
    scorer = load_scorer(args.scorer) if args.scorer else None
    cur = [j for j, t in enumerate(need) if t in prev]
    cur_tokens = [need[j] for j in cur]

    class Vocab:
        """The fixed vocabulary as a 'sampler': every scene gets all V trajectories."""
        mode = "vocab"

        def __init__(self, v):
            self.v = torch.from_numpy(np.load(f"outputs/16_fast_planner/vocab_{v}.npy")).float().cuda()

        def sample(self, b, K, steps, irp=0.0, generator=None):
            fut = self.v[:, None].expand(-1, len(b["row"]), -1, -1)
            return fut, fut.mean(0)

    def draw(model, data, K, irp, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        fut, ref = [], []
        for i in range(0, len(data["row"]), 50):
            b = batch_of(data, torch.arange(i, min(i + 50, len(data["row"])), device="cuda"))
            f, r = model.sample(b, K, args.steps, irp=irp, generator=g)
            fut.append(f)
            ref.append(r if r is not None else f.mean(0))
        return torch.cat(fut, 1), torch.cat(ref)

    out_rows = []
    path = Path(args.out)
    per_scene = path.parent / "per_scene"
    per_scene.mkdir(parents=True, exist_ok=True)
    for spec, label in zip(specs, labels):
        s = parse_spec(spec)
        if s["run"].startswith("vocab"):
            model, a = Vocab(int(s["run"][5:])), {"mode": "vocab"}
            K = len(model.v)
        else:
            model, a = load_model(Path("logs_exp") / s["run"] / "model.pt", raw=s["raw"])
            K = 1 if a["mode"] == "regress" else s["k"]

        # ------------------------------------------------------------ navtest -- #
        torch.cuda.synchronize()
        t0 = time.time()
        fut, ref = draw(model, test, K, s["irp"])
        torch.cuda.synchronize()
        ms = (time.time() - t0) / len(need) * 1000
        with torch.no_grad():
            feats = test["features"].float()
            xy = fut[..., :2]
            medoid = torch.cat([(xy[:, None, i:i + 25] - xy[None, :, i:i + 25]).norm(dim=-1).mean(-1).mean(1).argmin(0)
                                for i in range(0, xy.shape[1], 25)])
            mean = fut.mean(0)
            mean[..., 2] = torch.atan2(torch.sin(fut[..., 2]).mean(0), torch.cos(fut[..., 2]).mean(0))
            pick = lambda idx: fut[idx, torch.arange(fut.shape[1], device="cuda")]
            pv = verifier.predict_epdms({"features": feats}, fut)
            if scorer is not None:
                b = {k: (v.float() if k == "features" else v) for k, v in test.items() if k != "row"}
                ps = scorer.predict_epdms(b, fut)
            else:
                ps = pv
            plans = torch.stack([mean, ref, pick(medoid), pick(ps.argmax(0)), pick(pv.argmax(0))])
            cands = torch.cat([plans, fut]).transpose(0, 1).cpu().numpy()                # (N, R + K, 8, 3)
        t1 = time.time()
        df = scorer_pdm.score({t: cands[j] for j, t in enumerate(need)}, prev)
        R = len(RULES)
        sc = df.pivot(index="token", columns="k", values="score").loc[cur_tokens].to_numpy()
        samp = df[df.k >= R]
        row = {"label": label, "spec": spec, "mode": a["mode"], "K": K, "ms_per_scene": round(ms, 2)}
        for k, rule in enumerate(RULES):
            row[f"plan_{rule}"] = sc[:, k].mean()
        row["plan_oracle"] = sc[:, R:].max(1).mean()
        row["plan_oracle8"] = sc[:, R:R + 8].max(1).mean()
        row["sample_epdms"] = samp.score.mean()
        row["sample_bad"] = (samp.score < 0.5).mean()
        row["sample_offroad"] = (samp.drivable_area_compliance == 0).mean()
        row["sample_collision"] = (samp.no_at_fault_collisions == 0).mean()
        for m in ("ego_progress", "two_frame_extended_comfort", "history_comfort"):
            row[f"mean_{m}"] = df[df.k == 0][m].mean()

        # ---------------------------------------------------------------- val -- #
        if a["mode"] != "vocab":                     # a fixed vocabulary has no distribution to measure
            vf, vr = draw(model, val, K, s["irp"])
            vm = sample_metrics(vf, val["future"])
            for k in ("energy", "ade_min", "ade_mean", "ade_of_mean", "spread", "dispersion"):
                row[f"val_{k}"] = vm[k].mean().item()
            row["val_l2_4s_ref"] = (vr[:, -1, :2] - val["future"][:, -1, :2]).norm(dim=-1).mean().item()
            row["val_l2_4s_mean"] = (vf.mean(0)[:, -1, :2] - val["future"][:, -1, :2]).norm(dim=-1).mean().item()
        by_token = df.set_index(["token", "k"]).sort_index()
        sub = np.stack([by_token.loc[t][list(METRICS)].to_numpy() for t in cur_tokens])
        np.savez_compressed(per_scene / f"{label}.npz", tokens=np.array(cur_tokens),
                            plans=cands[cur][:, :R].astype(np.float16), samples=cands[cur][:, R:].astype(np.float16),
                            score=sc.astype(np.float32), sub=sub.astype(np.float16))
        out_rows.append(row)
        print(f"{label:28s} " + "  ".join(f"{k} {v:.3f}" for k, v in row.items() if isinstance(v, float))
              + f"  (pdm {time.time() - t1:.0f}s)", flush=True)
        old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["label"])
        pd.concat([old[~old.label.isin([label])], pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    scorer_pdm.close()
    print(f"-> {path}")


if __name__ == "__main__":
    main()
