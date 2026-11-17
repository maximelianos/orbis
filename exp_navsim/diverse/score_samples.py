"""How good is every single sample? PDM sub-scores of all K samples per navtest scene.

    python -m exp_navsim.diverse.score_samples --panels 17_full_h64_full/step0100000,17_ctrl_full/step0100000 \
        --labels full16,ctrl --out outputs/17_diversity/navtest_samples

The usual EPDMS scores one trajectory per scene. Here each of the K samples is a
candidate of its own (fast_pdm.FastPDM.candidate_scores on the 500-scene navtest subset;
no two-frame comfort, it pairs consecutive frames of one planner). Per model:

    sample_epdms    mean EPDMS over all samples: how good a random candidate is
    bad             share of samples with EPDMS < 0.5
    dac_fail        share of samples that leave the drivable area
    nc_fail         share of samples with an at-fault collision
    ddc_fail        share of samples driving against the traffic direction
    best_of_k       EPDMS of the best sample per scene (oracle selection)
    mean_traj       EPDMS of the sample mean (one trajectory)
    comfort_any     share of samples breaking a comfort bound (controls.comfort_flags)
    spread          mean pairwise ADE between samples (diversity)
    val/...         energy, minADE, spread, dispersion on 2000 held-out navtrain scenes
                    (logged futures there, no PDM): is the distribution still calibrated?

With --verifier, oversample * K samples are drawn and the verifier keeps the best K per
scene (verifier.filtered_sample); the label gets the suffix "+v".

and the same for the logged future (human) and constant velocity as references.
-> <out>/samples.csv, <out>/per_sample_<label>.npz (sub-scores, (K, N, metrics))
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exp_navsim.data.scenes import SceneFeatureDataset
from exp_navsim.diverse.controls import comfort_flags
from exp_navsim.diverse.plot_bev import load_model
from exp_navsim.diverse.train import batch_of, to_gpu
from exp_navsim.diverse.verifier import filtered_sample, load_verifier
from exp_navsim.eval.metrics import pairwise_mean, sample_metrics
from exp_navsim.eval.navsim_agent import mean_trajectory
from exp_navsim.planner.fast_pdm import METRICS, FastPDM, epdms, load_subset

SUBSET = Path("outputs/16_fast_planner/subset_500.json")


def summarize(sub, poses, v0):
    """sub (K, N, metrics) numpy, poses (K, N, 8, 3) numpy, v0 (N,) -> dict."""
    score = epdms(sub)
    m = lambda name: sub[..., METRICS.index(name)]
    t = torch.from_numpy(poses).float()
    flags = comfort_flags(t, torch.from_numpy(v0).float().expand(len(t), -1))
    out = {"sample_epdms": score.mean(), "bad": (score < 0.5).mean(), "dac_fail": (m("drivable_area_compliance") == 0).mean(),
           "nc_fail": (m("no_at_fault_collisions") == 0).mean(), "ddc_fail": (m("driving_direction_compliance") < 1).mean(),
           "best_of_k": score.max(0).mean(), "comfort_any": flags["any"].float().mean().item()}
    if len(t) > 1:
        out["spread"] = pairwise_mean(t[..., :2], lambda a, b: (a - b).norm(dim=-1).mean(-1)).mean().item()
    return out, score


def save(results, path):
    """Rows of this call replace rows of the same model in an existing csv."""
    df = pd.DataFrame(results)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old.model.isin(df.model)], df], ignore_index=True)
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", required=True, help="comma-separated run/checkpoint stems under logs_exp")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--mode", default="prop", help="prop, strat or nucleus<p>, e.g. nucleus0.8")
    ap.add_argument("--no-reference", action="store_true", help="skip human and constant velocity")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verifier", default=None, help="logs_exp/<run>/verifier.pt")
    ap.add_argument("--oversample", type=int, default=4)
    ap.add_argument("--accept", type=float, default=None, help="rejection threshold on predicted EPDMS; default top-K")
    ap.add_argument("--suffix", default=None, help="label suffix; default +v (top-K) or +r<accept>")
    ap.add_argument("--out", default="outputs/17_diversity/navtest_samples")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panels = args.panels.split(",")
    labels = args.labels.split(",") if args.labels else panels

    tokens, _ = load_subset(SUBSET)
    first = json.loads((Path("logs_exp") / panels[0].split("/")[0] / "config.json").read_text())
    ds = SceneFeatureDataset("data/navsim_index/navtest.pkl", features=first["features"].split(","), split="all",
                             require_future=False, feature_dir="data/features", history=4)
    pos = {t: r for r, t in zip(ds.rows, np.asarray(ds.index["token"])[ds.rows])}
    rows = np.array([pos[t] for t in tokens])
    data = to_gpu(ds, rows)
    data["future"] = torch.nan_to_num(data["future"])
    v0 = np.linalg.norm(ds.status[rows][:, :2], axis=1)
    val_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=first["features"].split(","), split="val",
                                 feature_dir="data/features", history=4)
    val = to_gpu(val_ds, np.sort(np.random.default_rng(0).permutation(val_ds.rows)[:2000]))
    verifier = load_verifier(args.verifier) if args.verifier else None
    suffix = "" if verifier is None else "+v" if args.accept is None else f"+r{args.accept:g}"
    suffix = suffix if args.suffix is None else args.suffix

    scorer = FastPDM("navtest", tokens, workers=args.workers)

    def score(poses):
        """(K, N, 8, 3) -> sub-scores (K, N, metrics)."""
        d = scorer.candidate_scores({t: poses[:, i].astype(np.float32) for i, t in enumerate(tokens)})
        return np.stack([d[t] for t in tokens], 1)

    ref = scorer.reference_poses(tokens)
    human = np.stack([ref[t][1] for t in tokens])[None]
    tt = 0.5 * np.arange(1, 9)
    cv = np.stack([np.stack([s * tt, np.zeros(8), np.zeros(8)], 1) for s in v0])[None]

    results = []
    for label, poses in () if args.no_reference else (("human", human), ("constant_velocity", cv)):
        r, _ = summarize(score(poses), poses, v0)
        results.append({"model": label, **r})
        print(label, {k: round(float(v), 3) for k, v in r.items()}, flush=True)
    for label, p in zip(labels, panels):
        run, ck = p.split("/")
        model, _ = load_model(Path("logs_exp") / run / "checkpoints" / f"{ck}.pt")
        label = label + suffix
        g = torch.Generator(device="cuda").manual_seed(0)

        def sample(data, chunk=125):
            out = []
            for i in range(0, len(data["row"]), chunk):
                b = batch_of(data, torch.arange(i, min(i + chunk, len(data["row"])), device="cuda"))
                out.append(filtered_sample(model, verifier, b, args.samples, args.steps, args.mode, args.oversample, g,
                                           args.accept))
            return torch.cat(out, 1)

        fut = sample(data).cpu().numpy()
        sub = score(fut)
        r, s = summarize(sub, fut, v0)
        r["mean_traj"] = epdms(score(mean_trajectory(fut)[None])).mean()
        vm = {k: v.mean().item() for k, v in sample_metrics(sample(val), val["future"]).items()}
        r.update({f"val/{k}": vm[k] for k in ("energy", "ade_min", "spread")})
        r["val/dispersion"] = vm["spread"] / vm["ade_mean"]
        results.append({"model": label, "panel": p, "mode": args.mode, **r})
        np.savez_compressed(out / f"per_sample_{label}.npz", token=np.array(tokens), sub=sub.astype(np.float32),
                            score=s.astype(np.float32), samples=fut.astype(np.float16))
        print(label, {k: round(float(v), 3) for k, v in r.items() if k != "panel"}, flush=True)
        save(results, out / "samples.csv")
    scorer.close()
    print(f"-> {out / 'samples.csv'}")


if __name__ == "__main__":
    main()
