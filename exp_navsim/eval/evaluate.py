"""Offline evaluation of probe checkpoints: sample metrics per checkpoint and split.

    python -m exp_navsim.eval.evaluate --run logs_exp/02_collapse --split val --ckpts all
    python -m exp_navsim.eval.evaluate --run logs_exp/06_dinov2 --split navtest --save-predictions

For every selected checkpoint, K futures are sampled for every scene of the split
and scored against the logged future (eval/metrics.py). One CSV row per checkpoint
goes to <run>/eval/<split>.csv with the dataset means, the NAVSIM-style L2 of the
mean prediction at 1, 2, 3 and 4 s, and the main metrics per driving command.
--save-predictions also writes <run>/eval/<split>_<ckpt>.npz with the tokens and
the sampled poses (K, N, 8, 3), which eval/pdm.py turns into EPDMS.

Splits: train_sub, val, holdout (the run's own datamodule, all scenes instead of the
validation subset) and the test indices navtest / navhard (features must be cached).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import BatchSampler, DataLoader, Subset

from exp_navsim.data.scenes import COMMANDS, SceneFeatureDataset
from exp_navsim.eval.metrics import sample_metrics
from exp_navsim.models.probe import TrajectoryProbe

MAIN = ("ade_mean", "ade_min", "fde_min", "energy", "spread")


def dataset_for(cfg, split, max_scenes=None, seed=0, holdout=None, cities=None, require_future=True):
    """holdout: (city, command) pairs replacing the run's own, e.g. to score a model
    trained on everything on the scenes another run held out. cities: replaces the
    run's city filter, e.g. to score a model on the city it never saw."""
    d = cfg.data
    holdout = holdout if holdout is not None else d.holdout
    cities = cities if cities is not None else d.cities
    common = dict(features=list(d.features), feature_dir=d.feature_dir, history=d.history, channels=d.channels,
                  val_fraction=d.val_fraction)
    if split in ("navtest", "navhard"):
        ds = SceneFeatureDataset(f"data/navsim_index/{split}.pkl", split="all", require_future=require_future,
                                 **common)
    else:
        name = "train" if split == "train_sub" else split
        ds = SceneFeatureDataset(d.index, split=name, cities=None if name.startswith("holdout") else cities,
                                 holdout=holdout, max_scenes=d.max_scenes if name == "train" else None,
                                 seed=d.seed, **common)
    if max_scenes is not None and len(ds) > max_scenes:
        ds = Subset(ds, np.sort(np.random.default_rng(seed).permutation(len(ds))[:max_scenes]))
    return ds


def checkpoints(run, which):
    ckpts = sorted((run / "checkpoints").glob("step*.ckpt"))
    if which == "all":
        return ckpts
    if which == "last":
        return [run / "checkpoints" / "last.ckpt"]
    return [run / "checkpoints" / f"{w}.ckpt" for w in which.split(",")]


@torch.no_grad()
def evaluate(model, ds, samples, steps, batch_size=512, keep=False):
    model.eval()
    g = torch.Generator(device=model.device).manual_seed(0)
    per_scene, preds, rows = [], [], []
    base, idx = (ds.dataset, ds.indices) if isinstance(ds, Subset) else (ds, range(len(ds)))
    for batch in DataLoader(base, sampler=BatchSampler(list(idx), batch_size, drop_last=False), batch_size=None):
        batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        s = model.sample(batch, samples, steps, generator=g)                  # (K, B, 8, 3)
        gt = batch["future"]
        m = {k: v.cpu() for k, v in sample_metrics(s, gt).items()}
        # NAVSIM-style L2 of the mean prediction at 1, 2, 3, 4 s (poses 2, 4, 6, 8).
        l2 = (s[..., :2].mean(0) - gt[..., :2]).norm(dim=-1).cpu()
        for t in (1, 2, 3, 4):
            m[f"l2_{t}s"] = l2[:, 2 * t - 1]
        m["command"] = batch["command"].argmax(1).cpu()
        per_scene.append(m)
        rows.append(batch["index"].cpu())
        if keep:
            preds.append(s.half().cpu())
    out = {k: torch.cat([m[k] for m in per_scene]).numpy() for k in per_scene[0]}
    out["row"] = torch.cat(rows).numpy()
    return out, (torch.cat(preds, 1).numpy() if keep else None)


def summarize(out):
    row = {k: float(np.nanmean(v)) for k, v in out.items() if k not in ("command", "row")}
    row["dispersion"] = row["spread"] / max(row["ade_mean"], 1e-3)
    for c, name in enumerate(COMMANDS[:3]):
        sel = out["command"] == c
        for k in MAIN:
            row[f"{k}_{name}"] = float(np.nanmean(out[k][sel])) if sel.any() else np.nan
    row["scenes"] = len(out["row"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--ckpts", default="last", help="all | last | comma-separated checkpoint names")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--steps", type=int, default=None, help="Euler steps; default: the model's")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--tag", default="", help="suffix of the output csv")
    ap.add_argument("--holdout", default=None, help="city:command,... replacing the run's holdout pairs")
    ap.add_argument("--cities", default=None, help="city,city,... replacing the run's city filter")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = OmegaConf.load(run / "config.yaml")
    holdout = [h.split(":") for h in args.holdout.split(",")] if args.holdout else None
    cities = args.cities.split(",") if args.cities else None
    # Predictions for the PDM score need every scene, also those without a logged future.
    ds = dataset_for(cfg, args.split, args.max_scenes, holdout=holdout, cities=cities,
                     require_future=not args.save_predictions)
    base = ds.dataset if isinstance(ds, Subset) else ds
    (run / "eval").mkdir(exist_ok=True)
    rows = []
    for ckpt in checkpoints(run, args.ckpts):
        model = TrajectoryProbe.load_from_checkpoint(ckpt, map_location="cuda")
        out, preds = evaluate(model, ds, args.samples, args.steps, keep=args.save_predictions)
        step = int(re.search(r"(\d+)", ckpt.stem).group(1)) if ckpt.stem != "last" else -1
        row = {"ckpt": ckpt.stem, "step": step, **summarize(out)}
        rows.append(row)
        print(f"{ckpt.stem}: " + "  ".join(f"{k} {row[k]:.3f}" for k in (*MAIN, "dispersion", "l2_4s")))
        if args.save_predictions:
            np.savez_compressed(run / "eval" / f"{args.split}_{ckpt.stem}.npz",
                                token=np.asarray(base.index["token"])[out["row"]], row=out["row"], samples=preds)
    path = run / "eval" / f"{args.split}{args.tag}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
