"""Compositional generalization of the fast planner: controlled sweeps over held-out (city, command) pairs.

    python -m exp_navsim.compositional.sweep --plan exp_navsim/compositional/plans/main.json

The data of one encoder is loaded once onto the GPU, then every config of the plan
with that encoder is trained (fast planner, regression head, ~2 min) and evaluated.
Finished configs (by name) are skipped, so a plan can be extended and rerun.

A config (all keys but name optional):
    name           run name
    features       cached feature, default wm11_ctx ('' = ego status and command only)
    holdout        [[city, command], ...] pairs removed from training ("composition" at test)
    fewshot        number of training scenes of the held-out pairs put back (random scenes)
    command_cities cities that keep their scenes of the held-out command (domain diversity);
                   default all. The held-out pair stays out regardless.
    command_quota  subsample training scenes of the held-out command to this many in total,
                   split equally over the cities that keep it (equal quantity across
                   diversity settings)
    mirror         probability of a left-right mirrored training sample: feature grid
                   flipped horizontally, lateral pose / heading / velocity / acceleration
                   negated, left and right commands swapped
    seed, steps
    pdm            false: skip the navtest EPDMS (val metrics only)

Evaluation per held-out pair and on the rest (outputs/22_compositional/results.csv):
    val            held-out navtrain logs: ADE, L2 at 4 s, lateral error at 4 s and the
                   signed lateral bias at 4 s (+ = predicted end left of the logged end)
    navtest        EPDMS with the official sub-scores (fast_pdm) on every navtest scene of
                   the pair and on the 500-scene navtest subset minus the pair's scenes
Learning curves: held-out pair and rest every --eval-every steps (curves.csv).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.planner.fast_pdm import FastPDM, load_subset
from exp_navsim.planner.subset import predecessors

OUT = Path("outputs/22_compositional")
COMMANDS = ("left", "straight", "right", "unknown")
DEFAULT = dict(features="wm11_ctx", holdout=[["sg-one-north", "left"]], fewshot=0, command_cities=None,
               command_quota=None, mirror=0.0, seed=0, steps=6000, pdm=True)


CITY = {"sg-one-north": "sg", "us-ma-boston": "bos", "us-nv-las-vegas-strip": "lv", "us-pa-pittsburgh-hazelwood": "pit"}


def pair_name(pair):
    return f"{CITY[pair[0]]}_{pair[1]}"


def pair_mask(index, rows, pair):
    city = np.asarray(index["city"])[rows]
    cmd = np.asarray(index["command"])[rows].argmax(1)
    return (city == pair[0]) & (cmd == COMMANDS.index(pair[1]))


def train_rows(ds, cfg, rng):
    """Training rows of a config: holdout, few-shot put-back, diversity and quota."""
    rows = ds.rows
    city = np.asarray(ds.index["city"])[rows]
    cmd = np.asarray(ds.index["command"])[rows].argmax(1)
    held = np.zeros(len(rows), bool)
    for p in cfg["holdout"]:
        held |= pair_mask(ds.index, rows, p)
    keep = ~held
    held_cmds = {COMMANDS.index(p[1]) for p in cfg["holdout"]}
    is_cmd = np.isin(cmd, list(held_cmds))
    if cfg["command_cities"] is not None:
        keep &= ~is_cmd | np.isin(city, cfg["command_cities"])
    if cfg["command_quota"] is not None:
        # the same total, split equally over the cities that keep the command
        kept_cities = sorted(set(city[keep & is_cmd]))
        for c in kept_cities:
            idx = np.flatnonzero(keep & is_cmd & (city == c))
            keep[rng.permutation(idx)[cfg["command_quota"] // len(kept_cities):]] = False
    if cfg["fewshot"]:
        back = rng.permutation(np.flatnonzero(held))[:cfg["fewshot"]]
        keep[back] = True
    return rows[keep]


def results_from_log(log=OUT / "sweep.log"):
    """results.csv rebuilt from the printed rows (one 'name=... key=value' line per run)."""
    rows = []
    for line in open(log):
        if line.startswith("name="):
            row = {}
            for kv in line.split():
                k, _, v = kv.partition("=")
                try:
                    row[k] = float(v) if v not in ("", "None") else None
                except ValueError:
                    row[k] = v
            rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("name", keep="last")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--pdm-workers", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = [{**DEFAULT, **c} for c in json.load(open(args.plan))]
    done = set(pd.read_csv(OUT / "results.csv")["name"]) if (OUT / "results.csv").exists() else set()
    plan = [c for c in plan if c["name"] not in done]
    if not plan:
        print("nothing to do")
        return

    # navtest scenes of every held-out pair in the plan + the generic subset, with predecessors.
    # The scoring pool forks, so it starts before CUDA is touched.
    import pickle
    test_ix = pickle.load(open("data/navsim_index/navtest.pkl", "rb"))
    all_rows = np.arange(len(test_ix["token"]))
    pairs = sorted({tuple(p) for c in plan for p in c["holdout"]})
    sub_tokens, _ = load_subset("outputs/16_fast_planner/subset_500.json")
    groups = {pair_name(p): [str(t) for t in np.asarray(test_ix["token"])[pair_mask(test_ix, all_rows, p)]]
              for p in pairs}
    in_pairs = set().union(*groups.values()) if groups else set()
    groups["rest"] = [t for t in sub_tokens if t not in in_pairs]
    prev_all, _, _ = predecessors([t for g in groups.values() for t in g])
    prev = {t: p for t, p in prev_all.items() if p is not None}
    scorer = FastPDM("navtest", sorted(set(prev) | set(prev.values()) | set(groups["rest"])),
                     workers=args.pdm_workers, preload=False)

    import torch

    from exp_navsim.data.scenes import SceneFeatureDataset
    from exp_navsim.planner.model import Planner, planner_loss

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"

    for features in dict.fromkeys(c["features"] for c in plan):
        feats = [f for f in features.split(",") if f]
        common = dict(features=feats, feature_dir="data/features", history=4)
        t0 = time.time()
        train_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="train", **common)
        val_ds = SceneFeatureDataset("data/navsim_index/navtrain.pkl", split="val", **common)
        test_ds = SceneFeatureDataset("data/navsim_index/navtest.pkl", split="all", require_future=False, **common)

        def gpu(ds, rows):
            x = np.concatenate([ds.features_of(rows[i:i + 8192]) for i in range(0, len(rows), 8192)])
            return {"features": torch.from_numpy(x).to(dev),
                    "future": torch.from_numpy(np.nan_to_num(ds.future[rows])).float().to(dev),
                    "past": torch.from_numpy(ds.past[rows]).float().to(dev),
                    "status": torch.from_numpy(ds.status[rows]).float().to(dev),
                    "command": torch.from_numpy(ds.command[rows]).float().to(dev)}

        # all training rows on the GPU once; a config indexes into them
        train_all = gpu(train_ds, train_ds.rows)
        pos_of = {r: i for i, r in enumerate(train_ds.rows)}
        val = gpu(val_ds, val_ds.rows)
        test = gpu(test_ds, test_ds.rows)
        test_tok = {t: i for i, t in enumerate(np.asarray(test_ds.index["token"])[test_ds.rows])}
        val_groups = {pair_name(p): np.flatnonzero(pair_mask(val_ds.index, val_ds.rows, p)) for p in pairs}
        in_val_pairs = np.zeros(len(val_ds.rows), bool)
        for g in val_groups.values():
            in_val_pairs[g] = True
        val_groups["rest"] = np.random.default_rng(0).permutation(np.flatnonzero(~in_val_pairs))[:4000]
        print(f"[data] {features or 'none'}: train {len(train_ds)} val {len(val_ds)} navtest {len(test_tok)}  "
              f"{time.time() - t0:.0f}s", flush=True)

        for cfg in [c for c in plan if c["features"] == features]:
            run_config(cfg, train_ds, train_all, pos_of, val, val_groups, test, test_tok, groups, prev, scorer,
                       args.eval_every, Planner, planner_loss)
        del train_all, val, test
        torch.cuda.empty_cache()
    scorer.close()


def mirror_batch(b, p, gen):
    import torch
    m = torch.rand(b["future"].shape[0], device=b["future"].device, generator=gen) < p
    if not m.any():
        return b
    b = dict(b)
    sign = torch.tensor([1.0, -1.0, -1.0], device=m.device)
    b["features"] = torch.where(m[:, None, None, None, None], b["features"].flip(-1), b["features"])
    b["future"] = torch.where(m[:, None, None], b["future"] * sign, b["future"])
    past = b["past"].view(b["past"].shape[0], -1, 3)
    b["past"] = torch.where(m[:, None, None], past * sign, past).view_as(b["past"])
    b["status"] = torch.where(m[:, None], b["status"] * torch.tensor([1.0, -1.0, 1.0, -1.0], device=m.device),
                              b["status"])
    b["command"] = torch.where(m[:, None], b["command"][:, [2, 1, 0, 3]], b["command"])
    return b


def run_config(cfg, train_ds, train_all, pos_of, val, val_groups, test, test_tok, groups, prev, scorer,
               eval_every, Planner, planner_loss):
    import torch

    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    gen = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    rows = train_rows(train_ds, cfg, rng)
    idx_all = torch.from_numpy(np.array([pos_of[r] for r in rows])).cuda()
    n_held = sum(int(pair_mask(train_ds.index, rows, p).sum()) for p in cfg["holdout"])
    feats = [f for f in cfg["features"].split(",") if f]
    T, C = train_all["features"].shape[1:3] if feats else (1, 0)
    model = Planner(C, T, vocab=None, dim=256, depth=4, decoder_depth=2, dropout=0.1,
                    use_image=bool(feats)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    steps = cfg["steps"]
    warm = min(300, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / max(warm, 1)) * 0.5 * (
        1 + math.cos(math.pi * min(s / steps, 1))))

    def batch_of(data, idx):
        b = {k: v[idx] for k, v in data.items()}
        b["features"] = b["features"].float()
        return b

    @torch.no_grad()
    def predict(data, idx):
        model.eval()
        out = torch.cat([model(batch_of(data, idx[i:i + 1024]))["traj"] for i in range(0, len(idx), 1024)])
        model.train()
        return out

    def val_metrics(tag):
        res = {}
        for g, idx in val_groups.items():
            idx_t = torch.from_numpy(idx).cuda()
            pred, gt = predict(val, idx_t), val["future"][idx_t]
            err = (pred[..., :2] - gt[..., :2]).norm(dim=-1)
            res[f"{tag}{g}_ade"] = float(err.mean())
            res[f"{tag}{g}_l2_4s"] = float(err[:, -1].mean())
            res[f"{tag}{g}_lat_4s"] = float((pred[:, -1, 1] - gt[:, -1, 1]).abs().mean())
            res[f"{tag}{g}_lat_bias"] = float((pred[:, -1, 1] - gt[:, -1, 1]).mean())
        return res

    t0 = time.time()
    curve = []
    model.train()
    for step in range(1, steps + 1):
        b = batch_of(train_all, idx_all[torch.randint(len(idx_all), (256,), device="cuda", generator=gen)])
        if cfg["mirror"]:
            b = mirror_batch(b, cfg["mirror"], gen)
        loss, _ = planner_loss(model, model(b), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            curve.append({"name": cfg["name"], "step": step, **val_metrics("")})
    train_time = time.time() - t0

    # navtest: one trajectory per scene for every pair token and its predecessor
    need = sorted(set(prev) | set(prev.values()) | set(groups["rest"])) if cfg["pdm"] else []
    need = [t for t in need if t in test_tok]
    trajs = {}
    if need:
        pred = predict(test, torch.tensor([test_tok[t] for t in need], device="cuda")).cpu().numpy()
        trajs = {t: pred[i][None] for i, t in enumerate(need)}
    row = {"name": cfg["name"], "features": cfg["features"] or "none",
           "holdout": ";".join(pair_name(p) for p in cfg["holdout"]), "fewshot": cfg["fewshot"],
           "command_cities": ";".join(CITY[c] for c in cfg["command_cities"]) if cfg["command_cities"]
           else "all", "command_quota": cfg["command_quota"], "mirror": cfg["mirror"], "seed": cfg["seed"],
           "steps": steps, "train_scenes": len(rows), "train_held": n_held, "train_min": train_time / 60,
           **val_metrics("val_")}
    own = ({pair_name(p) for p in cfg["holdout"]} | {"rest"}) if cfg["holdout"] else set(groups)
    for g, toks in groups.items():
        if not cfg["pdm"] or g not in own:
            continue
        cur = {t: prev.get(t) for t in toks if t in trajs}
        df = scorer.score({t: trajs[t] for t in set(cur) | {p for p in cur.values() if p in trajs}},
                          {t: p for t, p in cur.items() if p is not None and p in trajs})
        row[f"epdms_{g}"] = float(df["score"].mean())
        row[f"dac_{g}"] = float(df["drivable_area_compliance"].mean())
        row[f"n_{g}"] = len(df)
    if cfg["seed"] == 0:   # trajectories on the held-out val scenes for the plots
        save = {}
        for g, idx in val_groups.items():
            if g != "rest":
                save[g] = predict(val, torch.from_numpy(idx).cuda()).cpu().numpy()
                save[f"{g}_rows"] = idx
        (OUT / "pred").mkdir(exist_ok=True)
        np.savez_compressed(OUT / "pred" / f"{cfg['name']}.npz", **save)
    for f, rows_ in (("results.csv", [row]), ("curves.csv", curve)):
        df = pd.DataFrame(rows_)
        if (OUT / f).exists():   # rows differ in columns (pairs scored), so align instead of appending
            df = pd.concat([pd.read_csv(OUT / f), df], ignore_index=True)
        df.to_csv(OUT / f, index=False)
    print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
