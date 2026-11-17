"""GoalFlow-style goal-point head (arXiv 2503.05689) on the frozen ORBIS features.

GoalFlow scores a vocabulary of goal points with two heads: a distance (imitation) score
towards the logged end point and a drivable-area score, and conditions the planner on the
best one. Here the goal-conditioned v22 runs (24aa_goal_aug etc.) are the planner, and this
script gives them a goal without any logged information.

    # 1. goal vocabulary (k-means of the navtrain end poses) + drivable labels from the map
    python -m exp_navsim.v22.goal_head labels --k 1024 --workers 8
    python -m exp_navsim.v22.goal_head labels --k 1024 --workers 4 --scenes 2000   # timing probe
    # 2. train (navtrain train split, metrics on the train.py val split), then navhard goals
    python -m exp_navsim.v22.goal_head train --name 27a_goal_head --predict-navhard
    python -m exp_navsim.v22.goal_head predict logs_exp/27a_goal_head --lams 0,1,3

Vocabulary: outputs/27_nogoal/goal_vocab_{K}.npy, raw (x, y, heading) of the cluster centres
(k-means on x, y, 5 * heading). Labels: data/goal_dac/navtrain_{K}.npz with rows (index
rows), bits (packed (M, K) drivable flags of every vocab point in that scene) and log_end (is
the logged end point drivable). Drivable = inside one of the polygons of the PDM drivable
map (PDMDrivableMap.from_simulation: roadblocks, roadblock connectors, their lanes and lane
connectors, intersections, car parks). The polygons of a whole map go into one STRtree per
worker, so one vectorized query per scene labels all K points.

Model: the GaragePlanner trunk (model.py scene(): ego, command, image tokens, registers) of the
config given by --like, then K goal queries = MLP(vocab point) + ego token, --blocks
cross-attention blocks to the scene memory, and two logits per query: imitation (CE towards
softmax(-d^2 / tau), d^2 = position distance^2 + (heading_w * heading gap)^2 to the logged end
pose) and drivable (BCE with the map labels).

Inference: score(g) = log softmax_imi(g) + lam * log sigmoid(dac(g)); for every lam the top-1
goal goes to outputs/27_nogoal/goals_head{tag}_lam{lam}.npz (token, goal (N, 3), synthetic)
and the top-8 to ..._top8.npz (goal (N, 8, 3), goal_score (N, 8)) for sweep_rank --multi-goal.
"""

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from exp_navsim.data.scenes import load_index, log_in_split

OUT = Path("outputs/27_nogoal")
DAC = Path("data/goal_dac")
HEADING_W = 5.0                # metres per radian for the vocabulary clustering
INDEX = "data/navsim_index/navtrain.pkl"
_G = {}


# ------------------------------------------------------------------ vocabulary

def vocab_path(k):
    return OUT / f"goal_vocab_{k}.npy"


def load_vocab(k, seed=0):
    """(K, 3) goal vocabulary; built (k-means of the navtrain train-split end poses) if missing."""
    path = vocab_path(k)
    if path.exists():
        return np.load(path)
    from sklearn.cluster import KMeans
    ix = load_index(INDEX)
    train = np.array([log_in_split(l, "train", 0.1) for l in ix["log"]])
    end = np.asarray(ix["future"], np.float64)[:, -1]
    end = end[train & np.isfinite(end).all(1)]
    x = np.concatenate([end[:, :2], end[:, 2:] * HEADING_W], 1)
    t0 = time.time()
    km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(x)
    c = km.cluster_centers_
    vocab = np.concatenate([c[:, :2], c[:, 2:] / HEADING_W], 1).astype(np.float32)
    err = np.linalg.norm(end[:, :2] - vocab[km.labels_, :2], axis=1)
    print(f"[vocab] k-means of {len(end)} end poses into {k} goals in {time.time() - t0:.0f} s, "
          f"mean / p95 quantization error {err.mean():.2f} / {np.percentile(err, 95):.2f} m", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(path, vocab)
    return vocab


# ------------------------------------------------------------------ drivable labels

def _drivable_layers():
    """Vector layers of the nuPlan map that PDMDrivableMap.from_simulation uses."""
    return ("lane_groups_polygons", "lane_group_connectors", "lanes_polygons",
            "gen_lane_connectors_scaled_width_polygons", "intersections", "carpark_areas")


def _tree(map_name):
    """STRtree of every drivable polygon of one map; one map cached per worker."""
    if _G.get("map") != map_name:
        import shapely
        from exp_navsim.data import bev_extract
        bev_extract._MAP_API_CACHE.clear()                       # free the previous map
        api = bev_extract.get_map_api(map_name)
        geoms = np.concatenate([np.asarray(api._load_vector_map_layer(n).geometry.values)
                                for n in _drivable_layers()])
        _G["tree"] = shapely.STRtree(geoms)
        _G["map"] = map_name
        bev_extract._MAP_API_CACHE.clear()                       # the tree keeps the geometries
    return _G["tree"]


def _init(index, vocab):
    from exp_navsim.v22 import bev_targets
    bev_targets._init("navtrain", index)
    _G["index"], _G["vocab"] = index, vocab


def _label(row):
    """-> (row, packed K flags, log end drivable, error)."""
    from exp_navsim.v22 import bev_targets
    ix = _G["index"]
    token, log = str(ix["token"][row]), str(ix["log"][row])
    try:
        frame = bev_targets._log_frames(log)[token]
        tree = _tree(frame["map_location"])
    except Exception as e:                                        # missing map or frame
        return row, None, False, f"{type(e).__name__}: {e}"
    x, y, h = bev_targets._STATE["pose"](frame)
    end = np.asarray(ix["future"][row][-1, :2], np.float64)
    pts = np.concatenate([_G["vocab"][:, :2].astype(np.float64), np.nan_to_num(end)[None]])
    c, s = math.cos(h), math.sin(h)
    gx = x + c * pts[:, 0] - s * pts[:, 1]
    gy = y + s * pts[:, 0] + c * pts[:, 1]
    import shapely
    hit = tree.query(shapely.points(gx, gy), predicate="intersects")[0]
    inside = np.zeros(len(pts), bool)
    inside[hit] = True
    return row, np.packbits(inside[:-1]), bool(inside[-1] and np.isfinite(end).all()), None


def labels(args):
    vocab = load_vocab(args.k)
    ix = load_index(INDEX)
    rows = np.arange(len(ix["token"]))
    if args.scenes:
        # a consecutive block (whole logs, as in the full run), so the rate extrapolates
        start = int(np.random.default_rng(0).integers(0, max(1, len(rows) - args.scenes)))
        rows = rows[start:start + args.scenes]
    # sorted by city then log: a worker keeps one map tree and one log pickle at a time
    order = np.lexsort((np.asarray(ix["log"])[rows], np.asarray(ix["city"])[rows]))
    rows = rows[order]
    print(f"[labels] {len(rows)} scenes x {len(vocab)} goals, {args.workers} workers", flush=True)
    out_rows, bits, ends, errors = [], [], [], {}
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(ix, vocab)) as pool:
        for i, (row, packed, end, err) in enumerate(pool.imap(_label, rows, chunksize=32)):
            if err is not None:
                errors[err] = errors.get(err, 0) + 1
                continue
            out_rows.append(row)
            bits.append(packed)
            ends.append(end)
            if (i + 1) % 5000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"{i + 1}/{len(rows)}  {rate:.0f} scenes/s  eta {(len(rows) - i - 1) / rate / 60:.0f} min",
                      flush=True)
    dt = time.time() - t0
    n_all = len(ix["token"])
    print(f"[labels] {len(out_rows)} scenes in {dt:.0f} s ({len(rows) / dt:.1f} scenes/s, incl. map loading); "
          f"all {n_all} navtrain scenes at this rate: {n_all / (len(rows) / dt) / 60:.0f} min", flush=True)
    DAC.mkdir(parents=True, exist_ok=True)
    path = DAC / (f"navtrain_{args.k}.npz" if not args.scenes else f"navtrain_{args.k}_probe{args.scenes}.npz")
    o = np.argsort(out_rows)
    bits = np.stack(bits)[o]
    np.savez_compressed(path, rows=np.asarray(out_rows)[o], bits=bits, log_end=np.asarray(ends)[o],
                        vocab=vocab, k=args.k)
    flags = np.unpackbits(bits, axis=1)[:, :args.k]
    print(f"-> {path}: drivable share of the vocab points {flags.mean():.3f}, "
          f"of the logged end points {np.mean(ends):.3f}")
    if errors:
        print("errors:", errors)


# ------------------------------------------------------------------ model

def build_head(C, T, cfg, vocab, blocks=2, trunk_goal=False):
    import torch
    import torch.nn as nn
    from exp_navsim.v22.model import HEADING_SCALE, Block, GaragePlanner
    from exp_navsim.v22.runs import camera_kwargs, cams_of
    cam_kw = camera_kwargs(cfg)                       # {} for a front-only --like run

    class GoalHead(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg["dim"]
            self.trunk = GaragePlanner(C // len(cams_of(cfg)), T, dim=d, depth=cfg["depth"], heads=cfg["heads"], flow_depth=0,
                                       registers=cfg["registers"], dropout=cfg["dropout"], vocab=None,
                                       groups=("wp",), use_image=bool(cfg["features"]),
                                       use_status=not cfg["no_status"], use_past=not cfg["no_past"],
                                       use_command=not cfg["no_command"], goal=trunk_goal, **cam_kw)
            self.register_buffer("vocab", torch.as_tensor(vocab, dtype=torch.float32))
            self.q_in = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
            self.blocks = nn.ModuleList([Block(d, cfg["heads"], cfg["dropout"]) for _ in range(blocks)])
            self.norm = nn.LayerNorm(d)
            self.imi = nn.Linear(d, 1)
            self.dac = nn.Linear(d, 1)

        def forward(self, batch):
            """-> imitation logits (B, K), drivable logits (B, K)."""
            if self.trunk.goal_in is not None:                   # trunk of a goal run: null goal
                batch = dict(batch, goal=torch.full_like(batch["status"][:, :3], float("nan")))
            mem, ego, _ = self.trunk.scene(batch)
            v = self.vocab
            q = self.q_in(torch.cat([v[:, :2], v[:, 2:] * HEADING_SCALE], -1) / 10.0)[None] + ego
            for blk in self.blocks:
                q = blk(q, mem)
            q = self.norm(q)
            return self.imi(q)[..., 0], self.dac(q)[..., 0]

    return GoalHead()


def soft_target(vocab, end, tau=1.0, heading_w=HEADING_W):
    """softmax(-d^2 / tau) over the vocabulary; vocab (K, 3), end (B, 3) -> (B, K), and d^2."""
    import torch
    dxy = ((vocab[None, :, :2] - end[:, None, :2]) ** 2).sum(-1)
    dh = torch.remainder(vocab[None, :, 2] - end[:, None, 2] + math.pi, 2 * math.pi) - math.pi
    d2 = dxy + (heading_w * dh) ** 2
    return (-d2 / tau).softmax(-1), d2


def head_loss(imi, dac, end, label, mask, vocab, tau=1.0, heading_w=HEADING_W, w_dac=1.0):
    import torch
    import torch.nn.functional as F
    tgt, _ = soft_target(vocab, end, tau, heading_w)
    l_imi = -(tgt * imi.float().log_softmax(-1)).sum(-1).mean()
    if mask.any():
        l_dac = F.binary_cross_entropy_with_logits(dac[mask].float(), label[mask].float())
    else:
        l_dac = dac.sum() * 0.0
    return l_imi + w_dac * l_dac, {"imi": l_imi.detach(), "dac": l_dac.detach()}


def goal_scores(imi, dac, lam):
    import torch.nn.functional as F
    return imi.float().log_softmax(-1) + lam * F.logsigmoid(dac.float())


# ------------------------------------------------------------------ training

def load_labels(path, n_rows, k):
    """-> pos (n_rows,) index into flags or -1, flags (M, K) uint8, log_end (M,) bool."""
    d = np.load(path)
    assert int(d["k"]) == k, f"{path} is for K={int(d['k'])}"
    flags = np.unpackbits(d["bits"], axis=1)[:, :k]
    pos = np.full(n_rows, -1, np.int64)
    pos[d["rows"]] = np.arange(len(d["rows"]))
    return pos, flags, d["log_end"]


def train(args):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from exp_navsim.data.scenes import SceneFeatureDataset

    run = Path("logs_exp") / args.name
    if run.exists():
        sys.exit(f"{run} exists")
    cfg = json.loads((Path(args.like) / "config.json").read_text())
    vocab = load_vocab(args.k)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = args.device
    from exp_navsim.v22.runs import feature_dir, feature_names
    feats = feature_names(cfg)
    common = dict(features=feats, feature_dir=feature_dir(cfg), history=4)
    train_ds = SceneFeatureDataset(INDEX, split="train", max_scenes=args.max_scenes, seed=args.seed, **common)
    val_ds = SceneFeatureDataset(INDEX, split="val", **common)
    val_rows = np.sort(np.random.default_rng(0).permutation(val_ds.rows)[:4000])     # as train.py
    n_rows = len(train_ds.index["token"])
    pos, flags, log_end = load_labels(args.labels or DAC / f"navtrain_{args.k}.npz", n_rows, args.k)
    pos_t = torch.from_numpy(pos).to(dev)
    flags_t = torch.from_numpy(flags).to(dev)

    def gpu(ds, rows):
        x = np.concatenate([ds.features_of(rows[i:i + 8192]) for i in range(0, len(rows), 8192)])
        return {"features": torch.from_numpy(x).to(dev),
                "end": torch.from_numpy(ds.future[rows][:, -1]).float().to(dev),
                "past": torch.from_numpy(ds.past[rows]).float().to(dev),
                "status": torch.from_numpy(ds.status[rows]).float().to(dev),
                "command": torch.from_numpy(ds.command[rows]).float().to(dev),
                "row": torch.from_numpy(rows).to(dev)}

    t0 = time.time()
    tr, va = gpu(train_ds, train_ds.rows), gpu(val_ds, val_rows)
    has = pos[train_ds.rows] >= 0
    print(f"[data] train {len(train_ds.rows)} ({has.mean():.1%} with drivable labels)  val {len(val_rows)}  "
          f"features {tuple(tr['features'].shape[1:])}  {time.time() - t0:.0f}s", flush=True)

    T, C = tr["features"].shape[1:3] if feats else (1, 0)
    model = build_head(C, T, cfg, vocab, blocks=args.blocks, trunk_goal=bool(args.init)).to(dev)
    if args.init:
        sd = torch.load(Path(args.init) / "model.pt", map_location="cpu", weights_only=False)["model"]
        own = model.trunk.state_dict()
        take = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.trunk.load_state_dict(take, strict=False)
        print(f"[model] trunk initialised from {args.init}: {len(take)} / {len(own)} tensors", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:.2f}M parameters, {args.k} goal queries", flush=True)

    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({**vars(args), "like_config": cfg}, indent=1))
    (run / "README.md").write_text(
        f"# {args.name}\n\nGoalFlow-style goal-point head (v22/goal_head.py): K={args.k} goal vocabulary, "
        f"imitation CE (tau {args.tau} m^2, heading weight {args.heading_w}) + drivable BCE (w {args.w_dac}) "
        f"on the trunk of {args.like} (init: {args.init or 'none'}).\n\n{args.note}\n\n"
        f"```\npython -m exp_navsim.v22.goal_head {' '.join(sys.argv[1:])}\n```\n")
    tb = SummaryWriter(str(run / "tb"))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warm = min(300, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / max(warm, 1)) * 0.5 * (
        1 + math.cos(math.pi * min(s / args.steps, 1))))

    def batch_of(data, idx):
        b = {k: v[idx] for k, v in data.items()}
        b["features"] = b["features"].float()
        return b

    def labels_of(rows):
        p = pos_t[rows]
        return flags_t[p.clamp(min=0)], p >= 0

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        imis, dacs = [], []
        for i in range(0, len(va["row"]), 256):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                a, d = model(batch_of(va, torch.arange(i, min(i + 256, len(va["row"])), device=dev)))
            imis.append(a.float())
            dacs.append(d.float())
        model.train()
        imi, dac = torch.cat(imis), torch.cat(dacs)
        lab, mask = labels_of(va["row"])
        loss, logs = head_loss(imi, dac, va["end"], lab, mask, model.vocab, args.tau, args.heading_w, args.w_dac)
        _, d2 = soft_target(model.vocab, va["end"], args.tau, args.heading_w)
        near = d2.argmin(-1)
        ar = torch.arange(len(imi), device=dev)
        m = {"val/loss_imi": logs["imi"].item(), "val/loss_dac": logs["dac"].item(),
             "val/l2_nearest_vocab": (model.vocab[near, :2] - va["end"][:, :2]).norm(dim=-1).mean().item(),
             "val/dac_acc": ((dac > 0) == lab.bool())[mask].float().mean().item(),
             "val/log_end_drivable": float(log_end[pos[va["row"].cpu().numpy()]][mask.cpu().numpy()].mean())}
        for lam in [float(x) for x in args.lams.split(",")]:
            top = goal_scores(imi, dac, lam).argmax(-1)
            m[f"val/l2_top1_lam{lam:g}"] = (model.vocab[top, :2] - va["end"][:, :2]).norm(dim=-1).mean().item()
            m[f"val/drivable_top1_lam{lam:g}"] = lab[ar, top][mask].float().mean().item()
        for k, v in m.items():
            tb.add_scalar(k, v, step)
        print(f"eval @ {step}: " + "  ".join(f"{k[4:]} {v:.3f}" for k, v in m.items()), flush=True)
        return m

    t0 = time.time()
    n = len(tr["row"])
    for step in range(1, args.steps + 1):
        idx = torch.randint(n, (args.batch,), device=dev)
        b = batch_of(tr, idx)
        lab, mask = labels_of(b["row"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            imi, dac = model(b)
        loss, logs = head_loss(imi, dac, b["end"], lab, mask, model.vocab, args.tau, args.heading_w, args.w_dac)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            for k, v in logs.items():
                tb.add_scalar(f"train/{k}", v.item(), step)
            tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
        if step % 500 == 0:
            print(f"step {step:6d}/{args.steps}  loss {loss.item():.4f}  "
                  + "  ".join(f"{k} {v.item():.3f}" for k, v in logs.items())
                  + f"  {500 / (time.time() - t0):.1f} it/s", flush=True)
            t0 = time.time()
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(step)
    torch.save({"model": model.state_dict(), "args": vars(args), "cfg": cfg, "C": C, "T": T},
               run / "model.pt")
    (run / "val_metrics.json").write_text(json.dumps(m, indent=1))
    with open(run / "README.md", "a") as f:
        f.write("\n## Val metrics (final)\n\n" + "\n".join(f"- {k}: {v:.3f}" for k, v in m.items()) + "\n")
    tb.close()
    if args.predict_navhard:
        del tr
        torch.cuda.empty_cache() if dev == "cuda" else None
        predict(run, args.lams, args.tag, dev)


# ------------------------------------------------------------------ navhard goals

def predict(run, lams="0,1,3", tag="", dev="cuda"):
    import torch
    from exp_navsim.data.scenes import SceneFeatureDataset
    run = Path(run)
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    a, cfg = ck["args"], ck["cfg"]
    vocab = ck["model"]["vocab"].numpy()
    model = build_head(ck["C"], ck["T"], cfg, vocab, blocks=a["blocks"], trunk_goal=bool(a["init"])).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    from exp_navsim.v22.runs import feature_dir, feature_names
    ds = SceneFeatureDataset("data/navsim_index/navhard.pkl", split="all", require_future=False,
                             features=feature_names(cfg), feature_dir=feature_dir(cfg), history=4)
    rows = ds.rows
    tokens = np.asarray(ds.index["token"])[rows]
    syn = np.asarray(ds.index["synthetic"])[rows].astype(bool)
    imis, dacs = [], []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            r = rows[i:i + 256]
            b = {"features": torch.from_numpy(ds.features_of(r)).float().to(dev),
                 "past": torch.from_numpy(ds.past[r]).float().to(dev),
                 "status": torch.from_numpy(ds.status[r]).float().to(dev),
                 "command": torch.from_numpy(ds.command[r]).float().to(dev)}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                x, y = model(b)
            imis.append(x.float())
            dacs.append(y.float())
    imi, dac = torch.cat(imis), torch.cat(dacs)
    OUT.mkdir(parents=True, exist_ok=True)
    for lam in [float(x) for x in str(lams).split(",")]:
        s = goal_scores(imi, dac, lam)
        top = s.topk(8, -1)
        g = vocab[top.indices.cpu().numpy()]                                  # (N, 8, 3)
        base = OUT / f"goals_head{tag}_lam{lam:g}"
        np.savez(f"{base}.npz", token=tokens, goal=g[:, 0], synthetic=syn)
        np.savez(f"{base}_top8.npz", token=tokens, goal=g, goal_score=top.values.cpu().numpy(), synthetic=syn)
        dr = torch.sigmoid(dac.gather(1, top.indices[:, :1])).mean().item()
        print(f"-> {base}.npz / _top8.npz: {len(tokens)} tokens, median top-1 goal distance "
              f"{np.median(np.linalg.norm(g[:, 0, :2], axis=1)):.1f} m, mean p(drivable) of top-1 {dr:.3f}",
              flush=True)


# ------------------------------------------------------------------ cli

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("labels")
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--scenes", type=int, default=0, help="block of n consecutive scenes for a timing probe; 0 = all")
    p = sub.add_parser("train")
    p.add_argument("--name", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--like", default="logs_exp/24aa_goal_aug", help="run whose config gives features / trunk")
    p.add_argument("--init", default="", help="run whose trunk weights initialise ours (null goal)")
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--labels", default="", help="default data/goal_dac/navtrain_{K}.npz")
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--tau", type=float, default=1.0, help="m^2, soft imitation target temperature")
    p.add_argument("--heading-w", type=float, default=HEADING_W, help="metres per radian in d^2")
    p.add_argument("--w-dac", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lams", default="0,1,3")
    p.add_argument("--tag", default="", help="infix of the navhard goal files")
    p.add_argument("--predict-navhard", action="store_true")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("predict")
    p.add_argument("run")
    p.add_argument("--lams", default="0,1,3")
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.cmd == "labels":
        labels(args)
    elif args.cmd == "train":
        train(args)
    else:
        predict(args.run, args.lams, args.tag, args.device)


if __name__ == "__main__":
    main()
