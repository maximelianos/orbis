"""Dream features: roll the Orbis 2 L2 model out along K candidate trajectories per scene.

    # navtrain training scenes: top 4 + 4 random anchors of the PDM score banks
    python -m exp_navsim.orbis2.dream --split navtrain \
        --pdm-targets outputs/16_fast_planner/vocab256_scores_navtrain_sub.npz,outputs/16_fast_planner/vocab256_scores_navtrain_sub2.npz \
        --max-scenes 10000 --horizon 2 --out outputs/28_inputs/dream/navtrain_h2
    # navhard: top-K anchors of a planner's (N, 256) score matrix
    python -m exp_navsim.orbis2.dream --split navhard --scores <npz with token(s) + scores (N, 256)> \
        --horizon 2 --out outputs/28_inputs/dream/navhard_h2
    # navhard: top-K refined candidates of the v22 ensemble (navhard_candidates.py)
    python -m exp_navsim.orbis2.dream --split navhard --candidates outputs/28_inputs/candidates/navhard_near.npz \
        --horizon 2 --out outputs/28_inputs/dream/navhard_near_h2
    # timing only
    python -m exp_navsim.orbis2.dream --split navhard --random-anchors --max-scenes 16 --horizon 4 --out /tmp/x

Candidate trajectories are entries of the 256-anchor vocabulary (8 poses x, y,
heading at 0.5 s, ego frame, x forward, y left). For rollout step h (anchor time
0.5 h s) the L2 goal condition is the candidate path at 0.5 h + 0.1, ..., 0.5 h + 0.5 s
(poses linearly interpolated to 10 Hz, the origin prepended), expressed in the
candidate's pose at 0.5 h, i.e. re-anchored to the frame being predicted from. This
is what SpeedYawMovingGoalConditionPreprocessor computes from logged odometry
(goal_offsets 1..5, odometry_steps_per_image_frame 5, anchor = last context frame).

Rollout: context = the 4 history latents, then a sliding window of the 5 newest
frames (the L2 context length of the Orbis 2 hierarchy). Each step samples the next
frame with the few-NFE sampler (default Euler, 4 NFE, as L1/config_distill.yaml).
The K candidates of a scene share the context latents and the noise (common random
numbers), so their dreams differ only through the steering.

Features per step (--feat-mode):
    clean  (default) the generated frame as the newest context slot (t = 0) of the
           next pass, i.e. what the model makes of its own dream; projected with the
           Task-1 o2l2_b<b>_ctx PCA. The last step needs one extra pass, steered by
           the candidate's next 0.5 s (or NaN beyond its 4 s).
    noise  the next-frame slot at the first denoising step (o2l2_b<b>_next PCA).

Output (<out>/): dream_b11.npy, dream_b23.npy (N, K, H, 64, 9, 16) float16,
dream_lat.npy (N, K, H, 16, 18, 32) float16 (sem latents x 4), meta.npz with
tokens, rows, anchors (N, K) and, when known, the anchors' PDM sub-scores and EPDMS.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from exp_navsim.orbis2.common import Clips, NpyAppender, load_index, project, scene_noise, tokens_to_grid
from exp_navsim.orbis2.loader import CTX_FRAMES, NUM_GOALS, L2Model

VOCAB = Path("outputs/16_fast_planner/vocab_256.npy")
DREAM_BLOCKS = (11, 23)


# ---------------------------------------------------------------- candidates

def load_vocab(path=VOCAB):
    return np.load(path).astype(np.float32)                    # (256, 8, 3)


def topk_anchors(scores, k):
    """(N, V) scores -> (N, k) anchor ids, best first."""
    return np.argsort(-np.asarray(scores), axis=1, kind="stable")[:, :k]


def mixed_anchors(scores, k_top=4, k_rand=4, seed=0):
    """Top k_top by score plus k_rand random others per scene: (N, k_top + k_rand)."""
    rng = np.random.default_rng(seed)
    top = topk_anchors(scores, k_top)
    out = np.empty((len(top), k_top + k_rand), dtype=np.int64)
    V = scores.shape[1]
    for i, t in enumerate(top):
        rest = np.setdiff1d(np.arange(V), t)
        out[i] = np.concatenate([t, rng.choice(rest, k_rand, replace=False)])
    return out


def load_pdm_targets(paths):
    """Score banks of planner/score_vocab.py -> tokens, rows, sub-scores (N, 256, 8), EPDMS (N, 256)."""
    from exp_navsim.planner.fast_pdm import epdms
    tokens, rows, subs = [], [], []
    for p in paths:
        z = np.load(p)
        tokens.append(z["tokens"]), rows.append(z["rows"]), subs.append(z["scores"])
    tokens, rows, sub = np.concatenate(tokens), np.concatenate(rows), np.concatenate(subs)
    _, first = np.unique(tokens, return_index=True)            # the banks may overlap
    first = np.sort(first)
    sub = sub[first].astype(np.float32)
    return tokens[first], rows[first], sub, epdms(sub.astype(np.float64)).astype(np.float32)


def load_score_matrix(path, index):
    """npz with token(s) and scores (N, 256) or PDM sub-scores (N, 256, 8) -> rows, scores (N, 256)."""
    z = np.load(path)
    tokens = z["tokens"] if "tokens" in z.files else z["token"]
    scores = z["scores"].astype(np.float32)
    if scores.ndim == 3:
        from exp_navsim.planner.fast_pdm import epdms
        scores = epdms(scores.astype(np.float64)).astype(np.float32)
    row_of = {t: i for i, t in enumerate(index["token"])}
    return np.array([row_of[t] for t in tokens]), scores


def load_candidates(path, index, k):
    """navhard_candidates.py npz -> rows, anchor ids (N, k) and refined trajectories (N, k, 8, 3)
    of the k best candidates by mix score."""
    z = np.load(path)
    row_of = {t: i for i, t in enumerate(index["token"])}
    rows = np.array([row_of[t] for t in z["token"]])
    top = np.argsort(-z["mix"], axis=1, kind="stable")[:, :k]
    traj = np.take_along_axis(z["traj"], top[..., None, None], 1)
    return rows, top, traj


def goals_from_poses(poses, steps):
    """Candidate poses (K, 8, 3) at 0.5 s -> L2 goals (K, steps, 5, 2).

    Step h: positions at 0.5 h + 0.1 j s (j = 1..5) in the candidate frame at 0.5 h.
    NaN where the candidate (4 s) ends.
    """
    K = len(poses)
    p = np.concatenate([np.zeros((K, 1, 3), np.float32), poses], 1)       # (K, 9, 3), t = 0 .. 4 s
    p[:, :, 2] = np.unwrap(p[:, :, 2], axis=1)
    t_node = np.arange(9) * 0.5
    t_fine = np.arange(41) * 0.1
    fine = np.stack([np.stack([np.interp(t_fine, t_node, p[k, :, c]) for c in range(3)], -1)
                     for k in range(K)])                                    # (K, 41, 3)
    goals = np.full((K, steps, NUM_GOALS, 2), np.nan, np.float32)
    for h in range(steps):
        a = 5 * h
        if a + NUM_GOALS >= fine.shape[1]:
            break
        d = fine[:, a + 1:a + 1 + NUM_GOALS, :2] - fine[:, a:a + 1, :2]
        c, s = np.cos(fine[:, a, 2])[:, None], np.sin(fine[:, a, 2])[:, None]
        goals[:, h, :, 0] = c * d[..., 0] + s * d[..., 1]
        goals[:, h, :, 1] = -s * d[..., 0] + c * d[..., 1]
    return goals


# ---------------------------------------------------------------- rollout

class Dreamer:
    def __init__(self, l2, proj=None, horizon=2, nfe=4, sampler="euler", feat_mode="clean",
                 blocks=DREAM_BLOCKS):
        self.l2, self.proj, self.H, self.nfe = l2, proj, horizon, nfe
        self.sampler, self.feat_mode, self.blocks = sampler, feat_mode, blocks

    @torch.no_grad()
    def __call__(self, clips, rows, poses):
        """clips (S, 4, 3, H, W), rows (S,), poses (S, K, 8, 3) ->
        feats {block: (S, K, H, 64 or 768, 9, 16)}, latents (S, K, H, 16, 18, 32)."""
        l2, H = self.l2, self.H
        S, K = poses.shape[:2]
        lat = l2.encode(clips).repeat_interleave(K, 0)                       # (S K, 4, ...)
        goals = goals_from_poses(poses.reshape(S * K, 8, 3), H + 1)
        goals = torch.from_numpy(goals).to(l2.device)
        draws = self.nfe if self.sampler == "cd" else 1
        noise = scene_noise(rows, (H + 1) * draws).repeat_interleave(K, 0)   # shared by the K candidates
        noise = noise.view(S * K, H + 1, draws, *noise.shape[2:])
        ctx, gen, feats = lat, [], {b: [] for b in self.blocks}
        for h in range(H + 1):
            window = ctx[:, -CTX_FRAMES:]
            if h == H:
                if self.feat_mode == "clean":                                 # features of the last dream frame
                    seq = torch.cat([window, noise[:, h, :1].to(l2.device)], 1)
                    _, f = l2.velocity(seq, torch.ones(S * K, device=l2.device), goals[:, h], self.blocks)
                    self._keep(feats, f, -2)
                break
            x0, f = l2.sample_next(window, goals[:, h], noise[:, h], self.nfe, self.sampler, self.blocks)
            if self.feat_mode == "clean":
                if h > 0:
                    self._keep(feats, f, -2)
            else:
                self._keep(feats, f, -1)
            gen.append(x0)
            ctx = torch.cat([ctx, x0[:, None]], 1)
        out = {b: torch.stack(v, 1).unflatten(0, (S, K)) for b, v in feats.items()}
        return out, torch.stack(gen, 1).unflatten(0, (S, K))

    def _keep(self, feats, f, slot):
        for b in self.blocks:
            g = tokens_to_grid(f[b][:, slot])
            if self.proj is not None:
                g = project(g, self.proj[b])
            feats[b].append(g)


def load_projection(feat_mode, blocks, fit_dir="data/features/navtrain"):
    suffix = "ctx" if feat_mode == "clean" else "next"
    return {b: torch.load(Path(fit_dir) / f"o2l2_b{b}_{suffix}.pt") for b in blocks}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtrain")
    ap.add_argument("--index-dir", default="data/navsim_index")
    ap.add_argument("--out", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdm-targets", help="comma-separated score banks (score_vocab.py): top + random mix")
    src.add_argument("--scores", help="npz with token(s) + scores (N, 256): top-K")
    src.add_argument("--random-anchors", action="store_true", help="K random anchors (timing / debugging)")
    src.add_argument("--candidates", help="npz of navhard_candidates.py: top-K refined trajectories by mix")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k-top", type=int, default=4, help="with --pdm-targets: top anchors, the rest random")
    ap.add_argument("--horizon", type=int, default=2, help="rollout steps of 0.5 s (1..8)")
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--sampler", default="euler", choices=["euler", "cd"])
    ap.add_argument("--feat-mode", default="clean", choices=["clean", "noise"])
    ap.add_argument("--fit-dir", default="data/features/navtrain", help="Task-1 PCA projections")
    ap.add_argument("--no-pca", action="store_true", help="store raw pooled 768-dim features (debug)")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--scenes-per-batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-latents", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    assert 1 <= args.horizon <= 8

    index = load_index(args.split, args.index_dir)
    vocab = load_vocab()
    rng = np.random.default_rng(args.seed)
    sub = epd = traj = None
    if args.pdm_targets:
        tokens, rows, sub_all, epd_all = load_pdm_targets(args.pdm_targets.split(","))
        anchors = mixed_anchors(epd_all, args.k_top, args.k - args.k_top, args.seed)
        take = lambda a: np.take_along_axis(a, anchors if a.ndim == 2 else anchors[..., None], 1)
        sub, epd = take(sub_all), take(epd_all)
    elif args.scores:
        rows, scores = load_score_matrix(args.scores, index)
        anchors = topk_anchors(scores, args.k)
    elif args.candidates:
        rows, anchors, traj = load_candidates(args.candidates, index, args.k)
    else:
        rows = np.arange(len(index["token"]))
        anchors = np.stack([rng.choice(len(vocab), args.k, replace=False) for _ in rows])
    if args.max_scenes is not None and args.max_scenes < len(rows):
        keep = np.sort(rng.choice(len(rows), args.max_scenes, replace=False))
        rows, anchors = rows[keep], anchors[keep]
        traj = None if traj is None else traj[keep]
        sub = None if sub is None else sub[keep]
        epd = None if epd is None else epd[keep]
    N, K, H = len(rows), args.k, args.horizon
    # explicit per-scene trajectories (N, K, 8, 3); the vocabulary anchors otherwise
    poses_all = (vocab[anchors] if traj is None else traj).astype(np.float32)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(tokens=np.asarray(index["token"])[rows], rows=rows, anchors=anchors, poses=poses_all)
    if sub is not None:
        meta.update(sub_scores=sub.astype(np.float16), epdms=epd)
    np.savez(out / "meta.npz", **meta)

    l2 = L2Model("cuda", compile=not args.no_compile)
    proj = None if args.no_pca else load_projection(args.feat_mode, DREAM_BLOCKS, args.fit_dir)
    dreamer = Dreamer(l2, proj, H, args.nfe, args.sampler, args.feat_mode)
    C = 768 if proj is None else proj[DREAM_BLOCKS[0]]["basis"].shape[1]
    writers = {b: NpyAppender(out / f"dream_b{b}.npy", (N, K, H, C, 9, 16)) for b in DREAM_BLOCKS}
    if not args.no_latents:
        writers["lat"] = NpyAppender(out / "dream_lat.npy", (N, K, H, 16, 18, 32))

    loader = DataLoader(Clips(index, rows), batch_size=args.scenes_per_batch, num_workers=args.workers,
                        pin_memory=True)
    pos = 0
    t0 = time.time()
    for it, (clips, batch_rows) in enumerate(tqdm(loader, desc=f"dream {args.split} K={K} H={H}")):
        if it == 1:                                     # after the compilation of the first batch
            torch.cuda.synchronize()
            t1, n1 = time.time(), pos
        S = len(batch_rows)
        poses = poses_all[pos:pos + S]                                      # (S, K, 8, 3)
        feats, lat = dreamer(clips, batch_rows.tolist(), poses)
        for b in DREAM_BLOCKS:
            writers[b].append(feats[b].half().cpu().numpy())
        if "lat" in writers:
            writers["lat"].append(lat.half().cpu().numpy())
        pos += S
    torch.cuda.synchronize()
    secs = time.time() - t0
    steady = (time.time() - t1) / max(N - n1, 1) if N > args.scenes_per_batch else float("nan")
    for w in writers.values():
        w.close()
    info = dict(split=args.split, scenes=N, k=K, horizon=H, nfe=args.nfe, sampler=args.sampler,
                feat_mode=args.feat_mode, pca=proj is not None, seconds=round(secs, 1),
                seconds_per_scene=round(secs / N, 3),
                seconds_per_scene_steady=round(steady, 3), blocks=list(DREAM_BLOCKS))
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
