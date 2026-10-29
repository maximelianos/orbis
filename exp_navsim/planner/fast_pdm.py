"""In-process EPDMS on a small set of scenes, same numbers as the official one-stage script.

The official run (eval/pdm.py) spends most of its time loading scenes and metric caches;
here a pool of workers loads the metric caches of a fixed token set once and keeps
them, so scoring a new set of trajectories for 500 scenes takes seconds. Each worker
scores all candidates of a scene in one simulator / scorer call:

    proposals = [PDM-Closed, candidate_1, ..., candidate_K]

Everything in the PDM score is per proposal except ego progress, which the scorer
normalizes by the best progress over all proposals. The official script scores one
candidate next to PDM-Closed, so progress is renormalized here per pair
(max of the PDM-Closed and the candidate progress, each masked by its own
multiplicative metrics). The human penalty filter (a metric the human trajectory
fails is set to 1) is applied as in navsim/evaluate/pdm_score.py.

Two-frame extended comfort needs the simulated states of the previous frame under
the same planner: `score` takes the trajectories of every token in the subset json
(current and previous tokens), and returns rows for the current ones only.

    scorer = FastPDM("navtest", tokens)        # starts the pool, loads the caches
    df = scorer.score({token: (K, 8, 3) poses}, previous={token: prev_token})
    df.groupby("k").score.mean()

`candidate_scores` returns the per-candidate sub-scores without comfort across frames,
which is what a vocabulary scorer is trained on (exp_navsim/planner/score_vocab.py).
"""

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[2] / "navsim/navsim/planning/script/config/pdm_scoring"
CACHE = Path("outputs/navsim_exp/metric_cache")
METRICS = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
           "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
           "history_comfort")
MULTI = METRICS[:4]
WEIGHTS = {"ego_progress": 5.0, "time_to_collision_within_bound": 5.0, "lane_keeping": 2.0,
           "history_comfort": 2.0, "two_frame_extended_comfort": 2.0}

_W = {}  # per-worker state


def _init(split, tokens, preload=True):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from navsim.common.dataloader import MetricCacheLoader

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="default_run_pdm_score")
    _W["simulator"] = instantiate(cfg.simulator)
    _W["scorer"] = instantiate(cfg.scorer)
    _W["traffic"] = instantiate(cfg.traffic_agents_policy.non_reactive, _W["simulator"].proposal_sampling)
    loader = MetricCacheLoader(CACHE / split)
    _W["loader"] = loader
    _W["cache"] = {t: loader.get_from_token(t) for t in tokens} if preload else {}
    _W["human"] = {}


def _cache(token):
    """Metric cache of a token: kept from the start, or loaded for this call only."""
    return _W["cache"][token] if token in _W["cache"] else _W["loader"].get_from_token(token)


def _states(poses, ego_state, sampling):
    """poses: (8, 3) at 2 Hz, or a navsim Trajectory."""
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory

    traj = poses if isinstance(poses, Trajectory) else Trajectory(
        np.asarray(poses, np.float32), TrajectorySampling(time_horizon=4, interval_length=0.5))
    return get_trajectory_as_array(transform_trajectory(traj, ego_state), sampling, ego_state.time_point)


def _human_failures(token, mc):
    """Metric names the logged human trajectory fails (the filter sets them to 1)."""
    if token in _W["human"]:
        return _W["human"][token]
    from navsim.common.enums import SceneFrameType

    sim, scorer = _W["simulator"], _W["scorer"]
    failed = set()
    if scorer._config.human_penalty_filter and mc.scene_type == SceneFrameType.ORIGINAL:
        states = _states(mc.human_trajectory, mc.ego_state, sim.proposal_sampling)
        simulated = sim.simulate_proposals(states[None], mc.ego_state)
        tracks = _W["traffic"].simulate_environment(simulated[0], mc)
        row = scorer.score_proposals(simulated, mc.observation, mc.centerline, mc.route_lane_ids,
                                     mc.drivable_area_map, mc.map_parameters, tracks)[0]
        failed = {m for m in METRICS if row[m].iloc[0] == 0}
    _W["human"][token] = failed
    return failed


def _score_scene(args):
    """(token, (K, 8, 3) poses) -> (K, len(METRICS)) sub-scores, (K, 41, 11) simulated states."""
    from navsim.evaluate.pdm_score import get_trajectory_as_array

    token, poses = args
    mc, sim, scorer = _cache(token), _W["simulator"], _W["scorer"]
    sampling = sim.proposal_sampling
    pdm = get_trajectory_as_array(mc.trajectory, sampling, mc.ego_state.time_point)
    cand = np.stack([_states(p, mc.ego_state, sampling) for p in poses])
    simulated = sim.simulate_proposals(np.concatenate([pdm[None], cand]), mc.ego_state)
    tracks = _W["traffic"].simulate_environment(simulated[1], mc)   # log replay: does not depend on ego
    rows = scorer.score_proposals(simulated, mc.observation, mc.centerline, mc.route_lane_ids,
                                  mc.drivable_area_map, mc.map_parameters, tracks, mc.past_human_trajectory)
    out = np.array([[r[m].iloc[0] for m in METRICS] for r in rows[1:]], dtype=np.float64)

    # progress as if each candidate had been scored next to PDM-Closed alone
    raw = scorer._progress_raw
    mult = scorer._multi_metrics.prod(axis=0)
    norm = np.maximum(raw[0] * mult[0], raw[1:] * mult[1:])
    thr = scorer._config.progress_distance_threshold
    ep = np.where(norm > thr, np.clip(raw[1:] / np.maximum(norm, 1e-9), 0.0, 1.0), 1.0)
    out[:, METRICS.index("ego_progress")] = ep

    for m in _human_failures(token, mc):
        out[:, METRICS.index(m)] = 1.0
    meta = (mc.log_name, float(mc.timepoint.time_s))
    return token, out.astype(np.float32), simulated[1:], meta


def _reference_poses(token):
    """(PDM-Closed, human) trajectories of the metric cache as 8 poses at 2 Hz in the ego frame."""
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.evaluate.pdm_score import get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import \
        convert_absolute_to_relative_se2_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

    mc = _cache(token)
    states = get_trajectory_as_array(mc.trajectory, TrajectorySampling(num_poses=8, interval_length=0.5),
                                     mc.ego_state.time_point)[1:, StateIndex.STATE_SE2]
    pdm = convert_absolute_to_relative_se2_array(mc.ego_state.rear_axle, states)
    return token, pdm.astype(np.float32), np.asarray(mc.human_trajectory.poses, np.float32)


def epdms(sub, comfort=None):
    """sub (..., len(METRICS)), comfort (...) or None/NaN -> EPDMS (...)."""
    mult = np.prod(sub[..., :4], axis=-1)
    num = sum(WEIGHTS[m] * sub[..., METRICS.index(m)] for m in METRICS[4:])
    den = sum(WEIGHTS[m] for m in METRICS[4:])
    if comfort is not None:
        c = np.asarray(comfort, dtype=np.float64)
        ok = np.isfinite(c)
        num = num + np.where(ok, WEIGHTS["two_frame_extended_comfort"] * np.nan_to_num(c), 0.0)
        den = den + np.where(ok, WEIGHTS["two_frame_extended_comfort"], 0.0)
    return mult * num / den


class FastPDM:
    def __init__(self, split, tokens, workers=16, preload=True):
        """preload=False loads each metric cache when its scene is scored (large token sets)."""
        self.tokens = list(dict.fromkeys(tokens))
        self.workers = workers
        ctx = mp.get_context("fork")
        # One pool per token shard, so every worker only holds its own caches.
        shards = [self.tokens[i::workers] for i in range(workers)]
        self.owner = {t: i for i, s in enumerate(shards) for t in s}
        self.pools = [ctx.Pool(1, initializer=_init, initargs=(split, s, preload)) for s in shards]

    def close(self):
        for p in self.pools:
            p.terminate()

    def _map(self, items):
        jobs = {}
        for token, poses in items:
            jobs.setdefault(self.owner[token], []).append((token, poses))
        async_results = [self.pools[w].map_async(_score_scene, job) for w, job in jobs.items()]
        return [r for a in async_results for r in a.get()]

    def reference_poses(self, tokens):
        """{token: (PDM-Closed (8, 3), human (8, 3))} from the metric caches."""
        jobs = {}
        for t in tokens:
            jobs.setdefault(self.owner[t], []).append(t)
        res = [self.pools[w].map_async(_reference_poses, job) for w, job in jobs.items()]
        return {t: (p, h) for a in res for t, p, h in a.get()}

    def candidate_scores(self, trajectories):
        """{token: (K, 8, 3)} -> {token: (K, len(METRICS))} sub-scores (no two-frame comfort)."""
        return {t: s for t, s, _, _ in self._map(trajectories.items())}

    def score(self, trajectories, previous):
        """trajectories: {token: (K, 8, 3)} for current and previous tokens; previous:
        {current token: previous token}. -> one row per (current token, k) with the
        sub-scores, two_frame_extended_comfort and the EPDMS."""
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import \
            ego_is_two_frame_extended_comfort

        res = {t: (s, st, meta) for t, s, st, meta in self._map(trajectories.items())}
        interval = 0.1
        rows = []
        for t, p in previous.items():
            sub, states, (_, time) = res[t]
            K = sub.shape[0]
            comfort = np.full(K, np.nan)
            if p in res:
                psub, pstates, (_, ptime) = res[p]
                dt = time - ptime
                if 0 < dt < 0.55:
                    shift = round(dt / interval)
                    n = states.shape[1] - shift
                    comfort = ego_is_two_frame_extended_comfort(
                        states[:, :-shift].astype(np.float64), pstates[:, shift:].astype(np.float64),
                        np.arange(n) * interval).astype(np.float64)
            score = epdms(sub, comfort)
            for k in range(K):
                rows.append({"token": t, "k": k, **dict(zip(METRICS, sub[k])),
                             "two_frame_extended_comfort": comfort[k], "score": score[k]})
        return pd.DataFrame(rows)


def load_subset(path):
    import json
    d = json.load(open(path))
    return d["tokens"], dict(zip(d["tokens"], d["previous"]))


if __name__ == "__main__":
    # Check against an official run: python -m exp_navsim.planner.fast_pdm <predictions.npz> <sample> <official csv>
    import sys
    import time

    from exp_navsim.eval.navsim_agent import mean_trajectory

    npz, sample, csv = sys.argv[1], sys.argv[2], sys.argv[3]
    subset = sys.argv[4] if len(sys.argv) > 4 else "outputs/16_fast_planner/subset_500.json"
    tokens, previous = load_subset(subset)
    data = np.load(npz)
    samples = data["samples"].astype(np.float32)
    poses = mean_trajectory(samples) if sample == "mean" else samples[int(sample)]
    lookup = dict(zip(data["token"].tolist(), poses))
    t0 = time.time()
    scorer = FastPDM("navtest", tokens + list(previous.values()))
    t1 = time.time()
    trajs = {t: lookup[t][None] for t in set(tokens) | set(previous.values())}
    scorer.score(trajs, previous)                     # first call also loads the caches
    t2 = time.time()
    df = scorer.score(trajs, previous)
    t3 = time.time()
    off = pd.read_csv(csv).set_index("token").loc[df.token]
    print(f"start {t1 - t0:.1f}s  load + score {t2 - t1:.1f}s  score again {t3 - t2:.1f}s  scenes {len(df)}")
    for m in (*METRICS, "two_frame_extended_comfort", "score"):
        a, b = df[m].to_numpy(), off[m].to_numpy()
        print(f"{m:32s} fast {np.nanmean(a):.4f}  official {np.nanmean(b):.4f}  max |diff| {np.nanmax(np.abs(a - b)):.4f}")
    scorer.close()
