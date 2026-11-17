"""In-process NAVSIM v2.2 two-stage EPDMS on a few navhard groups, with the official numbers.

The official two-stage runner (eval/pdm.py --split navhard) scores 5,912 tokens and takes
about half an hour. Its EPDMS is a mean over 225 *groups*; a group is one pair of real
frames (now, previous) plus the synthetic frames that branch off them:

    group (now, prev) with second-stage pairs [(now_1, prev_1), ..., (now_B, prev_B)]
    score(group) = [ s(now) * avg_w s(now_i)  +  s(prev) * avg_w s(prev_i) ] / 2
    EPDMS        = mean over groups

so a subset of groups is an unbiased, exactly-aggregated estimate of the full split. This
module keeps the metric caches of the chosen groups in a pool of workers and reproduces the
aggregation of navsim's `create_scene_aggregators` / `calculate_individual_mapping_scores`:
the pseudo-closed-loop Gaussian weights of the stage-two frames, the two-frame extended
comfort of every pair, and the stage-one human penalty filter.

Unlike navtest (planner/fast_pdm.py, log-replay traffic), both stages run the *reactive* IDM
traffic policy, which reacts to the ego roll-out, so every candidate needs its own traffic
simulation and the candidates of a scene cannot share one scorer call.

**Branch pruning.** avg_w weights a stage-two frame by exp(-d^2 / 0.2), where d is the
distance from the end of the stage-one plan to the start of that frame. With sigma = 0.32 m
nearly all branches of a group get a weight of zero for a given plan, so only the few that
the plan actually reaches are scored: stage one first, then the branches whose relative
weight is above `prune`. This cuts a group from ~24 tokens to ~6 and is exact to ~1e-4
(`prune=0` scores every branch).

**Speed.** Almost all the time is the IDM simulation. The workers patch three exact hot
spots of it (_patch_simulation; FAST_EPDMS_PATCH=0 = off), round candidates to 1e-4 and
simulate duplicates once, and keep every result in an sqlite cache under DISK_CACHE
(FAST_EPDMS_DISK_CACHE=0 = off), so plans that were scored before cost nothing.

    groups = load_groups("outputs/24_v22/groups_32.json")
    scorer = FastEPDMS(groups)                       # starts the pool
    df = scorer.score({token: (K, 8, 3) poses})      # one row per (group, half, k)
    scorer.summary(df).loc[0, "score"]               # EPDMS of the first candidate
"""

import bisect
import hashlib
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.planner.fast_pdm import CACHE, CONFIG_DIR, METRICS, epdms

MAPPING = Path("navsim/navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml")
START_POINTS = Path("outputs/24_v22/navhard_start_points.json")
SIGMA_SQ = 0.1          # Gaussian kernel of the pseudo-closed-loop weights (SceneAggregator)
COLS = METRICS + ("two_frame_extended_comfort", "score")
_W = {}                 # per-worker state


# --------------------------------------------------------------------------------------- groups

def all_groups(path=MAPPING):
    """-> [(now, prev, [(now_i, prev_i), ...]), ...] from the devkit's reactive_all_mapping."""
    import yaml
    raw = yaml.safe_load(path.read_text())["reactive_all_mapping"]
    return [(str(now), str(prev), [(str(a), str(b)) for a, b in pairs]) for now, prev, pairs in raw]


def group_tokens(groups):
    """Every token a list of groups needs, in a stable order."""
    toks = []
    for now, prev, pairs in groups:
        toks += [now, prev] + [t for pair in pairs for t in pair]
    return list(dict.fromkeys(toks))


def save_groups(groups, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([[n, p, [list(x) for x in pairs]] for n, p, pairs in groups]))


def load_groups(path):
    return [(n, p, [tuple(x) for x in pairs]) for n, p, pairs in json.loads(Path(path).read_text())]


# --------------------------------------------------------------------------------------- worker

CACHE_KEEP = int(os.environ.get("FAST_EPDMS_CACHE_KEEP", 200))   # metric caches per worker
# on-disk results of (token, candidate); FAST_EPDMS_DISK_CACHE=0 turns it off
DISK_CACHE = Path(os.environ.get("FAST_EPDMS_DISK_CACHE_DIR", "outputs/navsim_exp/epdms_cache"))
DISK_VERSION = "v1"     # bump when the scorer config or the simulation changes
DECIMALS = 4            # candidates are rounded to 0.1 mm / 1e-4 rad before scoring (cache key)


def _simplify_once(occupancy_map, tolerance=1e-5):
    """navsim's simplify_occupancy_map_geometries without re-simplifying the same geometry.

    The IDM manager calls it once per agent and time step, so every geometry of the map was
    simplified again and again (O(agents^2) shapely calls per step, a third of the traffic
    simulation). Simplifying an already simplified geometry returns it unchanged, so the
    geometries this function produced itself are skipped; new or replaced ones still go
    through shapely exactly as before.
    """
    done = occupancy_map.__dict__.setdefault("_simplified", {})
    for gid in occupancy_map.get_all_ids():
        geom = occupancy_map.get(gid)
        if geom is None or done.get(gid) is geom:
            continue
        geom = geom.simplify(tolerance, preserve_topology=True)
        occupancy_map.set(gid, geom)
        done[gid] = geom
    return occupancy_map


class _LinearInterp:
    """scipy's interp1d(x, y, axis=0) for 2-D y, with the same arithmetic but less overhead.

    The IDM agents query their paths (nuplan InterpolatedPath / AngularInterpolator) a few
    hundred thousand times per token, mostly at single points, where scipy's checks and
    reshapes cost ~40 us per call. This repeats _call_linear and _check_bounds step by step,
    so the results are bit-identical.
    """

    def __init__(self, x, y):
        x, y = np.array(x), np.array(y)
        ind = np.argsort(x, kind="mergesort")
        x, y = x[ind], np.take(y, ind, axis=0)
        if not issubclass(y.dtype.type, np.inexact):
            y = y.astype(np.float64)
        self.x, self.extra = x, y.shape[1:]
        self._y = y.reshape((y.shape[0], -1))
        self._xs = x.tolist()

    def _scalar(self, xv):
        """One query point: the same float64 operations without the array bookkeeping."""
        xs = self._xs
        if xv < xs[0] or xv > xs[-1]:
            raise ValueError("A value in x_new is outside the interpolation range.")
        hi = min(max(bisect.bisect_left(xs, xv), 1), len(xs) - 1)
        y_lo = self._y[hi - 1]
        slope = (self._y[hi] - y_lo) / (xs[hi] - xs[hi - 1])
        return (slope * (xv - xs[hi - 1]) + y_lo).reshape(self.extra)

    def __call__(self, x_new):
        x_new = np.asarray(x_new)
        if not issubclass(x_new.dtype.type, np.inexact):
            x_new = x_new.astype(np.float64)
        if x_new.ndim == 0 and x_new.dtype == np.float64 and x_new == x_new:
            return self._scalar(float(x_new))
        shape, xn = x_new.shape, x_new.ravel()
        hi = np.searchsorted(self.x, xn).clip(1, len(self.x) - 1).astype(int)
        lo = hi - 1
        x_lo, y_lo = self.x[lo], self._y[lo]
        slope = (self._y[hi] - y_lo) / (self.x[hi] - x_lo)[:, None]
        y_new = slope * (xn - x_lo)[:, None] + y_lo
        if (xn < self.x[0]).any() or (xn > self.x[-1]).any():
            raise ValueError("A value in x_new is outside the interpolation range.")
        return y_new.reshape(shape + self.extra)


def _interp1d(x, y, kind="linear", axis=0, **kw):
    """Drop-in for scipy.interpolate.interp1d; the fast path only for the case nuplan uses."""
    from scipy.interpolate import interp1d
    if kind != "linear" or kw or np.ndim(y) < 2 or axis % np.ndim(y) != 0:
        return interp1d(x, y, kind=kind, axis=axis, **kw)
    return _LinearInterp(x, y)


_PARKED = {}    # (map, track, x, y) -> (in lanes, lateral deviation), pure map queries


def _parked_lookup(agent, map_api):
    """The map part of navsim's open-loop parked-vehicle test, memoized.

    NavsimIDMAgents.get_observation repeats it for every inactive vehicle at every time step
    of every candidate, although it only depends on the vehicle's logged pose and the map.
    """
    from nuplan.common.maps.abstract_map import SemanticMapLayer
    from navsim.planning.simulation.observation.navsim_idm.navsim_idm_agents_builder import get_starting_segment

    key = (map_api.map_name, agent.track_token, agent.center.x, agent.center.y, agent.center.heading)
    if key not in _PARKED:
        if len(_PARKED) > 200_000:
            _PARKED.clear()
        is_in_lanes = map_api.is_in_layer(agent.center, SemanticMapLayer.LANE) or map_api.is_in_layer(
            agent.center, SemanticMapLayer.INTERSECTION)
        route, _ = get_starting_segment(agent, map_api)
        lateral_deviation = None
        if route:
            state_on_path = route.baseline_path.get_nearest_pose_from_position(agent.center)
            lateral_deviation = np.hypot(state_on_path.x - agent.center.x, state_on_path.y - agent.center.y)
        _PARKED[key] = (is_in_lanes, lateral_deviation)
    return _PARKED[key]


def _get_observation(self, ego_state, vehicle_current_tracks, map_api, objects_future_tracks):
    """NavsimIDMAgents.get_observation (navsim v2.2) with the parked-vehicle map test memoized."""
    import hashlib as _hashlib
    import uuid
    from nuplan.common.actor_state.scene_object import SceneObjectMetadata

    detections = self._get_idm_agent_manager(ego_state, vehicle_current_tracks, map_api).get_active_agents(
        self.current_iteration, self._planned_trajectory_samples, self._planned_trajectory_sample_interval)
    for object in detections.tracked_objects.tracked_objects:
        object._metadata = SceneObjectMetadata(
            timestamp_us=ego_state.time_us,
            token=_hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[: len(object.metadata.token)],
            track_token=object.metadata.track_token,
            track_id=object.metadata.track_id,
            category_name=object.metadata.category_name)
    active = {a.track_token for a in detections.tracked_objects.tracked_objects}
    inactive_vehicle_agents = [a for a in vehicle_current_tracks.tracked_objects.tracked_objects
                               if a.track_token not in active]
    if self._add_open_loop_parked_vehicles and inactive_vehicle_agents:
        for agent in inactive_vehicle_agents:
            is_stationary = agent.velocity.magnitude() < 0.1
            is_in_lanes, lateral_deviation = _parked_lookup(agent, map_api)
            collides_with_active_agents = any(
                [agent.box.geometry.intersects(active_agent.box.geometry)
                 for active_agent in detections.tracked_objects.tracked_objects])
            if is_stationary and not is_in_lanes and not collides_with_active_agents:
                detections.tracked_objects.tracked_objects.append(agent)
                continue
            if (lateral_deviation is not None and lateral_deviation > self._idm_snap_threshold
                    and not collides_with_active_agents):
                detections.tracked_objects.tracked_objects.append(agent)
    if self._open_loop_detections_types:
        open_loop_detections = objects_future_tracks.get_tracked_objects_of_types(self._open_loop_detections_types)
        detections.tracked_objects.tracked_objects.extend(open_loop_detections)
    return detections


def _patch_simulation():
    """Exact speed-ups of the reactive IDM traffic simulation (FAST_EPDMS_PATCH=0: off)."""
    from types import SimpleNamespace
    import scipy.interpolate
    import nuplan.common.geometry.compute as compute
    import nuplan.planning.simulation.path.interpolated_path as interpolated_path
    import navsim.planning.simulation.observation.navsim_idm.navsim_idm_agent_manager as manager
    from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents

    manager.simplify_occupancy_map_geometries = _simplify_once
    compute.interp1d = _interp1d
    interpolated_path.sp_interp = SimpleNamespace(**{**vars(scipy.interpolate), "interp1d": _interp1d})
    NavsimIDMAgents.get_observation = _get_observation


def _init(split, tokens, disk_cache=True):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from navsim.common.dataloader import MetricCacheLoader

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import warnings
    # nuplan_map/utils.py:413 casts NaN to int on some lanes; harmless, but it floods the logs
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="nuplan")
    warnings.filterwarnings("ignore", message="invalid value encountered in cast")
    if os.environ.get("FAST_EPDMS_PATCH", "1") != "0":
        _patch_simulation()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="default_run_pdm_score")
    _W["simulator"] = instantiate(cfg.simulator)
    _W["scorer"] = instantiate(cfg.scorer)
    # both navhard stages use the reactive (IDM) policy, as in run_pdm_score.py
    _W["traffic"] = instantiate(cfg.traffic_agents_policy.reactive, _W["simulator"].proposal_sampling)
    _W["loader"] = MetricCacheLoader(CACHE / split)
    _W["cache"], _W["human"], _W["meta"] = {}, {}, {}
    _W["db"] = _open_db(split) if disk_cache else None


def _open_db(split):
    import sqlite3
    DISK_CACHE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DISK_CACHE / f"{split}_{DISK_VERSION}.sqlite", timeout=120)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS cand (key TEXT PRIMARY KEY, value BLOB)")
    db.execute("CREATE TABLE IF NOT EXISTS human (token TEXT PRIMARY KEY, failed TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS meta (token TEXT PRIMARY KEY, value TEXT)")
    db.commit()
    return db


def _db_get(keys):
    db = _W["db"]
    if db is None or not keys:
        return {}
    q = f"SELECT key, value FROM cand WHERE key IN ({','.join('?' * len(keys))})"
    return {k: pickle.loads(v) for k, v in db.execute(q, keys).fetchall()}


def _db_put(items):
    db = _W["db"]
    if db is None or not items:
        return
    db.executemany("INSERT OR IGNORE INTO cand VALUES (?, ?)",
                   [(k, pickle.dumps(v, protocol=4)) for k, v in items.items()])
    db.commit()


def _mc(token):
    """Metric cache of a token; loaded on first use and then kept in the worker."""
    if token not in _W["cache"]:
        # bounded: a worker of a whole-split pool would otherwise keep every cache it loaded,
        # and the job shares a 40 GB memory limit; evicted caches are simply reloaded
        while len(_W["cache"]) >= CACHE_KEEP:
            _W["cache"].pop(next(iter(_W["cache"])))
        _W["cache"][token] = _W["loader"].get_from_token(token)
    return _W["cache"][token]


def _start_point(token):
    mc = _mc(token)
    return token, [float(mc.ego_state.rear_axle.x), float(mc.ego_state.rear_axle.y)]


def _states(poses, ego_state, sampling):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory

    traj = poses if isinstance(poses, Trajectory) else Trajectory(
        np.asarray(poses, np.float32), TrajectorySampling(time_horizon=4, interval_length=0.5))
    return get_trajectory_as_array(transform_trajectory(traj, ego_state), sampling, ego_state.time_point)


def _human_failures(token):
    """Metric names the logged trajectory fails; the filter sets them to 1 (stage one only).

    It does not depend on the plan, so it is also kept in the disk cache.
    """
    if token in _W["human"]:
        return _W["human"][token]
    from navsim.common.enums import SceneFrameType

    db = _W["db"]
    row = db.execute("SELECT failed FROM human WHERE token = ?", (token,)).fetchone() if db else None
    if row is not None:
        _W["human"][token] = set(json.loads(row[0]))
        return _W["human"][token]
    mc, sim, scorer = _mc(token), _W["simulator"], _W["scorer"]
    failed = set()
    if scorer._config.human_penalty_filter and mc.scene_type == SceneFrameType.ORIGINAL:
        states = _states(mc.human_trajectory, mc.ego_state, sim.proposal_sampling)
        simulated = sim.simulate_proposals(states[None], mc.ego_state)
        tracks = _W["traffic"].simulate_environment(simulated[0], mc)
        row = scorer.score_proposals(simulated, mc.observation, mc.centerline, mc.route_lane_ids,
                                     mc.drivable_area_map, mc.map_parameters, tracks)[0]
        failed = {m for m in METRICS if row[m].iloc[0] == 0}
    _W["human"][token] = failed
    if db is not None:
        db.execute("INSERT OR IGNORE INTO human VALUES (?, ?)", (token, json.dumps(sorted(failed))))
        db.commit()
    return failed


def _simulate(mc, pdm, poses):
    """Sub-scores and simulated ego states of one candidate (with the PDM reference)."""
    sim, scorer = _W["simulator"], _W["scorer"]
    cand = _states(poses, mc.ego_state, sim.proposal_sampling)
    simulated = sim.simulate_proposals(np.stack([pdm, cand]), mc.ego_state)
    # the reactive policy reacts to the ego roll-out, so it runs per candidate
    tracks = _W["traffic"].simulate_environment(simulated[1], mc)
    row = scorer.score_proposals(simulated, mc.observation, mc.centerline, mc.route_lane_ids,
                                 mc.drivable_area_map, mc.map_parameters, tracks,
                                 mc.past_human_trajectory)[1]
    return (np.asarray([row[m].iloc[0] for m in METRICS], np.float64),
            np.asarray(simulated[1], np.float32))


def _meta(token):
    """(rear axle x, y, heading, time [s], is stage one) of a token, kept in the disk cache."""
    if token in _W["meta"]:
        return _W["meta"][token]
    from navsim.common.enums import SceneFrameType

    db = _W["db"]
    row = db.execute("SELECT value FROM meta WHERE token = ?", (token,)).fetchone() if db else None
    if row is not None:
        _W["meta"][token] = tuple(json.loads(row[0]))
        return _W["meta"][token]
    mc = _mc(token)
    a = mc.ego_state.rear_axle
    meta = (a.x, a.y, a.heading, float(mc.timepoint.time_s), bool(mc.scene_type == SceneFrameType.ORIGINAL))
    _W["meta"][token] = meta
    if db is not None:
        db.execute("INSERT OR IGNORE INTO meta VALUES (?, ?)", (token, json.dumps(meta)))
        db.commit()
    return meta


def _score_scene(args):
    """(token, (K, 8, 3) poses) -> sub-scores, simulated states and end points per candidate.

    Identical candidates (after rounding to DECIMALS) are simulated once, and candidates
    already in the disk cache are not simulated at all.
    """
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    from navsim.evaluate.pdm_score import get_trajectory_as_array

    token, poses = args
    poses = np.round(np.asarray(poses, np.float64), DECIMALS).astype(np.float32)
    keys = [f"{token}:{hashlib.sha1(p.tobytes()).hexdigest()}" for p in poses]
    known = _db_get(list(dict.fromkeys(keys)))
    hits = len(known)
    new = {}
    pdm = None
    for p, key in zip(poses, keys):
        if key in known:
            continue
        mc = _mc(token)          # only loaded when something has to be simulated
        if pdm is None:
            pdm = get_trajectory_as_array(mc.trajectory, _W["simulator"].proposal_sampling,
                                          mc.ego_state.time_point)
        known[key] = new[key] = _simulate(mc, pdm, p)
    _db_put(new)
    sub = np.stack([known[k][0] for k in keys])
    states = np.stack([known[k][1] for k in keys])
    for m in _human_failures(token):
        sub[:, METRICS.index(m)] = 1.0
    x, y, heading, time_s, stage_one = _meta(token)
    ends = []
    for p in poses:
        e = relative_to_absolute_poses(StateSE2(x, y, heading), [StateSE2(*np.asarray(p, np.float64)[-1])])[0]
        ends.append([e.x, e.y])
    return token, dict(sub=sub.astype(np.float32), states=states,
                       end=np.asarray(ends), time=time_s, stage_one=stage_one,
                       stats=np.array([len(keys), len(set(keys)), hits, len(new)]))


# ---------------------------------------------------------------------------------- aggregation

def _comfort(cur, prev, interval=0.1):
    """Two-frame extended comfort of every candidate, from the two frames' simulated states."""
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import \
        ego_is_two_frame_extended_comfort

    dt = cur["time"] - prev["time"]
    assert 0 < dt < 0.55, f"invalid observation interval {dt}"
    shift = round(dt / interval)
    a, b = cur["states"][:, :-shift].astype(np.float64), prev["states"][:, shift:].astype(np.float64)
    return ego_is_two_frame_extended_comfort(a, b, np.arange(a.shape[1]) * interval).astype(np.float64)


def _rows(res, token, comfort):
    """Per-candidate score columns of one token, with the two-frame comfort injected."""
    sub = res[token]["sub"].astype(np.float64)
    out = {m: sub[:, i] for i, m in enumerate(METRICS)}
    out["two_frame_extended_comfort"] = comfort
    out["score"] = epdms(sub, comfort)
    return out


def _raw_weights(end, starts):
    """Unnormalized Gaussian weights, (K, B), from plan end points (K, 2) and starts (B, 2)."""
    d = ((end[:, None, 0] - starts[None, :, 0]) ** 2 + (end[:, None, 1] - starts[None, :, 1]) ** 2)
    return np.exp(-d / (2 * SIGMA_SQ))


def _normalize(w):
    """navsim's normalization: uniform when the weights sum to (almost) zero."""
    s = w.sum(1, keepdims=True)
    bad = ~np.isfinite(s) | np.isclose(s, 0.0)
    return np.where(bad, 1.0 / w.shape[1], w / np.where(bad, 1.0, s))


class FastEPDMS:
    """EPDMS of candidate trajectories on a subset of navhard groups, exactly as the devkit."""

    def __init__(self, groups, split="navhard", workers=10, prune=1e-3, start_points=START_POINTS,
                 disk_cache=None):
        """disk_cache: keep every scored (token, candidate) in DISK_CACHE (default on; the
        environment variable FAST_EPDMS_DISK_CACHE=0 turns it off)."""
        if disk_cache is None:
            disk_cache = os.environ.get("FAST_EPDMS_DISK_CACHE", "1") != "0"
        self.groups = list(groups)
        self.tokens = group_tokens(self.groups)
        self.prune = prune
        ctx = mp.get_context("fork")
        workers = max(1, min(workers, len(self.tokens)))
        shards = [self.tokens[i::workers] for i in range(workers)]
        self.owner = {t: i for i, s in enumerate(shards) for t in s}
        self.pools = [ctx.Pool(1, initializer=_init, initargs=(split, s, disk_cache)) for s in shards]
        self.starts = self._start_points(start_points)

    def _start_points(self, path):
        """{token: (x, y)} of every stage-two frame, needed to prune the branches."""
        path = Path(path)
        known = json.loads(path.read_text()) if path.exists() else {}
        need = [t for t in self.tokens if t not in known]
        if need:
            print(f"[fast_epdms] start points of {len(need)} tokens", flush=True)
            jobs = {}
            for t in need:
                jobs.setdefault(self.owner[t], []).append(t)
            res = [self.pools[w].map_async(_start_point, job) for w, job in jobs.items()]
            known.update({t: xy for a in res for t, xy in a.get()})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(known))
        return {t: np.asarray(known[t]) for t in self.tokens}

    def close(self):
        for p in self.pools:
            p.terminate()

    def _map(self, items):
        jobs = {}
        for token, poses in items:
            jobs.setdefault(self.owner[token], []).append((token, poses))
        res = [self.pools[w].map_async(_score_scene, job) for w, job in jobs.items()]
        out = {t: r for a in res for t, r in a.get()}
        for r in out.values():
            self.stats += r.pop("stats")
        return out

    def _needed_pairs(self, res, groups):
        """Per group, the stage-two pairs that carry all but `prune` of the weight.

        The branches are taken in decreasing weight until 1 - prune of the mass is covered,
        so renormalizing over the kept ones changes a weight by at most `prune` relatively.
        A candidate whose weights sum to zero falls back to the uniform weights of the
        devkit and needs every branch.
        """
        out = []
        for now, prev, pairs in groups:
            keep = np.zeros(len(pairs), dtype=bool)
            for parent, which in ((now, 0), (prev, 1)):
                starts = np.stack([self.starts[p[which]] for p in pairs])
                w = _raw_weights(res[parent]["end"], starts)                      # (K, B)
                total = w.sum(1, keepdims=True)
                for k in range(len(w)):
                    if not np.isfinite(total[k, 0]) or np.isclose(total[k, 0], 0.0):
                        keep[:] = True                                            # uniform fallback
                        continue
                    order = np.argsort(-w[k])
                    cum = np.cumsum(w[k][order]) / total[k, 0]
                    keep[order[:int(np.searchsorted(cum, 1 - self.prune)) + 1]] = True
            out.append(np.flatnonzero(keep))
        return out

    def score(self, trajectories, groups=None, per_token=False):
        """{token: (K, 8, 3)} for every token of the groups -> one row per (group, half, k).

        `groups` defaults to every group the pool was built for; passing a subset of them
        scores only those (the pool and the start points cover a superset). Columns: the
        group sub-scores (stage one x stage two, as the devkit multiplies them), `score` =
        EPDMS of that half, and `<m>_stage_one` / `<m>_stage_two`. With `per_token=True` the
        per-token rows before the aggregation come too, which is what the official CSV holds.
        """
        groups = self.groups if groups is None else list(groups)
        self.stats = np.zeros(4, dtype=int)     # candidates, unique, disk hits, simulated
        if self.prune:
            stage_one = list(dict.fromkeys([t for g in groups for t in g[:2]]))
            res = self._map([(t, trajectories[t]) for t in stage_one])
            keep = self._needed_pairs(res, groups)
            need = {t for g, idx in zip(groups, keep) for j in idx for t in g[2][j]}
            res.update(self._map([(t, trajectories[t]) for t in sorted(need)]))
        else:
            keep = [np.arange(len(g[2])) for g in groups]
            res = self._map([(t, trajectories[t]) for t in group_tokens(groups)])
        self.n_scored = len(res)
        c, u, h, n = self.stats
        print(f"[fast_epdms] {len(res)} tokens, {c} candidates: {u} unique, {h} from the disk "
              f"cache, {n} simulated", flush=True)
        agg = self._aggregate(res, keep, groups)
        return (agg, self._token_rows(res, keep, groups)) if per_token else agg

    def _comforts(self, res, keep, groups):
        """Two-frame extended comfort of every scored token, per candidate."""
        comfort = {}
        for (now, prev, pairs), idx in zip(groups, keep):
            comfort[now] = comfort[prev] = _comfort(res[now], res[prev])
            for j in idx:
                a, b = pairs[j]
                comfort[a] = comfort[b] = _comfort(res[a], res[b])
        return comfort

    def _token_rows(self, res, keep, groups):
        comfort = self._comforts(res, keep, groups)
        rows = []
        for t, c in comfort.items():
            r = _rows(res, t, c)
            for k in range(len(c)):
                rows.append({"token": t, "k": k, "stage_one": res[t]["stage_one"],
                             **{m: r[m][k] for m in COLS}})
        return pd.DataFrame(rows)

    def _aggregate(self, res, keep, groups):
        comfort = self._comforts(res, keep, groups)
        rows = []
        for gi, ((now, prev, pairs), idx) in enumerate(zip(groups, keep)):
            for half, (parent, which) in enumerate(((now, 0), (prev, 1))):
                children = [pairs[j][which] for j in idx]
                s1 = _rows(res, parent, comfort[parent])
                w = _normalize(_raw_weights(res[parent]["end"],
                                            np.stack([self.starts[c] for c in children])))
                child = [_rows(res, c, comfort[c]) for c in children]
                row = {"group": gi, "half": half, "token": parent}
                for m in COLS:
                    s2 = sum(w[:, j] * child[j][m] for j in range(len(children)))
                    row[f"{m}_stage_one"], row[f"{m}_stage_two"] = s1[m], s2
                    row[m] = s1[m] * s2
                for k in range(len(s1["score"])):
                    rows.append({"k": k, **{a: (b[k] if np.ndim(b) else b) for a, b in row.items()}})
        return pd.DataFrame(rows)

    @staticmethod
    def summary(df):
        """Mean over groups (both halves) per candidate index: EPDMS and the stage components."""
        per_group = df.groupby(["group", "k"]).mean(numeric_only=True).reset_index()
        return per_group.drop(columns=["group", "half"]).groupby("k").mean()


if __name__ == "__main__":
    # Check against an official run:
    #   python -m exp_navsim.v22.fast_epdms <predictions.npz> <sample|mean> <official csv> [n_groups] [prune]
    import sys
    import time

    from exp_navsim.eval.navsim_agent import mean_trajectory

    npz, sample, csv = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    prune = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
    groups = all_groups()
    groups = groups if n <= 0 else groups[:n]
    data = np.load(npz)
    samples = data["samples"].astype(np.float32)
    poses = mean_trajectory(samples) if sample == "mean" else samples[int(sample)]
    lookup = dict(zip(data["token"].tolist(), poses))
    t0 = time.time()
    scorer = FastEPDMS(groups, prune=prune)
    t1 = time.time()
    df, tok = scorer.score({t: lookup[t][None] for t in group_tokens(groups)}, per_token=True)
    t2 = time.time()
    s = scorer.summary(df)
    print(f"start {t1 - t0:.0f}s  score {t2 - t1:.0f}s  groups {len(groups)}  "
          f"tokens {len(scorer.tokens)} -> scored {scorer.n_scored}  prune {prune}")
    print(f"fast EPDMS on these groups {100 * s.loc[0, 'score']:.3f}")

    off = pd.read_csv(csv)
    comb = off[off.token == "extended_pdm_score_combined"]
    if len(comb):
        print(f"official EPDMS on the full split {100 * float(comb['score'].iloc[0]):.3f}")
    off = off.set_index("token")
    tok = tok[tok.k == 0].set_index("token")
    for m in COLS:
        a, b = [], []
        for t, r in tok.iterrows():
            if t not in off.index:
                continue
            col = "score" if m == "score" else f"{m}_stage_one" if r["stage_one"] else f"{m}_stage_two"
            a.append(r[m]); b.append(off.loc[t, col])
        a, b = np.array(a, float), np.array(b, float)
        print(f"{m:32s} fast {np.nanmean(a):.4f}  official {np.nanmean(b):.4f}  "
              f"max |diff| {np.nanmax(np.abs(a - b)):.4f}  n {len(a)}")
    scorer.close()
