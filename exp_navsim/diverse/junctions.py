"""Detect crossroads in front of the ego from the nuPlan map, and assign trajectories to their branches.

    python -m exp_navsim.diverse.junctions build --split val [--limit 200]
    python -m exp_navsim.diverse.junctions build --index navtest --split all   # -> junctions_navtest.pkl
    python -m exp_navsim.diverse.junctions plot --out outputs/17_diversity/junctions/examples.png

Detection uses only the map and the current ego pose, never the logged future, so it
selects the scenes where several manoeuvres are possible, whatever the driver did:

1. ego lane: the lane or lane connector within 3 m of the ego whose heading is closest to
   the ego heading (within 45 deg). If it is a lane connector inside an intersection the
   ego is already in the junction and has committed to a branch (status "inside"), unless
   it is less than 2 m into it: then it waits at the entry and its incoming lane is used.
2. walk the lane graph forward from the ego lane and the other lanes of its roadblock
   (a turn may need a lane change first), up to H = 4 s * v + 10 m.
3. lane connectors that lie in an intersection polygon are the branches; only the first
   intersection reached counts. A branch's turn class comes from the heading change
   along its centreline: straight |d| < 30 deg, U-turn |d| > 150 deg, else left / right.
4. status "junction" if the first intersection is entered within H; the scene is a
   crossroad if left, straight and right are all available.

Per scene the build stores the branches as polylines in the current ego frame (connector
plus up to 20 m of the lane after it), so trajectories can be matched to them:
`classify` assigns a trajectory to the branch nearest to its final pose, or "before" if
it ends before the branches separate, or "off" if no branch is within 3.5 m.
-> outputs/17_diversity/junctions/junctions_<split>.pkl
"""

import argparse
import pickle
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OUT = Path("outputs/17_diversity/junctions")
LOGS = Path("navsim/download/trainval_navsim_logs/trainval")
LOGS_OF = {"navtrain": LOGS, "navtest": Path("navsim/download/test_navsim_logs/test")}
CLASSES = ("before", "left", "straight", "right", "uturn", "off")
TURNS = ("left", "straight", "right", "uturn")
SEPARATE = 4.0           # m along a branch before branches of one entry count as separated
MATCH = 3.5              # m, largest distance of a final pose to its branch
ENTRY = 2.0              # m into an intersection connector before the ego counts as inside


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def turn_class(path):
    d = wrap(path[-1].heading - path[0].heading)
    if abs(d) < np.pi / 6:
        return "straight"
    if abs(d) > 5 * np.pi / 6:
        return "uturn"
    return "left" if d > 0 else "right"


def to_ego(points, pose):
    """(N, 2) global -> ego frame of pose (x, y, yaw)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    d = np.asarray(points) - pose[:2]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def path_xy(edge, max_len=None):
    pts = np.array([[p.x, p.y] for p in edge.baseline_path.discrete_path])
    if max_len is not None:
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        pts = pts[seg <= max_len]
    return pts


def ego_lane(map_api, pose):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    pt = Point2D(*pose[:2])
    objs = map_api.get_proximal_map_objects(pt, 3.0, [L.LANE, L.LANE_CONNECTOR])
    best, best_cost = None, np.inf
    for layer in (L.LANE, L.LANE_CONNECTOR):
        for e in objs[layer]:
            p = e.baseline_path.get_nearest_pose_from_position(pt)
            dh = abs(wrap(p.heading - pose[2]))
            if dh > np.pi / 4:
                continue
            cost = np.hypot(p.x - pose[0], p.y - pose[1]) + 2.0 * dh
            if cost < best_cost:
                best, best_cost = e, cost
    return best


def intersection_of(map_api, edge, cache):
    """Id of the intersection polygon holding the middle of a lane connector, or None."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from shapely.geometry import Point as ShapelyPoint
    if edge.id not in cache:
        pts = path_xy(edge)
        mid = Point2D(*pts[len(pts) // 2])
        # The proximal query matches bounding boxes; keep only polygons that really hold the point.
        hits = map_api.get_proximal_map_objects(mid, 0.5, [L.INTERSECTION])[L.INTERSECTION]
        hits = [h for h in hits if h.polygon.distance(ShapelyPoint(mid.x, mid.y)) < 0.5]
        cache[edge.id] = hits[0].id if hits else None
        if hits:
            cache["poly", hits[0].id] = np.asarray(hits[0].polygon.exterior.coords)[:, :2]
    return cache[edge.id]


def detect(map_api, pose, speed, cache):
    """-> dict(status, dist, branches=[(turn, polyline ego frame)], classes)."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.abstract_map_objects import LaneConnector
    lane = ego_lane(map_api, pose)
    if lane is None:
        return {"status": "no_lane"}
    pt = Point2D(*pose[:2])
    if isinstance(lane, LaneConnector) and intersection_of(map_api, lane, cache) is not None:
        # Waiting at the entry: the nearest connector may be any branch starting there.
        if lane.baseline_path.get_nearest_arc_length_from_position(pt) < ENTRY and lane.incoming_edges:
            lane = lane.incoming_edges[0]
        else:
            inter = intersection_of(map_api, lane, cache)
            return {"status": "inside", "intersection": inter, "polygon": to_ego(cache["poly", inter], pose),
                    "lane": to_ego(path_xy(lane), pose)}
    horizon = 4.0 * max(speed, 0.0) + 10.0
    starts = [lane] if isinstance(lane, LaneConnector) else lane.parent.interior_edges
    queue = [(e, -e.baseline_path.get_nearest_arc_length_from_position(pt)) for e in starts]
    seen, found = set(), []                                          # (dist to entry, intersection, connector)
    while queue:
        edge, d0 = queue.pop()
        if edge.id in seen or d0 > horizon:
            continue
        seen.add(edge.id)
        if isinstance(edge, LaneConnector):
            inter = intersection_of(map_api, edge, cache)
            if inter is not None:
                found.append((max(d0, 0.0), inter, edge))
                continue                                             # branches end at the first junction
        d1 = d0 + edge.baseline_path.length
        if d1 < horizon:
            queue.extend((n, d1) for n in edge.outgoing_edges)
    if not found:
        return {"status": "no_junction"}
    first = min(found, key=lambda f: f[0])[1]
    branches, dist = [], np.inf
    for d, inter, c in found:
        if inter != first:
            continue
        dist = min(dist, d)
        pts = path_xy(c)
        nxt = c.outgoing_edges
        if nxt:
            pts = np.concatenate([pts, path_xy(nxt[0], 20.0)])
        branches.append((turn_class(c.baseline_path.discrete_path), to_ego(pts, pose).astype(np.float32)))
    classes = sorted({b[0] for b in branches}, key=TURNS.index)
    return {"status": "junction", "dist": float(dist), "branches": branches, "classes": classes,
            "intersection": first, "polygon": to_ego(cache["poly", first], pose)}


def classify(traj, branches):
    """traj (8, >=2) in the ego frame, branches from detect -> one of CLASSES."""
    from shapely.geometry import LineString, Point
    end = Point(*traj[-1, :2])
    best, best_d, separated = None, np.inf, False
    for turn, pts in branches:
        line = LineString(pts)
        d = line.distance(end)
        if line.project(end) >= SEPARATE:
            separated = True
            if d < best_d:
                best, best_d = turn, d
    if not separated:
        return "before"
    return best if best_d <= MATCH else "off"


def _log_job(args):
    """Detection for all scenes of one log -> list of (row, record)."""
    log, rows, tokens, speeds, future, logs = args
    from exp_navsim.data.bev_extract import get_map_api
    from exp_navsim.data.navsim_base import quat_to_yaw
    with open(logs / f"{log}.pkl", "rb") as f:
        frames = {fr["token"]: fr for fr in pickle.load(f)}
    out, cache = [], {}
    for row, tok, v, fut in zip(rows, tokens, speeds, future):
        fr = frames[tok]
        pose = np.array([*fr["ego2global_translation"][:2], quat_to_yaw(fr["ego2global_rotation"])])
        rec = detect(get_map_api(fr["map_location"]), pose, v, cache)
        if rec["status"] == "junction":
            rec["gt"] = classify(fut, rec["branches"])
        out.append((row, rec))
    return out


def build(split, limit, workers, index="navtrain"):
    from exp_navsim.data.scenes import SceneFeatureDataset
    ds = SceneFeatureDataset(f"data/navsim_index/{index}.pkl", features=[], split=split,
                             require_future=index == "navtrain")
    rows = ds.rows if limit is None else np.sort(np.random.default_rng(0).choice(ds.rows, limit, replace=False))
    by_log = defaultdict(list)
    for r in rows:
        by_log[ds.index["log"][r]].append(r)
    jobs = [(log, rs, [ds.index["token"][r] for r in rs], [float(ds.status[r][0]) for r in rs],
             [ds.future[r] for r in rs], LOGS_OF[index]) for log, rs in by_log.items()]
    records = {}
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_log_job, jobs)):
            records.update(res)
            if i % 10 == 0:
                print(f"{i + 1}/{len(jobs)} logs", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = split if index == "navtrain" else index
    path = OUT / f"junctions_{name}{'' if limit is None else f'_{limit}'}.pkl"
    with open(path, "wb") as f:
        pickle.dump(records, f)
    summarize(records)
    print(f"-> {path}")


def is_crossroad(rec):
    return rec["status"] == "junction" and {"left", "straight", "right"} <= set(rec["classes"])


def summarize(records):
    status = defaultdict(int)
    kinds = defaultdict(int)
    gt = defaultdict(int)
    for rec in records.values():
        status[rec["status"]] += 1
        if rec["status"] == "junction":
            kinds["+".join(rec["classes"])] += 1
            if is_crossroad(rec):
                gt[rec["gt"]] += 1
    n = len(records)
    print(f"{n} scenes: " + ", ".join(f"{k} {v} ({v / n:.1%})" for k, v in sorted(status.items())))
    print("junction kinds: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1])))
    print(f"crossroads: {sum(gt.values())}, logged future: " + ", ".join(f"{k} {v}" for k, v in gt.items()))


def plot(path, out, num, seed):
    """BEV of random crossroads, T-junctions and non-junction scenes with the detected branches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp_navsim.data.scenes import load_index
    from exp_navsim.diverse.plot_bev import load_scenes
    from navsim.visualization.bev import add_configured_bev_on_ax
    from navsim.visualization.plots import configure_ax, configure_bev_ax

    with open(path, "rb") as f:
        records = pickle.load(f)
    index = load_index("data/navsim_index/navtrain.pkl")
    rng = np.random.default_rng(seed)
    groups = {
        "crossroad": [r for r, x in records.items() if is_crossroad(x)],
        "other junction": [r for r, x in records.items() if x["status"] == "junction" and not is_crossroad(x)],
        "no junction / inside": [r for r, x in records.items() if x["status"] in ("no_junction", "inside")],
    }
    picks = {k: rng.choice(v, min(num, len(v)), replace=False) for k, v in groups.items()}
    rows = [r for v in picks.values() for r in v]
    scenes = load_scenes([index["token"][r] for r in rows], [index["log"][r] for r in rows])
    colors = {"left": "tab:green", "straight": "tab:blue", "right": "tab:orange", "uturn": "tab:purple"}
    fig, axes = plt.subplots(len(picks), num, figsize=(3.2 * num, 3.4 * len(picks)), squeeze=False)
    for i, (name, rs) in enumerate(picks.items()):
        for j, r in enumerate(rs):
            ax = axes[i, j]
            rec = records[r]
            add_configured_bev_on_ax(ax, scenes[index["token"][r]].map_api, scenes[index["token"][r]].frames[3])
            configure_bev_ax(ax)
            configure_ax(ax)
            if "polygon" in rec:
                ax.plot(rec["polygon"][:, 1], rec["polygon"][:, 0], color="magenta", lw=1.2, zorder=6)
            if "lane" in rec:
                ax.plot(rec["lane"][:, 1], rec["lane"][:, 0], color="red", lw=2.5, zorder=6)
            for turn, pts in rec.get("branches", []):
                ax.plot(pts[:, 1], pts[:, 0], color=colors[turn], lw=2.5, alpha=0.8, zorder=6)
            fut = index["future"][r]
            ax.plot(np.r_[0, fut[:, 1]], np.r_[0, fut[:, 0]], color="black", lw=2, zorder=7)
            ax.set_xlim(40, -40)
            ax.set_ylim(-15, 65)
            v = float(np.linalg.norm(index["velocity"][r]))
            title = f"{name}: {rec['status']}, v {v:.1f}"
            if rec["status"] == "junction":
                title += f"\n{'+'.join(rec['classes'])}, entry {rec['dist']:.0f} m, gt {rec['gt']}"
            ax.set_title(title, fontsize=7)
    fig.suptitle("detected branches: green left, blue straight, orange right, purple U-turn; black logged future",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "plot"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--index", default="navtrain", choices=["navtrain", "navtest"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--records", default=str(OUT / "junctions_val.pkl"))
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "examples.png"))
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.split, args.limit, args.workers, args.index)
    else:
        plot(args.records, args.out, args.num, args.seed)


if __name__ == "__main__":
    main()
