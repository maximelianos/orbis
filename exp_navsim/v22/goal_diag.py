"""Sanity checks of the navhard stage-2 goals of v22/goals.py.

    python -m exp_navsim.v22.goal_diag        # -> outputs/25_terra/stage2_goal/

Per synthetic token: the log frame that `corresponding_original_scene` names, the timestamp gap
between that log frame and the synthetic current frame, the offset of the synthetic ego from
the log ego, and the goal (distance, heading change, the mean speed it implies over 4 s next
to the current speed). Per stage-2 pair (now_i, prev_i) of a group: how many log frames apart
the two corresponding log frames are (a consistent pair is one frame = 0.5 s apart, like
the stage-1 pair) and how far apart the two goals are in global coordinates.

It also writes a second goal file, navhard_goals_near.npz: for a stage-2 token the goal is the
log pose 4 s after the log frame NEAREST to the synthetic ego (searched 16 frames around the
same-timestamp frame) instead of 4 s after the same-timestamp frame. The synthetic ego is
usually behind the log ego, so the same-timestamp goal asks it to catch up.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from exp_navsim.data.build_index import _pose
from exp_navsim.v22.fast_epdms import all_groups
from exp_navsim.v22.goals import AHEAD, LOGS, SYN

OUT = Path("outputs/25_terra/stage2_goal")


def to_global(origin, rel):
    """(x, y, h) in the frame of `origin` -> global."""
    c, s = np.cos(origin[2]), np.sin(origin[2])
    return np.array([origin[0] + c * rel[0] - s * rel[1], origin[1] + s * rel[0] + c * rel[1],
                     origin[2] + rel[2]])


def to_local(origin, pose):
    c, s = np.cos(origin[2]), np.sin(origin[2])
    dx, dy = pose[0] - origin[0], pose[1] - origin[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(pose[2] - origin[2])])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ix = pickle.load(open("data/navsim_index/navhard.pkl", "rb"))
    tokens = list(ix["token"])
    pos = {t: i for i, t in enumerate(tokens)}
    syn = np.asarray(ix["synthetic"])
    goals = np.load("outputs/25_terra/navhard_goals.npz")
    goal = dict(zip(goals["token"].tolist(), goals["goal"]))

    meta = {}
    need = set(np.asarray(tokens)[syn])
    for p in sorted(SYN.iterdir()):
        d = pickle.load(open(p, "rb"))
        m = d["scene_metadata"]
        if m["initial_token"] in need:
            f = d["frames"][-1]
            meta[m["initial_token"]] = dict(log=m["log_name"], orig=m["corresponding_original_scene"],
                                            orig_init=m.get("corresponding_original_initial_token"),
                                            ts=f["timestamp"], pose=np.asarray(f["ego_status"]["ego_pose"], float))

    logs = sorted({v["log"] for v in meta.values()} | set(np.asarray(ix["log"])[~syn]))
    where, frames, stamps = {}, {}, {}
    for log in logs:
        fr = pickle.load(open(LOGS / f"{log}.pkl", "rb"))
        frames[log] = np.array([_pose(f) for f in fr], float)
        stamps[log] = np.array([f["timestamp"] for f in fr], np.int64)
        for i, f in enumerate(fr):
            where[f["token"]] = (log, i)

    near = goals["goal"].copy()
    for j, t in enumerate(tokens):
        if syn[j] and t in meta and meta[t]["orig"] in where:
            log, i = where[meta[t]["orig"]]
            lo, hi = max(0, i - 16), min(len(frames[log]), i + 17)
            k = lo + int(np.argmin(np.hypot(*(frames[log][lo:hi, :2] - meta[t]["pose"][:2]).T)))
            near[j] = to_local(meta[t]["pose"], frames[log][k + AHEAD]) if k + AHEAD < len(frames[log]) \
                else np.nan
    np.savez_compressed("outputs/25_terra/navhard_goals_near.npz", token=goals["token"], goal=near,
                        synthetic=goals["synthetic"])
    nd = np.hypot(near[syn, 0], near[syn, 1])
    print(f"[near] stage 2 goal distance median {np.nanmedian(nd):.2f} m, "
          f"implied - current speed median {np.nanmedian(nd / 4 - np.linalg.norm(ix['velocity'][syn], axis=1)):.2f} m/s")

    rows = []
    for j, t in enumerate(tokens):
        g = goal.get(t, np.full(3, np.nan))
        v = float(np.linalg.norm(ix["velocity"][j]))
        r = dict(token=t, stage=2 if syn[j] else 1, speed=v, goal_x=g[0], goal_y=g[1], goal_h=g[2],
                 goal_dist=float(np.hypot(g[0], g[1])), goal_dh=float(abs(wrap(g[2]))),
                 implied_speed=float(np.hypot(g[0], g[1]) / 4.0))
        if syn[j] and t in meta and meta[t]["orig"] in where:
            m = meta[t]
            log, i = where[m["orig"]]
            off = to_local(frames[log][i], m["pose"])
            r.update(orig_dt_ms=(m["ts"] - stamps[log][i]) / 1e3, ego_off_x=off[0], ego_off_y=off[1],
                     ego_off_h=off[2], orig_frame=i, frames_left=len(frames[log]) - 1 - i)
            if m["orig_init"] in where:
                r["frames_after_init"] = i - where[m["orig_init"]][1]
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "goals_per_token.csv", index=False)

    # stage-2 pairs: log-frame distance of the two corresponding log frames, goal gap in global
    prow = []
    for gi, (now, prev, pairs) in enumerate(all_groups()):
        for a, b in pairs:
            if a not in meta or b not in meta or meta[a]["orig"] not in where or meta[b]["orig"] not in where:
                continue
            la, ia = where[meta[a]["orig"]]
            lb, ib = where[meta[b]["orig"]]
            ga = to_global(meta[a]["pose"], goal[a]) if np.isfinite(goal[a]).all() else np.full(3, np.nan)
            gb = to_global(meta[b]["pose"], goal[b]) if np.isfinite(goal[b]).all() else np.full(3, np.nan)
            prow.append(dict(group=gi, now=a, prev=b, frame_gap=ia - ib, same_log=la == lb,
                             ts_gap_ms=(meta[a]["ts"] - meta[b]["ts"]) / 1e3,
                             goal_gap=float(np.hypot(*(ga[:2] - gb[:2]))),
                             ego_step=float(np.hypot(*(meta[a]["pose"][:2] - meta[b]["pose"][:2])))))
    pr = pd.DataFrame(prow)
    pr.to_csv(OUT / "stage2_pairs.csv", index=False)

    lines = []
    q = lambda s: f"median {s.median():.2f}  p10 {s.quantile(.1):.2f}  p90 {s.quantile(.9):.2f}"
    for st in (1, 2):
        d = df[df.stage == st]
        lines.append(f"stage {st} ({len(d)} tokens)")
        lines.append(f"  goal distance [m]   {q(d.goal_dist)}")
        lines.append(f"  |heading change|    {q(np.degrees(d.goal_dh))} deg")
        lines.append(f"  current speed [m/s] {q(d.speed)}")
        lines.append(f"  goal dist / 4 s     {q(d.implied_speed)}")
        lines.append(f"  implied - current   {q(d.implied_speed - d.speed)}")
        lines.append(f"  goal behind ego (x < 0): {(d.goal_x < 0).mean():.3f}")
    d = df[df.stage == 2]
    lines.append("stage 2 vs the corresponding log frame")
    lines.append(f"  timestamp gap [ms]  {q(d.orig_dt_ms)}")
    lines.append(f"  ego offset x [m]    {q(d.ego_off_x)}")
    lines.append(f"  ego offset y [m]    {q(d.ego_off_y)}")
    lines.append(f"  |ego offset| [m]    {q(np.hypot(d.ego_off_x, d.ego_off_y))}")
    lines.append(f"  frames after the stage-1 frame: {d.frames_after_init.value_counts().head(6).to_dict()}")
    lines.append(f"stage-2 pairs ({len(pr)})")
    lines.append(f"  log frame gap now - prev: {pr.frame_gap.value_counts().head(6).to_dict()}")
    lines.append(f"  timestamp gap [ms]  {q(pr.ts_gap_ms)}")
    lines.append(f"  global goal gap [m] {q(pr.goal_gap)}")
    lines.append(f"  ego step [m]        {q(pr.ego_step)}")
    txt = "\n".join(lines)
    (OUT / "goal_stats.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
