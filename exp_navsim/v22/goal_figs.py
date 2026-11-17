"""Does following the log goal cause the stage-2 collisions? Per-token analysis + BEV figures.

    python -m exp_navsim.v22.goal_figs --run 24y_goal_select --rule mix_0.03

Reads the per-token rows and plans of two whole-split sweeps of the same run (v22/sweep_rank.py
--per-token): with the goal (tag _all) and with the null goal (tag _all_nogoal). On the
stage-2 tokens scored by both, it bins the sub-scores by the speed gap the goal implies
(goal distance / 4 s - current speed, v22/goal_diag.py) and by the end-point distance of
the plan to the goal. Then it draws BEVs of stage-2 scenes where the goal plan collides and the
goal-free plan does not: agents and map of the synthetic frame, the goal plan (red), the
goal-free plan (blue), the goal (star) and the log ego path of the same 4 s (green dashed,
the path the goal was taken from).

Writes outputs/25_terra/stage2_goal/{analysis.txt, bev_*.jpg}.
"""

import argparse
import pickle

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp_navsim.v22.goal_diag import OUT, to_local
from exp_navsim.data.build_index import _pose
from exp_navsim.v22.goals import AHEAD, LOGS, SYN

SWEEPS = OUT.parent / "sweeps"
SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound",
        "ego_progress", "lane_keeping", "two_frame_extended_comfort", "score"]


def plans(run, tag, rule):
    d = np.load(SWEEPS / f"{run}{tag}_plans.npz")
    j = list(d["rules"]).index(rule)
    return dict(zip(d["token"].tolist(), d["plans"][:, j].astype(np.float32)))


def analysis(run, rule, info):
    a = pd.read_csv(SWEEPS / f"{run}_all_tokens.csv")
    b = pd.read_csv(SWEEPS / f"{run}_all_nogoal_tokens.csv")
    a, b = a[(a.rule == rule) & ~a.stage_one], b[(b.rule == rule) & ~b.stage_one]
    m = a.merge(b, on="token", suffixes=("_g", "_n")).merge(info, on="token")
    pg = plans(run, "_all", rule)
    m["end_to_goal"] = [np.hypot(*(pg[t][-1, :2] - g)) for t, g in zip(m.token, m[["goal_x", "goal_y"]].values)]
    m["gap"] = m.implied_speed - m.speed
    lines = [f"{run} {rule}: {len(m)} stage-2 tokens scored with and without the goal (unweighted means)"]
    lines.append("bin                         n   " + "  ".join(f"{s[:6]:>13s}" for s in SUBS))
    lines.append("                                " + "  ".join(f"{'goal / none':>13s}" for _ in SUBS))

    def row(name, d):
        return f"{name:24s} {len(d):5d}   " + "  ".join(
            f"{100 * d[s + '_g'].mean():6.1f}/{100 * d[s + '_n'].mean():5.1f}" for s in SUBS)
    lines.append(row("all", m))
    for lo, hi in [(-99, 0), (0, 2), (2, 5), (5, 99)]:
        lines.append(row(f"speed gap {lo}..{hi} m/s", m[(m.gap >= lo) & (m.gap < hi)]))
    for lo, hi in [(0, 2), (2, 5), (5, 999)]:
        lines.append(row(f"plan end to goal {lo}..{hi} m", m[(m.end_to_goal >= lo) & (m.end_to_goal < hi)]))
    nc = m[(m.no_at_fault_collisions_g < 1) & (m.no_at_fault_collisions_n == 1)]
    lines.append(f"collide only with the goal: {len(nc)}, only without: "
                 f"{((m.no_at_fault_collisions_g == 1) & (m.no_at_fault_collisions_n < 1)).sum()}; "
                 f"median speed gap of the goal-only collisions {nc.gap.median():.2f} m/s "
                 f"(all {m.gap.median():.2f}); median plan end to goal {nc.end_to_goal.median():.1f} m")
    pn = plans(run, "_all_nogoal", rule)
    sp = lambda p: np.hypot(*(p[-1, :2])) / 4.0
    lines.append(f"mean plan speed over 4 s, stage-2 tokens: goal {np.mean([sp(pg[t]) for t in m.token]):.2f} "
                 f"m/s, no goal {np.mean([sp(pn[t]) for t in m.token]):.2f} m/s, current {m.speed.mean():.2f} m/s")
    return m, nc, "\n".join(lines)


def figures(run, rule, tokens, info):
    from navsim.visualization.bev import add_configured_bev_on_ax
    from navsim.visualization.plots import configure_ax, configure_bev_ax
    from exp_navsim.v22.figures import navhard_loader

    pg, pn = plans(run, "_all", rule), plans(run, "_all_nogoal", rule)
    loader = navhard_loader([], tokens)
    info = info.set_index("token")
    for t in tokens:
        scene = loader.get_scene_from_token(t)
        d = pickle.load(open(SYN / f"{scene.scene_metadata.scene_token}.pkl", "rb"))
        m = d["scene_metadata"]
        ego = np.asarray(d["frames"][-1]["ego_status"]["ego_pose"], float)
        fr = pickle.load(open(LOGS / f"{m['log_name']}.pkl", "rb"))
        i = [f["token"] for f in fr].index(m["corresponding_original_scene"])
        log = np.array([to_local(ego, _pose(f)) for f in fr[i:i + AHEAD + 1]])
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        add_configured_bev_on_ax(ax, scene.map_api, scene.frames[3])
        configure_bev_ax(ax)
        configure_ax(ax)
        ax.plot(log[:, 1], log[:, 0], "--", color="#2a9d3a", lw=2.5, zorder=6, label="log ego, same 4 s")
        ax.plot(log[0, 1], log[0, 0], "s", color="#2a9d3a", ms=8, zorder=6)
        ax.plot(np.r_[0, pn[t][:, 1]], np.r_[0, pn[t][:, 0]], "-o", color="#2f6fd0", lw=2.2, ms=3, zorder=7,
                label="plan, null goal")
        ax.plot(np.r_[0, pg[t][:, 1]], np.r_[0, pg[t][:, 0]], "-o", color="#d62828", lw=2.2, ms=3, zorder=8,
                label="plan, log goal")
        r = info.loc[t]
        ax.plot(r.goal_y, r.goal_x, "*", color="#f4a300", ms=18, mec="k", zorder=9, label="goal")
        ax.set_xlim(32, -32)
        ax.set_ylim(-14, 50)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        ax.set_title(f"{t}  speed {r.speed:.1f} m/s, goal needs {r.implied_speed:.1f} m/s", fontsize=9)
        fig.tight_layout()
        path = OUT / f"bev_{t}.jpg"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="24y_goal_select")
    ap.add_argument("--rule", default="mix_0.03")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    info = pd.read_csv(OUT / "goals_per_token.csv")
    m, nc, txt = analysis(args.run, args.rule, info)
    print(txt)
    (OUT / f"analysis_{args.run}_{args.rule}.txt").write_text(txt + "\n")
    pick = nc.sort_values("gap", ascending=False).token.tolist()[:args.n]
    figures(args.run, args.rule, pick, info)


if __name__ == "__main__":
    main()
