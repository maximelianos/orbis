"""Self-goal: turn the plans of a goal-free sweep into a goal file for a goal-conditioned run.

    python -m exp_navsim.v22.self_goal outputs/25_terra/sweeps/X_plans.npz --rule mix_0.03 \
        --out outputs/27_nogoal/goals_self.npz [--scale 1.0]

The goal of each token is the end pose (x, y, heading) of the chosen plan, optionally scaled
along the plan (scale > 1 extrapolates the last displacement, to look like a 4-6 s goal).
No logged information is used, so a planner fed with it stays goal-free.
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("plans")
ap.add_argument("--rule", default="mix_0.03")
ap.add_argument("--out", required=True)
ap.add_argument("--scale", type=float, default=1.0)
a = ap.parse_args()
d = np.load(a.plans, allow_pickle=True)
k = list(d["rules"]).index(a.rule)
p = d["plans"][:, k].astype(np.float32)                    # (N, 8, 3)
g = p[:, -1].copy()
if a.scale != 1.0:
    g[:, :2] = p[:, -1, :2] * a.scale
np.savez(a.out, token=d["token"], goal=g, synthetic=np.zeros(len(g), bool))
print(a.out, len(g), "median goal dist", np.median(np.linalg.norm(g[:, :2], axis=1)).round(2))
