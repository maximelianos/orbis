"""Figures of the scorer ensemble on the whole navhard split: a bar plot next to the
published entries, and the scenes for figures.py compare.

    python -m exp_navsim.v22.figs_ensemble bars
    python -m exp_navsim.v22.figs_ensemble pick       # writes figs_ensemble/scenes.csv

The scenes are drawn with figures.py (plans and token scores from the sweep_rank outputs):
    python -m exp_navsim.v22.figures compare --stage one --sweeps outputs/25_terra/sweeps \
        --runs 24aa_goal_aug__mix_0.03,24aa_goal_aug__mix_0.03,24aa_goal_aug__mix_0.03 \
        --tags _all_near,_all_near_ens5_prob,_all_ens5_prob_nogoal --tokens ...
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp_navsim.data.scenes import COMMANDS
from exp_navsim.v22.plots import BLUE, GREY, LITERATURE, ORANGE

TERRA = Path("outputs/25_terra")
SWEEPS = TERRA / "sweeps"
FIGS = TERRA / "figs_ensemble"
RULE = "mix_0.03"
RUN = "24aa_goal_aug"
# label, sweep csv; the EPDMS is read from the row of RULE
OURS = [("ours: single scorer 24aa, near goal", SWEEPS / "24aa_goal_aug_all_near.csv"),
        ("ours: 2-model ensemble, near goal", SWEEPS / "24aa_goal_aug_all_near_ens_ab_prob.csv"),
        ("ours: 5-model ensemble, near goal", SWEEPS / "24aa_goal_aug_all_near_ens5_prob.csv"),
        ("ours: 5-model ensemble, null goal", SWEEPS / "24aa_goal_aug_all_ens5_prob_nogoal.csv"),
        ("ours: best single, null goal (24w)", SWEEPS / "24w_goal_all_nogoal.csv"),
        ("ours: 5-model ensemble, refined targets", Path("outputs/26_onpolicy/sweeps/24al_refined_all_ens5r.csv"))]
# the three methods of the scene figures: label, sweep tag
METHODS = [("single", "_all_near"), ("ens5", "_all_near_ens5_prob"), ("ens5_nogoal", "_all_ens5_prob_nogoal")]
SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
        "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
        "history_comfort", "two_frame_extended_comfort"]


def epdms_of(path):
    df = pd.read_csv(path)
    return float(df[df.rule == RULE]["epdms"].iloc[-1])


def bars():
    rows = [(f"{n} (published)" + (" *" if m else ""), v, False) for n, v, m in LITERATURE]
    rows += [(n, epdms_of(p), True) for n, p in OURS]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.6, 0.4 * len(rows) + 1.3), dpi=140)
    y = np.arange(len(rows))
    best = max(v for _, v, ours in rows if ours)
    colors = [(ORANGE if v == best else BLUE) if ours else GREY for _, v, ours in rows]
    ax.barh(y, [v for _, v, _ in rows], color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in rows], fontsize=8)
    ax.invert_yaxis()
    for i, (_, v, _) in enumerate(rows):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=7.5, color="#333")
    ax.set_xlabel(f"navhard two-stage EPDMS, whole split (225 groups), rule {RULE}")
    ax.set_xlim(0, max(v for _, v, _ in rows) * 1.12)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("This work (blue, best in orange) next to published values (grey, py123d_garage table)\n"
                 "* = trained on the benchmark metric", fontsize=9)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "bars.png")
    print(f"-> {FIGS / 'bars.png'}")


def token_table():
    """One row per token scored by all three methods, with their sub-scores side by side."""
    out = None
    for name, tag in METHODS:
        df = pd.read_csv(SWEEPS / f"{RUN}{tag}_tokens.csv")
        df = df[df.rule == RULE][["token", "stage_one"] + SUBS + ["score"]]
        df = df.rename(columns={c: f"{c}_{name}" for c in SUBS + ["score"]})
        out = df if out is None else out.merge(df.drop(columns="stage_one"), on="token")
    ix = pickle.load(open("data/navsim_index/navhard.pkl", "rb"))
    pos = {str(t): i for i, t in enumerate(ix["token"])}
    rows = [pos[t] for t in out["token"]]
    fut = np.asarray(ix["future"])[rows]
    out["lateral_gt"] = np.abs(fut[:, -1, 1])                 # NaN for synthetic frames
    out["city"] = [ix["city"][r] for r in rows]
    out["command"] = [COMMANDS[np.asarray(ix["command"][r]).argmax()] for r in rows]
    out["gain"] = out["score_ens5"] - out["score_single"]
    return out


def pick():
    df = token_table()
    turn = df["command"].str.contains("left|right", case=False)
    s1, s2 = df[df.stage_one], df[~df.stage_one]
    # stage one: large ensemble gains, the most lateral ground truth first
    win1 = s1[s1.gain > 0.3].sort_values("lateral_gt", ascending=False).head(2)
    fail1 = s1[(s1.score_ens5 < 0.3) & (s1.lateral_gt > 2)].sort_values("score_ens5").head(1)
    # stage two has no logged future: turns by the command, one scene per city (the
    # synthetic branches of one start look alike)
    win2 = s2[turn[~df.stage_one]].sort_values("gain", ascending=False).drop_duplicates("city").head(2)
    fail2 = s2[turn[~df.stage_one] & (s2.score_single < 0.3)].sort_values("score_ens5").head(1)
    sel = pd.concat([win1.assign(kind="stage one, ensemble gain"),
                     fail1.assign(kind="stage one, ensemble fails"),
                     win2.assign(kind="stage two, ensemble gain"),
                     fail2.assign(kind="stage two, ensemble fails")])
    FIGS.mkdir(parents=True, exist_ok=True)
    sel.to_csv(FIGS / "scenes.csv", index=False)
    short = {f"{c}_{m}": f"{c[:4]}_{m[:2]}{m[-2:]}" for c in SUBS for m, _ in METHODS}
    cols = ["token", "kind", "city", "command", "lateral_gt"] + [f"score_{m}" for m, _ in METHODS] + \
           [f"{c}_{m}" for c in SUBS for m, _ in METHODS]
    print(sel[cols].rename(columns=short).round(2).T.to_string())
    print(f"-> {FIGS / 'scenes.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["bars", "pick"])
    a = ap.parse_args()
    bars() if a.action == "bars" else pick()
