"""Figures of outputs/27_nogoal: navhard EPDMS without a goal input.

    python -m exp_navsim.v22.figs_nogoal bars         # figs/bars.png, next to the published entries
    python -m exp_navsim.v22.figs_nogoal components   # figs/components.png, stage 1 / 2 sub-scores
    python -m exp_navsim.v22.figs_nogoal pick         # figs/scenes.csv, stage-one gains and losses
    python -m exp_navsim.v22.figs_nogoal scenes       # figs/scenes/<token>.jpg, camera + BEV

All numbers are read from the sweep_rank outputs (<run><tag>.csv, _tokens.csv, _plans.npz).
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from exp_navsim.data.scenes import COMMANDS
from exp_navsim.v22.plots import BLUE, GREEN, GREY, LITERATURE, ORANGE

OUT = Path("outputs/27_nogoal")
FIGS = OUT / "figs"
NOGOAL = OUT / "sweeps"
TERRA = Path("outputs/25_terra/sweeps")
RUN = "24aa_goal_aug"
INDEX = "data/navsim_index/navhard.pkl"
GOALS = OUT / "goals_head_lam1_top8.npz"

# label, sweep dir, tag, rule
NO_GOAL = [
    ("null goal, ens5 prob (start)", TERRA, "_all_ens5_prob_nogoal", "mix_0.03"),
    ("self-goal: own end point as goal", NOGOAL, "_selfgoal", "mix_0.03"),
    ("test-time refinement, 10 Adam steps", NOGOAL, "_ens5_null_opt10", "opt_mix_0.03"),
    ("1 predicted goal, minmult", NOGOAL, "_ens5_head_lam1_minmult", "mix_0.01"),
    ("null goal, minmult", NOGOAL, "_ens5_null_minmult_b", "mix_0.01"),
    ("null goal, minmult", NOGOAL, "_ens5_null_minmult", "mix_0.03"),
    ("4 predicted goals, null rescoring, prob", NOGOAL, "_ens5_mg4null_lam1_prob", "mg_null_mix_0.01"),
    ("4 jittered self-goals, null rescoring, minmult", NOGOAL, "_ens5_mg4null_selfjitter_minmult", "mg_null_mix_0.03"),
    ("2 predicted goals, null rescoring, minmult", NOGOAL, "_ens5_mg2null_lam1_minmult", "mg_null_mix_0.03"),
    ("4 predicted goals, null rescoring, minmult", NOGOAL, "_ens5_mg4_lam1_minmult", "mg_null_mix_0.03"),
    ("8 predicted goals, null rescoring, minmult", NOGOAL, "_ens5_mg8null_lam1_minmult", "mg_null_mix_0.03"),
]
WITH_GOAL = [("log goal (near), ens5 prob", TERRA, "_all_near_ens5_prob", "mix_0.03")]

# the methods of the component and scene figures: key, label, sweep dir, tag, rule, RGB
BASE = ("null", "null goal, ens5 prob", TERRA, "_all_ens5_prob_nogoal", "mix_0.03", (47, 111, 208))
MINMULT = ("minmult", "null goal, minmult", NOGOAL, "_ens5_null_minmult", "mix_0.03", (232, 140, 30))
MG4 = ("mg4", "4 predicted goals, null rescoring, minmult", NOGOAL, "_ens5_mg4_lam1_minmult",
       "mg_null_mix_0.03", (200, 40, 150))
NEAR = ("near", "log goal (near), ens5 prob", TERRA, "_all_near_ens5_prob", "mix_0.03", (40, 170, 60))
SCENE_METHODS = [BASE, MINMULT, MG4]

SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
        "ego_progress", "time_to_collision_within_bound", "lane_keeping", "two_frame_extended_comfort"]
SHORT = ["NC", "DAC", "DDC", "EP", "TTC", "LK", "EC"]
# sub-scores that count as failed when below 1 (EP is continuous and shown as a value)
FAIL = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC",
        "driving_direction_compliance": "DDC", "traffic_light_compliance": "TLC",
        "time_to_collision_within_bound": "TTC", "lane_keeping": "LK", "two_frame_extended_comfort": "EC"}


def sweep_row(sweeps, tag, rule):
    df = pd.read_csv(Path(sweeps) / f"{RUN}{tag}.csv")
    return df[df.rule == rule].iloc[-1]


def bars():
    rows = [(f"{n} (published)" + (" *" if m else ""), v, "lit") for n, v, m in LITERATURE]
    rows += [(f"{n}  [{r.replace('mg_null_', '').replace('opt_', '')}]", float(sweep_row(s, t, r)["epdms"]), "ours")
             for n, s, t, r in NO_GOAL]
    rows += [(f"{n}  [{r}]", float(sweep_row(s, t, r)["epdms"]), "goal") for n, s, t, r in WITH_GOAL]
    rows.sort(key=lambda r: r[1])
    best = max(v for _, v, k in rows if k == "ours")
    start = float(sweep_row(*BASE[2:5])["epdms"])
    colors = [GREY if k == "lit" else GREEN if k == "goal" else ORANGE if v == best else BLUE
              for _, v, k in rows]
    fig, ax = plt.subplots(figsize=(8.4, 0.36 * len(rows) + 1.6), dpi=140)
    y = np.arange(len(rows))
    ax.barh(y, [v for _, v, _ in rows], color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in rows], fontsize=8)
    ax.invert_yaxis()
    for i, (_, v, _) in enumerate(rows):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=7.5, color="#333")
    ax.axvline(start, color="#555", lw=0.8, ls="--", zorder=0)
    ax.text(start, -0.75, f" null-goal start {start:.1f}", fontsize=7, color="#555", va="center")
    ax.set_xlabel("navhard two-stage EPDMS, whole split (225 groups); [rule] = mix beta")
    ax.set_xlim(0, max(v for _, v, _ in rows) * 1.1)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GREEN, GREY)]
    ax.legend(handles, ["ours, no goal", "ours, no goal, best", "ours, uses the log goal",
                        "existing work (py123d_garage table)"],
              fontsize=7.5, loc="upper center", bbox_to_anchor=(0.35, -0.09), ncol=2, frameon=False)
    ax.set_title("navhard without a goal input (5-model ensemble of 24aa)\n"
                 "* = trained on the benchmark metric", fontsize=9)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "bars.png", bbox_inches="tight")
    print(f"-> {FIGS / 'bars.png'}")


def components():
    methods = [BASE, MINMULT, MG4, NEAR]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=140, sharey=True)
    x = np.arange(len(SUBS))
    w = 0.8 / len(methods)
    for ax, stage in zip(axes, ("s1", "s2")):
        for j, (_, lab, s, t, r, rgb) in enumerate(methods):
            row = sweep_row(s, t, r)
            vals = [float(row[f"{m}_{stage}"]) for m in SUBS]
            ax.bar(x + (j - (len(methods) - 1) / 2) * w, vals, w * 0.92, color=np.array(rgb) / 255,
                   label=f"{lab}  ({float(row['epdms']):.1f})" if stage == "s1" else None)
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT)
        ax.set_title("stage one (real frames)" if stage == "s1" else "stage two (synthetic frames)", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(40, 101)
    axes[0].set_ylabel("sub-score [%]")
    fig.legend(loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("navhard sub-scores, whole split (EPDMS in brackets); y axis starts at 40", fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "components.png")
    print(f"-> {FIGS / 'components.png'}")


def token_rows(method):
    key, _, s, t, r, _ = method
    df = pd.read_csv(Path(s) / f"{RUN}{t}_tokens.csv")
    df = df[(df.rule == r) & df.stage_one].drop_duplicates("token")
    cols = list(FAIL) + ["ego_progress", "score"]
    return df[["token"] + cols].rename(columns={c: f"{c}_{key}" for c in cols})


def index():
    ix = pickle.load(open(INDEX, "rb"))
    ix["pos"] = {str(t): i for i, t in enumerate(ix["token"])}
    return ix


def pick(n_gain=6, n_loss=4):
    df = token_rows(BASE)
    for m in SCENE_METHODS[1:]:
        df = df.merge(token_rows(m), on="token")
    ix = index()
    rows = [ix["pos"][t] for t in df["token"]]
    df["log"] = [ix["log"][r] for r in rows]
    df["city"] = [ix["city"][r] for r in rows]
    df["command"] = [COMMANDS[np.asarray(ix["command"][r]).argmax()] for r in rows]
    df["gain"] = df["score_mg4"] - df["score_null"]
    gain = df[df.gain > 0].sort_values("gain", ascending=False).drop_duplicates("log").head(n_gain)
    loss = df[df.gain < 0].sort_values("gain").drop_duplicates("log").head(n_loss)
    sel = pd.concat([gain.assign(kind="gain"), loss.assign(kind="loss")])
    FIGS.mkdir(parents=True, exist_ok=True)
    sel.to_csv(FIGS / "scenes.csv", index=False)
    print(f"stage-one tokens {len(df)}: mg4 better {(df.gain > 0).sum()}, worse {(df.gain < 0).sum()}")
    print(sel[["token", "kind", "city", "command", "score_null", "score_minmult", "score_mg4"]]
          .round(3).to_string(index=False))
    print(f"-> {FIGS / 'scenes.csv'}")


def sweep_plans(method):
    _, _, s, t, r, _ = method
    d = np.load(Path(s) / f"{RUN}{t}_plans.npz")
    j = list(d["rules"]).index(r)
    return dict(zip(d["token"].tolist(), d["plans"][:, j].astype(np.float32)))


def describe(row, key):
    fails = [n for c, n in FAIL.items() if row[f"{c}_{key}"] < 1]
    return (f"score {100 * row[f'score_{key}']:.1f}   EP {row[f'ego_progress_{key}']:.2f}   "
            f"fails: {', '.join(fails) if fails else 'none'}")


def scenes(n_goals=3):
    from exp_navsim.v22.figures import draw_plans, navhard_loader

    sel = pd.read_csv(FIGS / "scenes.csv", dtype={"token": str})
    plans = {m[0]: sweep_plans(m) for m in SCENE_METHODS}
    g = np.load(GOALS)
    goals = dict(zip(g["token"].tolist(), g["goal"][:, :4, :2]))
    star = set(sel[sel.kind == "gain"]["token"].head(n_goals))
    ix = index()
    loader = navhard_loader(sel["token"].tolist())
    out = FIGS / "scenes"
    out.mkdir(parents=True, exist_ok=True)
    for i, (_, row) in enumerate(sel.iterrows()):
        t = row["token"]
        scene = loader.get_scene_from_token(t)
        gt = np.asarray(ix["future"][ix["pos"][t]])
        gt = gt if np.isfinite(gt).all() else None
        lines = [(plans[m[0]][t], m[5], w) for m, w in zip(SCENE_METHODS, (10, 8, 6))]
        cam, bev, hw = draw_plans(scene, scene.frames[3], gt, lines, goals[t] if t in star else None)
        fig = plt.figure(figsize=(15, 7.3), dpi=100)
        ax_c = fig.add_axes((0.005, 0.01, 0.635, 0.70))
        ax_b = fig.add_axes((0.645, 0.01, 0.35, 0.70))
        for ax, im in ((ax_c, cam), (ax_b, bev)):
            ax.imshow(im)
            ax.set_axis_off()
        ax_b.set_title(f"BEV, {2 * hw:.0f} m wide", fontsize=9, loc="left")
        handles = [Line2D([], [], color=np.array((40, 170, 60)) / 255, lw=4)]
        labels = ["log (human driver)"]
        for m in SCENE_METHODS:
            handles.append(Line2D([], [], color=np.array(m[5]) / 255, lw=4))
            labels.append(f"{m[1]}:   {describe(row, m[0])}")
        if t in star:
            handles.append(Line2D([], [], ls="", marker="*", ms=14, color="#f4c430", mec="k"))
            labels.append("top-4 goals of the goal head (input of the mg4 passes)")
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.925), fontsize=10.5,
                   frameon=False, handlelength=2.5)
        fig.text(0.005, 0.975, f"{'GAIN' if row['kind'] == 'gain' else 'LOSS'} of mg4 over the null baseline: "
                 f"{100 * row['gain']:+.1f}     token {t}   {row['city']}   command {row['command']}   stage one",
                 fontsize=12, weight="bold", va="top")
        path = out / f"{i:02d}_{row['kind']}_{t}.jpg"
        fig.savefig(path, dpi=100, pil_kwargs={"quality": 88})
        plt.close(fig)
        print(f"-> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["bars", "components", "pick", "scenes", "all"])
    a = ap.parse_args()
    for name in (["bars", "components", "pick", "scenes"] if a.action == "all" else [a.action]):
        globals()[name]()
