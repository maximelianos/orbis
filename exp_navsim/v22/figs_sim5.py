"""The old 5-model scorer ensemble (24aa + 24ab, 24ai, 24aj, 24ak) next to the 5-seed
SimScale ensemble (28s_sim50 + s1..s4) on the whole navhard split, rule mix_0.03.

    python -m exp_navsim.v22.figs_sim5 bars     # bars.png
    python -m exp_navsim.v22.figs_sim5 subs     # subscores.csv, old vs sim5 per stage
    python -m exp_navsim.v22.figs_sim5 pick     # scenes_near.csv, scenes_null.csv
    python -m exp_navsim.v22.figs_sim5 draw     # figures.py compare on the picked scenes

The scenes are drawn with figures.compare (front camera + BEV, one row per ensemble).
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
from exp_navsim.v22.figs_ensemble import RULE, SUBS, epdms_of
from exp_navsim.v22.figures import compare
from exp_navsim.v22.plots import BLUE, ORANGE

FIGS = Path("outputs/28_inputs/figs_sim5")
TERRA = Path("outputs/25_terra/sweeps")
NOGOAL = Path("outputs/27_nogoal/sweeps")
INPUTS = Path("outputs/28_inputs/sweeps")
# setting: (old ensemble, sim5 ensemble), each (sweep dir, run, tag)
ENSEMBLES = {"near": ((TERRA, "24aa_goal_aug", "_all_near_ens5_prob"),
                      (INPUTS, "28s_sim50", "_all_near_ens_sim5_prob")),
             "null": ((NOGOAL, "24aa_goal_aug", "_ens5_null_minmult"),
                      (INPUTS, "28s_sim50", "_all_null_ens_sim5_minmult"))}
LABELS = ["old 5-model ensemble (24aa ab ai aj ak)", "SimScale 5-model ensemble (28s_sim50 + s1..s4)"]
# bars: label, near csv, null csv (None = not evaluated)
BARS = [("24aa single", TERRA / "24aa_goal_aug_all_near.csv", None),
        ("old 5-model ensemble", TERRA / "24aa_goal_aug_all_near_ens5_prob.csv",
         NOGOAL / "24aa_goal_aug_ens5_null_minmult.csv"),
        ("28s_sim50 single", INPUTS / "28s_sim50_all_near.csv", INPUTS / "28s_sim50_all_null.csv"),
        ("SimScale 5-model ensemble", INPUTS / "28s_sim50_all_near_ens_sim5_prob.csv",
         INPUTS / "28s_sim50_all_null_ens_sim5_minmult.csv")]
SHORT = dict(zip(SUBS, ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]))
# scenes per setting: (stage one gains, stage two gains, stage one losses, stage two losses)
COUNTS = {"near": (3, 3, 1, 2), "null": (2, 2, 1, 1)}


def csv_path(d, run, tag, suffix=""):
    return d / f"{run}{tag}{suffix}.csv"


def bars():
    x = np.arange(len(BARS))
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=140)
    for k, (name, color) in enumerate([("near goal", BLUE), ("null goal (minmult for ensembles)", ORANGE)]):
        vals = [epdms_of(r[1 + k]) if r[1 + k] is not None else np.nan for r in BARS]
        pos = x + (k - 0.5) * 0.38
        ax.bar(pos, np.nan_to_num(vals), width=0.36, color=color, label=name)
        for p, v in zip(pos, vals):
            ax.text(p, (v if np.isfinite(v) else 0) + 0.6, f"{v:.2f}" if np.isfinite(v) else "n/a",
                    ha="center", fontsize=8, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in BARS], fontsize=8.5)
    ax.set_ylabel(f"navhard two-stage EPDMS ({RULE})")
    ax.set_ylim(0, 60)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.set_title("Whole navhard split (225 groups): old vs SimScale scorers", fontsize=9.5)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "bars.png")
    print(f"-> {FIGS / 'bars.png'}")


def subs():
    """Sub-scores of the two ensembles per stage (from the main sweep csvs) and the mean
    per-token difference (from the _tokens.csv)."""
    rows = []
    for setting, (old, new) in ENSEMBLES.items():
        a = pd.read_csv(csv_path(*old)).query("rule == @RULE").iloc[-1]
        b = pd.read_csv(csv_path(*new)).query("rule == @RULE").iloc[-1]
        t = token_table(setting)
        for c in SUBS:
            row = {"setting": setting, "metric": SHORT[c]}
            for s in ("s1", "s2"):
                row[f"old_{s}"], row[f"sim5_{s}"] = a[f"{c}_{s}"], b[f"{c}_{s}"]
                row[f"diff_{s}"] = round(b[f"{c}_{s}"] - a[f"{c}_{s}"], 1)
            row["token_diff"] = round(100 * (t[f"{c}_new"] - t[f"{c}_old"]).mean(), 2)
            rows.append(row)
        rows.append({"setting": setting, "metric": "EPDMS", "old_s1": a["epdms"], "sim5_s1": b["epdms"],
                     "diff_s1": round(b["epdms"] - a["epdms"], 2)})
    df = pd.DataFrame(rows)
    FIGS.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIGS / "subscores.csv", index=False)
    print(df.to_string(index=False))
    print(f"-> {FIGS / 'subscores.csv'}")


def token_table(setting):
    """One row per token scored by both ensembles, sub-scores side by side (_old, _new)."""
    out = None
    for name, (d, run, tag) in zip(("old", "new"), ENSEMBLES[setting]):
        df = pd.read_csv(csv_path(d, run, tag, "_tokens"))
        df = df[df.rule == RULE][["token", "stage_one"] + SUBS + ["score"]]
        df = df.rename(columns={c: f"{c}_{name}" for c in SUBS + ["score"]})
        out = df if out is None else out.merge(df.drop(columns="stage_one"), on="token")
    ix = pickle.load(open("data/navsim_index/navhard.pkl", "rb"))
    pos = {str(t): i for i, t in enumerate(ix["token"])}
    rows = [pos[t] for t in out["token"]]
    out["lateral_gt"] = np.abs(np.asarray(ix["future"])[rows][:, -1, 1])   # NaN for synthetic frames
    out["city"] = [ix["city"][r] for r in rows]
    out["command"] = [COMMANDS[np.asarray(ix["command"][r]).argmax()] for r in rows]
    out["gain"] = out["score_new"] - out["score_old"]
    return out


def pick():
    for setting, (n_win1, n_win2, n_loss1, n_loss2) in COUNTS.items():
        df = token_table(setting)
        print(f"{setting}: {len(df)} tokens, mean gain {100 * df.gain.mean():.2f}, "
              f"better {(df.gain > 0.01).sum()}, worse {(df.gain < -0.01).sum()}")
        s1, s2 = df[df.stage_one], df[~df.stage_one]
        # stage one: largest gains, the most lateral ground truth first among ties;
        # stage two: one scene per city (the synthetic branches of one start look alike)
        win1 = s1[s1.gain > 0.3].sort_values(["gain", "lateral_gt"], ascending=False).head(n_win1)
        win2 = s2[s2.gain > 0.3].sort_values("gain", ascending=False).drop_duplicates("city").head(n_win2)
        loss1 = s1[s1.gain < -0.3].sort_values(["gain", "lateral_gt"], ascending=[True, False]).head(n_loss1)
        loss2 = s2[s2.gain < -0.3].sort_values("gain").drop_duplicates("city").head(n_loss2)
        sel = pd.concat([win1.assign(kind="gain", stage="one"), win2.assign(kind="gain", stage="two"),
                         loss1.assign(kind="loss", stage="one"), loss2.assign(kind="loss", stage="two")])
        FIGS.mkdir(parents=True, exist_ok=True)
        sel.to_csv(FIGS / f"scenes_{setting}.csv", index=False)
        cols = ["token", "kind", "stage", "city", "command", "score_old", "score_new"] + \
               [f"{c}_{m}" for c in SUBS for m in ("old", "new")]
        print(sel[cols].rename(columns={f"{c}_{m}": f"{SHORT[c]}_{m}" for c in SUBS for m in ("old", "new")})
              .round(2).to_string(index=False))
        print(f"-> {FIGS / f'scenes_{setting}.csv'}")


def draw():
    """figures.py compare per setting, kind and stage; prints the equivalent command lines."""
    for setting, ens in ENSEMBLES.items():
        sel = pd.read_csv(FIGS / f"scenes_{setting}.csv", dtype={"token": str})
        runs = [f"{run}__{RULE}" for _, run, _ in ens]
        tags = [tag for _, _, tag in ens]
        sweeps = ",".join(str(d) for d, _, _ in ens)
        for (kind, stage), g in sel.groupby(["kind", "stage"]):
            out = FIGS / f"{setting}_{kind}"
            tokens = g["token"].tolist()
            print(f"python -m exp_navsim.v22.figures compare --stage {stage} --sweeps {sweeps} "
                  f"--runs {','.join(runs)} --tags {','.join(tags)} --tokens {','.join(tokens)} "
                  f"--labels '{','.join(LABELS)}' --out {out}")
            compare(runs, tokens, out, stage, LABELS, tags, sweeps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["bars", "subs", "pick", "draw"])
    a = ap.parse_args()
    {"bars": bars, "subs": subs, "pick": pick, "draw": draw}[a.action]()
