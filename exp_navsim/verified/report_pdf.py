"""PDF report of the 18_verified_sampling methods: tables, then BEV pages with the same names.

    python -m exp_navsim.verified.report_pdf --crossroads 50 --out outputs/18_verified_sampling/report.pdf

Needs report_samples.csv (sample_eval.py --methods), report_plans.csv (plan_eval.py
--methods) and report_crossroads.csv (crossroads_eval.py --methods --sets val,navtest).

Pages:
1. the methods (methods.py) and what the columns mean
2. the result tables, one row per method
3. crossroads: --crossroads scenes where the logged future takes one of the branches,
   drawn at random (seeded, at most one per log) from the validation and navtest crossroads, 4 per page
4. the 8 validation scenes of bev_guidance.png
Every BEV page: front camera, then one column per method of methods.PLOTTED, 32 samples
in red, logged future in black, crossroad branches shaded (green left, blue straight,
orange right).
"""

import argparse
import pickle
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from exp_navsim.data.resize_frames import frame_file
from exp_navsim.data.scenes import COMMANDS, SceneFeatureDataset
from exp_navsim.diverse.crossroads import COLORS, TURN3, crossroad_rows
from exp_navsim.diverse.junctions import OUT as JOUT, classify
from exp_navsim.diverse.plot_bev import draw
from exp_navsim.diverse.train import to_gpu
from exp_navsim.verified.methods import METHODS, PLOTTED, VERIFIER_V3, by_key
from exp_navsim.verified.sampler import Sampler
from exp_navsim.viz.plot_samples import pick_scenes

OUT = Path("outputs/18_verified_sampling")
DOWNLOAD = Path("navsim/download")
SOURCES = {"val": ("navtrain", DOWNLOAD / "trainval_navsim_logs/trainval", DOWNLOAD / "trainval_sensor_blobs/trainval"),
           "navtest": ("navtest", DOWNLOAD / "test_navsim_logs/test", DOWNLOAD / "test_sensor_blobs/test")}
A4 = (11.69, 8.27)                                              # landscape


def dataset(name):
    index = SOURCES[name][0]
    if index == "navtest":
        return SceneFeatureDataset("data/navsim_index/navtest.pkl", features=["wm11_ctx"], split="all",
                                   require_future=False)
    return SceneFeatureDataset("data/navsim_index/navtrain.pkl", features=["wm11_ctx"], split="val")


def load_scenes(name, tokens, logs):
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    _, log_dir, sensor_dir = SOURCES[name]
    loader = SceneLoader(data_path=log_dir, original_sensor_path=sensor_dir,
                         scene_filter=SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                                                  log_names=sorted(set(logs)), tokens=list(tokens)),
                         sensor_config=SensorConfig.build_no_sensors())
    return {t: loader.get_scene_from_token(t) for t in tokens}


# ------------------------------------------------------------------ pages -- #
def text_page(pdf, title, lines):
    fig = plt.figure(figsize=A4)
    fig.text(0.04, 0.95, title, fontsize=16, weight="bold", va="top")
    y = 0.90
    for line in lines:
        bold = line.startswith("**")
        line = line.strip("*")
        for part in textwrap.wrap(line, 160) or [""]:
            fig.text(0.04, y, part, fontsize=8.5, weight="bold" if bold else "normal", va="top", family="monospace")
            y -= 0.0215
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, title, df, note):
    fig = plt.figure(figsize=A4)
    fig.text(0.04, 0.95, title, fontsize=14, weight="bold", va="top")
    ax = fig.add_axes((0.02, 0.25, 0.96, 0.62))
    ax.axis("off")
    tab = ax.table(cellText=df.values, colLabels=df.columns, loc="upper center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.9)
    for (r, c), cell in tab.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.8)
        if c == 0:
            cell.set_text_props(ha="left")
    tab.auto_set_column_width(list(range(len(df.columns))))
    y = 0.22
    for line in note:
        for part in textwrap.wrap(line, 160) or [""]:
            fig.text(0.04, y, part, fontsize=9, va="top")
            y -= 0.024
    pdf.savefig(fig)
    plt.close(fig)


def bev_pages(pdf, title, items, preds, scenes, records=None, per_page=4):
    """items: list of (set name, row, token, image path, caption, logged future);
    preds: {key: list of (K, 8, 3) per item}; records: {(set, row): junction record} or None."""
    cols = 1 + len(PLOTTED)
    for p in range(0, len(items), per_page):
        chunk = list(range(p, min(p + per_page, len(items))))
        fig, axes = plt.subplots(per_page, cols, figsize=(A4[0], A4[1] * 1.0), squeeze=False,
                                 gridspec_kw={"width_ratios": [1.35] + [1] * len(PLOTTED)})
        fig.suptitle("\n".join(textwrap.wrap(f"{title}  (page {p // per_page + 1} of "
                                             f"{(len(items) + per_page - 1) // per_page})", 170)), fontsize=9, y=0.995)
        for i in range(per_page):
            if i >= len(chunk):
                for ax in axes[i]:
                    ax.axis("off")
                continue
            j = chunk[i]
            name, row, tok, img, caption, _ = items[j]
            ax = axes[i, 0]
            ax.imshow(Image.open(img))
            ax.set_title(caption, fontsize=7)
            ax.axis("off")
            rec = records[(name, row)] if records else None
            for c, key in enumerate(PLOTTED):
                ax = axes[i, c + 1]
                s = preds[key][j]
                draw(ax, scenes[tok], s, np.zeros(len(s), int), items[j][5], 1)
                sub = by_key(key)[1] if i == 0 else ""
                if rec is not None:
                    for turn, pts in rec["branches"]:
                        if turn in COLORS:
                            ax.plot(pts[:, 1], pts[:, 0], color=COLORS[turn], lw=5, alpha=0.25, zorder=3)
                    cls = [classify(t, rec["branches"]) for t in s]
                    counts = " ".join(f"{c0[0].upper()}{cls.count(c0)}" for c0 in TURN3 + ("before", "off")
                                      if cls.count(c0))
                    sub = (sub + "\n" if sub else "") + counts
                    ax.set_xlim(25, -25)
                    ax.set_ylim(-5, 45)
                ax.set_title(sub, fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        pdf.savefig(fig)
        plt.close(fig)


def sample_items(items, data_by_set, pos_by_set, samples):
    """-> {key: [ (K, 8, 3) per item ]} for the plotted methods."""
    preds = {}
    for key in PLOTTED:
        sampler = Sampler(by_key(key)[2], VERIFIER_V3)
        per_set = {n: sampler(d, samples).cpu().numpy() for n, d in data_by_set.items()}
        preds[key] = [per_set[it[0]][:, pos_by_set[it[0]][it[1]]] for it in items]
        print(f"[sampled] {key}", flush=True)
    return preds


# ------------------------------------------------------------------- main -- #
def fmt(x, pct=False):
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossroads", type=int, default=50)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "report.pdf"))
    args = ap.parse_args()

    smp = pd.read_csv(OUT / "report_samples.csv").set_index("model")
    plans = pd.read_csv(OUT / "report_plans.csv").pivot(index="spec", columns="rule", values="epdms")
    cross = pd.read_csv(OUT / "report_crossroads.csv").set_index("model")
    cross_n = (OUT / "report_crossroads.log").read_text().split("crossroads: ")[1].splitlines()[0]

    with PdfPages(args.out) as pdf:
        lines = ["**Methods (the same letters and names in every table and figure)", ""]
        for key, name, spec, desc in METHODS:
            lines += [f"{name:30s} {desc}", f"{'':30s} spec: {spec}", ""]
        lines += ["All generators: trained on all 92k navtrain scenes, 100k steps, input = WM block-11 image features only.",
                  "Verifier (D, E, F): logs_exp/18_verifier_v3, predicts the PDM sub-scores of a trajectory from the image,",
                  "trained on 10k cached navtrain scenes (vocabulary and generator samples). 32 samples per scene everywhere.",
                  "", "**Columns",
                  "bad              share of samples with EPDMS < 0.5 when each sample is scored as a plan (navtest 500 subset)",
                  "off-road         share of samples leaving the drivable area; collision: at-fault collision",
                  "sample EPDMS     mean EPDMS over all samples (how good a random candidate is)",
                  "best of 32       EPDMS of the best sample per scene (oracle)",
                  "spread           mean pairwise distance between samples [m] (diversity)",
                  "energy / minADE  against the logged future on 2000 held-out navtrain scenes (lower is better)",
                  "dispersion       spread / mean error; 1 = calibrated, < 1 overconfident",
                  "plan EPDMS       official EPDMS with two-frame comfort of ONE plan per scene: the mean of the 32 samples,",
                  "                 a random sample, or the verifier's top-1",
                  "branches         at crossroads where the driver takes a branch: how many of left / straight / right get",
                  "                 >= 1 sample (max 3); p(taken) = share of samples on the branch the driver took"]
        text_page(pdf, "Verified sampling of the trajectory distribution (outputs/18_verified_sampling)", lines)

        rows = []
        for key, name, _, _ in METHODS:
            s = smp.loc[key]
            rows.append([name, fmt(s["bad"], True), fmt(s["dac_fail"], True), fmt(s["nc_fail"], True),
                         fmt(s["sample_epdms"]), fmt(s["best_of_k"]), f"{s['spread']:.2f}", fmt(s["comfort_any"], True),
                         f"{s['val/energy']:.3f}", fmt(s["val/ade_min"]), f"{s['val/dispersion']:.2f}"])
        df = pd.DataFrame(rows, columns=["method", "bad", "off-road", "collision", "sample\nEPDMS", "best\nof 32",
                                         "spread\n[m]", "comfort\nviolations", "val\nenergy", "val\nminADE",
                                         "val\ndispersion"])
        table_page(pdf, "1. Every sample as a plan (navtest 500 subset) and calibration (2000 held-out navtrain scenes)",
                   df, ["Human driving (logged future) on the same subset: 0.2% bad, 0% off-road, sample EPDMS 0.955.",
                        "Lower bad / off-road / energy is better; higher spread means more diverse samples."])

        rows = []
        for key, name, _, _ in METHODS:
            c, p = cross.loc[key], plans.loc[key]
            rows.append([name, fmt(p["mean"]), fmt(p["sample"]), fmt(p["verifier"]),
                         f"{c['taken/branches@1']:.2f}", f"{c['taken/branches@10%']:.2f}", fmt(c["taken/p_taken"], True),
                         fmt(c["taken/off"], True), f"{c['taken/energy']:.3f}"])
        df = pd.DataFrame(rows, columns=["method", "plan EPDMS\nmean", "plan EPDMS\nrandom", "plan EPDMS\nverifier top-1",
                                         "branches\n>= 1 sample", "branches\n>= 10%", "p(taken\nbranch)",
                                         "off lanes", "energy at\ncrossroads"])
        table_page(pdf, "2. One plan per scene (navtest 500 subset) and crossroads", df,
                   ["References for plan EPDMS on the same subset: PDM-Closed 0.871 (uses the map and the true agents), "
                    "regression planner with ego status 0.887, constant velocity 0.209.",
                    f"Crossroads: {cross_n}; the branch columns use the 'taken' scenes."])

        # crossroad pages
        rng = np.random.default_rng(args.seed)
        pool = []
        recs = {}
        for name in ("val", "navtest"):
            with open(JOUT / f"junctions_{name}.pkl", "rb") as f:
                records = pickle.load(f)
            rows_, taken = crossroad_rows(records)
            pool += [(name, int(r)) for r in rows_[taken]]
            recs.update({(name, int(r)): records[r] for r in rows_[taken]})
        # at most one scene per log: consecutive frames of one log are near-copies
        logs_of = {"val": np.asarray(dataset("val").index["log"]), "navtest": np.asarray(dataset("navtest").index["log"])}
        pick, seen = [], set()
        for i in rng.permutation(len(pool)):
            name, r = pool[i]
            if (name, logs_of[name][r]) not in seen:
                seen.add((name, logs_of[name][r]))
                pick.append(pool[i])
            if len(pick) == args.crossroads:
                break
        pick.sort()
        items, data_by_set, pos_by_set, scenes = [], {}, {}, {}
        for name in ("val", "navtest"):
            rows_ = np.array([r for n, r in pick if n == name])
            if not len(rows_):
                continue
            ds = dataset(name)
            data_by_set[name] = to_gpu(ds, rows_)
            pos_by_set[name] = {int(r): i for i, r in enumerate(rows_)}
            toks = [str(ds.index["token"][r]) for r in rows_]
            scenes.update(load_scenes(name, toks, [ds.index["log"][r] for r in rows_]))
            for r, t in zip(rows_, toks):
                cmd = COMMANDS[int(ds.command[r].argmax())]
                cap = f"{name}  cmd {cmd}  v = {np.linalg.norm(ds.status[r][:2]):.1f} m/s\ndriver: {recs[(name, int(r))]['gt']}"
                items.append((name, int(r), t, frame_file(ds.index["split"], ds.index["frame_ids"][r][-1]), cap,
                              np.nan_to_num(ds.future[r])))
        preds = sample_items(items, data_by_set, pos_by_set, args.samples)
        bev_pages(pdf, "3. Crossroads where the driver takes a branch: samples red, driver black, branches shaded "
                       "(green L, blue S, orange R); panel title = samples per branch (B before, O off)",
                  items, preds, scenes, recs)

        # the scenes of bev_guidance.png
        ds = dataset("val")
        rows_ = ds.rows[pick_scenes(ds, 8, 3)]
        data = to_gpu(ds, rows_)
        toks = [str(ds.index["token"][r]) for r in rows_]
        scenes = load_scenes("val", toks, [ds.index["log"][r] for r in rows_])
        items = [("val", int(r), t, frame_file(ds.index["split"], ds.index["frame_ids"][r][-1]),
                  f"val  cmd {COMMANDS[int(ds.command[r].argmax())]}  v = {np.linalg.norm(ds.status[r][:2]):.1f} m/s",
                  ds.future[r]) for r, t in zip(rows_, toks)]
        preds = sample_items(items, {"val": data}, {"val": {int(r): i for i, r in enumerate(rows_)}}, args.samples)
        bev_pages(pdf, "4. Validation scenes with a turn or a stop (the scenes of bev_guidance.png): "
                       "samples red, driver black", items, preds, scenes)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
