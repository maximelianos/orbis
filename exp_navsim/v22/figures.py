"""Front camera + BEV figures of navhard scenes: method comparison and diverse sampling.

    # scenes where the runs disagree most in EPDMS, from the per-group files of score_runs
    python -m exp_navsim.v22.figures pick --runs 16_regress_wm11ctx_d256,24c_joint,24g_metric
    # one panel row per method, front camera left, BEV right
    python -m exp_navsim.v22.figures compare --runs 16_regress_wm11ctx_d256,24c_joint,24g_metric
    # the same scene with temperature 0 (one plan) and temperature 1 (K samples)
    python -m exp_navsim.v22.figures diverse --run 24c_joint --temperature 1.0 --samples 16

The drawing is the one of viz/video.py: the trajectories are projected onto the ground
plane of the front camera with the log's calibration and lens distortion, and drawn again
on navsim's BEV. Green is the logged future, red the plan, light red the samples.

navhard has two kinds of frame. Stage-one tokens are real test-log frames; stage-two tokens
are synthetic renderings from the navhard download, which is where our planner loses most
of its score, so `--stage two` makes the figures from those.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp_navsim.data.scenes import COMMANDS
from exp_navsim.v22.fast_epdms import all_groups
from exp_navsim.viz.video import DOWNLOAD, draw_bev, draw_camera

LOGS = DOWNLOAD / "test_navsim_logs/test"
SENSORS = DOWNLOAD / "test_sensor_blobs/test"
SYNTH_SENSORS = DOWNLOAD / "navhard_two_stage/sensor_blobs"
SYNTH_SCENES = DOWNLOAD / "navhard_two_stage/synthetic_scene_pickles"
INDEX = "data/navsim_index/navhard.pkl"
OUT = Path("outputs/24_v22")
FILTER = Path("navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml")


def navhard_loader(tokens, synthetic=()):
    """SceneLoader for navhard. Real frames go in `tokens`, synthetic ones in
    `synthetic_scene_tokens`; the loader keeps the two lists apart."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    scene_filter = instantiate(OmegaConf.load(FILTER))
    scene_filter.tokens = list(tokens) or None
    scene_filter.synthetic_scene_tokens = list(synthetic) or None
    sensors = SensorConfig(cam_f0=[3], cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False,
                           cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)
    return SceneLoader(data_path=LOGS, original_sensor_path=SENSORS,
                       synthetic_sensor_path=SYNTH_SENSORS, synthetic_scenes_path=SYNTH_SCENES,
                       scene_filter=scene_filter, sensor_config=sensors)


def token_scores(names):
    """{run: {token: EPDMS of the group half that this stage-one token leads}} from score_runs."""
    out = {}
    for n in names:
        df = pd.read_csv(OUT / "per_group" / f"{n}.csv")
        df = df[df.k == 0]
        out[n] = dict(zip(df["token"], df["score"]))
    return out


def sweep_plans(name, tag, sweeps):
    """Plans of run__rule from a sweep_rank output sweeps/<run><tag>_plans.npz."""
    run, rule = name.split("__")
    d = np.load(Path(sweeps) / f"{run}{tag}_plans.npz")
    j = list(d["rules"]).index(rule)
    return dict(zip(d["token"].tolist(), d["plans"][:, j].astype(np.float32)))


def sweep_token_scores(name, tag, sweeps):
    """{token: EPDMS of that token} of run__rule from sweeps/<run><tag>_tokens.csv (--per-token)."""
    run, rule = name.split("__")
    df = pd.read_csv(Path(sweeps) / f"{run}{tag}_tokens.csv")
    df = df[df.rule == rule]
    return dict(zip(df["token"], df["score"]))


def predictions(name, k=0):
    """Plans of a run for every navhard token; k = None gives all saved samples."""
    if "__" in name:                                   # run__rule of v22/sweep_rank.py
        run, rule = name.split("__")
        tag = "_all" if rule.endswith("_all") else ""
        d = np.load(OUT / "sweeps" / f"{run}{tag}_plans.npz")
        j = list(d["rules"]).index(rule[:-4] if tag else rule)
        return dict(zip(d["token"].tolist(), d["plans"][:, j].astype(np.float32)))
    run = Path("logs_exp") / name
    if not run.exists() and name.endswith("_all"):    # score_runs name of a whole-split rescoring
        run = Path("logs_exp") / name[:-4]
    d = np.load(run / "navhard_top8.npz")
    s = d["samples"].astype(np.float32)
    return dict(zip(d["token"].tolist(), s.transpose(1, 0, 2, 3) if k is None else s[k]))


def index():
    ix = pickle.load(open(INDEX, "rb"))
    ix["pos"] = {str(t): i for i, t in enumerate(ix["token"])}
    return ix


def pick(runs, n=8):
    """Stage-one tokens ranked by how much the last run beats the first one."""
    sc = token_scores(runs)
    base, new = sc[runs[0]], sc[runs[-1]]
    rows = [{"token": t, **{r: sc[r].get(t, np.nan) for r in runs},
             "gain": new.get(t, np.nan) - base.get(t, np.nan)}
            for t in base if t in new]
    df = pd.DataFrame(rows).dropna().sort_values("gain")
    ix = index()
    df["command"] = [COMMANDS[np.asarray(ix["command"][ix["pos"][t]]).argmax()] for t in df["token"]]
    df["city"] = [ix["city"][ix["pos"][t]] for t in df["token"]]
    out = pd.concat([df.tail(n), df.head(n)])
    out.to_csv(OUT / "figure_scenes.csv", index=False)
    print(out.to_string(index=False))
    return out


def stage_two_token(token):
    """One synthetic frame that branches off a stage-one token, or None."""
    for now, prev, pairs in all_groups():
        if token == now:
            return pairs[len(pairs) // 2][0]
        if token == prev:
            return pairs[len(pairs) // 2][1]
    return None


def panel(scene, frame, gt, plan, samples, label, width=1500):
    """Front camera (left) + BEV (right) with a caption strip above."""
    img = draw_camera(frame.cameras.cam_f0.image, frame, gt, plan, samples)
    h = 540
    img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)
    bev = draw_bev(scene, gt if gt is not None else np.zeros((8, 3)), plan, samples, size_px=h)
    canvas = np.concatenate([img, bev], 1)
    canvas = cv2.resize(canvas, (width, int(canvas.shape[0] * width / canvas.shape[1])))
    head = np.full((38, canvas.shape[1], 3), 255, np.uint8)
    cv2.putText(head, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return np.concatenate([head, canvas], 0)


def draw_plans(scene, frame, gt, plans, goals=None, h=540):
    """Front camera and BEV images with several plans in their own colours.
    plans: list of (trajectory (8, 3), RGB tuple 0..255, line width); the first is drawn
    at the bottom. goals: optional (G, 2+) ego-frame points drawn as stars on the BEV.
    The BEV is zoomed to the drawn points (half width 10..32 m); returns (camera, bev, half width)."""
    from navsim.visualization.bev import add_configured_bev_on_ax
    from navsim.visualization.plots import configure_ax, configure_bev_ax
    from exp_navsim.viz.video import GREEN, densify, ground_height, project

    img = frame.cameras.cam_f0.image.copy()
    z = ground_height(frame)
    cam = frame.cameras.cam_f0
    lines = ([(gt, GREEN, 9)] if gt is not None else []) + list(plans)
    for traj, color, width in lines:
        cv2.polylines(img, project(densify(traj), z, cam, img.shape[:2]), False, color, width, cv2.LINE_AA)
    img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)

    fig, ax = plt.subplots(figsize=(h / 100, h / 100), dpi=100)
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[3])
    configure_bev_ax(ax)
    configure_ax(ax)
    for k, (traj, color, width) in enumerate(lines):
        ax.plot(np.r_[0, traj[:, 1]], np.r_[0, traj[:, 0]], color=np.array(color) / 255,
                lw=width / 3, zorder=6 + k)
        ax.plot(traj[-1, 1], traj[-1, 0], "o", color=np.array(color) / 255, ms=4, zorder=6 + k)
    if goals is not None:
        ax.plot(goals[:, 1], goals[:, 0], "*", color="#f4c430", ms=15, mec="k", mew=0.8, zorder=20)
    pts = np.concatenate([traj[:, :2] for traj, _, _ in lines] + ([goals[:, :2]] if goals is not None else []))
    hw = float(np.clip(max(np.abs(pts[:, 1]).max() + 4, (pts[:, 0].max() + 4) / 1.56), 10, 32))
    ax.set_xlim(hw, -hw)
    ax.set_ylim(-0.44 * hw, 1.56 * hw)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    bev = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img, bev, hw


def compare(runs, tokens, out, stage="one", labels=None, tags=None, sweeps=None):
    """One row per run for every token; the rows of a token are stacked into one image.
    With `tags` (one per run) the plans and per-token scores come from the sweep_rank
    outputs in `sweeps` instead (run__rule with <run><tag>_plans.npz and _tokens.csv)."""
    labels = labels or runs
    ix = index()
    # with --stage two a stage-one token is replaced by one of its synthetic branches,
    # a synthetic token is drawn as it is
    use = [t if stage == "one" or ix["synthetic"][ix["pos"][t]] else stage_two_token(t) for t in tokens]
    use = [t for t in use if t is not None]
    loader = navhard_loader([], use) if stage == "two" else navhard_loader(use)
    if tags:
        # the same run__rule can come with several tags, so key by both; `sweeps` is one
        # dir for all runs or a comma-separated list with one dir per run
        dirs = sweeps.split(",")
        dirs = dirs * len(runs) if len(dirs) == 1 else dirs
        preds = {r + g: sweep_plans(r, g, d) for r, g, d in zip(runs, tags, dirs)}
        sc = {r + g: sweep_token_scores(r, g, d) for r, g, d in zip(runs, tags, dirs)}
        runs = [r + g for r, g in zip(runs, tags)]
    else:
        preds = {r: predictions(r) for r in runs}
        sc = token_scores([r for r in runs if (OUT / "per_group" / f"{r}.csv").exists()])
    score_name = "token EPDMS" if tags else "group EPDMS"
    Path(out).mkdir(parents=True, exist_ok=True)
    for t in use:
        scene = loader.get_scene_from_token(t)
        frame = scene.frames[3]
        row = ix["pos"][t]
        gt = np.asarray(ix["future"][row])
        gt = None if not np.isfinite(gt).all() else gt
        rows = []
        for r, lab in zip(runs, labels):
            s = sc.get(r, {}).get(t)
            cap = f"{lab}" + (f"   {score_name} {100 * s:.1f}" if s is not None else "")
            cap += f"   {ix['city'][row]}   command {COMMANDS[np.asarray(ix['command'][row]).argmax()]}"
            cap += f"   stage {'one' if not ix['synthetic'][row] else 'two'}"
            rows.append(panel(scene, frame, gt, preds[r].get(t), np.zeros((0, 8, 3)), cap))
        img = np.concatenate(rows, 0)
        path = Path(out) / f"{t}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"-> {path}")


def diverse(run, tokens, out, temperature=1.0, n_samples=16, steps=8, stage="one"):
    """The deterministic plan next to `n_samples` plans drawn at a positive temperature."""
    import torch

    from exp_navsim.data.scenes import SceneFeatureDataset
    from exp_navsim.v22.runs import build_planner, feature_dir, feature_names

    ckpt = torch.load(Path("logs_exp") / run / "model.pt", map_location="cpu", weights_only=False)
    a = ckpt["args"]
    feats = feature_names(a)
    ds = SceneFeatureDataset(INDEX, split="all", require_future=False, features=feats,
                             feature_dir=feature_dir(a), history=4)
    vocab = np.load(OUT.parent / "16_fast_planner" / f"vocab_{a['vocab']}.npy") if a["vocab"] else None
    T, C = ds.features_of(ds.rows[:1]).shape[1:3]
    model = build_planner(a, C, T, vocab)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ix = index()
    use = [t if stage == "one" else stage_two_token(t) for t in tokens]
    use = [t for t in use if t is not None]
    rows = np.array([ix["pos"][t] for t in use])
    batch = {"features": torch.from_numpy(ds.features_of(rows)).float(),
             "future": torch.from_numpy(np.nan_to_num(ix["future"][rows])).float(),
             "past": torch.from_numpy(ix["past"][rows]).float(),
             "status": torch.from_numpy(np.concatenate([ix["velocity"], ix["acceleration"]], 1)[rows]).float(),
             "command": torch.from_numpy(np.asarray(ix["command"])[rows]).float(),
             # goal runs: the logged end pose (NaN for synthetic scenes = the null goal)
             "goal": torch.from_numpy(np.asarray(ix["future"])[rows][:, -1]).float()}
    with torch.no_grad():
        det = model.sample(batch, 1, steps=steps, temperature=0.0)[0].numpy()
        smp = model.sample(batch, n_samples, steps=steps, temperature=temperature).numpy()
        top = None
        if model.vocab is not None and model.scorer is not None:
            # the metric planner: its plan is the best anchor, and the next ones are the
            # alternative modes it would also accept
            top = model.predict(batch, 8, plan="vocab", rank="mix").numpy()
            det = top[0]
    loader = navhard_loader([], use) if stage == "two" else navhard_loader(use)
    Path(out).mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(use):
        scene = loader.get_scene_from_token(t)
        frame = scene.frames[3]
        gt = np.asarray(ix["future"][ix["pos"][t]])
        gt = None if not np.isfinite(gt).all() else gt
        a_ = panel(scene, frame, gt, det[i], np.zeros((0, 8, 3)),
                   f"{run}   the driven plan" + ("" if top is not None else f" (temperature 0, {steps} Euler steps)"))
        rows_ = [a_]
        if top is not None:
            rows_.append(panel(scene, frame, gt, det[i], top[1:, i],
                               f"{run}   the 8 best anchors by log EPDMS + 0.03 log p_imitation"))
        b_ = panel(scene, frame, gt, det[i], smp[:, i],
                   f"{run}   temperature {temperature}: {n_samples} samples, spread "
                   f"{smp[:, i, -1, 1].std():.2f} m at 4 s")
        path = Path(out) / f"{t}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows_ + [b_], 0), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["pick", "compare", "diverse"])
    ap.add_argument("--runs", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--run", default="")
    ap.add_argument("--tokens", default="", help="comma-separated; default: figure_scenes.csv")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--stage", default="one", choices=("one", "two"))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tags", default="", help="compare: one sweep_rank tag per run (run__rule), "
                    "e.g. _all_near,_all_near_ens5_prob; plans and token scores from --sweeps")
    ap.add_argument("--sweeps", default=str(OUT / "sweeps"), help="one dir, or one per run (comma-separated)")
    args = ap.parse_args()
    runs = [r for r in args.runs.split(",") if r]
    labels = [l for l in args.labels.split(",") if l] or None

    if args.action == "pick":
        pick(runs, args.n)
        return
    if args.tokens:
        tokens = args.tokens.split(",")
    else:
        df = pd.read_csv(OUT / "figure_scenes.csv")
        tokens = df.sort_values("gain", ascending=False)["token"].astype(str).tolist()[:args.n]
    if args.action == "compare":
        tags = args.tags.split(",") if args.tags else None
        compare(runs, tokens, args.out or OUT / f"figures_compare_{args.stage}", args.stage, labels,
                tags, args.sweeps)
    else:
        diverse(args.run, tokens, args.out or OUT / "figures_diverse", args.temperature,
                args.samples, stage=args.stage)


if __name__ == "__main__":
    main()
