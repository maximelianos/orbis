"""Front camera + BEV videos of navtest sequences at 2 fps, logged future green, predictions red.

    # candidate windows (crossroads, planner failures, Singapore left turns) -> candidates.csv
    python -m exp_navsim.viz.video select --out outputs/23_site/videos
    # one video: 16 consecutive navtest scenes of a log, starting at index row --start
    python -m exp_navsim.viz.video render --start 5120 --frames 16 --out outputs/23_site/videos/x.webm

navtest keeps only some frames of a log, so consecutive navtest scenes can be seconds
to minutes apart and such a video jumps. The navtest_seq split has every frame of the
log segments in SEGMENTS:

    python -m exp_navsim.viz.video segments                    # scene filter -> outputs/23_site/videos
    python -m exp_navsim.data.build_index --split navtest_seq  # index + front frames
    python -m exp_navsim.encoders.cache_features --encoder wm --split navtest_seq
    python -m exp_navsim.viz.video cache                       # metric cache of the segment scenes
    python -m exp_navsim.planner.predict --run logs_exp/16_regress_wm11ctx_d256 --split navtest_seq
    python -m exp_navsim.eval.evaluate --run logs_exp/15_wm11ctx_status_cmd --split navtest_seq --ckpts last --save-predictions
    python -m exp_navsim.viz.video score                       # planner EPDMS per frame (FastPDM)
    python -m exp_navsim.viz.video render --split navtest_seq --segment sg_left --out outputs/23_site/videos/sg_left.webm

Every frame is one scene (frames are 0.5 s apart in navtest_seq). Left: the CAM_F0 image with the
trajectories drawn on the ground (ego frame -> camera with the log's calibration and lens
distortion; ground height from the annotated boxes). Right: navsim's BEV (map, agents)
with the same trajectories. Red thin: samples of the diffusion probe (--samples npz),
red thick: the fast planner (--planner npz). Green: the logged future. The header gives
time, driving command and the planner's official EPDMS for the scene.
Videos are VP9 WebM (the OpenCV build here has no H.264 encoder).
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

DOWNLOAD = Path("/scratch/global/velikanm/navsim/download")
LOGS = DOWNLOAD / "test_navsim_logs/test"
SENSORS = DOWNLOAD / "test_sensor_blobs/test"
INDEX = "data/navsim_index/navtest.pkl"
PLANNER = "logs_exp/16_regress_wm11ctx_d256/navtest_top8.npz"
PLANNER_CSV = "logs_exp/16_regress_wm11ctx_d256/pdm/navtest_top8_0"
SAMPLES = "logs_exp/15_wm11ctx_status_cmd/eval/navtest_last.npz"
JUNCTIONS = "outputs/17_diversity/junctions/junctions_navtest.pkl"
SEQ = dict(index="data/navsim_index/navtest_seq.pkl",
           planner="logs_exp/16_regress_wm11ctx_d256/navtest_seq_top8.npz",
           samples="logs_exp/15_wm11ctx_status_cmd/eval/navtest_seq_last.npz",
           scores="outputs/23_site/videos/navtest_seq_scores.csv",
           filter="outputs/23_site/videos/navtest_seq_filter.yaml")
# name: (log, first frame, last frame) of the log pickle, every frame becomes a scene
SEGMENTS = {
    "crossroad_right": ("2021.09.16.19.49.00_veh-42_00990_01609", 630, 684),
    "crossroad_left": ("2021.09.16.19.27.01_veh-45_01749_03230", 2782, 2830),
    "sg_left_a": ("2021.10.06.08.16.17_veh-52_00922_01296", 556, 600),
    "sg_left_b": ("2021.10.06.08.16.17_veh-52_00922_01296", 716, 756),
}
GREEN, RED, LIGHT_RED = (40, 170, 60), (220, 40, 40), (255, 120, 120)   # RGB


def load_predictions(path, sample=None):
    d = np.load(path)
    s = d["samples"].astype(np.float32)
    return dict(zip(d["token"].tolist(), s[sample] if sample is not None else s.transpose(1, 0, 2, 3)))


def paths(split):
    if split == "navtest_seq":
        return SEQ
    return dict(index=INDEX, planner=PLANNER, samples=SAMPLES)


def scene_table(split="navtest"):
    """One row per index row: log, city, command, crossroad, planner EPDMS."""
    ix = pickle.load(open(paths(split)["index"], "rb"))
    if split == "navtest_seq":
        pdm = pd.read_csv(SEQ["scores"]).set_index("token")
        junc = {}
    else:
        junc = pickle.load(open(JUNCTIONS, "rb"))
        csv = sorted(Path(PLANNER_CSV).glob("*.csv"))[-1]
        pdm = pd.read_csv(csv, index_col=0)
        pdm = pdm[pdm["token"] != "average_all_frames"].set_index("token")
    df = pd.DataFrame({"token": ix["token"], "log": ix["log"], "city": ix["city"],
                       "command": [COMMANDS[i] for i in np.asarray(ix["command"]).argmax(1)]})
    from exp_navsim.diverse.junctions import is_crossroad
    df["crossroad"] = [is_crossroad(junc.get(i, {"status": ""})) for i in range(len(df))]
    df["junction"] = [junc.get(i, {}).get("status", "") for i in range(len(df))]
    df["epdms"] = df["token"].map(pdm["score"])
    df["dac"] = df["token"].map(pdm["drivable_area_compliance"])
    df["nc"] = df["token"].map(pdm["no_at_fault_collisions"])
    fut = np.asarray(ix["future"])
    df["gt_lat"] = fut[:, -1, 1]
    df["gt_heading"] = fut[:, -1, 2]
    samples = load_predictions(paths(split)["samples"])
    spread = []
    for t in df["token"]:
        s = samples.get(t)
        spread.append(float(s[:, -1, 1].std()) if s is not None else np.nan)
    df["sample_lat_std"] = spread
    return df


def select(out, frames=16):
    """Windows of `frames` consecutive scenes of one log, scored for three kinds of scene."""
    df = scene_table()
    rows = []
    for log, g in df.groupby("log", sort=False):
        idx = g.index.to_numpy()
        for a in range(0, len(idx) - frames + 1, 2):
            w = df.loc[idx[a:a + frames]]
            turn = w["command"].isin(["left", "right"])
            rows.append({"start": idx[a], "log": log, "city": w["city"].iloc[0],
                         "crossroad_turns": int((w["crossroad"] & turn).sum()),
                         "turn_angle": float(np.abs(w["gt_heading"]).max()),
                         "epdms_min": float(w["epdms"].min()), "epdms_mean": float(w["epdms"].mean()),
                         "bad_frames": int((w["epdms"] < 0.3).sum()),
                         "dac_fail": int((w["dac"] < 1).sum()), "nc_fail": int((w["nc"] < 1).sum()),
                         "sample_spread": float(w["sample_lat_std"].max()),
                         "sg_left": int(((w["city"] == "sg-one-north") & (w["command"] == "left")).sum())})
    c = pd.DataFrame(rows)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    c.to_csv(out / "candidates.csv", index=False)
    show = dict(
        crossroad_success=c[(c.crossroad_turns >= 4) & (c.turn_angle > 1.0)].sort_values(
            ["epdms_min", "sample_spread"], ascending=False),
        crossroad_spread=c[c.crossroad_turns >= 4].sort_values("sample_spread", ascending=False),
        failure=c[c.bad_frames >= 3].sort_values(["bad_frames", "epdms_mean"], ascending=[False, True]),
        sg_left=c[(c.sg_left >= 6) & (c.turn_angle > 1.0)].sort_values("epdms_mean"),
    )
    for k, v in show.items():
        print(f"== {k} ({len(v)})")
        print(v.head(8).to_string(index=False))


def ground_height(frame):
    boxes = frame.annotations.boxes
    if len(boxes) == 0:
        return -0.35
    bottom = boxes[:, 2] - boxes[:, 5] / 2
    near = np.hypot(boxes[:, 0], boxes[:, 1]) < 40
    return float(np.clip(np.median(bottom[near]) if near.any() else np.median(bottom), -1.0, 0.3))


def densify(traj, n=60):
    pts = np.concatenate([np.zeros((1, 2)), traj[:, :2]])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if d[-1] < 1e-3:
        return pts
    s = np.linspace(0, d[-1], n)
    return np.stack([np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])], 1)


def project(xy, z, cam, shape):
    """Ego-frame ground points (N, 2) -> list of pixel polylines, one per run of visible points.
    Points outside the field of view are dropped before the lens distortion, whose
    polynomial folds far-off points back into the image."""
    R, t = cam.sensor2lidar_rotation, cam.sensor2lidar_translation
    pts = np.concatenate([xy, np.full((len(xy), 1), z)], 1)
    pc = (pts - t) @ R                                      # lidar(=ego) -> camera frame
    depth = np.maximum(pc[:, 2], 1e-6)
    half_w = shape[1] / 2 / cam.intrinsics[0, 0]
    half_h = shape[0] / 2 / cam.intrinsics[1, 1]
    keep = (pc[:, 2] > 2.0) & (np.abs(pc[:, 0] / depth) < 1.15 * half_w) & (np.abs(pc[:, 1] / depth) < 1.15 * half_h)
    idx = np.flatnonzero(keep)
    if len(idx) < 2:
        return []
    px, _ = cv2.projectPoints(pc[idx], np.zeros(3), np.zeros(3), cam.intrinsics, cam.distortion)
    px = px[:, 0]
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    return [seg.astype(np.int32) for seg in np.split(px, breaks) if len(seg) > 1]


def draw_camera(img, frame, gt, planner, samples):
    img = img.copy()
    z = ground_height(frame)
    cam = frame.cameras.cam_f0
    overlay = img.copy()
    for s in samples:
        cv2.polylines(overlay, project(densify(s), z, cam, img.shape[:2]), False, LIGHT_RED, 3, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    for traj, color, width in ((gt, GREEN, 9), (planner, RED, 7)):
        if traj is None:
            continue
        cv2.polylines(img, project(densify(traj), z, cam, img.shape[:2]), False, color, width, cv2.LINE_AA)
    return img


def draw_bev(scene, gt, planner, samples, size_px=540):
    from navsim.visualization.bev import add_configured_bev_on_ax
    from navsim.visualization.plots import configure_ax, configure_bev_ax
    fig, ax = plt.subplots(figsize=(size_px / 100, size_px / 100), dpi=100)
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[3])
    configure_bev_ax(ax)
    configure_ax(ax)
    for s in samples:
        ax.plot(np.r_[0, s[:, 1]], np.r_[0, s[:, 0]], color=np.array(LIGHT_RED) / 255, lw=1.0, alpha=0.7, zorder=5)
    ax.plot(np.r_[0, gt[:, 1]], np.r_[0, gt[:, 0]], color=np.array(GREEN) / 255, lw=3.0, zorder=6)
    if planner is not None:
        ax.plot(np.r_[0, planner[:, 1]], np.r_[0, planner[:, 0]], color=np.array(RED) / 255, lw=2.5, zorder=7)
    ax.set_xlim(32, -32)
    ax.set_ylim(-14, 50)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    bev = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return bev


def segment_frames(log, first, last):
    frames = pickle.load(open(LOGS / f"{log}.pkl", "rb"))
    return [f["token"] for f in frames[first:last + 1]]


def write_segments():
    """Scene filter of navtest_seq: every frame of the SEGMENTS (build_index reads it)."""
    import yaml
    tokens = [t for log, a, b in SEGMENTS.values() for t in segment_frames(log, a, b)]
    cfg = {"_target_": "navsim.common.dataclasses.SceneFilter", "_convert_": "all", "num_history_frames": 4,
           "num_future_frames": 10, "frame_interval": 1, "has_route": True, "max_scenes": None,
           "log_names": sorted({log for log, _, _ in SEGMENTS.values()}), "tokens": tokens}
    Path(SEQ["filter"]).parent.mkdir(parents=True, exist_ok=True)
    Path(SEQ["filter"]).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{len(tokens)} frames -> {SEQ['filter']}")


def metric_cache(workers=8):
    """Metric cache of the navtest_seq scenes, as run_metric_caching.py does for navtest."""
    import os
    import yaml
    from exp_navsim.eval.pdm import CACHE
    from exp_navsim.planner.fast_pdm import CONFIG_DIR
    os.environ["NAVSIM_EXP_ROOT"] = str(Path("outputs/navsim_exp").resolve())
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(DOWNLOAD))
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from navsim.planning.metric_caching.caching import cache_data
    from navsim.planning.script.builders.worker_pool_builder import build_worker
    flt = yaml.safe_load(open(SEQ["filter"]))
    with initialize_config_dir(config_dir=str(CONFIG_DIR.parent / "metric_caching"), version_base=None):
        cfg = compose(config_name="default_metric_caching", overrides=[
            "train_test_split=navtest", f"navsim_log_path={LOGS}",
            f"metric_cache_path={(CACHE / 'navtest_seq').resolve()}", "worker=single_machine_thread_pool",
            "worker.use_process_pool=true", f"worker.max_workers={workers}"])
    with open_dict(cfg):
        cfg.train_test_split.scene_filter.tokens = flt["tokens"]
        cfg.train_test_split.scene_filter.log_names = flt["log_names"]
    cache_data(cfg=cfg, worker=build_worker(cfg))


def score_segments():
    """Official sub-scores of the planner on every navtest_seq frame, two-frame comfort
    against the previous frame."""
    from exp_navsim.planner.fast_pdm import FastPDM
    ix = pickle.load(open(SEQ["index"], "rb"))
    planner = load_predictions(SEQ["planner"], 0)
    tokens = [t for t in ix["token"] if t in planner]
    previous = {t: p for p, t, lp, l in zip(ix["token"][:-1], ix["token"][1:], ix["log"][:-1], ix["log"][1:])
                if lp == l and t in planner and p in planner}
    for t in tokens:
        previous.setdefault(t, t)          # first frame of a segment: comfort against itself is skipped
    scorer = FastPDM("navtest_seq", tokens, workers=8)
    df = scorer.score({t: planner[t][None] for t in tokens}, previous)
    scorer.close()
    df.to_csv(SEQ["scores"], index=False)
    print(df[["score", "drivable_area_compliance", "no_at_fault_collisions"]].describe().round(3))


def render(start, frames, out, num_samples=16, title="", split="navtest"):
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader

    df = scene_table(split)
    w = df.loc[start:start + frames - 1]
    assert w["log"].nunique() == 1, "window crosses a log boundary"
    tokens = w["token"].tolist()
    sensors = SensorConfig(cam_f0=[3], cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False,
                           cam_r2=False, cam_b0=False, lidar_pc=False)
    loader = SceneLoader(data_path=LOGS, original_sensor_path=SENSORS,
                         scene_filter=SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1,
                                                  log_names=[w["log"].iloc[0]], tokens=tokens),
                         sensor_config=sensors)
    ix = pickle.load(open(paths(split)["index"], "rb"))
    planner = load_predictions(paths(split)["planner"], 0)
    samples = load_predictions(paths(split)["samples"])
    writer, size = None, None
    stills = []
    for k, (row, r) in enumerate(w.iterrows()):
        scene = loader.get_scene_from_token(r["token"])
        frame = scene.frames[3]
        gt = np.asarray(ix["future"][row])
        smp = samples.get(r["token"], np.zeros((0, 8, 3)))[:num_samples]
        img = draw_camera(frame.cameras.cam_f0.image, frame, gt, planner.get(r["token"]), smp)
        img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        bev = draw_bev(scene, gt, planner.get(r["token"]), smp)
        canvas = np.concatenate([img, bev], 1)
        head = np.full((44, canvas.shape[1], 3), 255, np.uint8)
        text = (f"{title}   t = {k * 0.5:4.1f} s   command: {r['command']}   "
                f"planner EPDMS {r['epdms']:.2f}" + ("   [crossroad]" if r["crossroad"] else ""))
        cv2.putText(head, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
        canvas = np.concatenate([head, canvas], 0)
        if writer is None:
            size = (canvas.shape[1], canvas.shape[0])
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"VP90"), 2, size)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        stills.append(canvas)
    writer.release()
    # poster frame and a contact sheet of every 4th frame for checking
    cv2.imwrite(str(Path(out).with_suffix(".jpg")), cv2.cvtColor(stills[len(stills) // 2], cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    sheet = np.concatenate([cv2.resize(s, (s.shape[1] // 2, s.shape[0] // 2)) for s in stills[::4]], 0)
    cv2.imwrite(str(Path(out).with_name(Path(out).stem + "_sheet.jpg")), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{len(stills)} frames -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["select", "segments", "cache", "score", "render"])
    ap.add_argument("--out", default="outputs/23_site/videos")
    ap.add_argument("--split", default="navtest", choices=["navtest", "navtest_seq"])
    ap.add_argument("--start", type=int)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--segment", choices=list(SEGMENTS), help="navtest_seq: render this whole segment")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    if args.action == "select":
        select(args.out, args.frames)
    elif args.action == "segments":
        write_segments()
    elif args.action == "cache":
        metric_cache()
    elif args.action == "score":
        score_segments()
    else:
        start, frames = args.start, args.frames
        if args.segment:
            log, a, b = SEGMENTS[args.segment]
            ix = pickle.load(open(SEQ["index"], "rb"))
            rows = [i for i, (t, l) in enumerate(zip(ix["token"], ix["log"]))
                    if l == log and t in set(segment_frames(log, a, b))]
            start, frames = rows[0], len(rows)
        render(start, frames, args.out, title=args.title, split=args.split)


if __name__ == "__main__":
    main()
