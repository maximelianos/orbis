"""Camera geometry of the 8 NAVSIM cameras and the viewing ray of every token cell.

    python -m exp_navsim.v22.cameras          # -> data/features/camera_calib.json

The calibration is read from the `cams` entry of the log pkl frames: intrinsics (3 x 3, at
the 1920 x 1080 sensor resolution), OpenCV distortion (k1, k2, p1, p2, k3) and the
sensor-to-lidar rotation and translation. The lidar frame is the ego frame here
(lidar2ego is the identity in the OpenScene logs; main() checks it), with x forward,
y left, z up.

The reference calibration is the first navtrain log; main() also compares 5 logs of
different vehicles and the first and last frame of a log.

token_rays(calib, cams) gives the unit viewing direction in the ego frame of the centre of
every cell of the cached 9 x 16 grid. The images were resized from 1920 x 1080 to
512 x 288 without a crop (resize_frames.py, the same 16:9 aspect), the tokens are 16 px
patches of that image (18 x 32) and the cache pools them 2 x 2, so a cell is 32 px at
512 x 288, i.e. 120 px of the sensor image.
"""

import json
import pickle
from pathlib import Path

import numpy as np

from exp_navsim.data.build_index import SPLITS

CAMS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")
SENSOR_SIZE = (1080, 1920)           # (H, W) of the raw images
RESIZED = (288, 512)                 # the encoders' input, resize_frames.SIZE
GRID = (9, 16)                       # cached feature grid
CALIB = Path("data/features/camera_calib.json")


def _calib_of(frame):
    out = {}
    for cam in CAMS:
        c = frame["cams"][cam]
        out[cam] = {"intrinsics": np.asarray(c["cam_intrinsic"], float).tolist(),
                    "distortion": np.asarray(c["distortion"], float).tolist(),
                    "sensor2lidar_rotation": np.asarray(c["sensor2lidar_rotation"], float).tolist(),
                    "sensor2lidar_translation": np.asarray(c["sensor2lidar_translation"], float).tolist()}
    return out


def _rot_deg(a, b):
    """Angle in degrees of the rotation between two rotation matrices."""
    r = np.asarray(a).T @ np.asarray(b)
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def _diff(ref, other):
    """Per camera: max abs intrinsics difference (px), rotation angle (deg), translation (m)."""
    return {cam: {"intrinsics_px": float(np.abs(np.subtract(ref[cam]["intrinsics"], other[cam]["intrinsics"])).max()),
                  "distortion": float(np.abs(np.subtract(ref[cam]["distortion"], other[cam]["distortion"])).max()),
                  "rotation_deg": _rot_deg(ref[cam]["sensor2lidar_rotation"], other[cam]["sensor2lidar_rotation"]),
                  "translation_m": float(np.linalg.norm(np.subtract(ref[cam]["sensor2lidar_translation"],
                                                                     other[cam]["sensor2lidar_translation"])))}
            for cam in CAMS}


def load_calib(path=CALIB):
    return json.loads(Path(path).read_text())["cams"]


def cell_centres(grid=GRID, resized=RESIZED, sensor=SENSOR_SIZE):
    """(h * w, 2) sensor-pixel (u, v) of the cell centres of the cached grid."""
    s = sensor[1] / resized[1]
    assert abs(sensor[0] / resized[0] - s) < 1e-6, "resize_frames keeps the aspect, no crop"
    ch, cw = resized[0] / grid[0], resized[1] / grid[1]
    v, u = np.meshgrid((np.arange(grid[0]) + 0.5) * ch * s, (np.arange(grid[1]) + 0.5) * cw * s, indexing="ij")
    return np.stack([u.ravel(), v.ravel()], -1)


def token_rays(calib, cams, grid=GRID):
    """-> (len(cams), h, w, 3) float32 unit rays in the ego frame (x fwd, y left, z up).

    calib: load_calib() dict; cams: names like "CAM_L0" or "L0". The cell centre is
    undistorted with the OpenCV model of the camera, lifted to the camera frame (x right,
    y down, z along the optical axis) and rotated by sensor2lidar_rotation.
    """
    import cv2
    uv = cell_centres(grid).astype(np.float64)
    out = []
    for cam in cams:
        c = calib[cam if cam.startswith("CAM_") else f"CAM_{cam}"]
        k, d = np.asarray(c["intrinsics"]), np.asarray(c["distortion"])
        xy = cv2.undistortPoints(uv[:, None], k, d)[:, 0]
        ray = np.concatenate([xy, np.ones((len(xy), 1))], -1) @ np.asarray(c["sensor2lidar_rotation"]).T
        out.append((ray / np.linalg.norm(ray, axis=-1, keepdims=True)).reshape(*grid, 3))
    return np.stack(out).astype(np.float32)


def main():
    logs_dir = SPLITS["navtrain"]["logs"]
    with open("data/navsim_index/navtrain.pkl", "rb") as f:
        index = pickle.load(f)
    logs = sorted(set(index["log"]))
    # one log per vehicle, the first 5 vehicles
    by_vehicle = {}
    for log in logs:
        by_vehicle.setdefault(log.split("_")[1], log)
    picked = [by_vehicle[v] for v in sorted(by_vehicle)][:5]
    calibs, lidar2ego = {}, {}
    for log in picked:
        with open(logs_dir / f"{log}.pkl", "rb") as f:
            frames = pickle.load(f)
        calibs[log] = _calib_of(frames[0])
        last = _calib_of(frames[-1])
        lidar2ego[log] = {"translation": np.asarray(frames[0]["lidar2ego_translation"], float).tolist(),
                          "rotation": np.asarray(frames[0]["lidar2ego_rotation"], float).tolist(),
                          "first_vs_last_frame": _diff(calibs[log], last)}
    ref_log = picked[0]
    ref = calibs[ref_log]
    variation = {log: _diff(ref, c) for log, c in calibs.items() if log != ref_log}
    worst = {cam: {k: max(v[cam][k] for v in variation.values()) for k in variation[picked[1]][cam]}
             for cam in CAMS}
    rays = token_rays(ref, CAMS)
    centre = {cam: rays[i, GRID[0] // 2, GRID[1] // 2].round(3).tolist() for i, cam in enumerate(CAMS)}
    yaw = {cam: float(np.degrees(np.arctan2(c[1], c[0]))) for cam, c in centre.items()}
    out = {"reference_log": ref_log, "sensor_size_hw": SENSOR_SIZE, "resized_hw": RESIZED, "grid": GRID,
           "cams": ref, "logs": picked, "lidar2ego": lidar2ego, "variation_vs_reference": variation,
           "max_variation": worst, "centre_ray": centre, "centre_yaw_deg": yaw}
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps(out, indent=1))
    print(f"reference {ref_log}, compared with {picked[1:]}")
    for cam in CAMS:
        w = worst[cam]
        print(f"{cam}: centre yaw {yaw[cam]:7.1f} deg, f {ref[cam]['intrinsics'][0][0]:.0f} px | max over logs: "
              f"K {w['intrinsics_px']:.1f} px, dist {w['distortion']:.3f}, R {w['rotation_deg']:.2f} deg, "
              f"t {w['translation_m']:.3f} m")
    for log, l2e in lidar2ego.items():
        fl = max(v["rotation_deg"] for v in l2e["first_vs_last_frame"].values())
        print(f"{log}: lidar2ego t {l2e['translation']} q {l2e['rotation']}, first vs last frame R {fl:.3f} deg")
    print(f"-> {CALIB}")


if __name__ == "__main__":
    main()
