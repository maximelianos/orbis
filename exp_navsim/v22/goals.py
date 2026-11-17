"""Goal input of every navhard token: the logged ego pose 4 s ahead, in the current ego frame.

    python -m exp_navsim.v22.goals            # -> outputs/25_terra/navhard_goals.npz

py123d_garage feeds the end point of the logged ego trajectory as its navigation "goal".
Stage-1 navhard scenes are log scenes, so their goal is the last pose of the logged future
(index["future"][:, -1]). Stage-2 scenes are synthetic: they have no future frames
(num_future_frames = 0), but their metadata names `corresponding_original_scene`, the log
frame with the same timestamp. Their goal is the log pose 8 frames (4 s) after that frame,
transformed from global coordinates into the synthetic ego frame (the synthetic ego pose
deviates from the log pose, by up to a few metres). The same log lookup is run on the
stage-1 tokens as a check against the index future.
"""

import pickle
from pathlib import Path

import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2

from exp_navsim.data.build_index import DOWNLOAD, _pose
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

LOGS = DOWNLOAD / "test_navsim_logs/test"
SYN = DOWNLOAD / "navhard_two_stage/synthetic_scene_pickles"
OUT = Path("outputs/25_terra/navhard_goals.npz")
AHEAD = 8                   # 8 frames at 2 Hz = the 4 s planning horizon


def main():
    index = pickle.load(open("data/navsim_index/navhard.pkl", "rb"))
    tokens = list(index["token"])
    syn = np.asarray(index["synthetic"])
    need = set(np.asarray(tokens)[syn])

    # synthetic scenes: current global pose and the log frame with the same timestamp
    meta = {}
    for p in sorted(SYN.iterdir()):                      # one flat directory of pickles
        d = pickle.load(open(p, "rb"))
        m = d["scene_metadata"]
        if m["initial_token"] in need:
            meta[m["initial_token"]] = (m["log_name"], m["corresponding_original_scene"],
                                        np.asarray(d["frames"][-1]["ego_status"]["ego_pose"], np.float64))
    print(f"[goals] synthetic metadata of {len(meta)} / {len(need)} tokens", flush=True)

    logs = sorted({v[0] for v in meta.values()} | set(np.asarray(index["log"])[~syn]))
    where = {}                                            # token -> (log, frame index)
    frames = {}
    for log in logs:
        fr = pickle.load(open(LOGS / f"{log}.pkl", "rb"))
        frames[log] = np.array([_pose(f) for f in fr], np.float64)
        for i, f in enumerate(fr):
            where[f["token"]] = (log, i)

    def ahead(origin_pose, tok):
        """Log pose AHEAD frames after `tok`, relative to `origin_pose`; NaN past the log end."""
        if tok not in where:
            return np.full(3, np.nan)
        log, i = where[tok]
        if i + AHEAD >= len(frames[log]):
            return np.full(3, np.nan)
        return convert_absolute_to_relative_se2_array(StateSE2(*origin_pose), frames[log][i + AHEAD][None])[0]

    goal = np.full((len(tokens), 3), np.nan, np.float32)
    check = []
    for j, t in enumerate(tokens):
        if syn[j]:
            if t in meta:
                log, orig, pose = meta[t]
                goal[j] = ahead(pose, orig)
        else:
            goal[j] = index["future"][j, -1]
            if t in where:
                log, i = where[t]
                check.append(np.abs(ahead(frames[log][i], t) - goal[j]).max())
    ok = np.isfinite(goal).all(1)
    print(f"[goals] stage 1: {ok[~syn].sum()} / {(~syn).sum()} goals, log lookup vs index future "
          f"max |diff| {np.nanmax(check):.2e} over {len(check)} tokens", flush=True)
    print(f"[goals] stage 2: {ok[syn].sum()} / {syn.sum()} goals; "
          f"|goal xy| median {np.median(np.linalg.norm(goal[syn & ok, :2], axis=1)):.1f} m "
          f"(stage 1 {np.median(np.linalg.norm(goal[~syn, :2], axis=1)):.1f} m)", flush=True)
    # tokens without a goal (log ends within 4 s) fall back to the null goal at test time
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, token=np.asarray(tokens), goal=goal, synthetic=syn)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
