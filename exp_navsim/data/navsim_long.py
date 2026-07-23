"""Long (whole-episode) dataloader for NAVSIM navhard_two_stage.

    python -m exp_navsim.data.navsim_long   # quick smoke check of one episode

One index == one whole episode. See navsim_base.NavsimLongBase for the return
dict contract and the shared machinery; this file only adds the navhard-specific
episode enumeration (one meta pickle -> one episode = its longest scene).

Config options mirror configs/long_data.yaml: omit num_frames, use
stored_data_frame_rate + frame_rate (subsampling), omit aug (fixed
resize+center-crop). A train/validation split is available via `split` +
`val_fraction` (deterministic, hash-based in/out check on the scene token).

------------------------------------------------------------------------------
DATA-FORMAT ASSUMPTION (see readme.md). Each openscene_meta_datas/<token>.pkl is
a pickled list[dict] of frames. Per-frame keys used (from
navsim.common.dataclasses): ego2global_translation, ego2global_rotation,
ego_dynamic_state, cams {CAM_F0: {"data_path": <rel>}, ...}, scene_token,
frame_idx. Camera data_path is relative to navhard_two_stage/sensor_blobs.
------------------------------------------------------------------------------
"""

import pickle
from pathlib import Path

import numpy as np

# Re-export the shared helpers so existing imports keep working
# (latent_dataset.py does `from exp_navsim.data.navsim_long import ...`).
from exp_navsim.data.navsim_base import (  # noqa: F401
    NavsimLongBase,
    SURROUND_CAMERAS,
    FRONT_CAMERA,
    quat_to_yaw,
    poses_to_local_traj,
    traj_to_velocity,
    build_image_transform,
    in_split,
)


class NavsimLongDataset(NavsimLongBase):
    """One whole navhard episode per index (episode = longest scene in a pickle)."""

    def __init__(
        self,
        data_root,                     # navsim/download/navhard_two_stage
        size=256,
        stored_data_frame_rate=2,
        frame_rate=2,
        front_camera=FRONT_CAMERA,
        load_surround=True,
        split="all",                   # "all" | "train" | "val"
        val_fraction=0.1,
        max_episodes=None,             # optional cap (debugging)
    ):
        data_root = Path(data_root)
        super().__init__(
            sensor_root=data_root / "sensor_blobs",
            size=size,
            stored_data_frame_rate=stored_data_frame_rate,
            frame_rate=frame_rate,
            front_camera=front_camera,
            load_surround=load_surround,
        )
        self.meta_dir = data_root / "openscene_meta_datas"

        # One episode per meta pickle. Sorted for a deterministic, cache-friendly
        # order (latents are reused between runs by episode index). Split by the
        # file stem (== scene token) so train/val never overlap.
        paths = sorted(self.meta_dir.glob("*.pkl"))
        self.episodes = [p for p in paths if in_split(p.stem, split, val_fraction)]
        if max_episodes is not None:
            self.episodes = self.episodes[:max_episodes]
        assert len(self.episodes) > 0, f"No episodes under {self.meta_dir} for split={split}"

    # backward-compatible alias (older scripts referenced `episode_paths`)
    @property
    def episode_paths(self):
        return self.episodes

    def episode_token(self, idx):
        return self.episodes[idx].stem      # file stem == scene token

    def _episode_frames(self, path):
        """Frames of the longest scene in `path`, sorted by frame_idx."""
        with open(path, "rb") as f:
            frame_list = pickle.load(f)
        # A meta pickle may contain several scene_tokens; keep the longest scene
        # so poses form a single continuous episode.
        groups = {}
        for fr in frame_list:
            groups.setdefault(fr.get("scene_token", "single"), []).append(fr)
        frames = max(groups.values(), key=len)
        frames.sort(key=lambda fr: fr.get("frame_idx", 0))
        return frames, {"path": str(path), "token": path.stem}


def _main():
    """Smoke test: build the dataset from config and inspect one episode."""
    from omegaconf import OmegaConf
    from exp_navsim.data.navsim_base import NavsimLongBase

    cfg = OmegaConf.load("exp_navsim/config.yaml")
    ds = NavsimLongBase.from_config(cfg)
    print(f"episodes: {len(ds)}")
    s = ds[0]
    for k in ("images", "trajectory", "velocity"):
        print(f"  {k}: {np.asarray(s[k]).shape}")
    print(f"  metadata: {s['metadata']}")


if __name__ == "__main__":
    _main()
