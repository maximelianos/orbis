import os

import torch


def _latest_run_checkpoint(logdir):
    if not os.path.exists(logdir):
        raise ValueError(f"Logdir {logdir} does not exist")

    subdirs = [
        d
        for d in os.listdir(logdir)
        if os.path.isdir(os.path.join(logdir, d))
    ]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {logdir}")

    subdirs.sort()
    ckpt_path = os.path.join(logdir, subdirs[-1], "checkpoints", "last.ckpt")
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint not found at {ckpt_path}")
    print(f"Using last checkpoint from {ckpt_path}")
    return ckpt_path


def resolve_checkpoint(ckpt=None, last_ckpt=False, logdir="logs_nuplan"):
    # ckpt: "logs_navsim", "logs_navsim/<run>/", or the checkpoints/ subdir
    if last_ckpt or ckpt is None:
        return _latest_run_checkpoint(logdir)

    if os.path.isdir(ckpt):
        # Accept a run dir (logs_navsim/<run>/) or its checkpoints/ subdir.
        for candidate in (os.path.join(ckpt, "last.ckpt"),
                          os.path.join(ckpt, "checkpoints", "last.ckpt")):
            if os.path.exists(candidate):
                print(f"Using last checkpoint from {candidate}")
                return candidate
        # Otherwise treat ckpt as a logdir (logs_navsim/) holding run subdirs
        # and use the latest one's checkpoints/last.ckpt.
        return _latest_run_checkpoint(ckpt)

    if not os.path.exists(ckpt):
        raise ValueError(f"Checkpoint not found at {ckpt}")

    return ckpt


def load_model_checkpoint(model, ckpt_path, map_location="cpu", strict=False):
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    model.eval()
    print("Model loaded successfully")
    return model
