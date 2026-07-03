import os

import torch


def resolve_checkpoint(ckpt=None, last_ckpt=False, logdir="logs_nuplan"):
    if last_ckpt:
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

    if ckpt is None:
        raise ValueError("Either ckpt must be provided or last_ckpt must be true")

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
