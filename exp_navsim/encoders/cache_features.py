"""Encode the history images of a scene index once and store compact feature grids.

    python -m exp_navsim.encoders.cache_features --encoder dinov2 --split navtrain
    python -m exp_navsim.encoders.cache_features --encoder dinov2 --split navhard

Every feature ends up on the same interface, so encoders are compared through
the same probe input:
    (N_items, C, 9, 16) float16    an 18 x 32 grid average-pooled 2 x 2
with C = pca_dim (64) for wide features (DINOv2 768, V-JEPA 2 1024, world model
768) and the native 16 channels for the tokenizer latents. PCA components are
sorted, so the first k channels are the k-dim PCA and smaller widths need no
second cache.

Items are frames (frame_paths of the index) for per-frame encoders, and scenes for
clip encoders (V-JEPA 2, world model), whose features depend on the whole history.

Frames are read from the local 288 x 512 copies of data/resize_frames.py.

Pass 1 fits the projection on a random subset of training items: the PCA basis and
one global scale that brings the features to unit variance. navhard reuses the
navtrain projection (--fit-from). Pass 2 encodes everything and writes the arrays.

Output: data/features/<split>/<name>.npy, and <name>.pt with the projection.
"""

import argparse
import gc
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from exp_navsim.data.resize_frames import frame_file
from exp_navsim.encoders.frame_encoders import FRAME_ENCODERS, build_encoder

SIZE = (288, 512)
POOL = 2


def load_image(path):
    """A frame of data/frames (already 288 x 512) -> (3, H, W) in [-1, 1].

    A test-split frame that was missing from the download has no file; it gets a
    blank image here and is never read, because its scene points to a neighbour.
    """
    if not path.exists():
        return torch.zeros(3, *SIZE)
    img = np.asarray(Image.open(path).convert("RGB"))
    assert img.shape[:2] == SIZE, f"{path} is {img.shape[:2]}, run exp_navsim.data.resize_frames"
    return torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 127.5 - 1


class Items(Dataset):
    """Frames (per-frame encoders) or 4-frame clips (clip encoders) of an index."""

    def __init__(self, index, clips, ids=None):
        self.split = index["split"]
        self.num_frames = len(index["frame_paths"])
        self.frame_ids = index["frame_ids"]
        self.clips = clips
        n = len(self.frame_ids) if clips else self.num_frames
        self.ids = np.arange(n) if ids is None else ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        i = self.ids[i]
        if self.clips:
            return torch.stack([load_image(frame_file(self.split, f)) for f in self.frame_ids[i]])
        return load_image(frame_file(self.split, i))


def train_items(index, clips, val_fraction):
    """Items that belong to training logs only (the projection must not see val)."""
    from exp_navsim.data.scenes import log_in_split
    train = np.array([log_in_split(log, "train", val_fraction) for log in index["log"]])
    if clips:
        return np.flatnonzero(train)
    return np.unique(index["frame_ids"][train].ravel())


def pooled(feats):
    return {k: F.avg_pool2d(v, POOL) for k, v in feats.items()}


@torch.no_grad()
def fit_projection(encoder, loader, pca_dim, max_vectors=400_000):
    """Per feature: mean, PCA basis (C, k) or None, and a scalar scale."""
    chunks = {}
    for x in tqdm(loader, desc="fit"):
        for k, v in pooled(encoder(x.cuda())).items():
            chunks.setdefault(k, []).append(v.permute(0, 2, 3, 1).flatten(0, 2).cpu())
    proj = {}
    for k, vs in chunks.items():
        v = torch.cat(vs)
        v = v[torch.randperm(len(v))[:max_vectors]].cuda().double()
        mean = v.mean(0)
        v = v - mean
        basis = None
        if v.shape[1] > pca_dim:
            # Eigenvectors of the covariance, largest variance first.
            evals, evecs = torch.linalg.eigh(v.T @ v / len(v))
            order = evals.argsort(descending=True)[:pca_dim]
            basis = evecs[:, order]
            explained = (evals[order].sum() / evals.sum()).item()
            v = v @ basis
            print(f"[{k}] PCA {v.shape[1]}/{len(mean)} channels, explained variance {explained:.3f}")
        scale = v.std().item()
        proj[k] = {"mean": mean.float().cpu(), "basis": None if basis is None else basis.float().cpu(),
                   "scale": scale}
    return proj


def project(v, p):
    """(N, C, h, w) -> (N, k, h, w) with projection p."""
    x = v.permute(0, 2, 3, 1) - p["mean"].to(v.device)
    if p["basis"] is not None:
        x = x @ p["basis"].to(v.device)
    return (x / p["scale"]).permute(0, 3, 1, 2)


class NpyAppender:
    """Writes an .npy of known shape batch by batch, without a memmap.

    The pages of a writable memmap stay dirty in this process until the end and
    count as its memory; twelve world-model features are 22 GB.
    """

    def __init__(self, path, shape, dtype=np.float16):
        self.file = open(path, "wb")
        np.lib.format.write_array_header_1_0(
            self.file, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False,
                        "shape": tuple(shape)})
        self.shape, self.rows = tuple(shape), 0

    def append(self, a):
        self.file.write(np.ascontiguousarray(a).tobytes())
        self.rows += len(a)

    def close(self):
        assert self.rows == self.shape[0], f"wrote {self.rows} rows, header says {self.shape[0]}"
        self.file.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["orbis", "dinov2", "vjepa2", "wm"])
    ap.add_argument("--split", default="navtrain")
    ap.add_argument("--index-dir", default="data/navsim_index")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--fit-from", default="navtrain", help="split whose projection is reused")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--fit-items", type=int, default=3000)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reuse-fit", action="store_true", help="keep an existing projection of this split")
    args = ap.parse_args()

    with open(Path(args.index_dir) / f"{args.split}.pkl", "rb") as f:
        index = pickle.load(f)
    clips = args.encoder not in FRAME_ENCODERS
    encoder = build_encoder(args.encoder)
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    loader_kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True)

    fit_dir = Path(args.out) / args.fit_from
    fitted = all((fit_dir / f"{k}.pt").exists() for k in encoder.names)
    if args.split == args.fit_from and not (fitted and args.reuse_fit):
        rng = np.random.default_rng(0)
        ids = rng.permutation(train_items(index, clips, args.val_fraction))[:args.fit_items]
        proj = fit_projection(encoder, DataLoader(Items(index, clips, ids), **loader_kw), args.pca_dim)
        for k, p in proj.items():
            torch.save(p, fit_dir / f"{k}.pt")
        del proj
        gc.collect()                # before the main loader forks its workers
    proj = {k: torch.load(fit_dir / f"{k}.pt") for k in encoder.names}

    items = Items(index, clips)
    writers = {}
    for x in tqdm(DataLoader(items, **loader_kw), desc=f"{args.encoder} {args.split}"):
        for k, v in pooled(encoder(x.cuda())).items():
            v = project(v, proj[k]).half().cpu().numpy()
            if k not in writers:
                writers[k] = NpyAppender(out / f"{k}.npy", (len(items), *v.shape[1:]))
            writers[k].append(v)
    for k, w in writers.items():
        w.close()
        meta = {"items": "scenes" if clips else "frames", "shape": list(w.shape), "encoder": args.encoder,
                "pca": proj[k]["basis"] is not None, "fit_from": args.fit_from}
        (out / f"{k}.json").write_text(json.dumps(meta))
        print(f"{k}: {w.shape} -> {out / k}.npy")


if __name__ == "__main__":
    main()
