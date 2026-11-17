"""Encode the history images of a scene index once and store compact feature grids.

    python -m exp_navsim.encoders.cache_features --encoder dinov2 --split navtrain
    python -m exp_navsim.encoders.cache_features --encoder dinov2 --split navhard
    python -m exp_navsim.encoders.cache_features --encoder wm --features wm11_ctx --camera CAM_L0 --split navtrain

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

Bottleneck variants: --pca-dim k (default 64) and --pool p (default 2; 1 keeps the 18 x 32
grid). A non-default setting adds a suffix to the output and projection names, e.g.
wm11_ctx_p256 (k 256) or wm11_ctx_g18 (p 1), so every variant gets its own projection fit on
--fit-from and nothing of the default cache is overwritten:
    ... --encoder wm --features wm11_ctx --split navtrain --pca-dim 256
    ... --encoder wm --features wm11_ctx --split navhard --pca-dim 256    # reuses navtrain/wm11_ctx_p256.pt

Other cameras (--camera CAM_L0 etc., clip encoders only): the history images of that
camera are found through the logs (camera_paths.py) and decoded from the raw 1920 x 1080
JPEGs with the policy of resize_frames.py (draft decode at half scale, bilinear resize to
512 x 288, no crop). The projection of the front camera is reused as it is (never refit),
so the channels of all cameras are comparable. Output names get the camera suffix,
e.g. wm11_ctx_L0.npy. A scene without any image of that camera on disk gets NaN features.

--features wm11_ctx keeps only those outputs; for the world model it also stops the
backbone after the last block needed (block 11: 12 of 24 blocks).
"""

import argparse
import gc
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from exp_navsim.data.resize_frames import SIZE as PIL_SIZE, frame_file
from exp_navsim.encoders.camera_paths import cam_name, cam_short, history_paths
from exp_navsim.encoders.frame_encoders import FRAME_ENCODERS, build_encoder

SIZE = (288, 512)
PATCH = 16


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


def load_raw(path):
    """A raw sensor JPEG -> (3, 288, 512) in [-1, 1], resized like resize_frames._resize.
    None (no image of this camera) -> a blank image; its scene is written as NaN."""
    if path is None:
        return torch.zeros(3, *SIZE)
    img = Image.open(path)
    img.draft("RGB", (PIL_SIZE[0] * 2, PIL_SIZE[1] * 2))
    img = img.convert("RGB").resize(PIL_SIZE, Image.BILINEAR, reducing_gap=2.0)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 127.5 - 1


class CameraClips(Dataset):
    """4-frame clips of a non-front camera, from history_paths (one list of 4 per scene)."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.stack([load_raw(p) for p in self.paths[i]])


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


def variant_suffix(pca_dim, pool):
    """"" for the default bottleneck (PCA 64, pool 2), else e.g. "_p256" or "_g18"."""
    s = "" if pca_dim == 64 else f"_p{pca_dim}"
    if pool != 2:
        s += f"_g{SIZE[0] // PATCH // pool}"
    return s


@torch.no_grad()
def fit_projection(encoder, loader, pca_dim, pool=2, max_vectors=400_000):
    """Per feature: mean, PCA basis (C, k) or None, and a scalar scale."""
    chunks = {}
    for x in tqdm(loader, desc="fit"):
        for k, v in encoder(x.cuda()).items():
            v = F.avg_pool2d(v, pool).permute(0, 2, 3, 1).flatten(0, 2)
            if pool < 2:
                # keep as many vectors per item as the pooled grid has (host memory of the fit)
                v = v[torch.randperm(len(v), device=v.device)[:len(v) * pool * pool // 4]]
            chunks.setdefault(k, []).append(v.cpu())
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
    ap.add_argument("--pool", type=int, default=2, choices=[1, 2],
                    help="average pooling of the 18 x 32 token grid (1 keeps it)")
    ap.add_argument("--fit-items", type=int, default=3000)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reuse-fit", action="store_true", help="keep an existing projection of this split")
    ap.add_argument("--camera", default="CAM_F0", help="CAM_F0 | CAM_L0 | ... (non-front: clip encoders)")
    ap.add_argument("--features", default=None,
                    help="comma-separated outputs to keep, e.g. wm11_ctx (default: all of the encoder)")
    ap.add_argument("--proj-dir", default=None,
                    help="directory of the <name>.pt projections (default: <out>/<fit-from>)")
    ap.add_argument("--max-scenes", type=int, default=None, help="first n items only (smoke test)")
    args = ap.parse_args()

    with open(Path(args.index_dir) / f"{args.split}.pkl", "rb") as f:
        index = pickle.load(f)
    clips = args.encoder not in FRAME_ENCODERS
    camera = cam_name(args.camera)
    front = camera == "CAM_F0"
    assert front or clips, "other cameras are only implemented for clip encoders"
    # paths are looked up (worker pool) before CUDA is touched
    paths = None
    if not front:
        rows = np.arange(len(index["token"]))[:args.max_scenes]
        paths, missing = history_paths(index, camera, rows)
        none = sum(p[0] is None for p in paths)
        print(f"[{args.split} {camera}] {len(paths)} scenes, missing history images {missing.sum()} / "
              f"{4 * len(paths)} (filled from the nearest frame), scenes without any image {none} (-> NaN)",
              flush=True)
    names = args.features.split(",") if args.features else None
    kw = {}
    if names and args.encoder == "wm":
        kw["blocks"] = tuple(sorted({int(re.match(r"wm(\d+)", n).group(1)) for n in names}))
    encoder = build_encoder(args.encoder, **kw)
    names = tuple(names or encoder.names)
    variant = variant_suffix(args.pca_dim, args.pool)
    suffix = variant + ("" if front else f"_{cam_short(camera)}")
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    loader_kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                     persistent_workers=False, prefetch_factor=4 if args.workers else None)

    fit_dir = Path(args.proj_dir) if args.proj_dir else Path(args.out) / args.fit_from
    fitted = all((fit_dir / f"{k}{variant}.pt").exists() for k in names)
    assert front or fitted, f"camera {camera} reuses the front projections, missing in {fit_dir}"
    if front and args.split == args.fit_from and not (fitted and args.reuse_fit):
        rng = np.random.default_rng(0)
        ids = rng.permutation(train_items(index, clips, args.val_fraction))[:args.fit_items]
        proj = fit_projection(encoder, DataLoader(Items(index, clips, ids), **loader_kw), args.pca_dim, args.pool)
        for k, p in proj.items():
            torch.save(p, fit_dir / f"{k}{variant}.pt")
        del proj
        gc.collect()                # before the main loader forks its workers
    proj = {k: torch.load(fit_dir / f"{k}{variant}.pt") for k in names}
    for k in names:
        p = proj[k]
        k_out = len(p["mean"]) if p["basis"] is None else p["basis"].shape[1]
        assert p["basis"] is None or k_out == args.pca_dim, \
            f"{fit_dir / k}{variant}.pt has {k_out} channels, --pca-dim is {args.pca_dim}"

    if front:
        n = len(index["frame_ids"]) if clips else len(index["frame_paths"])
        items = Items(index, clips, np.arange(n)[:args.max_scenes])
    else:
        items = CameraClips(paths)
    blank = None if front else torch.tensor([p[0] is None for p in paths])
    writers = {}
    t_start, t_warm, done = time.time(), None, 0
    for b, x in enumerate(tqdm(DataLoader(items, **loader_kw), desc=f"{args.encoder} {args.split} {camera}")):
        feats = encoder(x.cuda())
        for k in names:
            v = project(F.avg_pool2d(feats[k], args.pool), proj[k])
            if blank is not None:
                nan = blank[done:done + len(v)].to(v.device)
                v = torch.where(nan[:, None, None, None], torch.full_like(v, float("nan")), v)
            v = v.half().cpu().numpy()
            if k not in writers:
                writers[k] = NpyAppender(out / f"{k}{suffix}.npy", (len(items), *v.shape[1:]))
            writers[k].append(v)
        done += len(x)
        if b == 1:
            t_warm, n_warm = time.time(), done
    t_end = time.time()
    rate = (done - n_warm) / (t_end - t_warm) if t_warm and done > n_warm else done / (t_end - t_start)
    full = len(index["token"]) if clips else len(index["frame_paths"])
    print(f"[speed] {done} items in {t_end - t_start:.0f}s, steady {rate:.1f} items/s "
          f"-> {full} items of {args.split} in about {full / rate / 3600:.2f} h", flush=True)
    for k, w in writers.items():
        w.close()
        meta = {"items": "scenes" if clips else "frames", "shape": list(w.shape), "encoder": args.encoder,
                "pca": proj[k]["basis"] is not None, "fit_from": args.fit_from, "camera": camera}
        if variant:
            meta.update(pca_dim=args.pca_dim, pool=args.pool)
        if blank is not None:
            meta["nan_scenes"] = int(blank.sum())
        (out / f"{k}{suffix}.json").write_text(json.dumps(meta))
        print(f"{k}{suffix}: {w.shape} -> {out / k}{suffix}.npy")


if __name__ == "__main__":
    main()
