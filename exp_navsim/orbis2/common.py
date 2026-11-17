"""Scene clips, pooling, PCA projection and the .npy writer shared by cache_l2 and dream.

Same conventions as exp_navsim/encoders/cache_features.py (copied, not imported,
because that file is being edited elsewhere): frames from data/frames (288 x 512),
oldest first, in [-1, 1]; features avg-pooled 2 x 2 to 9 x 16, PCA'd to 64 channels
and divided by one global scale.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from exp_navsim.data.resize_frames import frame_file

SIZE = (288, 512)
GRID = (18, 32)
POOL = 2


def load_index(split, index_dir="data/navsim_index"):
    with open(Path(index_dir) / f"{split}.pkl", "rb") as f:
        return pickle.load(f)


def load_image(path):
    """A 288 x 512 frame -> (3, H, W) in [-1, 1]; a missing test frame -> zeros."""
    if not path.exists():
        return torch.zeros(3, *SIZE)
    img = np.asarray(Image.open(path).convert("RGB"))
    assert img.shape[:2] == SIZE, f"{path} is {img.shape[:2]}"
    return torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 127.5 - 1


class Clips(Dataset):
    """The 4 history frames of index rows `ids`: (4, 3, 288, 512), plus the row."""

    def __init__(self, index, ids=None):
        self.split = index["split"]
        self.frame_ids = index["frame_ids"]
        self.ids = np.arange(len(self.frame_ids)) if ids is None else np.asarray(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        row = int(self.ids[i])
        clip = torch.stack([load_image(frame_file(self.split, f)) for f in self.frame_ids[row]])
        return clip, row


def tokens_to_grid(tok):
    """(B, 18*32, D) block tokens -> pooled (B, D, 9, 16) float32."""
    B, N, D = tok.shape
    x = tok.float().transpose(1, 2).reshape(B, D, *GRID)
    return F.avg_pool2d(x, POOL)


class CovarianceFit:
    """Streaming mean / covariance of pooled token vectors (float64 on the GPU)."""

    def __init__(self, dim, device):
        self.n = 0
        self.s = torch.zeros(dim, dtype=torch.float64, device=device)
        self.ss = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    def add(self, grid):
        v = grid.permute(0, 2, 3, 1).reshape(-1, grid.shape[1]).double()
        self.n += len(v)
        self.s += v.sum(0)
        self.ss += v.T @ v

    def projection(self, k):
        """{mean, basis (D, k), scale} with the cache_features.py conventions."""
        mean = self.s / self.n
        cov = self.ss / self.n - torch.outer(mean, mean)
        evals, evecs = torch.linalg.eigh(cov)
        order = evals.argsort(descending=True)[:k]
        basis = evecs[:, order]
        # std of all projected entries (zero-mean per channel) = sqrt(mean eigenvalue)
        scale = evals[order].clamp_min(0).mean().sqrt().item()
        explained = (evals[order].sum() / evals.sum()).item()
        return {"mean": mean.float().cpu(), "basis": basis.float().cpu(), "scale": scale,
                "explained": explained, "fit_vectors": self.n}


def project(grid, p):
    """(N, D, h, w) -> (N, k, h, w)."""
    x = grid.permute(0, 2, 3, 1) - p["mean"].to(grid.device)
    x = x @ p["basis"].to(grid.device)
    return (x / p["scale"]).permute(0, 3, 1, 2)


class NpyAppender:
    """Writes an .npy of known shape batch by batch (no writable memmap)."""

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


def write_meta(path, **meta):
    Path(path).write_text(json.dumps(meta))


def scene_noise(rows, n, shape=(16, *GRID), salt=0):
    """Deterministic standard-normal draws per scene: (len(rows), n, *shape)."""
    out = []
    for r in rows:
        g = torch.Generator().manual_seed(int(r) * 1000 + salt)
        out.append(torch.randn(n, *shape, generator=g))
    return torch.stack(out)
