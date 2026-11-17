"""LoRA post-training of the ORBIS world-model encoder behind the v22 planner.

The cached wm11_ctx feature (encoders/cache_features.py) is reproduced online and made
trainable: the tokenizer encodes the 4 history frames, the world-model backbone runs over
[4 observed frames, noise next frame] at t = 1 up to block 11, the last-observed-frame slot
is LayerNormed, avg-pooled 2 x 2 and projected with the PCA of wm11_ctx.pt. LoRA adapters
(A random, B zero, so step 0 is the frozen encoder) sit on the attention qkv / proj linears
(and with mlp=True the MLP fc1 / fc2) of the last K blocks up to 11. Blocks before them and
the tokenizer run frozen under no_grad in fp16 autocast, as in the cache; the LoRA blocks run
in bf16 autocast with gradient checkpointing.

Noise of the next-frame slot: the cache drew randn(32, ...) from a generator seeded 0 for
every batch of 32 scenes, so scene `row` of a split got slot row % 32 of that draw. The
encoder keeps that bank and indexes it by row % 32, so its features equal the cache.

    enc = OnlineWM(k=6, rank=16)
    feats = enc(clips_uint8, missing, slot)      # (B, 1, 64, 9, 16) float32

CLI:
    python -m exp_navsim.v22.lora check --split navtrain --n 128       # step-0 equivalence
    python -m exp_navsim.v22.lora cache --run logs_exp/<run> --split navhard
"""

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

from exp_navsim.data.resize_frames import frame_file

SIZE = (288, 512)
BLOCK = 11                      # wm11_ctx
NOISE_BANK = 32                 # cache_features default --batch
FEATURE = "wm11_ctx"


class LoRALinear(nn.Module):
    """base(x) + scale * B(A(x)); A kaiming-uniform, B zero (exact identity at init)."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def lora_targets(block, mlp):
    """Names of the linears of one SwinSTDiT block that get an adapter."""
    names = ["space_attn.attn.qkv", "space_attn.attn.proj", "time_attn.qkv", "time_attn.proj"]
    if mlp:
        names += ["space_mlp.fc1", "space_mlp.fc2", "time_mlp.fc1", "time_mlp.fc2"]
    return names


def set_submodule(root, name, module):
    parent, _, leaf = name.rpartition(".")
    setattr(root.get_submodule(parent), leaf, module)


class OnlineWM(nn.Module):
    """Tokenizer + world-model blocks 0..11 (last k with LoRA) + LayerNorm, pool, PCA."""

    def __init__(self, k=6, rank=16, alpha=None, mlp=False, train_proj=False,
                 proj_path="data/features/navtrain/wm11_ctx.pt", device="cuda", compile=True,
                 exp_dir="logs_wm/orbis_288x512", tokenizer_dir="logs_tk/tokenizer_288x512"):
        super().__init__()
        from exp_navsim.encoders.orbis_io import build_encoder
        from exp_navsim.encoders.wm_runner import WorldModelRunner
        assert 0 <= k <= BLOCK + 1
        self.k, self.rank, self.mlp = k, rank, mlp
        self.alpha = float(alpha if alpha is not None else rank)
        self.tokenizer = build_encoder(exp_dir=tokenizer_dir, device=device)     # plain object, frozen
        self.runner = WorldModelRunner(exp_dir, "checkpoints/last.ckpt", "config.yaml", True, device,
                                       blocks=(BLOCK,))
        self.net = self.runner.net                       # blocks 0..11, eval mode, frozen
        self.first = BLOCK + 1 - k                       # first LoRA block
        for i in range(self.first, BLOCK + 1):
            for name in lora_targets(self.net.blocks[i], mlp):
                set_submodule(self.net.blocks[i], name,
                              LoRALinear(self.net.blocks[i].get_submodule(name), rank, self.alpha))
        p = torch.load(proj_path)
        self.train_proj = train_proj
        for key in ("mean", "basis"):
            t = p[key].float()
            if train_proj:
                setattr(self, key, nn.Parameter(t))
            else:
                self.register_buffer(key, t)
        scale = torch.tensor(float(p["scale"]))
        if train_proj:
            self.log_scale = nn.Parameter(scale.log())
        else:
            self.register_buffer("log_scale", scale.log())
        self.to(device)
        self.device = torch.device(device)
        # speed (2.2x on an H100): the cuDNN SDPA kernel is slow on the 5-frame time attention, and
        # torch.compile fuses the many small elementwise ops of the blocks. Compiled callables are kept
        # outside the module tree, so state_dict names stay those of the plain blocks.
        torch.backends.cuda.enable_cudnn_sdp(False)
        # 12 block instances share code objects: one cache entry per block x grad mode x batch size
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 1024)
        self.run_blocks = [torch.compile(b) if compile else b for b in self.net.blocks]
        tok = self.tokenizer.tokenizer.encode
        self.tok_encode = torch.compile(tok) if compile else tok
        gen = torch.Generator(device=self.device).manual_seed(0)
        C = self.net.x_embedder.proj.in_channels                         # rec + sem latents, 32
        H, W = self.net.x_embedder.grid_size
        self.register_buffer("noise", torch.randn(NOISE_BANK, 1, C, H, W, device=self.device, generator=gen),
                             persistent=False)
        n = sum(p.numel() for p in self.trainable())
        print(f"[lora] blocks {self.first}..{BLOCK} rank {rank} alpha {self.alpha} mlp {mlp} "
              f"projection {'trainable' if train_proj else 'frozen'}: {n / 1e6:.2f} M trainable params", flush=True)

    def train(self, mode=True):
        """The backbone stays in eval mode (its preprocess_inputs drops / noises the context in training)."""
        super().train(mode)
        self.net.eval()
        return self

    def trainable(self):
        out = [p for n, p in self.named_parameters() if n.endswith(".A") or n.endswith(".B")]
        if self.train_proj:
            out += [self.mean, self.basis, self.log_scale]
        return out

    def lora_state(self):
        """Trainable tensors only (the frozen ~350 M stay out of checkpoints)."""
        keep = {n for n, p in self.named_parameters() if n.endswith(".A") or n.endswith(".B")}
        if self.train_proj:
            keep |= {"mean", "basis", "log_scale"}
        return {n: t.detach().cpu() for n, t in self.state_dict().items() if n in keep}

    def config(self):
        return {"k": self.k, "rank": self.rank, "alpha": self.alpha, "mlp": self.mlp,
                "train_proj": self.train_proj}

    def forward(self, clips, missing, slot):
        """clips (B, 4, 3, 288, 512) uint8, missing (B, 4) bool, slot (B,) noise slot
        -> (B, 1, 64, 9, 16) float32."""
        B, T = clips.shape[:2]
        x = clips.to(self.device, non_blocking=True).float() / 127.5 - 1
        x = torch.where(missing.to(self.device)[:, :, None, None, None], torch.zeros_like(x), x)
        net = self.net
        with torch.no_grad():
            with torch.autocast("cuda", torch.float16):          # as loaders_for_projects Encoder.encode
                q_rec, q_sem = self.tok_encode(x.flatten(0, 1))["continuous"]
            q_rec, q_sem = (q.float().clone().unflatten(0, (B, T)) for q in (q_rec, q_sem))
            ctx = self.runner.pack(q_rec, q_sem)
            t = torch.full((B,), self.runner.timescale, device=self.device)
            noise = self.noise[slot.to(self.device)]
            with torch.autocast("cuda", torch.float16):
                h = net.preprocess_inputs(noise, ctx, t, torch.full((B,), 2.0, device=self.device))
                c = net.get_condition_embeddings(t)
                for i in range(self.first):
                    h = self.run_blocks[i](h, c)
            h, c = h.float(), c.float()
        grad = torch.is_grad_enabled()
        with torch.autocast("cuda", torch.bfloat16):
            for i in range(self.first, BLOCK + 1):
                if grad:
                    h = checkpoint(self.run_blocks[i], h, c, use_reentrant=False)
                else:
                    h = self.run_blocks[i](h, c)
        f = F.layer_norm(h.float(), h.shape[-1:])[:, -2]                        # (B, HW, D) last observed slot
        H, W = net.x_embedder.grid_size
        f = F.avg_pool2d(f.transpose(1, 2).unflatten(-1, (H, W)), 2)            # (B, D, 9, 16)
        f = (f.permute(0, 2, 3, 1) - self.mean) @ self.basis / self.log_scale.exp()
        return f.permute(0, 3, 1, 2)[:, None]


def load_frame(path):
    """A 288 x 512 frame -> (uint8 (3, H, W), missing). Same pixels as cache_features.load_image."""
    if not path.exists():
        return torch.zeros(3, *SIZE, dtype=torch.uint8), True
    img = np.asarray(Image.open(path).convert("RGB"))
    assert img.shape[:2] == SIZE, f"{path} is {img.shape[:2]}"
    return torch.from_numpy(img.copy()).permute(2, 0, 1), False


class Clips(Dataset):
    """Item i of a table of scenes: split name, 4 history frame ids, noise slot.

    Item -> (position, uint8 clip (4, 3, 288, 512), missing (4,), noise slot); a batch_sampler
    loader collates them to (B,), (B, 4, 3, 288, 512), (B, 4), (B,)."""

    def __init__(self, splits, split_of, frame_ids, slot):
        self.splits, self.split_of, self.frame_ids, self.slot = splits, split_of, frame_ids, slot

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, i):
        frames = [load_frame(frame_file(self.splits[self.split_of[i]], int(f))) for f in self.frame_ids[i]]
        return (i, torch.stack([f[0] for f in frames]), torch.tensor([f[1] for f in frames]),
                int(self.slot[i]))


def table(index, rows, split_id=0):
    """(split_of, frame_ids (N, 4), slot) of index rows."""
    rows = np.asarray(rows)
    return (np.full(len(rows), split_id), np.asarray(index["frame_ids"])[rows], rows % NOISE_BANK)


def loader(ds, batches, workers):
    """batches: an iterable of lists of positions; default collate stacks the items."""
    return DataLoader(ds, batch_sampler=batches, num_workers=workers, pin_memory=True,
                      prefetch_factor=4 if workers else None, persistent_workers=False)


@torch.no_grad()
def encode_rows(enc, index, rows, batch=64, workers=8, desc=""):
    """Features of index rows (in order) -> float16 numpy (N, 64, 9, 16)."""
    split_of, fids, slot = table(index, rows)
    ds = Clips([index["split"]], split_of, fids, slot)
    order = [list(range(i, min(i + batch, len(rows)))) for i in range(0, len(rows), batch)]
    out = []
    t0 = time.time()
    for _, clips, missing, s in loader(ds, order, workers):
        out.append(enc(clips, missing, s)[:, 0].half().cpu().numpy())
    print(f"[lora] encoded {len(rows)} {desc} scenes in {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(out)


def write_cache(enc, split, name, run_name="", batch=64, workers=8, out="data/features"):
    """Encode every scene of an index with the (tuned) encoder -> <out>/<split>/<name>.npy + .json."""
    with open(f"data/navsim_index/{split}.pkl", "rb") as f:
        index = pickle.load(f)
    x = encode_rows(enc, index, np.arange(len(index["token"])), batch, workers, split)
    path = Path(out) / split / f"{name}.npy"
    np.save(path, x)
    meta = {"items": "scenes", "shape": list(x.shape), "encoder": "wm", "pca": True, "fit_from": "navtrain",
            "camera": "CAM_F0", "lora_run": run_name, "lora": enc.config()}
    path.with_suffix(".json").write_text(json.dumps(meta))
    print(f"[lora] {name}: {x.shape} -> {path}", flush=True)
    return x


def from_run(run, device="cuda"):
    """The tuned encoder of a run dir (lora.pt)."""
    ck = torch.load(Path(run) / "lora.pt", map_location="cpu")
    enc = OnlineWM(device=device, **ck["config"])
    missing, unexpected = enc.load_state_dict(ck["lora"], strict=False)
    assert not unexpected and set(ck["lora"]) <= set(enc.state_dict()), unexpected
    enc.eval()
    return enc, ck


def check(args):
    """Step-0 equivalence: online features of the zero-LoRA encoder vs the cached rows."""
    with open(f"data/navsim_index/{args.split}.pkl", "rb") as f:
        index = pickle.load(f)
    n = len(index["token"])
    rows = np.sort(np.random.default_rng(0).choice(n - n % NOISE_BANK, args.n, replace=False))
    cache = np.load(f"data/features/{args.split}/{FEATURE}.npy", mmap_mode="r")[rows].astype(np.float32)
    enc = OnlineWM(k=args.k, rank=args.rank, mlp=args.mlp, compile=not args.no_compile)
    if args.random_b:              # sanity: a nonzero adapter must change the features
        for name, p in enc.named_parameters():
            if name.endswith(".B"):
                nn.init.normal_(p, std=args.random_b)
    on = encode_rows(enc, index, rows, desc=args.split).astype(np.float32)
    d = on - cache
    cos = (on * cache).reshape(len(rows), -1).sum(1) / (np.linalg.norm(on.reshape(len(rows), -1), axis=1)
                                                        * np.linalg.norm(cache.reshape(len(rows), -1), axis=1))
    print(f"[check] {args.split} {len(rows)} scenes, k {args.k}: |cache| mean {np.abs(cache).mean():.4f}, "
          f"|online - cache| mean {np.abs(d).mean():.5f} max {np.abs(d).max():.4f}, "
          f"relative L2 {np.linalg.norm(d) / np.linalg.norm(cache):.5f}, cosine min {cos.min():.6f} mean {cos.mean():.6f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check", "cache"))
    ap.add_argument("--split", default="navtrain")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--random-b", type=float, default=0.0, help="check: B ~ N(0, std) instead of 0")
    ap.add_argument("--run", default=None, help="cache: run dir with lora.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.cmd == "check":
        check(args)
    else:
        enc, ck = from_run(args.run)
        write_cache(enc, args.split, ck["feature"], Path(args.run).name, args.batch, args.workers)


if __name__ == "__main__":
    main()
