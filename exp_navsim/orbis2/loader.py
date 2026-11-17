"""Load the Orbis 2 L2 predictor (orbis2_code/, untouched) next to the ORBIS-1 repo.

    from exp_navsim.orbis2.loader import L2Model
    l2 = L2Model("cuda")
    lat = l2.encode(clip)                          clip (B, T, 3, 288, 512) in [-1, 1]
    v, feats = l2.velocity(seq, t, goals, blocks)  one DiT pass
    x0, feats = l2.sample_next(ctx, goals, ...)    few-step sample of the next frame
    img = l2.decode(lat)                           (B, T, 3, 288, 512) in [-1, 1]

orbis2_code has top-level packages data/, models/, modules/, networks/ and util.py,
the same names as the ORBIS-1 repo root (all namespace packages, so Python would
merge their __path__). `orbis2_imports()` swaps the ORBIS-1 modules out of
sys.modules, puts orbis2_code first on sys.path (and the repo root off it), imports,
and swaps back. The objects built inside keep their own module globals, so they
work after the swap; the Orbis 2 modules are kept aside for a later re-entry.

The L2 checkpoint (logs_orbis2/L2) is the consistency-distilled student
(cd_model_v2.ConsistencyDistillPredictorModule). Its config points the teacher at
the same 9.6 GB file; we build the module without a teacher and load only the
`ema_vit.*` (the deliverable) and `ae.*` weights with mmap, so CPU RAM stays low.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
ORBIS2 = REPO / "orbis2_code"
MODELS = REPO / "logs_orbis2"
CLASH = ("data", "models", "modules", "networks", "util", "evaluate")
_ORBIS2_MODULES = {}

FRAME_RATE = 2.0          # L2 runs at 2 Hz, like the NAVSIM cameras
NUM_GOALS = 5             # goal_offsets 1..5 of 0.1 s: the ego path of the next 0.5 s
CTX_FRAMES = 5            # L2 context length in the L1 hierarchy (L1/config*.yaml)
CD_GRIDS = {1: [0.99], 2: [0.99, 0.8333], 4: [0.99, 0.9375, 0.8333, 0.625]}  # cd_model_v2


def _clashes(name):
    return name.split(".")[0] in CLASH


@contextmanager
def orbis2_imports():
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if _clashes(k)}
    sys.modules.update(_ORBIS2_MODULES)
    old_path = list(sys.path)
    sys.path[:] = [str(ORBIS2)] + [p for p in sys.path
                                   if Path(p or ".").resolve() != REPO]
    try:
        yield
    finally:
        sys.path[:] = old_path
        for k in [k for k in sys.modules if _clashes(k)]:
            _ORBIS2_MODULES[k] = sys.modules.pop(k)
        sys.modules.update(saved)


def build_l2_module(folder=MODELS / "L2"):
    """The PredictorModule with ema_vit + tokenizer weights, on CPU, eval mode."""
    import timm
    os.environ.setdefault("ORBIS2_MODELS_DIR", str(MODELS))
    create_model = timm.create_model

    def create_model_compat(name, *args, **kwargs):
        # timm >= 1.0.2x rejects "timm/<arch>.<tag>" without an "hf-hub:" prefix; the
        # tokenizer builds its MAE encoders with pretrained=False, so the bare name is enough.
        if isinstance(name, str) and name.startswith("timm/") and not kwargs.get("pretrained", False):
            name = name[len("timm/"):]
        return create_model(name, *args, **kwargs)

    timm.create_model = create_model_compat
    # the tokenizer ckpt (4.7 GB, mostly optimizer state) is read with mmap, not into RAM
    torch_load = torch.load
    torch.load = lambda f, *a, **kw: torch_load(f, *a, **{"mmap": isinstance(f, (str, Path)), **kw})
    try:
        with orbis2_imports():
            from omegaconf import OmegaConf
            from util import instantiate_from_config
            cfg = OmegaConf.load(Path(folder) / "config.yaml")
            cfg.model.params.teacher_ckpt_path = None
            module = instantiate_from_config(cfg.model)
    finally:
        timm.create_model = create_model
        torch.load = torch_load
    state = torch.load(Path(folder) / "checkpoints/last.ckpt", map_location="cpu",
                       weights_only=True, mmap=True)["state_dict"]
    state = {k: v for k, v in state.items() if k.startswith(("ema_vit.", "ae."))}
    module.vit = module.ema_vit          # drop the training weights; the sampler uses ema_vit
    res = module.load_state_dict(state, strict=False)
    bad = [k for k in res.missing_keys if k.startswith(("ema_vit.", "ae."))]
    assert not bad, bad
    del state
    return module.eval().requires_grad_(False)


def release_cpu_memory():
    """Give freed heap pages back to the OS (before DataLoader workers fork)."""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


class L2Model:
    def __init__(self, device="cuda", dtype=torch.float16, compile=True):
        self.module = build_l2_module()
        self.module.to(device)
        release_cpu_memory()
        self.net = self.module.ema_vit
        if compile:
            # ~3.5x faster on the H100 (the eager DiT is bound by layer norms and
            # adaLN elementwise ops); about 1 min of compilation per new input shape.
            self.net = torch.compile(self.net, dynamic=False)
        self.device, self.dtype = device, dtype
        self.depth = len(self.net.blocks)

    def autocast(self):
        return torch.autocast("cuda", dtype=self.dtype, enabled=str(self.device).startswith("cuda"))

    @torch.no_grad()
    def encode(self, clip):
        """(B, T, 3, 288, 512) in [-1, 1] -> sem latents x enc_scale_dino, (B, T, 16, 18, 32)."""
        with self.autocast():
            return self.module.first_stage.encode_frames(clip.to(self.device)).float()

    @torch.no_grad()
    def decode(self, lat):
        """(B, T, 16, 18, 32) -> (B, T, 3, 288, 512) in about [-1, 1] (sem decoder)."""
        with self.autocast():
            return self.module.first_stage.decode_frames(lat.to(self.device)).float().clamp(-1, 1)

    @torch.no_grad()
    def velocity(self, seq, t, goals, blocks=()):
        """One DiT pass.

        seq (B, F, 16, 18, 32): clean context frames + the noisy next frame last.
        t (B,): flow time of the last frame (1 = noise); context frames get t = 0
        (per-frame timestep conditioning). goals (B, 5, 2) metres in the frame of the
        last context frame, NaN = no steering.
        Returns v of the last frame (x0 = x_t + t * v) and {block: (B, F, 18*32, 768)}.
        """
        B, F = seq.shape[:2]
        model_t = torch.zeros(B, F, device=self.device)
        model_t[:, -1] = t
        fr = torch.full((B,), FRAME_RATE, device=self.device)
        want = bool(blocks)
        with self.autocast():
            out = self.net(seq, t=model_t, frame_rate=fr, return_features=want,
                           steering=goals.to(self.device, torch.float32))
        if want:
            out, feats = out
            return out[:, -1].float(), {b: feats[b] for b in blocks}
        return out[:, -1].float(), {}

    @torch.no_grad()
    def sample_next(self, ctx, goals, noise, nfe=4, sampler="euler", blocks=()):
        """Sample the next latent frame given clean context latents.

        ctx (B, F, 16, 18, 32), goals (B, 5, 2), noise (B, nfe, 16, 18, 32) standard normal
        (euler uses noise[:, 0]; cd uses one fresh draw per step).
        sampler "euler": the module's configured FlowMatchingSamplerEuler on
        linspace(1, 0, nfe + 1) (what L1/config_distill.yaml uses, l2_pred_NFE = 4).
        sampler "cd": cd_model_v2.ConsistencyDistillationSampler grids (re-noise the
        endpoint estimate with fresh noise at each grid time).
        Returns x0 (B, 16, 18, 32) and the block features of the first pass (t at the
        noise end), {block: (B, F + 1, 576, 768)}.
        """
        B = ctx.shape[0]
        noise = noise.to(self.device)
        x = noise[:, :1]
        feats = None
        if sampler == "euler":
            ts = torch.linspace(1.0, 0.0, nfe + 1).tolist()
            for i in range(nfe):
                v, f = self.velocity(torch.cat([ctx, x], 1), torch.full((B,), ts[i], device=self.device),
                                     goals, blocks if i == 0 else ())
                feats = f if i == 0 else feats
                x = x + v[:, None] * (ts[i] - ts[i + 1])
            return x[:, 0], feats
        grid = CD_GRIDS.get(nfe) or torch.linspace(0.99, 0.01, nfe + 1)[:-1].tolist()
        x0 = None
        for i, tk in enumerate(grid):
            if i > 0:
                x = (1 - tk) * x0 + tk * noise[:, i:i + 1]
            v, f = self.velocity(torch.cat([ctx, x], 1), torch.full((B,), tk, device=self.device),
                                 goals, blocks if i == 0 else ())
            feats = f if i == 0 else feats
            x0 = x + tk * v[:, None]
        return x0[:, 0], feats


def null_goals(batch, device="cpu"):
    return torch.full((batch, NUM_GOALS, 2), float("nan"), device=device)
