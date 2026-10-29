"""Flow-matching trajectory probe on frozen encoder features (NAVSIM protocol).

Input: cached feature grids of the history frames (data/scenes.py), optionally the
ego status and the driving command. Output: a distribution over the next 4 s,
8 poses (x, y, heading), sampled by integrating the flow-matching ODE on the
trajectory code z of models/codec.py.

    x_t = (1 - t) z + sigma(t) eps,  sigma(t) = sigma_min + t (1 - sigma_min)
    target velocity u = z - (1 - sigma_min) eps,  loss = ||u_theta(x_t, t | cond) - u||^2

Validation runs on several loaders (data/datamodule.py names them, e.g. "val" for
held-out logs and "train_sub" for a fixed set of training scenes). For each it
logs the flow-matching loss with fixed t and noise, and the sample metrics of
eval/metrics.py over num_val_samples futures per scene. The train_sub numbers
next to the val ones show memorization: on seen scenes the spread collapses and
the error goes to zero long before it does on unseen ones.
"""

import math

import torch
import pytorch_lightning as pl

from exp_navsim.eval.metrics import sample_metrics
from exp_navsim.models.codec import build_codec
from exp_navsim.models.denoiser import ProbeDenoiser


class TrajectoryProbe(pl.LightningModule):
    def __init__(self, in_channels, num_frames, hidden_dim=16, image_tokens=16, ff_dim=32, num_heads=2,
                 depth=4, dropout=0.0, use_status=False, use_command=False, codec="poly", poly_order=3,
                 sigma_min=1e-5, num_diffusion_steps=10, num_val_samples=16, lr=5e-4, weight_decay=0.01,
                 warmup_steps=500, min_lr_multiplier=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.codec = build_codec(codec, poly_order)
        self.denoiser = ProbeDenoiser(self.codec.latent_dim, in_channels, num_frames, hidden_dim, image_tokens,
                                      ff_dim, num_heads, depth, use_status, use_command, dropout)
        self.val_names = ["val"]

    # ----------------------------------------------------------------- setup -- #
    def on_fit_start(self):
        dm = self.trainer.datamodule
        self.val_names = dm.val_names
        if not bool(self.codec.fitted):
            ds = dm.train_dataset
            futures = torch.from_numpy(ds.future[ds.rows]).to(self.device)
            stats = self.codec.fit(futures)
            print(f"[probe] codec {self.hparams.codec} fitted on {len(futures)} futures: {stats}")
        n = sum(p.numel() for p in self.denoiser.parameters())
        print(f"[probe] denoiser parameters: {n}")

    # --------------------------------------------------------------- training -- #
    def add_noise(self, z, t, noise):
        s = self.hparams.sigma_min
        return (1 - t[:, None]) * z + (s + t[:, None] * (1 - s)) * noise

    def fm_loss(self, batch, t=None, noise=None):
        z = self.codec.encode(batch["future"])
        t = torch.rand(z.shape[0], device=z.device) if t is None else t
        noise = torch.randn_like(z) if noise is None else noise
        target = z - (1.0 - self.hparams.sigma_min) * noise
        pred = self.denoiser(self.denoiser.condition(batch), self.add_noise(z, t, noise), t)
        return ((pred - target) ** 2).mean()

    def training_step(self, batch, batch_idx):
        loss = self.fm_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    # ------------------------------------------------------------- sampling -- #
    @torch.no_grad()
    def sample(self, batch, num_samples, num_steps=None, generator=None):
        """-> (K, B, 8, 3) futures. Euler from t = 1 (noise) to t = 0."""
        steps = num_steps or self.hparams.num_diffusion_steps
        cond = self.denoiser.condition(batch)
        B, K = cond.shape[0], num_samples
        cond = cond.repeat(K, 1, 1)                                       # (K * B, L, D), sample-major
        z = torch.randn(K * B, self.codec.latent_dim, device=cond.device, generator=generator)
        ts = torch.linspace(1, 0, steps + 1, device=cond.device)
        for i in range(steps):
            # dx/dt = -u for x_t = (1 - t) z + sigma(t) eps, so a step from t down to t - dt adds u dt.
            z = z + self.denoiser(cond, z, ts[i].expand(K * B)) * (ts[i] - ts[i + 1])
        return self.codec.decode(z).unflatten(0, (K, B))

    # ----------------------------------------------------------- validation -- #
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        name = self.val_names[dataloader_idx]
        g = torch.Generator(device=self.device).manual_seed(batch_idx)
        B = batch["future"].shape[0]
        t = torch.rand(B, device=self.device, generator=g)
        noise = torch.randn(B, self.codec.latent_dim, device=self.device, generator=g)
        log = dict(on_epoch=True, add_dataloader_idx=False, batch_size=B)
        self.log(f"{name}/loss", self.fm_loss(batch, t, noise), prog_bar=name == "val", **log)
        samples = self.sample(batch, self.hparams.num_val_samples, generator=g)
        metrics = sample_metrics(samples, batch["future"])
        for k, v in metrics.items():
            if k != "dispersion":
                self.log(f"{name}/{k}", v.mean(), **log)

    def on_validation_epoch_end(self):
        # Dispersion as a ratio of means, so near-zero-error scenes do not dominate it.
        m = self.trainer.callback_metrics
        for name in self.val_names:
            if f"{name}/spread" in m:
                self.log(f"{name}/dispersion", m[f"{name}/spread"] / m[f"{name}/ade_mean"].clamp(min=1e-3))

    # ------------------------------------------------------------ optimizer -- #
    def configure_optimizers(self):
        hp = self.hparams
        optimizer = torch.optim.AdamW(self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

        def lr_lambda(step):
            # Linear warmup, then cosine decay down to min_lr_multiplier.
            if step < hp.warmup_steps:
                return step / max(hp.warmup_steps, 1)
            progress = min((step - hp.warmup_steps) / max(self.trainer.max_steps - hp.warmup_steps, 1), 1.0)
            return hp.min_lr_multiplier + (1 - hp.min_lr_multiplier) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
