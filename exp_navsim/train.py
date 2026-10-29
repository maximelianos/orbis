"""Train a trajectory probe on cached features.

    python -m exp_navsim.train -c exp_navsim/configs/probe.yaml --name 01_dinov2 \
        --note "DINOv2 baseline" data.features=[dinov2] trainer.max_steps=100000

Positional key=value arguments override the config (OmegaConf dotlist). The run goes
to <logdir>/<name> with the resolved config.yaml, a README.md holding the note and
the command, TensorBoard logs in tb/, and checkpoints every trainer.ckpt_every steps
(all kept: the collapse analysis reads them in order).
"""

import argparse
import shlex
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from exp_navsim.data.datamodule import SceneDataModule
from exp_navsim.models.probe import TrajectoryProbe


class PrintProgress(Callback):
    """One plain line per `every` steps and per validation, readable in a tmux pane behind tee."""

    def __init__(self, every=500):
        self.every, self.t0 = every, time.time()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.every == 0:
            rate = self.every / max(time.time() - self.t0, 1e-6)
            self.t0 = time.time()
            print(f"step {step:7d}/{trainer.max_steps}  loss {outputs['loss'].item():.4f}  {rate:.1f} it/s", flush=True)

    def on_validation_end(self, trainer, module):
        m = trainer.callback_metrics
        line = "  ".join(f"{k} {m[k].item():.3f}" for k in sorted(m)
                         if k.split("/")[-1] in ("energy", "spread", "dispersion", "ade_min"))
        print(f"val @ {trainer.global_step}: {line}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="exp_navsim/configs/probe.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--logdir", default="logs_exp")
    ap.add_argument("--note", default="")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/checkpoints/last.ckpt")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    run = Path(args.logdir) / args.name
    ckpt = run / "checkpoints" / "last.ckpt"
    if run.exists() and not args.resume:
        sys.exit(f"{run} exists; pick another --name or pass --resume")
    run.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run / "config.yaml")
    if not args.resume:
        (run / "README.md").write_text(
            f"# {args.name}\n\n{args.note}\n\nOverrides: `{' '.join(args.overrides) or 'none'}`\n\n"
            f"```\npython -m exp_navsim.train {' '.join(shlex.quote(a) for a in sys.argv[1:])}\n```\n")

    seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    dm = SceneDataModule(**cfg.data)
    model = TrajectoryProbe(in_channels=dm.in_channels, num_frames=dm.num_frames, **cfg.model)

    t = cfg.trainer
    trainer = Trainer(
        max_steps=t.max_steps,
        val_check_interval=t.val_every,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        logger=TensorBoardLogger(str(run), name="tb", version=""),
        callbacks=[ModelCheckpoint(dirpath=run / "checkpoints", filename="step{step:07d}",
                                   every_n_train_steps=t.ckpt_every, save_top_k=-1, save_last=True,
                                   auto_insert_metric_name=False),
                   LearningRateMonitor(logging_interval="step"), PrintProgress()],
        accelerator="gpu", devices=1, precision=t.get("precision", "32-true"),
        enable_progress_bar=False,
    )
    trainer.fit(model, dm, ckpt_path=str(ckpt) if args.resume else None)


if __name__ == "__main__":
    main()
