"""
Training script for NuPlan trajectory prediction model.

Usage:
    # Train from scratch
    python train_nuplan.py --config configs/nuplan_simple_diffusion.yaml --epochs 10
    
    python train_nuplan.py --config configs/simple_trajectory.yaml --max_steps 35000
    
    python train_nuplan.py --config configs/whole_trajectory.yaml --max_steps 5000
    
    python train_nuplan.py --config configs/game.yaml --max_steps 5000
    
    python train_nuplan.py --config configs/nuplan.yaml --max_steps 5000
    python evaluate/test_nuplan.py --config configs/nuplan.yaml --last_ckpt --logdir logs_nuplan --num_samples 10
    
    # Resume training
    python train_nuplan.py --config configs/nuplan.yaml --resume logs/trajectory_model/checkpoints/last.ckpt
    
    # Test only
    python train_nuplan.py --config configs/nuplan.yaml --test
"""

import argparse
import datetime
import os
import sys
import logging
import torch

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from util import instantiate_from_config


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_parser(**parser_kwargs):
    """Command line argument parser."""
    parser = argparse.ArgumentParser(**parser_kwargs)
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="path to config file",
    )
    
    parser.add_argument(
        "-r", "--resume",
        type=str,
        default=None,
        help="path to checkpoint to resume from",
    )
    
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="number of gpus (default: 1)",
    )
    
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=1,
        help="number of nodes (default: 1)",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="number of epochs",
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="number of steps",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="run test only (no training)",
    )
    
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs_nuplan",
        help="directory for logs",
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="experiment name (default: timestamp + config name)",
    )
    
    return parser


if __name__ == "__main__":
    # Parse arguments
    parser = get_parser()
    opt = parser.parse_args()
    
    # Set random seed
    seed_everything(opt.seed, workers=True)
    
    # Load config
    logger.info("Loading config from %s", opt.config)
    config = OmegaConf.load(opt.config)
    
    # Setup experiment name and logdir
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if opt.name:
        exp_name = opt.name
    else:
        config_name = os.path.splitext(os.path.basename(opt.config))[0]
        exp_name = f"{now}_{config_name}"
    
    logdir = os.path.join(opt.logdir, exp_name)
    ckptdir = os.path.join(logdir, "checkpoints")
    os.makedirs(ckptdir, exist_ok=True)
    
    logger.info("Logging to %s", logdir)
    
    # Save config to logdir
    config_save_path = os.path.join(logdir, "config.yaml")
    OmegaConf.save(config, config_save_path)
    logger.info("Saved config to %s", config_save_path)
    
    # Instantiate model
    logger.info("Creating model")
    model = instantiate_from_config(config.model)
    
    # Instantiate data module
    logger.info("Creating data module")
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=ckptdir,
            filename="epoch-{epoch:03d}",
            monitor="train/loss", # TODO
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Setup logger
    tb_logger = TensorBoardLogger(logdir, name="tb")
    
    # Get GPU/node configuration
    ngpu = opt.n_gpus
    nodes = opt.n_nodes
    
    # configure learning rate
    batch_size, base_lr, adjust_learning_rate = config.data.params.batch_size, config.model.base_learning_rate, config.model.adjust_learning_rate
    grad_acc_steps = max(config.model.params.get("grad_acc_steps", 1), config.model.get("grad_acc_steps", 1))

    ngpu = opt.n_gpus


    model.num_iters_per_epoch = len(data.datasets["train"]) // (config.data.params.batch_size * opt.n_gpus * grad_acc_steps)
    logger.info("Num iters per epoch: %s", model.num_iters_per_epoch)

    # Scale learning rate based on effective batch size if enabled
    if adjust_learning_rate:
        # Linear scaling rule: lr scales with effective batch size
        model.learning_rate = base_lr * batch_size * ngpu * grad_acc_steps * opt.n_nodes
        logger.info("Setting learning rate to %s = %s (num_gpus) * %s (batchsize) * %s (base_lr) * %s (grad accumulation factor)",
        model.learning_rate, ngpu, batch_size, base_lr, grad_acc_steps)
    else: 
        # Use base learning rate without scaling
        model.learning_rate = base_lr
        logger.info("Setting learning rate to %s", model.learning_rate)
    
    logger.info("Cuda is available: %s", torch.cuda.is_available())
    
    # Setup trainer
    trainer_kwargs = {
        #"max_epochs": opt.epochs,
        "max_steps": opt.max_steps,
        "accelerator": "auto",
        #"strategy": "auto",
        "logger": tb_logger,
        "callbacks": callbacks,
        "precision": config.model.get("precision", "16-mixed"),
        "gradient_clip_val": config.model.get("gradient_clip_val", None),
        "accumulate_grad_batches": grad_acc_steps,
        "check_val_every_n_epoch": config.get("check_val_every_n_epoch", 1),
        "log_every_n_steps": 50,
        # distributed  training
        "devices": opt.n_gpus, # distributed training
        #"num_nodes": opt.n_nodes, # distributed training
        #"strategy": DDPStrategy(find_unused_parameters=False) if gpus > 1 or nodes > 1 else "auto", # distributed training
    }
    
    trainer = Trainer(**trainer_kwargs)
    
    # Run training or testing
    if opt.test:
        logger.info("Running test only")
        trainer.test(model, data, ckpt_path=opt.resume)
    else:
        logger.info("Starting training")
        trainer.fit(model, data, ckpt_path=opt.resume)
        
        # Run final test
        #logger.info("Running final test")
        #trainer.test(model, data)
