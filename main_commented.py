import argparse
import datetime
import glob
import os
import sys
import logging
import torch
import warnings

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DDPStrategy

from util import *


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



def get_callbacks(opt, logdir, ckptdir, config, lightning_config, now):
    """Initialize PyTorch Lightning callbacks from configuration.
    
    Args:
        opt: Command-line arguments
        logdir: Directory for logs
        ckptdir: Directory for checkpoints
        config: Model and data configuration
        lightning_config: PyTorch Lightning specific configuration
        now: Timestamp string for current run
        
    Returns:
        List of instantiated callback objects
    """
    # Prepare override parameters to be merged with callback configs
    overrides = {
        "logdir": logdir,
        "ckptdir": ckptdir,
        "cfgdir": os.path.join(logdir, "configs"),
        "now": now,
        "lightning_config": lightning_config,
        "config": config,
        "resume": opt.resume,
        "log_img_frequency": opt.log_img_frequency,
        "periodic_checkpoint": opt.periodic_checkpoint,
        "log_ckpt_frequency": opt.log_ckpt_frequency,
        "increase_log_steps": opt.increase_log_steps,
        "tsne_epoch_frequency": opt.tsne_epoch_frequency,
        # Use slower refresh rate on SLURM clusters to reduce I/O overhead
        "bar_refresh_rate": 100 if os.environ.get("SLURM_JOB_ID") else 1,
    }
    
    # Load callbacks configuration from model config or default base config
    if "callbacks" in config:
        callbacks_config = config.callbacks
    else:
        callbacks_config = OmegaConf.load("callbacks/configs/base.yaml").callbacks

    # Instantiate each callback with merged configuration
    callbacks = []
    for cb in callbacks_config:
        callbacks.append(instantiate_from_config(OmegaConf.merge(overrides, cb)))

    return callbacks



def get_parser(**parser_kwargs):
    def str2bool(v):
        """Convert string to boolean value for argparse."""
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-g",
        "--n_gpus",
        type=int,
        default=None,
        help="number of gpus per node",
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=None,
        help="number of nodes",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="check validation every n epochs",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="number of epochs",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "--profiler",
        type=str2bool,
        nargs="?",
        default=True,
    )
    parser.add_argument(
        "--tsne_epoch_frequency", 
        type=int, 
        default=0,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--periodic_checkpoint", 
        type=int, 
        default=1,
    )
    parser.add_argument(
        "--enable_codebook_usage_logger",
        action="store_true",
        help="Enable CodebookUsageLogger callback"
    )
    parser.add_argument(
        "--increase_log_steps",
        action="store_true",
        help="Increase log steps exponentially for ImageLogger",
    )
    parser.add_argument(
        "--log_img_frequency",
        type=int,
        default=2000,
        help="Frequency (in steps) for logging images",
    )
    parser.add_argument(
        "--log_ckpt_frequency",
        type=int,
        default=2000,
        help="Frequency (in steps) for saving checkpoints",
    )
    return parser



if __name__ == "__main__":
    # Generate timestamp for this training run
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Add current directory to Python path for importing local modules
    sys.path.append(os.getcwd())

    # Parse command-line arguments
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # Load and merge configuration files
    # Base configs are loaded from left to right, with later configs overriding earlier ones
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    # Parse additional command-line overrides (e.g., --key value)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    # Extract Lightning-specific configuration
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    lightning_config.trainer = trainer_config
    

    # Determine number of GPUs: use SLURM environment variables if available,
    # otherwise use all available CUDA devices
    if opt.n_gpus is None:
        opt.n_gpus = int(os.environ.get("SLURM_NTASKS_PER_NODE", torch.cuda.device_count()))
    
    # Determine number of nodes: use SLURM environment variables if available,
    # otherwise default to single-node training
    if opt.n_nodes is None:
        opt.n_nodes = int(os.environ.get("SLURM_NNODES", 1))

    # Validate that --name and --resume aren't both specified
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both. "
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    
    # Select appropriate work directory based on model type
    # First-stage models (tokenizers) use TK_WORK_DIR, second-stage models use WM_WORK_DIR
    if "first_stage" in config.model.target and "second_stage" not in config.model.target:
        workdir_env_name = "TK_WORK_DIR"
    else:
        workdir_env_name = "WM_WORK_DIR"
    
    workdir_env_name = os.environ.get(workdir_env_name)
    
    # Fall back to default logs directory if environment variable is not set
    if workdir_env_name is None:
        default_path = os.path.join(os.getcwd(), 'logs')
        workdir_env_name = default_path
        warnings.warn(
            f"Environment variables 'TK_WORK_DIR' or 'WM_WORK_DIR' are not set. "
            f"Using default work directory: {default_path}"
        )
    
        
    # Handle resuming from existing checkpoint or creating new training run
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            # If resume path is a checkpoint file, extract the log directory
            paths = opt.resume.split("/")
            log_name = workdir_env_name.split("/")[-1]
            idx = len(paths) - paths[::-1].index(log_name) + 1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
            logger.info("Resuming from checkpoint %s, logdir: %s", opt.resume, logdir)
        else:
            # If resume path is a directory, look for last.ckpt inside it
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
            logger.info("Resuming from folder %s, logdir: %s", opt.resume, logdir)
        
        opt.resume_from_checkpoint = ckpt

        # Load configuration files from the resumed run
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base

        # Use the resumed run's name
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        # Create a new run name based on timestamp and config
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix + "_" + get_jobid()

        logdir = os.path.join(workdir_env_name, nowname)
    
    logger.info(">>> Logging to %s", logdir)

    # Set up directories for checkpoints and configs
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    
    # Set random seed for reproducibility across all libraries
    seed_everything(opt.seed, workers=True)

    # Instantiate the model from configuration
    model = instantiate_from_config(config.model)

    # trainer and callbacks
    trainer_kwargs = dict()
    trainer_kwargs["logger"] = TensorBoardLogger(logdir, name="tb")
    logger.info("Logging to %s", logdir)
    trainer_kwargs["devices"] = opt.n_gpus
    trainer_kwargs["num_nodes"] = opt.n_nodes
    trainer_kwargs["check_val_every_n_epoch"] = opt.check_val_every_n_epoch
    trainer_kwargs["max_epochs"] = opt.num_epochs
    trainer_kwargs["precision"] = config.model.get("precision", "16-mixed")
    trainer_kwargs["gradient_clip_val"] = config.model.get("grad_clip", None)
    trainer_kwargs["gradient_clip_algorithm"] = config.model.get("gradient_clip_algorithm", None)
    trainer_kwargs["accumulate_grad_batches"] = config.model.get("grad_acc_steps", 1)
    
    # Configure DDP strategy for distributed training
    trainer_kwargs["strategy"] = DDPStrategy(
        find_unused_parameters=config.model.get("find_unused_parameters", False),
        static_graph=config.model.get("static_graph", False)
    )   
    
    # Enable PyTorch profiler for performance analysis if requested
    if opt.profiler:
        profiler = PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
            schedule=torch.profiler.schedule(skip_first=5, wait=1, warmup=1, active=3, repeat=2)
        )
        trainer_kwargs["profiler"] = profiler

    # Initialize callbacks for logging, checkpointing, etc.
    trainer_kwargs["callbacks"] = get_callbacks(opt, logdir, ckptdir, config, lightning_config, now)
    
    # Create the trainer with all configured options
    trainer = Trainer(**trainer_kwargs)
    
    # Initialize data module and prepare datasets
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    # configure learning rate
    bs, base_lr, adjust_learning_rate = config.data.params.batch_size, config.model.base_learning_rate, config.model.adjust_learning_rate
    grad_acc_steps = max(config.model.params.get("grad_acc_steps", 1), config.model.get("grad_acc_steps", 1))

    ngpu = opt.n_gpus


    model.num_iters_per_epoch = len(data.datasets["train"]) // (config.data.params.batch_size * opt.n_gpus * grad_acc_steps)
    logger.info("Num iters per epoch: %s", model.num_iters_per_epoch)

    # Scale learning rate based on effective batch size if enabled
    if adjust_learning_rate:
        # Linear scaling rule: lr scales with effective batch size
        model.learning_rate = base_lr * bs * ngpu * grad_acc_steps * opt.n_nodes
        logger.info("Setting learning rate to %s = %s (num_gpus) * %s (batchsize) * %s (base_lr) * %s (grad accumulation factor)",
        model.learning_rate, ngpu, bs, base_lr, grad_acc_steps)
    else: 
        # Use base learning rate without scaling
        model.learning_rate = base_lr
        logger.info("Setting learning rate to %s", model.learning_rate)


    # Set up signal handlers for interactive checkpointing and debugging
    def melk(*args, **kwargs):
        """Save checkpoint on SIGUSR1 signal (useful for preemption on clusters)."""
        if trainer.global_rank == 0:
            logger.warning("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        """Drop into debugger on SIGUSR2 signal (only on rank 0)."""
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()

    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # Start training
    try:
        if opt.resume:
            # Resume training from checkpoint
            trainer.fit(model, data, ckpt_path=opt.resume_from_checkpoint)
        else:
            # Start training from scratch
            trainer.fit(model, data)
    except Exception:
        # Save checkpoint on exception before re-raising
        melk()
        raise
