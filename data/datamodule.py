from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl

from omegaconf import ListConfig, DictConfig

from main import logger, instantiate_from_config, get_obj_from_str


class DataModuleFromConfig(pl.LightningDataModule):
    """
    LightningDataModule that instantiates datasets from Conf configs.

    Supports a single dataset or a list of datasets (concatenated) per split.
    DDP-friendly: optionally uses a DistributedSampler in debug mode.

    Args:
        batch_size: Training batch size.
        val_batch_size: Validation/Test batch size (defaults to `batch_size`).
        train: Config (DictConfig) or list of configs for the train split.
        validation: Config (DictConfig) or list for the validation split.
        test: Config (DictConfig) or list for the test split.
        wrap: Reserved; not implemented.
        num_workers: Dataloader workers (defaults to `batch_size * 2`).
        dbg: If True, force a DistributedSampler when DDP is initialized.
        pin_memory: Pin host memory for faster H2D transfers.
        persistent_workers: Keep workers alive between iterations (if workers > 0).
        drop_last_train: Drop last incomplete train batch.
        shuffle_train: Shuffle training data.
        collate_fn: Dotted path to a collate function (e.g.
            "exp_navsim.episode.collate.pad_collate" for variable-length episodes);
            None uses the default PyTorch collate.
    """
    def __init__(self, batch_size, val_batch_size=None, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, dbg=False, persistent_workers=True,
                 collate_fn=None):
        super().__init__()
        # Dotted path to a custom collate function (e.g. for variable-length
        # episodes); None keeps the default PyTorch collate.
        self.collate_fn = get_obj_from_str(collate_fn) if collate_fn is not None else None
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        # keeps worker processes (and any in-worker state, e.g. the sample buffer)
        # alive across epochs instead of respawning them at each epoch start
        self.persistent_workers = persistent_workers and self.num_workers > 0
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap
        self.dbg = dbg
        
        if self.wrap:
            raise NotImplementedError("Wrapped datasets not implemented")

    def setup(self, stage=None):
        self.datasets = dict()
        for k, cfg in self.dataset_configs.items():
            logger.info("Loading dataset: %s", k)
            if isinstance(cfg, (list, ListConfig)):
                datasets = [instantiate_from_config(c) for c in cfg]
                self.datasets[k] = ConcatDataset(datasets)
                # [logger.info(d) for d in datasets]
            elif isinstance(cfg, DictConfig):
                ds = instantiate_from_config(cfg)
                self.datasets[k] = ds
                # logger.info(ds)
            else:
                raise ValueError(f"Invalid dataset config: {cfg}")

    def _train_dataloader(self):
        # dbg: build the DistributedSampler manually. Sampler does the shuffling and partitions
        # data across DDP ranks
        # else: Lightning injects the DistributedSampler automatically
        if self.dbg:
            # TODO originally shuffle=True in sampler
            sampler = DistributedSampler(self.datasets["train"], shuffle=False) # if self.trainer.use_ddp else None
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=False, pin_memory=True, drop_last=True, sampler=sampler,
                              persistent_workers=self.persistent_workers, collate_fn=self.collate_fn)
        else:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=True,
                            persistent_workers=self.persistent_workers, collate_fn=self.collate_fn)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.val_batch_size,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.persistent_workers, collate_fn=self.collate_fn)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.val_batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers, collate_fn=self.collate_fn)