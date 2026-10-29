"""Data module of the probe: one training split and several named validation loaders.

    train       training logs (minus holdout pairs, optionally a random subset)
    val         held-out logs of the same split, the in-distribution test
    train_sub   a fixed subset of the training scenes, to watch memorization
    holdout     the held-out (city, command) pairs in validation logs (compositional test)
    navtest     navtest scenes, if their features are cached: the test logs
    navhard     navhard_two_stage stage-1 scenes, if their features are cached

The validation loaders are random fixed subsets of val_scenes scenes each, so a
validation round costs the same whatever the split sizes are.
"""

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from exp_navsim.data.scenes import SceneFeatureDataset


class SceneDataModule(pl.LightningDataModule):
    def __init__(self, index, features, feature_dir="data/features", history=4, channels=None,
                 val_fraction=0.1, cities=None, holdout=None, max_scenes=None, seed=0, batch_size=256,
                 val_scenes=2000, num_workers=4, test_splits=("navtest", "navhard")):
        super().__init__()
        common = dict(features=list(features), feature_dir=feature_dir, history=history, channels=channels,
                      val_fraction=val_fraction, holdout=holdout)
        self.train_dataset = SceneFeatureDataset(index, split="train", cities=cities, max_scenes=max_scenes,
                                                 seed=seed, **common)
        vals = {"val": SceneFeatureDataset(index, split="val", cities=cities, **common),
                "train_sub": self.train_dataset}
        if holdout:
            vals["holdout"] = SceneFeatureDataset(index, split="holdout_val", **common)
        for name in test_splits:
            index_path = Path("data/navsim_index") / f"{name}.pkl"
            if index_path.exists() and all((Path(feature_dir) / name / f"{f}.npy").exists() for f in features):
                vals[name] = SceneFeatureDataset(index_path, split="all", **{**common, "holdout": None})
        rng = np.random.default_rng(seed)
        self.val_datasets = {k: (v, np.sort(rng.permutation(len(v))[:val_scenes])) for k, v in vals.items()}
        self.val_names = list(self.val_datasets)
        self.batch_size, self.num_workers = batch_size, num_workers

        x = self.train_dataset[0]["features"]
        self.num_frames, self.in_channels = x.shape[0], x.shape[1]

    # The datasets gather a whole batch at once: the sampler yields index lists and
    # batch_size=None turns off the per-sample collate.
    def train_dataloader(self):
        sampler = BatchSampler(RandomSampler(self.train_dataset), self.batch_size, drop_last=True)
        return DataLoader(self.train_dataset, sampler=sampler, batch_size=None,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return [DataLoader(ds, sampler=BatchSampler(idx.tolist(), self.batch_size, drop_last=False),
                           batch_size=None) for ds, idx in self.val_datasets.values()]
