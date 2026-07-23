"""In-memory ring-buffer dataset, copied from brain_matching/data/buffer_dataset.py.

The buffer serves cached samples and only periodically draws fresh ones from the
(slow) source dataset, so a GPU-side consumer (e.g. the on-the-fly image encoder)
is not starved by disk/decoding. The source dataset itself is unrelated to
brain_matching -- only the buffer is reused here.
"""

import numpy as np
from torch.utils.data import Dataset

from util import instantiate_from_config


class BufferDataset(Dataset):
    """Generic wrapper that serves samples from an in-memory buffer."""

    def __init__(
        self,
        source_dataset,
        initial_buffer_size: int,
        max_buffer_size: int,
        replace_rate: float,   # fraction of replace_count
        replace_count: int,
        enable_buffer: bool,
        dataset_length: int = None,   # if None, use the source dataset's length
    ):
        super().__init__()
        # Accept either an instantiated dataset or a config dict, so the source
        # can be nested directly in the YAML config.
        if hasattr(source_dataset, "get") and source_dataset.get("target") is not None:
            source_dataset = instantiate_from_config(source_dataset)
        self.source_dataset = source_dataset
        self.initial_buffer_size = initial_buffer_size
        self.max_buffer_size = max_buffer_size
        self.dataset_size = dataset_length if dataset_length is not None else len(source_dataset)
        self.replace_rate = max(int(replace_rate * replace_count), replace_count)
        self.replace_count = replace_count
        self.enable_buffer = enable_buffer

        self._samples_since_replace = 0
        self._source_idx = 0
        self._target_idx = 0
        if self.enable_buffer:
            self._buffer = [self._get_sample() for _ in range(self.initial_buffer_size)]

    def __len__(self):
        return self.dataset_size

    def _get_sample(self):
        sample = self.source_dataset[self._source_idx]
        self._source_idx = (self._source_idx + 1) % len(self.source_dataset)
        return sample

    def __getitem__(self, idx):
        # Buffering disabled: serve directly from the source dataset.
        if not self.enable_buffer:
            return self.source_dataset[idx % self.dataset_size]

        sample = self._buffer[idx % len(self._buffer)]

        # Refill with fresh samples once we have consumed enough from the buffer.
        self._samples_since_replace += 1
        if self._samples_since_replace >= self.replace_rate:
            for _ in range(self.replace_count):
                sample = self._get_sample()
                if len(self._buffer) < self.max_buffer_size:
                    self._buffer.append(sample)
                else:
                    self._buffer[self._target_idx] = sample
                    self._target_idx = (self._target_idx + 1) % len(self._buffer)
                self._samples_since_replace = 0

        return sample
