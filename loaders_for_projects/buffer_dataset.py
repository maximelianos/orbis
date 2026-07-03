import numpy as np
from torch.utils.data import Dataset


class BufferDataset(Dataset):
    """Backward-compatible wrapper class for an existing dataset instance.
    """
    def __init__(
        self,
        source_dataset,
        initial_buffer_size: int = 100,
        replace_rate: float = 2,  # fraction of replace_count
        seed: int = 0,
        max_buffer_size: int = 1000,
        dataset_size: int = -1,
        replace_count: int = 10,
    ):
        super().__init__()
        self.source_dataset = source_dataset
        self.initial_buffer_size = initial_buffer_size
        self.max_buffer_size = max_buffer_size
        self.dataset_size = len(source_dataset) if dataset_size == -1 else dataset_size
        self.replace_rate = max(int(replace_rate * replace_count), 1)
        self.replace_count = replace_count
        self.rng = np.random.default_rng(seed)

        self._samples_since_replace = 0
        self._source_idx = 0
        self._buffer = [self._draw_source_sample() for _ in range(self.initial_buffer_size)]

    def __len__(self):
        return self.dataset_size

    def _draw_source_sample(self):
        sample = self.source_dataset[self._source_idx]
        self._source_idx = (self._source_idx + 1) % len(self.source_dataset)
        return sample

    def _maybe_replace(self):
        self._samples_since_replace += 1
        if self._samples_since_replace >= self.replace_rate:
            # sample many at once
            for _ in range(self.replace_count):
                sample = self._draw_source_sample()
                if len(self._buffer) < self.max_buffer_size:
                    # append sample to buffer, if buffer is not full
                    self._buffer.append(sample)
                else:
                    # replace a sample
                    replace_idx = int(self.rng.integers(0, len(self._buffer)))
                    self._buffer[replace_idx] = sample
                self._samples_since_replace = 0

    def __getitem__(self, idx):
        idx = idx % len(self._buffer)
        sample = self._buffer[idx]
        self._maybe_replace()
        return sample
