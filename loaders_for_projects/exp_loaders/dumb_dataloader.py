"""
Dummy dataloader
"""

import os
import sys
import torch
import numpy as np

class SimpleDataset():
    """
    This dataset always returns [1]
    """
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __getitem__(self, idx):
        sample = dict()
        sample['value'] = np.array([1,], dtype=np.float32)
        return sample

    def __len__(self):
        return 1000