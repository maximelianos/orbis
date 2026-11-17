"""Unpickling of the downloaded SimScale files with a whitelist of classes.

pickle.load calls whatever function a file names, so the untrusted logs and reward
files are read with find_class limited to the numpy array machinery.
"""

import importlib
import io
import pickle

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
}


def find_class(module, name):
    if (module, name) not in ALLOWED:
        raise pickle.UnpicklingError(f"blocked class {module}.{name}")
    return getattr(importlib.import_module(module), name)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return find_class(module, name)


def safe_loads(data):
    return SafeUnpickler(io.BytesIO(data)).load()
