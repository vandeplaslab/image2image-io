"""Lazy HDF5 wrapper around image data."""
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np


class LazyImageWrapper:
    """Lazy wrapper around hdf5 file."""

    def __init__(self, path: Path, key: str, x_coordinates: np.ndarray, y_coordinates: np.ndarray):
        self.path = path
        self.key = key
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates

    @contextmanager
    def open(self):
        """Open the file."""
        with h5py.File(self.path, "r") as fh:
            yield fh[self.key]
