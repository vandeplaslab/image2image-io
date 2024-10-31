"""Lazy HDF5 wrapper around image data."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from image2image_io.utils.utilities import reshape, reshape_batch


class LazyImageWrapper:
    """Lazy wrapper around hdf5 file."""

    array: np.ndarray | None = None

    def __init__(
        self, path: Path, key: str, channel_names: list[str], x_coordinates: np.ndarray, y_coordinates: np.ndarray
    ):
        self.path = path
        self.key = key
        self.channel_names = channel_names
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates

    @contextmanager
    def open(self):
        """Open the file."""
        with h5py.File(self.path, "r") as fh:
            yield fh

    @contextmanager
    def lazy_array(self) -> ty.Generator[np.ndarray, None, None]:
        """Get reference to the peak's data without actually loading it into memory."""
        with self.open() as h5:
            yield h5[self.key]

    def __getitem__(self, item: int) -> np.ndarray:
        """Return item."""
        with self.lazy_array() as array:
            return reshape(self.x_coordinates, self.y_coordinates, array[:, item])

    @property
    def shape(self) -> tuple[int, int]:
        """Return shape of the array."""
        with self.lazy_array() as array:
            return array.shape  # type: ignore[return-value]

    @property
    def dtype(self) -> np.dtype:
        """Return dtype of the array."""
        with self.lazy_array() as peaks:
            return peaks.dtype  # type: ignore[no-any-return]

    def get_image(self) -> np.ndarray:
        """Return 3d version of the data."""
        if self.array is None:
            with self.open() as h5:
                array = np.zeros(self.shape, dtype=self.dtype)
                for sl in h5[self.key].iter_chunks():
                    array[sl] = h5[self.key][sl]
            self.array = reshape_batch(self.x_coordinates, self.y_coordinates, array)
        return self.array
