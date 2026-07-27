"""Reader for imspy AnnData H5AD files."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import h5py
import numpy as np
from koyo.typing import PathLike

from image2image_io.readers._base_reader import BaseReader


class H5ADFormatError(ValueError):
    """Raised when an H5AD file does not match the supported imspy schema."""


def _raise_format_error(message: str) -> ty.NoReturn:
    """Raise an H5AD schema error with a descriptive message."""
    raise H5ADFormatError(message)


def _decode_strings(dataset: h5py.Dataset) -> list[str]:
    """Decode an HDF5 string dataset."""
    values = np.asarray(dataset[:]).reshape(-1).tolist()
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _read_scalar(handle: h5py.File, path: str) -> float | None:
    """Read an optional scalar value from an HDF5 file."""
    if path not in handle:
        return None
    node = handle[path]
    if not isinstance(node, h5py.Dataset):
        _raise_format_error(f"AnnData metadata '{path}' must be a scalar dataset.")
    value = np.asarray(node[()])
    if value.size != 1:
        _raise_format_error(f"AnnData metadata '{path}' must be a scalar.")
    return float(value.reshape(-1)[0])


def _read_var_names(handle: h5py.File) -> list[str]:
    """Read channel names from the AnnData variable index."""
    if "var" not in handle:
        _raise_format_error("AnnData file is missing the 'var' group.")
    var = handle["var"]
    if not isinstance(var, h5py.Group):
        _raise_format_error("AnnData 'var' must be a group.")
    index_key = var.attrs.get("_index", "_index")
    if isinstance(index_key, bytes):
        index_key = index_key.decode("utf-8")
    if index_key not in var or not isinstance(var[index_key], h5py.Dataset):
        _raise_format_error("AnnData 'var' group is missing its variable index dataset.")
    return _decode_strings(var[index_key])


def _read_coordinates(handle: h5py.File) -> np.ndarray:
    """Read and validate imspy y/x spatial coordinates."""
    if "obsm/spatial" not in handle:
        _raise_format_error("AnnData file is missing 'obsm/spatial' coordinates.")
    node = handle["obsm/spatial"]
    if not isinstance(node, h5py.Dataset):
        _raise_format_error("AnnData 'obsm/spatial' must be a dense dataset.")
    coordinates = np.asarray(node[:])
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        _raise_format_error(
            "AnnData 'obsm/spatial' must have shape (n_observations, 2) in y/x order; "
            f"got {coordinates.shape}."
        )
    if not np.issubdtype(coordinates.dtype, np.number) or not np.all(np.isfinite(coordinates)):
        _raise_format_error("AnnData 'obsm/spatial' coordinates must be finite numeric values.")
    rounded = np.rint(coordinates)
    if not np.array_equal(coordinates, rounded):
        _raise_format_error("AnnData 'obsm/spatial' coordinates must be integral pixel indices.")
    return rounded.astype(np.int64, copy=False)


def _read_image_shape(handle: h5py.File, coordinates: np.ndarray) -> tuple[int, int]:
    """Read the imspy image shape, falling back to coordinate extents."""
    height = _read_scalar(handle, "uns/spatial/imspy/metadata/height")
    width = _read_scalar(handle, "uns/spatial/imspy/metadata/width")
    if height is None or width is None:
        if not len(coordinates):
            _raise_format_error("Cannot infer the AnnData image shape from empty spatial coordinates.")
        if height is None:
            height = float(np.max(coordinates[:, 0]) + 1)
        if width is None:
            width = float(np.max(coordinates[:, 1]) + 1)
    if not height.is_integer() or not width.is_integer() or height <= 0 or width <= 0:
        _raise_format_error(f"AnnData image shape must contain positive integers; got ({height}, {width}).")
    return int(height), int(width)


def _read_resolution(handle: h5py.File) -> float:
    """Read the imspy pixel size, falling back to one micron."""
    resolution = _read_scalar(handle, "uns/spatial/imspy/scalefactors/pixel_size_um")
    if resolution is None:
        resolution = 1.0
    if not np.isfinite(resolution) or resolution <= 0:
        _raise_format_error(f"AnnData pixel size must be positive; got {resolution}.")
    return resolution


class H5ADReader(BaseReader):
    """Read dense centroid images exported by imspy as AnnData H5AD."""

    reader = "h5ad"
    reader_type = "image"
    lazy = True
    _is_rgb = False
    _array_dtype = np.dtype("float32")
    _channel_names: list[str]
    _matrix: h5py.Dataset
    x: np.ndarray
    y: np.ndarray

    def __init__(
        self,
        path: PathLike,
        key: str | None = None,
        auto_pyramid: bool | None = None,
        reader_kws: dict[str, ty.Any] | None = None,
    ) -> None:
        super().__init__(path, key=key, auto_pyramid=auto_pyramid, reader_kws=reader_kws)
        self.fh = h5py.File(Path(path), "r")
        try:
            self._initialize_from_file()
        except Exception:
            self.fh.close()
            self.fh = None
            raise

    def _initialize_from_file(self) -> None:
        """Read metadata and validate the supported imspy schema."""
        handle = ty.cast(h5py.File, self.fh)
        if "X" not in handle:
            _raise_format_error("AnnData file is missing the 'X' intensity matrix.")
        matrix = handle["X"]
        if not isinstance(matrix, h5py.Dataset):
            _raise_format_error("AnnData 'X' must be a dense dataset; sparse matrices are not supported.")
        if matrix.ndim != 2:
            _raise_format_error(f"AnnData 'X' must be two-dimensional; got shape {matrix.shape}.")
        if not np.issubdtype(matrix.dtype, np.number):
            _raise_format_error(f"AnnData 'X' must contain numeric intensities; got dtype {matrix.dtype}.")

        coordinates = _read_coordinates(handle)
        channel_names = _read_var_names(handle)
        if matrix.shape[0] != len(coordinates):
            _raise_format_error(
                "AnnData observation count does not match its spatial coordinates: "
                f"X has {matrix.shape[0]} rows but obsm/spatial has {len(coordinates)}."
            )
        if matrix.shape[1] != len(channel_names):
            _raise_format_error(
                "AnnData variable count does not match its variable index: "
                f"X has {matrix.shape[1]} columns but var has {len(channel_names)} names."
            )
        if not channel_names:
            _raise_format_error("AnnData 'var' must contain at least one channel name.")

        image_shape = _read_image_shape(handle, coordinates)
        y, x = coordinates[:, 0], coordinates[:, 1]
        if np.any(y < 0) or np.any(x < 0) or np.any(y >= image_shape[0]) or np.any(x >= image_shape[1]):
            _raise_format_error(f"AnnData spatial coordinates fall outside the image shape {image_shape}.")

        self._matrix = matrix
        self.y = y
        self.x = x
        self._channel_names = channel_names
        self._image_shape = image_shape
        self._array_shape = (*image_shape, len(channel_names))
        self.resolution = _read_resolution(handle)

    def _reshape_channel(self, index: int) -> np.ndarray:
        """Load and reshape one centroid channel into image space."""
        if index < 0 or index >= len(self.channel_names):
            message = f"Channel index out of range: {index}"
            raise IndexError(message)
        values = np.asarray(self._matrix[:, index], dtype=np.float32)
        image = np.full(self.image_shape, np.nan, dtype=np.float32)
        image[self.y, self.x] = values
        return image

    def get_channel(self, index: int, pyramid: int = 0, split_rgb: bool | None = None) -> np.ndarray:
        """Return one centroid image without loading the other channels."""
        if pyramid != 0:
            message = f"Pyramid index out of range: {pyramid}"
            raise IndexError(message)
        return self.get_channel_pyramid(index)[0]

    def get_channel_pyramid(self, index: int) -> list[np.ndarray]:
        """Return the single-level pyramid for one centroid channel."""
        name = self.channel_names[index]
        return [self.preprocessor(name, self._reshape_channel(index))]

    def get_image(self) -> np.ndarray:
        """Return all centroid channels as a channel-last image stack."""
        return np.dstack([self._reshape_channel(index) for index in range(self.n_channels)])

    def get_dask_pyr(self) -> list[np.ndarray]:
        """Return the single-level full image pyramid."""
        return [self.get_image()]

    def get_channel_axis_and_n_channels(self, shape: tuple[int, ...] | None = None) -> tuple[int, int]:
        """Return the channel-last axis and channel count."""
        return 2, self.n_channels
