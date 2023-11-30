"""Utilities."""
from __future__ import annotations

import typing as ty

import numpy as np

if ty.TYPE_CHECKING:
    from skimage.transform import ProjectiveTransform


def format_mz(mz: float) -> str:
    """Format m/z value."""
    return f"m/z {mz:.3f}"


def get_shape_of_image(array_or_shape: np.ndarray | tuple[int, ...]) -> tuple[int, int | None, tuple[int, int]]:
    """Return shape of an image."""
    if isinstance(array_or_shape, tuple):
        ndim = len(array_or_shape)
        shape = array_or_shape
    else:
        ndim = array_or_shape.ndim
        shape = array_or_shape.shape

    shape = list(shape)  # type: ignore[assignment]
    if ndim == 3:
        channel_axis = int(np.argmin(shape))
        n_channels = int(shape[channel_axis])
        shape.pop(channel_axis)
    else:
        n_channels = 1
        channel_axis = None
    return n_channels, channel_axis, tuple(shape)


def get_flat_shape_of_image(array_or_shape: np.ndarray | tuple[int, ...]) -> tuple[int, int]:
    """Return shape of an image."""
    n_channels, _, shape = get_shape_of_image(array_or_shape)
    n_px = int(np.prod(shape))
    return n_channels, n_px


def compute_transform(src: np.ndarray, dst: np.ndarray, transform_type: str = "affine") -> ProjectiveTransform:
    """Compute transform."""
    from skimage.transform import estimate_transform

    if len(dst) != len(src):
        raise ValueError(f"The number of fixed and moving points is not equal. (moving={len(dst)}; fixed={len(src)})")
    return estimate_transform(transform_type, src, dst)


def get_dtype_for_array(array: np.ndarray) -> np.dtype:
    """Return smallest possible data type for shape."""
    n = array.shape[1]
    if np.issubdtype(array.dtype, np.integer):
        if n < np.iinfo(np.uint8).max:
            return np.uint8
        elif n < np.iinfo(np.uint16).max:
            return np.uint16
        elif n < np.iinfo(np.uint32).max:
            return np.uint32
        elif n < np.iinfo(np.uint64).max:
            return np.uint64
    else:
        if n < np.finfo(np.float32).max:
            return np.float32
        elif n < np.finfo(np.float64).max:
            return np.float64


def reshape_fortran(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Reshape data to Fortran (MATLAB) ordering."""
    return x.T.reshape(shape[::-1]).T


def reshape(x: np.ndarray, y: np.ndarray, array: np.ndarray, fill_value: float = 0) -> np.ndarray:
    """Reshape array."""
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    shape = (ymax - ymin + 1, xmax - xmin + 1)
    dtype = np.float32 if np.isnan(fill_value) else array.dtype
    new_array = np.full(shape, fill_value=fill_value, dtype=dtype)
    new_array[y - ymin, x - xmin] = array
    return new_array


def reshape_batch(x: np.ndarray, y: np.ndarray, array: np.ndarray, fill_value: float = 0) -> np.ndarray:
    """Batch reshaping of images."""
    if array.ndim != 2:
        raise ValueError("Expected 2-D array.")
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    y = y - ymin
    x = x - xmin
    n = array.shape[1]
    shape = (n, ymax - ymin + 1, xmax - xmin + 1)
    dtype = np.float32 if np.isnan(fill_value) else array.dtype
    im = np.full(shape, fill_value=fill_value, dtype=dtype)
    for i in range(n):
        im[i, y, x] = array[:, i]
    return im


def get_yx_coordinates_from_shape(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Get coordinates from image shape."""
    _y, _x = np.indices(shape)
    yx_coordinates = np.c_[np.ravel(_y), np.ravel(_x)]
    return yx_coordinates[:, 0], yx_coordinates[:, 1]
