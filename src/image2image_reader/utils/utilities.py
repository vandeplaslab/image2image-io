"""Utilities."""
import typing as ty

import numpy as np

if ty.TYPE_CHECKING:
    from skimage.transform import ProjectiveTransform


def format_mz(mz: float) -> str:
    """Format m/z value."""
    return f"m/z {mz:.3f}"


def get_shape_of_image(array: np.ndarray) -> tuple[int, ty.Optional[int], tuple[int, ...]]:
    """Return shape of an image."""
    if array.ndim == 3:
        shape = list(array.shape)
        channel_axis = int(np.argmin(shape))
        n_channels = int(shape[channel_axis])
        shape.pop(channel_axis)
    else:
        shape = list(array.shape)
        n_channels = 1
        channel_axis = None
    return n_channels, channel_axis, tuple(shape)


def get_flat_shape_of_image(array: np.ndarray) -> tuple[int, int]:
    """Return shape of an image."""
    n_channels, _, shape = get_shape_of_image(array)
    n_px = int(np.prod(shape))
    return n_channels, n_px


def compute_transform(src: np.ndarray, dst: np.ndarray, transform_type: str = "affine") -> "ProjectiveTransform":
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


def reshape_fortran(x: np.ndarray, shape: ty.Tuple[int, int]) -> np.ndarray:
    """Reshape data to Fortran (MATLAB) ordering."""
    return x.T.reshape(shape[::-1]).T
