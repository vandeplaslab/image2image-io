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
