"""Utilities."""

from __future__ import annotations

import typing as ty
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from dask import array as da
from koyo.typing import PathLike

if ty.TYPE_CHECKING:
    from skimage.transform import ProjectiveTransform


def resize(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize an image which could be grayscale, RGB or multi-channel."""
    import cv2

    n_channels, channel_axis, _ = get_shape_of_image(array)
    array = ensure_numpy_array(array)
    if channel_axis is None or channel_axis == 2:
        array = cv2.resize(array, shape[::-1], interpolation=cv2.INTER_LINEAR)
    else:
        array_ = np.empty((n_channels, *shape), dtype=array.dtype)
        for i in range(array.shape[0]):
            array_[i] = cv2.resize(array[i], shape[::-1], interpolation=cv2.INTER_LINEAR)
        array = array_
    return array


def write_thumbnail(path: PathLike, output_dir: PathLike, with_title: bool, first_only: bool = False) -> None:
    """Write thumbnail."""
    import matplotlib as mpl

    from image2image_io.readers import get_simple_reader

    mpl.use("agg")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = get_simple_reader(path, auto_pyramid=False)
    if reader.is_rgb:
        image = reader.pyramid[-1]
        filename = output_dir / f"{reader.name}_rgb_thumbnail.jpg"
        make_thumbnail(filename, image, f"{reader.name}", with_title)
    else:
        for channel_id, channel_name in enumerate(reader._channel_names):
            filename = output_dir / f"{reader.name}_index={channel_id}_name={channel_name}_thumbnail.jpg"
            image = reader.get_channel(channel_id, -1)
            make_thumbnail(filename, image, f"{reader.name}\n{channel_id} / {channel_name}", with_title)
            if first_only:
                break
    del reader


def make_thumbnail(filename: Path, image: np.ndarray, title: str, with_title: bool):
    """Create thumbnail of an image."""
    import matplotlib.pyplot as plt

    _fig, ax = plt.subplots(figsize=(12, 12))
    if image.ndim == 2:
        vmax = np.percentile(np.ravel(image), 99.0)
        ax.imshow(image, cmap="turbo", aspect="equal", vmax=vmax)
    else:
        ax.imshow(image, aspect="equal")
    if with_title:
        ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def xmlstr_to_dict(xmlstr: str) -> dict:
    """Convert xml string to dict."""
    return etree_to_dict(ET.fromstring(xmlstr))


def etree_to_dict(t: ET) -> dict:
    """Convert ElementTree to dict."""
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(("@" + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]["#text"] = text
        else:
            d[t.tag] = text
    return d


def format_merge_channel_names(
    channel_names: list[list[str]] | None, n_ch: int, channel_ids: list[list[int]] | None
) -> list[str]:
    """Format channel names and ensure number of channel names matches number of channels or default to C1, C2, C3, etc.

    Parameters
    ----------
    channel_names:list
        list of str that are channel names
    n_ch: int
        number of channels detected in image
    channel_ids: list[list[int]] | None
        list of channel ids

    Returns
    -------
    channel_names:
        list of str that are formatted
    """
    if channel_names is None:
        channel_names = [f"C{idx}" for idx in range(n_ch)]
    if channel_ids is not None:
        channel_names_ = []
        if len(channel_ids) != len(channel_names):
            raise ValueError("The number of channel ids does not match the number of channel names.")
        for index, channel_ids_ in enumerate(channel_ids):
            channel_names_.extend([channel_names[index][idx_] for idx_ in channel_ids_])
        channel_names = channel_names_
    else:
        if isinstance(channel_names, list) and isinstance(channel_names[0], list):
            channel_names = [name for names in channel_names for name in names]
        if n_ch != len(channel_names):
            raise ValueError("The number of channels does not match the number of channel names.")
    return channel_names


def format_mz(mz: float) -> str:
    """Format m/z value."""
    return f"m/z {mz:.3f}"


def guess_rgb(shape: tuple[int, ...]) -> bool:
    """Guess if the passed shape comes from rgb data.

    If last dim is 3 or 4 assume the data is rgb, including rgba.

    Parameters
    ----------
    shape : list of int
        Shape of the data that should be checked.
    """
    if hasattr(shape, "shape"):
        shape = shape.shape
    ndim = len(shape)
    last_dim = shape[-1]
    rgb = False
    if ndim > 2 and last_dim < 5:
        rgb = True
    return rgb


def get_shape_of_image(array_or_shape: np.ndarray | tuple[int, ...]) -> tuple[int, int | None, tuple[int, int]]:
    """Return shape of an image."""
    if isinstance(array_or_shape, tuple):
        ndim = len(array_or_shape)
        shape = array_or_shape
    else:
        ndim = array_or_shape.ndim
        shape = array_or_shape.shape

    shape = list(shape)  # type: ignore[assignment]
    if ndim == 2:
        n_channels = 1
        channel_axis = None
    elif ndim == 3:
        if shape[2] in [3, 4]:  # rgb or rgba
            channel_axis = 2
            n_channels = shape[2]
        else:
            channel_axis = 0
            n_channels = shape[0]
        # channel_axis = int(np.argmin(shape))
        # n_channels = int(shape[channel_axis])
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
        if n < np.iinfo(np.uint16).max:
            return np.uint16
        if n < np.iinfo(np.uint32).max:
            return np.uint32
        if n < np.iinfo(np.uint64).max:
            return np.uint64
        return array.dtype
    if n < np.finfo(np.float32).max:
        return np.float32
    if n < np.finfo(np.float64).max:
        return np.float64
    return array.dtype


def reshape_fortran(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Reshape data to Fortran (MATLAB) ordering."""
    return x.T.reshape(shape[::-1]).T


def reshape(x: np.ndarray, y: np.ndarray, array: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """Reshape array."""
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    shape = (ymax - ymin + 1, xmax - xmin + 1)
    dtype = np.float32 if np.isnan(fill_value) else array.dtype
    new_array = np.full(shape, fill_value=fill_value, dtype=dtype)
    new_array[y - ymin, x - xmin] = array
    return new_array


def reshape_batch(x: np.ndarray, y: np.ndarray, array: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
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


def calc_pyramid_levels(xy_final_shape: np.ndarray, tile_size: int) -> list[tuple[int, int]]:
    """Calculate the number of pyramids for a given image dimension, and tile size.

    Stops when further downsampling would be smaller than tile_size.

    Parameters
    ----------
    xy_final_shape:np.ndarray
        final shape in xy order
    tile_size: int
        size of the tiles in the pyramidal layers

    Returns
    -------
    res_shapes:list
        list of tuples of the shapes of the downsampled images

    """
    res_shape = xy_final_shape[::-1]
    res_shapes = [(int(res_shape[0]), int(res_shape[1]))]

    while all(res_shape > tile_size):
        res_shape = res_shape // 2
        res_shapes.append((int(res_shape[0]), int(res_shape[1])))
    return res_shapes[:-1]


def get_pyramid_info(
    y_size: int, x_size: int, n_ch: int, tile_size: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int, int]]]:
    """
    Get pyramidal info for OME-tiff output.

    Parameters
    ----------
    y_size: int
        y dimension of base layer
    x_size:int
        x dimension of base layer
    n_ch:int
        number of channels in the image
    tile_size:int
        tile size of the image

    Returns
    -------
    pyr_levels
        pyramidal levels
    pyr_shapes:
        OME-zarr pyramid shapes for all levels

    """
    yx_size = np.asarray([y_size, x_size], dtype=np.int32)
    pyr_levels = calc_pyramid_levels(yx_size, tile_size)
    pyr_shapes = [(1, n_ch, 1, int(pl[0]), int(pl[1])) for pl in pyr_levels]
    return pyr_levels, pyr_shapes


def ensure_dask_array(image: np.ndarray | da.Array) -> da.Array:
    """Ensure array is a dask array."""
    if isinstance(image, da.core.Array):
        return image
    # handles np.ndarray _and_ another array like objects.
    return da.from_array(image)


def ensure_numpy_array(image: np.ndarray | da.Array) -> np.ndarray:
    """Ensure an array is a numpy array."""
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, da.core.Array):
        return image.compute()
    return np.array(image)


def grayscale(rgb_image: np.ndarray | da.Array, is_interleaved: bool = False) -> np.ndarray:
    """
    Convert RGB image data to greyscale.

    Parameters
    ----------
    rgb_image: np.ndarray
        image data
    is_interleaved: bool
        whether the image is interleaved

    Returns
    -------
    image:np.ndarray
        returns 8-bit greyscale image for 24-bit RGB image
    """
    if is_interleaved:
        result = (
            (rgb_image[..., 0] * 0.2125).astype(np.uint8)
            + (rgb_image[..., 1] * 0.7154).astype(np.uint8)
            + (rgb_image[..., 2] * 0.0721).astype(np.uint8)
        )
    else:
        result = (
            (rgb_image[0, ...] * 0.2125).astype(np.uint8)
            + (rgb_image[1, ...] * 0.7154).astype(np.uint8)
            + (rgb_image[2, ...] * 0.0721).astype(np.uint8)
        )
    return result


def check_df_columns(
    df: pd.DataFrame,
    required: list[str],
    either: list[tuple[str, ...]] | None = None,
    either_dtype: tuple[np.dtype, ...] | None = None,
) -> bool:
    """Check if a DataFrame has the required columns."""
    if not all(col in df.columns for col in required):
        return False
    if either is not None:
        if either_dtype is not None:
            return all(any((col in df.columns) and (df[col].dtype in either_dtype) for col in cols) for cols in either)
        return all(any(col in df.columns for col in cols) for cols in either)
    return True


def get_column_name(df: pd.DataFrame, options: list[str]) -> str:
    """Get columns from a DataFrame."""
    for key in options:
        if key in df.columns:
            return key
    raise ValueError(f"None of the options {options} are in the DataFrame. Available columns are {df.columns}")


def sort_pyramid(pyramid: list[da.Array]) -> list[da.Array]:
    """Sort pyramid levels from highest to lowest resolution."""
    return sorted(pyramid, key=lambda x: x.size, reverse=True)
