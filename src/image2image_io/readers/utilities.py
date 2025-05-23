"""Tiff file utilities."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import SimpleITK as sitk
import zarr
from koyo.timer import MeasureTimer
from loguru import logger
from tifffile import TiffFile, imread, xml2dict

from image2image_io.utils.utilities import guess_rgb

logger = logger.bind(src="Tiff")


def tifffile_to_dask(im_fp: str | Path, largest_series: int) -> list[da.Array]:
    """Convert a tifffile to a dask array."""
    with MeasureTimer() as timer:
        imdata = zarr.open(imread(im_fp, aszarr=True, series=largest_series))
    logger.trace(f"Loaded OME-TIFF data in {timer()}")
    with MeasureTimer() as timer:
        if isinstance(imdata, zarr.hierarchy.Group):
            imdata = [da.from_zarr(imdata[z]) for z in imdata.array_keys()]
        else:
            imdata = [da.from_zarr(imdata)]
    logger.trace(f"Converted OME-TIFF data to dask in {timer()}")
    return imdata


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


def centered_transform(
    image_size,
    image_spacing,
    rotation_angle,
) -> np.ndarray:
    """Centered rotation transform."""
    angle = np.deg2rad(rotation_angle)

    sina = np.sin(angle)
    cosa = np.cos(angle)

    # build rot mat
    rot_mat = np.eye(3)
    rot_mat[0, 0] = cosa
    rot_mat[1, 1] = cosa
    rot_mat[1, 0] = sina
    rot_mat[0, 1] = -sina

    # recenter transform
    center_point = np.multiply(image_size, image_spacing) / 2
    center_point = np.append(center_point, 0)
    translation = center_point - np.dot(rot_mat, center_point)
    rot_mat[:2, 2] = translation[:2]

    return rot_mat


def get_largest_pyramid_series_tiff(image_filepath):
    """Determine the largest series for .scn files by examining metadata.

    For other multi-series files, find the one with the most pixels.

    Parameters
    ----------
    image_filepath: str
        path to the image file

    Returns
    -------
    largest_series:int
        index of the largest series in the image data
    """
    fp_ext = Path(image_filepath).suffix.lower()
    tf_im = TiffFile(image_filepath)
    if fp_ext == ".scn":
        scn_meta = xml2dict(tf_im.scn_metadata)
        image_meta = scn_meta.get("scn").get("collection").get("image")
        largest_series = np.argmax(
            [im.get("scanSettings").get("objectiveSettings").get("objective") for im in image_meta]
        )
    else:
        largest_series = np.argmax([np.prod(np.asarray(series.shape), dtype=np.int64) for series in tf_im.series])
    tf_im.close()
    tf_im.filehandle.close()
    del tf_im
    return largest_series


def tifffile_zarr_backend(image_filepath, largest_series, preprocessing, force_rgb=None):
    """
    Read image with tifffile and use zarr to read data into memory.

    Parameters
    ----------
    image_filepath: str
        path to the image file
    largest_series: int
        index of the largest series in the image
    preprocessing:
        whether to do some read-time pre-processing
        - greyscale conversion (at the tile level)
        - read individual or range of channels (at the tile level)
    force_rgb: bool
        whether to force the image to be read as rgb

    Returns
    -------
    image: sitk.Image
        image ready for another registration pre-processing

    """
    zarr_series = imread(image_filepath, aszarr=True, series=largest_series)
    zarr_store = zarr.open(zarr_series)
    zarr_im = zarr_get_base_pyr_layer(zarr_store)
    return read_preprocess_array(zarr_im, preprocessing=preprocessing, force_rgb=force_rgb)


def tifffile_dask_backend(image_filepath, largest_series, preprocessing, force_rgb=None):
    """
    Read image with tifffile and use dask to read data into memory.

    Parameters
    ----------
    image_filepath: str
        path to the image file
    largest_series: int
        index of the largest series in the image
    preprocessing:
        whether to do some read-time pre-processing
        - greyscale conversion (at the tile level)
        - read individual or range of channels (at the tile level)
    force_rgb: bool
        whether to force the image to be read as rgb

    Returns
    -------
    image: sitk.Image
        image ready for other registration pre-processing

    """
    zarr_series = imread(image_filepath, aszarr=True, series=largest_series)
    zarr_store = zarr.open(zarr_series)
    dask_im = da.squeeze(da.from_zarr(zarr_get_base_pyr_layer(zarr_store)))
    return read_preprocess_array(dask_im, preprocessing=preprocessing, force_rgb=force_rgb)


def zarr_get_base_pyr_layer(zarr_store):
    """
    Find the base pyramid layer of a zarr store.

    Parameters
    ----------
    zarr_store
        zarr store

    Returns
    -------
    zarr_im: zarr.core.Array
        zarr array of base layer
    """
    if isinstance(zarr_store, zarr.hierarchy.Group):
        zarr_im = zarr_store[str(0)]
    elif isinstance(zarr_store, zarr.core.Array):
        zarr_im = zarr_store
    else:
        raise ValueError("Could not find base pyramid layer.")
    return zarr_im


def ensure_dask_array(image):
    """Ensure array is a dask array."""
    if isinstance(image, da.core.Array):
        return image

    if isinstance(image, zarr.Array):
        return da.from_zarr(image)

    # handles np.ndarray _and_ other array like objects.
    return da.from_array(image)


def read_preprocess_array(array, preprocessing, force_rgb=None):
    """Read np.array, zarr.Array, or dask.array image into memory with preprocessing for registration."""
    is_interleaved = guess_rgb(array.shape)
    is_rgb = is_interleaved if not force_rgb else force_rgb

    if is_rgb:
        if preprocessing:
            image_out = np.asarray(grayscale(array, is_interleaved=is_interleaved))
            image_out = sitk.GetImageFromArray(image_out)
        else:
            image_out = np.asarray(array)
            if not is_interleaved:
                image_out = np.rollaxis(image_out, 0, 3)
            image_out = sitk.GetImageFromArray(image_out, isVector=True)

    elif len(array.shape) == 2:
        image_out = sitk.GetImageFromArray(np.asarray(array))

    else:
        if preprocessing:
            if preprocessing.ch_indices and len(array.shape) > 2:
                chs = list(preprocessing.ch_indices)
                array = array[chs, :, :]

        image_out = sitk.GetImageFromArray(np.squeeze(np.asarray(array)))

    return image_out


def get_tifffile_info(image_filepath):
    """Get tiff file information."""
    largest_series = get_largest_pyramid_series_tiff(image_filepath)
    zarr_im = zarr.open(imread(image_filepath, aszarr=True, series=largest_series))
    zarr_im = zarr_get_base_pyr_layer(zarr_im)
    image_shape = image_shape_ = np.squeeze(zarr_im.shape)
    if len(image_shape_) == 2:
        image_shape_ = np.concatenate([[1], image_shape_])
    return image_shape, image_shape_, zarr_im.dtype, largest_series


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


def tf_zarr_read_single_ch(image_filepath, channel_idx, is_rgb, is_rgb_interleaved=True):
    """
    Reads a single channel using zarr or dask in combination with tifffile.

    Parameters
    ----------
    image_filepath:str
        file path to image
    channel_idx:int
        index of the channel to be read
    is_rgb:bool
        whether image is rgb interleaved
    is_rgb_interleaved:bool
        whether image is rgb interleaved

    Returns
    -------
    im:np.ndarray
        image as a np.ndarray
    """
    largest_series = get_largest_pyramid_series_tiff(image_filepath)
    zarr_im = zarr.open(imread(image_filepath, aszarr=True, series=largest_series))
    zarr_im = zarr_get_base_pyr_layer(zarr_im)
    try:
        im = da.squeeze(da.from_zarr(zarr_im))
        if is_rgb and is_rgb_interleaved:
            im = im[:, :, channel_idx].compute()
        elif len(im.shape) > 2:
            im = im[channel_idx, :, :].compute()
        else:
            im = im.compute()

    except ValueError:
        im = zarr_im
        if is_rgb is True and is_rgb_interleaved:
            im = im[:, :, channel_idx]
        elif len(im.shape) > 2:
            im = im[channel_idx, :, :].compute()
        else:
            im = im.compute()
    return im


def yield_tiles(z, tile_size, is_rgb):
    """Return tiles."""
    for y in range(0, z.shape[0], tile_size):
        for x in range(0, z.shape[1], tile_size):
            if is_rgb:
                yield z[y : y + tile_size, x : x + tile_size, :].compute()


def compute_sub_res(zarray, ds_factor, tile_size, is_rgb, im_dtype) -> da.Array:
    """Compute sub-resolution image."""
    if is_rgb:
        resampling_axis = {0: 2**ds_factor, 1: 2**ds_factor, 2: 1}
        tiling = (tile_size, tile_size, 3)
    else:
        resampling_axis = {0: 1, 1: 2**ds_factor, 2: 2**ds_factor}
        tiling = (1, tile_size, tile_size)

    resampled_zarray_subres = da.coarsen(np.mean, zarray, resampling_axis, trim_excess=True)
    resampled_zarray_subres = resampled_zarray_subres.astype(im_dtype)
    resampled_zarray_subres = resampled_zarray_subres.rechunk(tiling)
    return resampled_zarray_subres


def prepare_ome_xml_str(
    y_size: int, x_size: int, n_ch: int, im_dtype: np.dtype[ty.Any], is_rgb: bool, **ome_metadata: ty.Any
) -> str:
    """Prepare OME-XML metadata."""
    from tifffile import OmeXml

    omexml = OmeXml()
    if is_rgb:
        stored_shape = (1, 1, 1, y_size, x_size, n_ch)
        im_shape = (y_size, x_size, n_ch)
    else:
        stored_shape = (n_ch, 1, 1, y_size, x_size, 1)
        im_shape = (n_ch, y_size, x_size)

    omexml.addimage(
        dtype=im_dtype,
        shape=im_shape,
        # specify how the image is stored in the TIFF file
        storedshape=stored_shape,
        **ome_metadata,
    )

    return omexml.tostring().encode("utf8")


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
