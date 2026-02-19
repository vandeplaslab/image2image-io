"""Combine multiple files into one.

Unlike the merge method, which appends image channels from multiple files, this method combines files
by adding them together (or max int projection) to create a single image.
"""

from __future__ import annotations

import typing as ty
from pathlib import Path

import cv2
import numpy as np
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from skimage.exposure import match_histograms


def match_histograms_cv2(
    src: np.ndarray,
    ref: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    ref_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Histogram-match src to ref using OpenCV-style LUT mapping.

    Notes:
    - Fastest for uint8/uint16.
    - For float images, we quantize to uint16 internally and map back.
    - mask / ref_mask let you ignore background (useful for IMS sparse pixels).
    """
    if src.shape != ref.shape:
        # shape doesn't need to match, but mask shapes must match their image
        pass

    # Choose working depth
    if src.dtype == np.uint8 and ref.dtype == np.uint8:
        levels = 256
        src_u = src
        ref_u = ref
        to_src = None
    elif src.dtype == np.uint16 and ref.dtype == np.uint16:
        levels = 65536
        src_u = src
        ref_u = ref
        to_src = None
    else:
        # Quantize floats/other ints to uint16 based on finite min/max
        src_f = src.astype(np.float32, copy=False)
        ref_f = ref.astype(np.float32, copy=False)

        s_min, s_max = np.nanmin(src_f), np.nanmax(src_f)
        r_min, r_max = np.nanmin(ref_f), np.nanmax(ref_f)

        # handle degenerate
        if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
            return src.copy()
        if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max <= r_min:
            return src.copy()

        levels = 65536
        src_u = np.clip((src_f - s_min) / (s_max - s_min) * (levels - 1), 0, levels - 1).astype(np.uint16)
        ref_u = np.clip((ref_f - r_min) / (r_max - r_min) * (levels - 1), 0, levels - 1).astype(np.uint16)

        # mapping back to original float scale of src
        to_src = (s_min, s_max)

    # Histograms (masked if provided)
    if mask is None:
        h_src = cv2.calcHist([src_u], [0], None, [levels], [0, levels]).ravel()
    else:
        m = mask.astype(np.uint8) * 255
        h_src = cv2.calcHist([src_u], [0], m, [levels], [0, levels]).ravel()

    if ref_mask is None:
        h_ref = cv2.calcHist([ref_u], [0], None, [levels], [0, levels]).ravel()
    else:
        rm = ref_mask.astype(np.uint8) * 255
        h_ref = cv2.calcHist([ref_u], [0], rm, [levels], [0, levels]).ravel()

    # CDFs
    cdf_src = np.cumsum(h_src)
    cdf_ref = np.cumsum(h_ref)
    if cdf_src[-1] == 0 or cdf_ref[-1] == 0:
        return src.copy()
    cdf_src /= cdf_src[-1]
    cdf_ref /= cdf_ref[-1]

    # LUT: for each src level, find the closest ref level with same CDF
    lut = np.searchsorted(cdf_ref, cdf_src, side="left")
    lut = np.clip(lut, 0, levels - 1)

    # Apply LUT
    if levels == 256:
        lut_cv = lut.astype(np.uint8)
        out_u = cv2.LUT(src_u, lut_cv)
    else:
        # cv2.LUT supports 8-bit LUT; for uint16 use numpy take (still fast)
        out_u = lut[src_u]

    # Convert back if we quantized floats
    if to_src is not None:
        s_min, s_max = to_src
        out_f = (out_u.astype(np.float32) / (levels - 1)) * (s_max - s_min) + s_min
        return out_f.astype(src.dtype, copy=False)

    return out_u.astype(src.dtype, copy=False)


def match_histograms_nan_safe(img, ref, *, channel_axis=-1, fill="min"):
    """Like skimage's match_histograms but handles NaNs by filling them with a specified value."""
    img = np.asarray(img)
    ref = np.asarray(ref)

    def _fill(x):
        x = x.astype(np.float32, copy=False)
        finite = np.isfinite(x)
        if not finite.any():
            return np.zeros_like(x, dtype=np.float32)
        if fill == "min":
            v = np.nanmin(x)
        elif fill == "median":
            v = np.nanmedian(x)
        elif isinstance(fill, (int, float)):
            v = float(fill)
        else:
            raise ValueError("fill must be 'min', 'median', or a number")
        return np.where(finite, x, v)

    img_f = _fill(img)
    ref_f = _fill(ref)

    out = match_histograms(img_f, ref_f, channel_axis=channel_axis)

    # If you want to preserve NaN locations from the original:
    out = out.astype(np.float32, copy=False)
    out[~np.isfinite(img.astype(np.float32, copy=False))] = np.nan
    return out


def match_histograms_auto(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match histogram."""
    dtype = img.dtype
    is_rgb = img.ndim == 3 and img.shape[2] == 3
    if dtype.kind == "u":
        img = img.astype(np.float32)
    # match histogram for each channel individually
    if img.ndim == 2:
        img = match_histograms(np.asarray(img), np.asarray(ref), channel_axis=None)
    for i in range(img.shape[-1]):
        img[..., i] = match_histograms(np.asarray(img[..., i]), np.asarray(ref[..., i]), channel_axis=None)
    # now ensure that the data type is correct
    if is_rgb:
        img = np.clip(img, 0, 255).astype(dtype)
    return img


def _match_channel(source, reference):
    """Internal helper for single channel matching."""
    orig_shape = source.shape
    orig_dtype = source.dtype

    # Flatten for histogram processing
    source_flat = source.ravel()
    ref_flat = reference.ravel()

    # For performance on 20k x 20k, we compute the histogram
    # using a fixed number of bins based on bit-depth.
    if orig_dtype == np.uint8:
        bins = 256
        hist_range = [0, 256]
    elif orig_dtype == np.uint16:
        bins = 65536
        hist_range = [0, 65536]
    else:  # float32
        bins = 1024  # Sufficient for most microscopy dynamic ranges
        hist_range = [source_flat.min(), source_flat.max()]

    # Compute histograms and CDFs
    s_hist, s_bins = np.histogram(source_flat, bins, range=hist_range)
    r_hist, r_bins = np.histogram(ref_flat, bins, range=hist_range)

    s_cdf = s_hist.cumsum()
    s_cdf = (s_cdf - s_cdf.min()) / (s_cdf.max() - s_cdf.min())

    r_cdf = r_hist.cumsum()
    r_cdf = (r_cdf - r_cdf.min()) / (r_cdf.max() - r_cdf.min())

    # Create the mapping (Interpolate source CDF to reference values)
    # This finds where the source CDF values would fall on the reference CDF
    bin_centers = (s_bins[:-1] + s_bins[1:]) / 2
    ref_bin_centers = (r_bins[:-1] + r_bins[1:]) / 2

    # Use numpy's fast interpolation as a Look-Up Table
    lookup_table = np.interp(s_cdf, r_cdf, ref_bin_centers)

    # Apply the mapping
    # For uint8/uint16, we can use a literal LUT. For float, we interpolate.
    matched = np.interp(source_flat, bin_centers, lookup_table)

    return matched.reshape(orig_shape).astype(orig_dtype)


def match_histograms_cv2_alt(image, reference):
    """
    Adjust the pixel values of 'image' to match the histogram of 'reference'.

    Args:
        image (np.ndarray): Source image (uint8, uint16, or float32).
        reference (np.ndarray): Reference image to match.

    Returns:
        np.ndarray: The matched image in the original input dtype.
    """
    # Determine if image is RGB or Grayscale
    is_rgb = len(image.shape) == 3 and image.shape[2] == 3

    if is_rgb:
        # For RGB (H&E or PAS), process channels independently
        # Alternatively, convert to LAB and match only the 'L' channel
        matched = np.empty_like(image)
        for i in range(3):
            matched[..., i] = _match_channel(image[..., i], reference[..., i])
        return matched
    return _match_channel(image, reference)


def reduce(
    arrays: np.ndarray,
    reduce_func: ty.Literal["sum", "mean", "max"] = "sum",
    *,
    match_histogram: bool = True,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Reduction method."""
    if match_histogram:
        ref = arrays[0] if reference is None else reference

        for i in range(len(arrays)):
            if ref is not arrays[i]:
                arrays[i] = match_histograms_cv2_alt(arrays[i], ref)

    arrays = np.stack(arrays, axis=0)
    if reduce_func == "sum":
        return np.asarray(arrays.sum(axis=0, dtype=arrays.dtype))
    if reduce_func == "mean":
        return np.asarray(arrays.mean(axis=0, dtype=arrays.dtype))
    if reduce_func == "max":
        return np.asarray(arrays.max(axis=0))
    raise ValueError(f"Invalid reduce function: {reduce_func}")


def combine(
    name: str,
    paths: list[PathLike],
    output_dir: PathLike,
    as_uint8: bool | None = None,
    overwrite: bool = False,
    reduce_func: ty.Literal["sum", "mean", "max"] = "max",
    match_histogram: bool = True,
) -> Path:
    """Combine multiple images."""
    from image2image_io.readers import get_simple_reader
    from image2image_io.writers import OmeTiffWriter

    paths = [Path(path) for path in paths]
    output_dir = Path(output_dir)

    pixel_sizes = []
    channel_names = []
    image_shapes = []
    is_rgb = []
    readers = []
    with MeasureTimer() as timer:
        for path_ in paths:
            path = Path(path_)
            reader = get_simple_reader(path, init_pyramid=False, auto_pyramid=False)
            readers.append(reader)
            pixel_sizes.append(round(reader.resolution, 3))
            channel_names.append(tuple(reader.channel_names))
            image_shapes.append(reader.image_shape)
            is_rgb.append(reader.is_rgb)

    # check that all images have the same shape
    if len(set(image_shapes)) > 1:
        logger.error(f"All images must have the same shape to be combined. ({image_shapes})")
        raise ValueError(f"All images must have the same shape to be combined. ({image_shapes})")
    if len(set(pixel_sizes)) > 1:
        logger.error(f"All images must have the same pixel size to be combined. ({pixel_sizes})")
        raise ValueError(f"All images must have the same pixel size to be combined. ({pixel_sizes})")
    if len(set(channel_names)) > 1:
        logger.error(f"All images must have the same channel names to be combined. ({channel_names})")
        raise ValueError(f"All images must have the same channel names to be combined. ({channel_names})")
    logger.info(f"Loaded {len(channel_names)} images in {timer()}.")

    # if the image is RGB, we need to export it slightly differently than if it's a multi-channel image
    is_rgb = is_rgb[0]

    output_filename = output_dir / f"{name}-{reduce_func}.ome.tiff"
    writer = OmeTiffWriter(reader=readers[0])
    with writer.exporter(
        name=f"{name}-{reduce_func}",
        output_dir=output_dir,
        as_uint8=as_uint8,
        write_pyramid=True,
        overwrite=overwrite,
    ) as writer:
        if not writer:
            logger.error("Failed to create writer.")
            return None

        if is_rgb:
            writer.append_rgb(
                reduce(
                    [reader.get_channel(0, pyramid=0, split_rgb=False) for reader in readers],
                    reduce_func=reduce_func,
                    match_histogram=match_histogram,
                )
            )
        else:
            for channel_id, channel_name in enumerate(channel_names):
                writer.append_channel(
                    channel_id,
                    channel_name,
                    reduce(
                        [reader.get_channel(channel_id, pyramid=0) for reader in readers],
                        reduce_func=reduce_func,
                        match_histogram=match_histogram,
                    ),
                )
    return output_filename
