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


def reduce(
    arrays: np.ndarray,
    reduce_func: ty.Literal["sum", "mean", "max"] = "sum",
    *,
    match_histogram: bool = True,
    reference: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    arrays = np.asarray(arrays)
    if arrays.ndim != 3:
        raise ValueError("Expected shape (n_images, H, W)")

    if match_histogram:
        ref = arrays[0] if reference is None else reference
        arrays = np.stack(
            [
                ref if i == 0 and reference is None else match_histograms_cv2(arrays[i], ref, mask=mask)
                for i in range(arrays.shape[0])
            ],
            axis=0,
        )

    if reduce_func == "sum":
        return arrays.sum(axis=0)
    if reduce_func == "mean":
        return arrays.mean(axis=0)
    if reduce_func == "max":
        return arrays.max(axis=0)
    raise ValueError(f"Invalid reduce function: {reduce_func}")


def combine(
    name: str,
    paths: list[PathLike],
    output_dir: PathLike,
    as_uint8: bool | None = None,
    overwrite: bool = False,
    reduce_func: ty.Literal["sum", "mean", "max"] = "sum",
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
            pixel_sizes.append(reader.resolution)
            channel_names.append(reader.channel_names)
            image_shapes.append(reader.image_shape)
            is_rgb.append(reader.is_rgb)

    # check that all images have the same shape
    if len(set(image_shapes)) > 1:
        logger.error("All images must have the same shape to be combined.")
        raise ValueError("All images must have the same shape to be combined.")
    if len(set(pixel_sizes)) > 1:
        logger.error("All images must have the same pixel size to be combined.")
        raise ValueError("All images must have the same pixel size to be combined.")
    if len(set(channel_names)) > 1:
        logger.error("All images must have the same channel names to be combined.")
        raise ValueError("All images must have the same channel names to be combined.")
    logger.info(f"Loaded {len(channel_names)} images in {timer()}.")

    # if the image is RGB, we need to export it slightly differently than if it's a multi-channel image
    is_rgb = is_rgb[0]

    output_filename = output_dir / f"{name}_combined.ome.tiff"
    writer = OmeTiffWriter(reader=readers[0])
    with writer.exporter(
        name=f"{name}_combined",
        output_dir=output_dir,
        as_uint8=as_uint8,
        write_pyramid=True,
        overwrite=overwrite,
    ) as writer:
        if not writer:
            logger.error("Failed to create writer.")
            return None

        if not is_rgb:
            for channel_id, channel_name in enumerate(channel_names):
                writer.append_channel(
                    channel_id,
                    channel_name,
                    reduce([reader.get_channel(channel_id, pyramid=0) for reader in readers], reduce_func=reduce_func),
                )
        else:
            writer.append_rgb(
                reduce(
                    [reader.get_channel(0, pyramid=0, split_rgb=False) for reader in readers], reduce_func=reduce_func
                )
            )
    return output_filename
