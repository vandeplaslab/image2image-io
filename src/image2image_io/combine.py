"""Combine multiple files into one.

Unlike the merge method, which appends image channels from multiple files, this method combines files
by adding them together (or max int projection) to create a single image.
"""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger


def reduce(arrays: list[np.ndarray], reduce_func: ty.Literal["sum", "mean", "max"] = "sum") -> np.ndarray:
    """Reduce arrays."""
    print(arrays)
    if reduce_func == "sum":
        return np.sum(arrays, axis=0, dtype=arrays[0].dtype)
    if reduce_func == "mean":
        return np.mean(arrays, axis=0, dtype=arrays[0].dtype)
    if reduce_func == "max":
        return np.max(arrays, axis=0)  # , dtype=arrays[0].dtype)
    raise ValueError(f"Invalid reduce function: {reduce_func}")


def combine(
    name: str,
    paths: list[PathLike],
    output_dir: PathLike,
    as_uint8: bool | None = None,
    overwrite: bool = False,
    reduce_func: ty.Literal["sum", "mean", "max"] = "max",
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
