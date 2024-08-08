"""Merge utility class."""

from __future__ import annotations

import typing as ty
from pathlib import Path
from warnings import warn

import numpy as np
from loguru import logger

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader


class MergeImages:
    """Merge multiple images."""

    def __init__(
        self,
        paths_or_readers: list[Path | str | BaseReader],
        pixel_sizes: list[int | float],
        channel_names: list[list[str]] | None = None,
        channel_ids: list[list[int]] | None = None,
        channel_colors: list[list[str]] | None = None,
    ):
        from image2image_io.readers import BaseReader, get_simple_reader

        if not isinstance(paths_or_readers, list):
            raise ValueError("MergeImages requires a list of images to merge")
        if not isinstance(pixel_sizes, list):
            raise ValueError("MergeImages requires a list of image resolutions for each image to merge")

        if channel_names is None:
            channel_names = [None] * len(paths_or_readers)
        if channel_ids is None:
            channel_ids = [None] * len(paths_or_readers)
        if channel_colors is None:
            channel_colors = [None] * len(paths_or_readers)

        readers = []
        for _index, (path, pixel_size, channel_names_, channel_ids_, channel_colors_) in enumerate(
            zip(paths_or_readers, pixel_sizes, channel_names, channel_ids, channel_colors)
        ):
            if isinstance(path, BaseReader):
                reader = path
            else:
                reader: BaseReader = get_simple_reader(path)
            if channel_colors_:
                reader._channel_colors = channel_colors_
            if channel_names_:
                reader._channel_names = channel_names_
            if channel_ids_:
                reader._channel_ids = channel_ids_
            reader.resolution = pixel_size
            if reader.channel_names is None:  # or len(reader.channel_names) != reader.n_channels:
                reader._channel_names = [f"C{idx}" for idx in range(0, reader.n_channels)]
                logger.trace(
                    f"Channel names not provided for {reader.path}, using {reader.channel_names} ({channel_names_})"
                )
            readers.append(reader)
        if not all(im.dtype == readers[0].dtype for im in readers):
            warn("MergeImages created with mixed data types, writing will cast to the largest data type", stacklevel=2)
        if any(im.is_rgb for im in readers):
            warn(
                "MergeImages does not support writing merged interleaved RGB. Data will be written as multi-channel",
                stacklevel=2,
            )

        self.readers: list[BaseReader] = readers
        self.paths: list[str | Path | BaseReader] = paths_or_readers
        self.dtype: np.dtype = self.readers[0].dtype
        self.is_rgb: bool = False
        self.n_channels: int = np.sum([reader.n_channels for reader in self.readers])
        self.channel_names: list[list[str]] = [reader.channel_names for reader in self.readers]
        self.channel_ids: list[list[int]] = [reader.channel_ids for reader in self.readers]
        self.original_size_transform = None

    @property
    def channel_names_flat(self) -> list[str]:
        """Return flat channel names."""
        return [name for names in self.channel_names for name in names]
