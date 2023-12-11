"""Merge utility class."""
from __future__ import annotations

from pathlib import Path
from warnings import warn

import numpy as np

from image2image_io._reader import get_simple_reader
from image2image_io.readers import BaseReader


class MergeImages:
    """Merge multiple images."""

    def __init__(
        self,
        paths: list[Path | str],
        pixel_sizes: list[int | float],
        channel_names: list[list[str]] | None = None,
        channel_colors: list[list[str]] | None = None,
    ):
        if not isinstance(paths, list):
            raise ValueError("MergeImages requires a list of images to merge")
        if not isinstance(pixel_sizes, list):
            raise ValueError("MergeImages requires a list of image resolutions for each image to merge")

        if channel_names is None:
            channel_names = [None] * len(paths)
        if channel_colors is None:
            channel_colors = [None] * len(paths)

        readers = []
        for _index, image_data in enumerate(zip(paths, pixel_sizes, channel_names, channel_colors)):
            path, pixel_size, channel_names_, channel_colors_ = image_data
            reader: BaseReader = get_simple_reader(path)
            reader._channel_colors = channel_colors_
            reader._channel_names = channel_names_
            reader.resolution = pixel_size
            if reader.channel_names is None or len(reader.channel_names) != reader.n_channels:
                reader._channel_names = [f"C{idx}" for idx in range(0, reader.n_channels)]
            readers.append(reader)

        if all(im.dtype == readers[0].dtype for im in readers) is False:
            warn("MergeImages created with mixed data types, writing will cast to the largest data type")
        if any(im.is_rgb for im in readers) is True:
            warn("MergeImages does not support writing merged interleaved RGB. Data will be written as multi-channel")

        self.readers = readers
        self.paths = paths
        self.dtype = self.readers[0].dtype
        self.is_rgb = False

        self.n_channels = np.sum([reader.n_channels for reader in self.readers])
        self.channel_names = [reader.channel_names for reader in self.readers]
        self.original_size_transform = None
