"""Tiff file wrapper.

Copied from:
https://github.com/NHPatterson/napari-imsmicrolink/blob/master/src/napari_imsmicrolink/data/tifffile_reader.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

from loguru import logger
from ome_types import from_xml
from tifffile import TiffFile

from image2image_io.config import CONFIG
from image2image_io.readers._base_reader import BaseReader
from image2image_io.readers.tiff_utils import (
    ometiff_ch_names,
    ometiff_xy_pixel_sizes,
    svs_xy_pixel_sizes,
    tifftag_xy_pixel_sizes,
)
from image2image_io.readers.utilities import get_tifffile_info, guess_rgb, tf_zarr_read_single_ch, tifffile_to_dask

logger = logger.bind(src="Tiff")


class TiffImageReader(BaseReader):
    """TIFF image wrapper."""

    fh: TiffFile
    reader = "tifffile"

    def __init__(
        self,
        path,
        key: str | None = None,
        init_pyramid: bool | None = None,
        auto_pyramid: bool | None = None,
    ):
        super().__init__(path, key, auto_pyramid=auto_pyramid)
        self.fh = TiffFile(self.path)

        self.im_dims, self.im_dtype, self.largest_series = self._get_image_info()
        self.im_dims = tuple(self.im_dims)
        self._is_rgb = guess_rgb(self.im_dims)

        self.resolution = self._get_im_res()
        self._channel_names = self._get_channel_names()
        self._channel_colors = None
        logger.trace(
            f"{path}: RGB={self.is_rgb}; dims={self.im_dims}; px={self.resolution}; n_ch={len(self._channel_names)}"
        )
        init_pyramid = init_pyramid if init_pyramid is not None else CONFIG.init_pyramid
        if init_pyramid:
            self._pyramid = self.pyramid

    @property
    def n_channels(self):
        """Return number of channels."""
        return self.im_dims[2] if self.is_rgb else self.im_dims[0]

    def get_dask_pyr(self):
        """Get instance of Dask pyramid."""
        d_pyr = tifffile_to_dask(self.path, self.largest_series)
        if self.is_rgb and guess_rgb(d_pyr[0].shape) is True:
            d_pyr[0] = d_pyr[0].rechunk((2048, 2048, 1))
        elif len(d_pyr[0].shape) > 2:
            d_pyr[0] = d_pyr[0].rechunk((1, 2048, 2048))
        else:
            d_pyr[0] = d_pyr[0].rechunk((2048, 2048))
        return d_pyr

    def _get_im_res(self) -> float:
        if Path(self.path).suffix.lower() in [".scn", ".ndpi"]:
            return tifftag_xy_pixel_sizes(
                self.fh,
                self.largest_series,
                0,
            )[0]
        elif Path(self.path).suffix.lower() in [".svs"]:
            return svs_xy_pixel_sizes(
                self.fh,
                self.largest_series,
                0,
            )[0]
        elif self.fh.ome_metadata:
            return ometiff_xy_pixel_sizes(
                from_xml(self.fh.ome_metadata, parser="lxml"),
                self.largest_series,
            )[0]
        else:
            try:
                return tifftag_xy_pixel_sizes(
                    self.fh,
                    self.largest_series,
                    0,
                )[0]
            except KeyError:
                warnings.warn(
                    "Unable to parse pixel resolution information from file" " defaulting to 1",
                    stacklevel=2,
                )
                return 1.0

    def _get_channel_names(self):
        channel_names = []
        if self.fh.ome_metadata:
            channel_names = ometiff_ch_names(from_xml(self.fh.ome_metadata, parser="lxml"), self.largest_series)
            if self.n_channels > len(channel_names):
                channel_names = []

        if not channel_names:
            channel_names = []
            for idx, _ch in enumerate(range(self.n_channels)):
                channel_names.append(f"C{str(idx + 1).zfill(2)}")
        return channel_names

    def _get_image_info(self):
        if len(self.fh.series) > 1:
            warnings.warn(
                "The tiff contains multiple series, the largest series will be read by default",
                stacklevel=2,
            )

        im_dims, im_dtype, largest_series = get_tifffile_info(self.path)
        if not CONFIG.auto_pyramid:
            largest_series = 0
        return im_dims, im_dtype, largest_series
