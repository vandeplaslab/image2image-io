"""Tiff file wrapper.

Copied from:
https://github.com/NHPatterson/napari-imsmicrolink/blob/master/src/napari_imsmicrolink/data/tifffile_reader.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

from koyo.typing import PathLike
from loguru import logger
from ome_types import from_xml
from tifffile import TiffFile

from image2image_io.config import CONFIG
from image2image_io.readers._base_reader import BaseReader
from image2image_io.readers.tiff_utils import (
    ometiff_ch_names,
    ometiff_xy_pixel_sizes,
    qptiff_channel_names,
    svs_xy_pixel_sizes,
    tifftag_xy_pixel_sizes,
)
from image2image_io.readers.utilities import get_tifffile_info, tifffile_to_dask
from image2image_io.utils.utilities import guess_rgb

logger = logger.bind(src="Tiff")


class TiffImageReader(BaseReader):
    """TIFF image wrapper."""

    fh: TiffFile
    reader = "tifffile"

    def __init__(
        self,
        path: PathLike,
        key: str | None = None,
        init_pyramid: bool | None = None,
        auto_pyramid: bool | None = None,
    ):
        super().__init__(path, key, auto_pyramid=auto_pyramid)
        self.fh = TiffFile(self.path)

        self.shape, self._array_dtype, self.largest_series = self._get_image_info()
        self._image_shape = self.shape[0:2] if self.is_rgb else (self.shape[1::] if len(self.shape) > 2 else self.shape)

        self.resolution = self._get_im_res()
        self._channel_names = self._get_channel_names()
        self._channel_colors = None
        CONFIG.trace(
            f"{path}: RGB={self.is_rgb}; dims={self.shape}; px={self.resolution:.3f}; n_ch={len(self._channel_names)}"
        )
        init_pyramid = init_pyramid if init_pyramid is not None else CONFIG.init_pyramid
        if init_pyramid:
            self._pyramid = self.pyramid

    @property
    def n_channels(self) -> int:
        """Return number of channels."""
        if self.is_rgb:
            return self.shape[2]
        _, n_channels = self.get_channel_axis_and_n_channels()
        return n_channels

    def get_dask_pyr(self) -> list:
        """Get instance of Dask pyramid."""
        d_pyr = tifffile_to_dask(self.path, self.largest_series)
        channel_axis, _ = self.get_channel_axis_and_n_channels(shape=d_pyr[0].shape)
        if self.is_rgb and guess_rgb(d_pyr[0].shape):
            d_pyr[0] = d_pyr[0].rechunk((2048, 2048, 1))
        elif len(d_pyr[0].shape) > 2:
            if channel_axis == 0:
                d_pyr[0] = d_pyr[0].rechunk((1, 2048, 2048))
            else:
                d_pyr[0] = d_pyr[0].rechunk((2048, 2048, 1))
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
                    "Unable to parse pixel resolution information from file defaulting to 1",
                    stacklevel=2,
                )
                return 1.0

    def _get_channel_names(self) -> list[str]:
        channel_names = []
        if any(suffix in self.path.name for suffix in [".qptiff", ".qptiff.intermediate", ".qptiff.raw"]):
            channel_names = qptiff_channel_names(self.fh)
        if self.fh.ome_metadata and not channel_names:
            channel_names = ometiff_ch_names(from_xml(self.fh.ome_metadata, parser="lxml"), self.largest_series)
            if self.n_channels > len(channel_names):
                channel_names = []

        if not channel_names or len(channel_names) != self.n_channels:
            if self.is_rgb:
                channel_names = ["R", "G", "B"] if CONFIG.split_rgb else ["RGB"]
            else:
                channel_names = []
                for idx, _ch in enumerate(range(self.n_channels)):
                    channel_names.append(f"C{str(idx + 1).zfill(2)}")
        return channel_names

    def _get_image_info(self) -> tuple:
        if len(self.fh.series) > 1:
            warnings.warn(
                "The tiff contains multiple series, the largest series will be read by default",
                stacklevel=2,
            )

        array_shape, _, array_dtype, largest_series = get_tifffile_info(self.path)
        if not CONFIG.auto_pyramid:
            largest_series = 0
        return array_shape, array_dtype, largest_series

    def get_channel_axis_and_n_channels(self, shape: tuple | None = None) -> tuple[int | None, int]:
        """Return channel axis and number of channels."""
        if shape is None:
            shape = self.shape
        ndim = len(shape)
        # 2D images will be returned as they are
        if ndim == 3:
            if self.is_rgb:
                return 2, 3
            # elif np.argmin(shape) == 2:
            #     return 2, shape[2]
            else:
                return 0, shape[0]
        return None, 1
