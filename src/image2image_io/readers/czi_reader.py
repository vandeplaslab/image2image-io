"""CZI reader."""
from __future__ import annotations

import typing as ty

import numpy as np
import zarr
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger

from image2image_io.config import CONFIG
from image2image_io.readers._base_reader import BaseReader
from image2image_io.readers._czi import CziFile, CziSceneFile
from image2image_io.readers.utilities import guess_rgb

logger = logger.bind(src="CZI")


class CziMixin:
    """Mixin class for CZI images."""

    fh: CziFile | CziSceneFile
    im_dims: tuple[int, ...]
    is_rgb: bool
    _is_rgb: bool | None = None
    auto_pyramid: bool | None = None
    n_channels: int

    def _get_channel_names(self) -> list[str]:
        """Return list of channel names."""
        czi_meta = self.xml_metadata()
        channels_meta = czi_meta["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]
        channel_names: list[str] = []
        for ch in channels_meta:
            if isinstance(ch, dict):
                channel_names.append(str(ch.get("ShortName")))
            else:
                channel_names.append(str(ch))
        if len(channel_names) != self.n_channels:
            # logger.warning(
            #     f"Number of channels ({self.n_channels}) does not match number of channel names ({channel_names})"
            # )
            channel_names = []
            if self.is_rgb:
                channel_names = ["R", "G", "B"]
            else:
                for idx, _ch in enumerate(range(self.n_channels)):
                    channel_names.append(f"C{str(idx + 1).zfill(2)}")
        return channel_names

    def _get_pixel_size(self) -> float:
        czi_meta = self.xml_metadata()
        pixel_scaling_str = czi_meta["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][0]["Value"]
        return float(pixel_scaling_str) * 1_000_000

    def xml_metadata(self) -> dict[str, ty.Any]:
        """Return XML metadata."""
        from image2image_io.utils.utilities import xmlstr_to_dict

        # from tifffile import xml2dict

        return xmlstr_to_dict(self.fh.metadata())

    def _get_image_info(self) -> tuple:
        # if RGB need to get 0
        if self.fh.shape[-1] > 1:
            ch_dim_idx = self.fh.axes.index("0")
        else:
            ch_dim_idx = self.fh.axes.index("C")
        y_dim_idx = self.fh.axes.index("Y")
        x_dim_idx = self.fh.axes.index("X")
        if self.fh.shape[-1] > 1:
            im_dims = np.array(self.fh.shape)[[y_dim_idx, x_dim_idx, ch_dim_idx]]
        else:
            im_dims = np.array(self.fh.shape)[[ch_dim_idx, y_dim_idx, x_dim_idx]]
        return ch_dim_idx, y_dim_idx, x_dim_idx, im_dims, self.fh.dtype


class CziImageReader(BaseReader, CziMixin):  # type: ignore[misc]
    """CZI file wrapper."""

    fh: CziFile

    def __init__(
        self,
        path: PathLike,
        key: str | None = None,
        init_pyramid: bool | None = None,
        auto_pyramid: bool | None = None,
    ):
        super().__init__(path, key, auto_pyramid=auto_pyramid)
        self.fh = CziFile(self.path)
        self.n_scenes = CziSceneFile.get_num_scenes(self.path)

        *_, self.im_dims, self._im_dtype = self._get_image_info()
        self.im_dims = self._image_shape = tuple(self.im_dims)
        self._is_rgb = guess_rgb(self.im_dims)
        self.resolution = self._get_pixel_size()
        self._channel_names = self._get_channel_names()
        logger.trace(
            f"{path}: RGB={self.is_rgb}; dims={self.im_dims}; px={self.resolution:.3f}; n_ch={len(self._channel_names)}"
        )

        init_pyramid = init_pyramid if init_pyramid is not None else CONFIG.init_pyramid
        if init_pyramid:
            with MeasureTimer() as timer:
                self._pyramid = self.pyramid
            logger.trace(f"{path}: pyramid={len(self._pyramid)} in {timer()}")

    @property
    def n_channels(self):
        """Return number of channels."""
        return self.im_dims[2] if self.is_rgb else self.im_dims[0]

    def get_dask_pyr(self) -> list:
        """Get instance of Dask pyramid."""
        auto_pyramid = self.auto_pyramid if self.auto_pyramid is not None else CONFIG.auto_pyramid
        return self.fh.zarr_pyramidize_czi(zarr.storage.TempStore(), auto_pyramid)


class CziSceneImageReader(BaseReader, CziMixin):  # type: ignore[misc]
    """Multi-scene reader."""

    fh: CziSceneFile

    def __init__(
        self,
        path: PathLike,
        key: str | None = None,
        scene_index: int = 0,
        init_pyramid: bool | None = None,
        auto_pyramid: bool | None = None,
    ):
        super().__init__(path, key, reader_kws={"scene_index": scene_index}, auto_pyramid=auto_pyramid)
        self.fh = CziSceneFile(self.path, scene_index=scene_index)

        *_, self.im_dims, self._im_dtype = self._get_image_info()
        self.im_dims = tuple(self.im_dims)
        self._is_rgb = self.fh.is_rgb
        self.resolution = self._get_pixel_size()
        self._channel_names = self._get_channel_names()
        logger.trace(
            f"{path}: RGB={self.is_rgb}; dims={self.im_dims}; px={self.resolution:.3f}; n_ch={len(self._channel_names)}"
            f"; scene={scene_index}"
        )

        init_pyramid = init_pyramid if init_pyramid is not None else CONFIG.init_pyramid
        if init_pyramid:
            with MeasureTimer() as timer:
                self._pyramid = self.pyramid
            logger.trace(f"{path}: pyramid={len(self._pyramid)} in {timer()}")

    @property
    def n_channels(self):
        """Return number of channels."""
        return self.im_dims[2] if self.is_rgb else self.im_dims[0]

    def get_dask_pyr(self) -> list:
        """Get instance of Dask pyramid."""
        auto_pyramid = self.auto_pyramid if self.auto_pyramid is not None else CONFIG.auto_pyramid
        return self.fh.zarr_pyramidize_czi(zarr.storage.TempStore(), auto_pyramid)
