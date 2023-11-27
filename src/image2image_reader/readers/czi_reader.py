"""CZI reader."""
from __future__ import annotations

import numpy as np
import zarr
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from tifffile import xml2dict

from image2image_reader.config import CONFIG
from image2image_reader.readers._base_reader import BaseReader
from image2image_reader.readers._czi import CziFile, CziSceneFile
from image2image_reader.readers.utilities import guess_rgb

logger = logger.bind(src="CZI")


class CziImageReader(BaseReader):
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
        self.im_dims = tuple(self.im_dims)
        self._is_rgb = guess_rgb(self.im_dims)

        czi_meta = xml2dict(self.fh.metadata())
        pixel_scaling_str = czi_meta["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][0]["Value"]
        pixel_scaling = float(pixel_scaling_str) * 1_000_000
        self.resolution = pixel_scaling
        channels_meta = czi_meta["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]
        logger.trace(f"{path}: RGB={self.is_rgb}; dims={self.im_dims}; px={self.resolution}")

        channel_names: list[str] = []
        for ch in channels_meta:
            if isinstance(ch, dict):
                channel_names.append(str(ch.get("ShortName")))
            else:
                channel_names.append(str(ch))
        self._channel_names = channel_names

        init_pyramid = init_pyramid if init_pyramid is not None else CONFIG.init_pyramid
        if init_pyramid:
            with MeasureTimer() as timer:
                self._pyramid = self.pyramid
            logger.trace(f"{path}: pyramid={len(self._pyramid)} in {timer()}")

    def get_dask_pyr(self) -> list:
        """Get instance of Dask pyramid."""
        auto_pyramid = self.auto_pyramid if self.auto_pyramid is not None else CONFIG.auto_pyramid
        return self.fh.zarr_pyramidalize_czi(zarr.storage.TempStore(), auto_pyramid)

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


class CziSceneImageReader(BaseReader):
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

        czi_meta = xml2dict(self.fh.metadata())
        pixel_scaling_str = czi_meta["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][0]["Value"]
        pixel_scaling = float(pixel_scaling_str) * 1_000_000
        self.resolution = pixel_scaling
        logger.trace(f"{path}: RGB={self.is_rgb}; dims={self.im_dims}; scene={scene_index}; px={pixel_scaling}")

        channel_names = []
        channels_meta = czi_meta["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]
        for ch in channels_meta:
            if isinstance(ch, dict):
                channel_names.append(str(ch.get("ShortName")))
            else:
                channel_names.append(str(ch))
        self._channel_names = channel_names

        init_pyramid = init_pyramid if init_pyramid is not None else CONFIG.init_pyramid
        if init_pyramid:
            with MeasureTimer() as timer:
                self._pyramid = self.pyramid
            logger.trace(f"{path}: pyramid={len(self._pyramid)} in {timer()}")

    def get_dask_pyr(self) -> list:
        """Get instance of Dask pyramid."""
        auto_pyramid = self.auto_pyramid if self.auto_pyramid is not None else CONFIG.auto_pyramid
        return self.fh.zarr_pyramidalize_czi(zarr.storage.TempStore(), auto_pyramid)

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
