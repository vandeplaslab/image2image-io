"""Base image wrapper."""

from __future__ import annotations

import math
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger

from image2image_io.config import CONFIG
from image2image_io.enums import DEFAULT_TRANSFORM_NAME
from image2image_io.models.transform import TransformData
from image2image_io.readers.utilities import guess_rgb

logger = logger.bind(src="Reader")


def check_if_open(path: Path) -> None:
    """Check if file is open."""
    import psutil

    # using psutil to check if file is open in the current process
    current_process_id = psutil.Process().pid
    proc = psutil.Process(current_process_id)
    for file in proc.open_files():
        if file.path == str(path):
            logger.warning(f"File {path} is still open in the current process...")


class BaseReader:
    """Base class for some of the other image readers."""

    _pyramid = None
    _is_rgb: bool | None = None
    _im_dtype: np.dtype | None = None
    _im_shape: tuple[int, ...] | None = None
    _image_shape: tuple[int, int] | None = None
    auto_pyramid: bool | None = None
    reader: str = "base"
    reader_type: str = "image"
    n_scenes: int = 1
    _scene_ids: list[int] | None = None
    lazy: bool = False
    fh: ty.Any | None = None
    allow_extraction: bool = False
    _resolution: float = 1.0
    _channel_names: list[str] | None = None
    _channel_colors: list[str] | None = None
    _channel_ids: list[int] | None = None

    def __init__(
        self, path: PathLike, key: str | None = None, reader_kws: dict | None = None, auto_pyramid: bool | None = None
    ):
        # This is the direct path to the image
        self.path = Path(path)
        # This is the attribute we will use to identify the image
        self.auto_pyramid = auto_pyramid
        self.key = key or self.path.name
        self.reader_kws = reader_kws or {}
        self.transform_data: TransformData = TransformData()
        self.transform_name = DEFAULT_TRANSFORM_NAME

    def find_channel_name(self, name: str) -> int | None:
        """Find cycle by name or part of the name."""
        if name in self.channel_names:
            return self.channel_names.index(name)
        for i, n in enumerate(self.channel_names):
            if name in n:
                return i
        return None

    def print_info(self):
        """Print information about the image."""
        print(f"Image: {self.path}")
        print(f"  - Resolution: {self.resolution}")
        print(f"  - Shape: {self.shape}")
        print(f"  - Channels: {self.n_channels}")
        print(f"  - RGB: {self.is_rgb}")
        for i, channel in enumerate(self.channel_names):
            print(f"  - Channel {i}: {channel}")
        print(f"  - Transform: {self.transform}")

    @property
    def transform(self) -> np.ndarray:
        """Return transform."""
        transform: np.ndarray = self.transform_data.transform.params
        return transform

    @transform.setter
    def transform(self, value: np.ndarray) -> None:
        assert value.shape == (3, 3)
        self.transform_data._transform = value

    def is_identity_transform(self) -> bool:
        """Return whether transform is identity."""
        if self.transform_data.transform:
            return np.allclose(self.transform_data.transform.params, np.eye(3))
        return np.allclose(self.transform, np.eye(3))

    @property
    def inv_resolution(self) -> float:
        """Return inverse resolution."""
        return 1 / self.resolution

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape of the image, including channels, etc."""
        if self._im_shape is None:
            self._im_shape = self.pyramid[0].shape
        return self._im_shape

    @property
    def image_shape(self) -> tuple[int, int]:
        """Image shape."""
        if self._image_shape is None:
            from image2image_io.utils.utilities import get_shape_of_image

            self._image_shape = get_shape_of_image(self.shape)[-1]
        return self._image_shape

    @property
    def channel_names(self) -> list[str]:
        """Return channel names."""
        return self._channel_names

    @property
    def channel_ids(self) -> list[int]:
        """Return channel indices."""
        if self._channel_ids is None:
            self._channel_ids = list(range(len(self._channel_names)))
        return self._channel_ids

    @channel_ids.setter
    def channel_ids(self, value: list[int]) -> None:
        """Set channel indices."""
        if not isinstance(value, list):
            value = list(value)
        assert len(value) <= len(
            self._channel_names
        ), f"Too many channels selected: {len(value)} > {len(self._channel_names)}"
        assert max(value) < len(
            self._channel_names
        ), f"Channel index out of range: {max(value)} > {len(self._channel_names)}"
        self._channel_ids = value

    @property
    def scene_ids(self) -> list[int]:
        """Return channel indices."""
        if self._scene_ids is None:
            self._scene_ids = list(range(self.n_scenes))
        return self._scene_ids

    @scene_ids.setter
    def scene_ids(self, value: list[int]) -> None:
        """Set channel indices."""
        if not isinstance(value, list):
            value = list(value)
        assert len(value) <= self.n_scenes, f"Too many channels selected: {len(value)} > {self.n_scenes}"
        assert max(value) < self.n_scenes, f"Channel index out of range: {max(value)} > {self.n_scenes}"
        self._scene_ids = value

    @property
    def channel_colors(self) -> list[str]:
        """Return channel names."""
        return self._channel_colors

    def channel_to_index(self, channel: str) -> int:
        """Return index of a channel."""
        if channel in self.channel_names:
            return self.channel_names.index(channel)
        return -1

    @property
    def n_channels(self) -> int:
        """Return number of channels."""
        return len(self.channel_names)

    @property
    def dtype(self) -> np.dtype:
        """Return dtype."""
        if self._im_dtype is None:
            self._im_dtype = self.pyramid[0].dtype
        return self._im_dtype

    @property
    def is_rgb(self) -> bool:
        """Return whether image is RGB."""
        if self._is_rgb is None:
            self._is_rgb = guess_rgb(self.pyramid[0].shape)
        return self._is_rgb

    @property
    def scale(self) -> tuple[float, float]:
        """Return scale."""
        return self.resolution, self.resolution

    @property
    def resolution(self) -> float:
        """Return resolution."""
        return self._resolution or 1.0

    @resolution.setter
    def resolution(self, value: float) -> None:
        self._resolution = value
        if self.transform_data:
            self.transform_data.moving_resolution = value

    @property
    def name(self) -> str:
        """Return name of the input path."""
        return self.path.name

    @property
    def stem(self) -> str:
        """Return name of the input path."""
        return self.path.stem

    def flat_array(
        self, channel_indices: list[int] | None = None, index: int = 0
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Return a flat array."""
        from image2image_io.utils.utilities import get_shape_of_image

        array = self.pyramid[index]
        if hasattr(array, "compute"):
            array = array.compute()

        if channel_indices is not None:
            channel_indices = list(channel_indices)
            assert all(channel_index in self.channel_ids for channel_index in channel_indices), "Invalid channel index"
        else:
            channel_indices = self.channel_ids

        # sub-select channels
        if array.ndim == 3:
            if self.is_rgb:
                array = array[:, :, channel_indices]
            else:
                array = array[channel_indices, :, :]

        # reshape array
        n_channels, _, shape = get_shape_of_image(array)
        if array.ndim == 3:
            array = array.reshape(-1, n_channels)
        else:
            array = array.reshape(-1, 1)
        return array, shape

    def close(self) -> None:
        """Close the file handle."""
        if self.fh and hasattr(self.fh, "close") and callable(self.fh.close):
            self.fh.close()
        if self.fh and hasattr(self.fh, "filehandle"):
            self.fh.filehandle.close()
        self.fh = None
        self._pyramid = None
        # check_if_open(self.path)
        logger.trace("Closed file handles")

    @property
    def pyramid(self) -> list:
        """Pyramid."""
        if self._pyramid is None:
            self._pyramid = self.get_dask_pyr()
        return self._pyramid

    def get_dask_pyr(self) -> list[ty.Any]:
        """Get dask representation of the pyramid."""
        raise NotImplementedError("Must implement method")

    def crop(self, left: int, right: int, top: int, bottom: int) -> np.ndarray:
        """Crop image."""
        top, bottom = sorted([top, bottom])
        left, right = sorted([left, right])
        left = math.floor(left * self.inv_resolution)
        right = math.ceil(right * self.inv_resolution)
        top = math.floor(top * self.inv_resolution)
        bottom = math.ceil(bottom * self.inv_resolution)
        array = self.pyramid[0]
        if array.ndim == 2:
            array_ = array[top:bottom, left:right]
        elif array.ndim == 3:
            shape = array.shape
            channel_axis, _ = self.get_channel_axis_and_n_channels()
            # channel_axis = int(np.argmin(shape))
            if channel_axis == 0:
                array_ = array[:, top:bottom, left:right]
            elif channel_axis == 1:
                array_ = array[top:bottom, :, left:right]
            elif channel_axis == 2:
                array_ = array[top:bottom, left:right, :]
            else:
                raise ValueError(f"Array has unsupported shape: {array.shape}")
        else:
            raise ValueError(f"Array has unsupported shape: {array.shape}")
        # check whether an array is dask array - if so, we need to compute it
        if hasattr(array_, "compute"):
            array_ = array_.compute()
        return array_  # type: ignore[no-any-return]

    def warp(self, array: np.ndarray) -> np.ndarray:
        """Warp array."""
        from image2image_io.utils.mask import transform_mask

        transform = self.transform_data.compute(yx=True, px=True).params
        transformed_mask = transform_mask(array, transform, self.image_shape)
        return transformed_mask

    def get_channel_axis_and_n_channels(self, shape: tuple | None = None) -> tuple[int | None, int]:
        """Return channel axis and number of channels."""
        if shape is None:
            shape = self.shape
        ndim = len(shape)
        # 2D images will be returned as they are
        if ndim == 2:
            channel_axis = None
            n_channels = 1
        # 3D images will be split into channels
        elif ndim == 3:
            if shape[2] in [3, 4]:  # rgb
                channel_axis = 2
                n_channels = shape[2]
            elif np.argmin(shape) == 2:
                channel_axis = 2
                n_channels = shape[2]
            else:
                channel_axis = 0
                n_channels = shape[0]
        else:
            raise ValueError(f"Array has unsupported shape: {shape}")
        return channel_axis, n_channels

    def get_channel(self, index: int, pyramid: int = 0) -> np.ndarray:
        """Return channel."""
        array: np.ndarray = self.pyramid[pyramid]
        channel_axis, n_channels = self.get_channel_axis_and_n_channels()
        if channel_axis is None:
            return array
        if channel_axis == 0:
            return array[index]
        elif channel_axis == 1:
            return array[:, index]
        elif channel_axis == 2:
            return array[:, :, index]
        raise ValueError(f"Array has unsupported shape: {array.shape}")

    def get_channel_pyramid(self, index: int) -> list[np.ndarray]:
        """Return channel pyramid."""
        array = self.pyramid
        channel_axis, n_channels = self.get_channel_axis_and_n_channels()
        if channel_axis is None:
            return array
        if self.is_rgb and not CONFIG.split_rgb:
            return array
        if channel_axis == 0:
            return [a[index] for a in array]
        elif channel_axis == 1:
            return [a[:, index] for a in array]
        elif channel_axis == 2:
            return [a[:, :, index] for a in array]
        else:
            raise ValueError("Could not retrieve channel pyramid.")

    @property
    def n_in_pyramid(self) -> int:
        """Return number of images in the pyramid."""
        return len(self.pyramid)

    def scale_for_pyramid(self, pyramid: int = 0) -> tuple[float, float]:
        """Return scale for pyramid."""
        if pyramid < 0:
            pyramid = list(range(self.n_in_pyramid))[pyramid]
            # pyramid = range(self.n_in_pyramid)[pyramid] + 1
        resolution = self.resolution * 2**pyramid
        return resolution, resolution

    def __del__(self):
        """Close the file handle."""
        self.close()

    def to_ome_tiff(
        self,
        path: PathLike,
        as_uint8: bool = False,
        channel_ids: list[int] | None = None,
        channel_names: list[str] | None = None,
    ) -> Path:
        """Write image as OME-TIFF."""
        from image2image_io.writers import write_ome_tiff_alt

        return write_ome_tiff_alt(path, self, as_uint8=as_uint8, channel_names=channel_names, channel_ids=channel_ids)
