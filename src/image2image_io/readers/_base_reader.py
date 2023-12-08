"""Base image wrapper."""
from __future__ import annotations

import math
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike

from image2image_io.enums import DEFAULT_TRANSFORM_NAME
from image2image_io.models.transform import TransformData
from image2image_io.readers.utilities import guess_rgb


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
    lazy: bool = False
    fh: ty.Any | None = None
    allow_extraction: bool = False
    _resolution: float = 1.0
    _channel_names: list[str]
    _channel_colors: list[str]

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
    def channel_colors(self) -> list[str]:
        """Return channel names."""
        return self._channel_colors

    def channel_to_index(self, channel: str) -> int:
        """Return index of a channel."""
        return self.channel_names.index(channel)

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

    def flat_array(self, index: int = 0) -> tuple[np.ndarray, tuple[int, int]]:
        """Return a flat array."""
        from image2image_io.utils.utilities import get_shape_of_image

        array = self.pyramid[index]
        if hasattr(array, "compute"):
            array = array.compute()
        n_channels, _, shape = get_shape_of_image(array)
        if array.ndim == 3:
            array = array.reshape(-1, n_channels)
        else:
            array = array.reshape(-1, 1)
        return array, shape

    def close(self) -> None:
        """Close the file handle."""
        if self.fh and hasattr(self.fh, "close"):
            self.fh.close()
        self.fh = None
        self._pyramid = None

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
            channel_axis = int(np.argmin(shape))
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

    def get_channel_axis_and_n_channels(self) -> tuple[int | None, int]:
        """Return channel axis and number of channels."""
        shape = self.shape
        ndim = len(shape)
        # 2D images will be returned as they are
        if ndim == 2:
            channel_axis = None
            n_channels = 1
        # 3D images will be split into channels
        else:
            channel_axis = int(np.argmin(shape))
            n_channels = shape[channel_axis]
        return channel_axis, n_channels

    def get_channel(self, index: int, pyramid: int = 0) -> np.ndarray:
        """Return channel."""
        array = self.pyramid[pyramid]
        channel_axis, n_channels = self.get_channel_axis_and_n_channels()
        if channel_axis is None:
            return array
        if channel_axis == 0:
            return array[index]
        elif channel_axis == 1:
            return array[:, index]
        elif channel_axis == 2:
            return array[:, :, index]
        else:
            raise ValueError(f"Array has unsupported shape: {array.shape}")

    def get_channel_pyramid(self, index: int) -> list[np.ndarray]:
        """Return channel pyramid."""
        array = self.pyramid
        channel_axis, n_channels = self.get_channel_axis_and_n_channels()
        if channel_axis is None:
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
            pyramid = range(1, self.n_in_pyramid + 1)[pyramid]
        resolution = self.resolution * 2**pyramid
        return resolution, resolution
