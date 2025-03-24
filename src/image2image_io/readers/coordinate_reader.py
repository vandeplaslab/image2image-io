"""Coordinate wrapper."""

from __future__ import annotations

import typing as ty
from functools import lru_cache
from pathlib import Path

import numpy as np
from koyo.timer import MeasureTimer
from koyo.typing import PathLike

from image2image_io.config import CONFIG
from image2image_io.readers._base_reader import BaseReader
from image2image_io.utils.lazy import LazyImageWrapper
from image2image_io.utils.utilities import format_mz

if ty.TYPE_CHECKING:
    from imzy._readers._base import BaseReader as BaseImzyReader


def set_dimensions(reader: CoordinateImageReader | LazyCoordinateImageReader) -> None:
    """Set dimension information."""
    x, y = reader.x, reader.y
    reader.xmin, reader.xmax = np.min(x), np.max(x)
    reader.ymin, reader.ymax = np.min(y), np.max(y)
    reader._image_shape = (reader.ymax - reader.ymin + 1, reader.xmax - reader.xmin + 1)


def get_image(array_or_reader: np.ndarray | BaseImzyReader) -> np.ndarray:
    """Return image for the array/image."""
    if not isinstance(array_or_reader, np.ndarray):
        array_or_reader = array_or_reader.reshape(array_or_reader.get_tic())
    return array_or_reader.astype(np.float32)  # type: ignore[no-any-return]


class CoordinateImagerMixin:
    """Mixin class to reduce amount of duplicate code."""

    xmin: int
    xmax: int
    ymin: int
    ymax: int
    x: np.ndarray
    y: np.ndarray
    is_fixed: bool = False
    image_shape: tuple[int, int]

    def get_random_image(self) -> np.ndarray:
        """Return random ion image."""
        array = np.full(self.image_shape, np.nan)
        array[self.y - self.ymin, self.x - self.xmin] = np.random.randint(128, 255, size=len(self.x)) / 255
        return array


class CoordinateImageReader(BaseReader, CoordinateImagerMixin):  # type: ignore[misc]
    """Reader for data that has defined coordinates."""

    def __init__(
        self,
        path: PathLike,
        x: np.ndarray,
        y: np.ndarray,
        key: str | None = None,
        resolution: float = 1.0,
        array_or_reader: np.ndarray | BaseImzyReader | None = None,
        data: dict[str, np.ndarray] | None = None,
        auto_pyramid: bool | None = None,
        reader_kws: dict | None = None,
    ):
        super().__init__(path, key, auto_pyramid=auto_pyramid, reader_kws=reader_kws)
        self.x = x
        self.y = y
        self.resolution = resolution
        self.reader = None if isinstance(array_or_reader, np.ndarray) else array_or_reader
        self.allow_extraction = self.reader is not None
        self._is_rgb = False
        self._array_dtype = np.float32  # type: ignore[assignment]

        self.data = data or {}
        if self.name not in self.data:
            name = "tic" if self.reader is not None else self.name
            self.data[name] = get_image(array_or_reader)
            self.get_image.cache_clear()  # clear cache
        set_dimensions(self)

    @property
    def _channel_names(self) -> list[str]:
        return list(self.data.keys())

    @property
    def channel_names(self) -> list[str]:
        """List of channel names."""
        return list(self.data.keys())

    @property
    def pyramid(self) -> list:
        """Pyramid."""
        return self.get_dask_pyr()

    def extract(self, mzs: np.ndarray, ppm: float = 10.0) -> tuple[Path, list[str]]:
        """Extract ion images."""
        if self.reader is None:
            raise ValueError("Cannot extract ion images from a numpy array.")
        mzs = np.atleast_1d(mzs)
        CONFIG.trace(f"Extracting {len(mzs)} ion images from {self.path.name} ({self.name})")

        with MeasureTimer() as timer:
            images = self.reader.get_ion_images(mzs, ppm=ppm)
        CONFIG.trace(f"Extracted {len(mzs)} ion images in {timer}")

        labels = []
        for i, mz in enumerate(mzs):
            label = format_mz(mz)
            self.data[label] = images[i]
            labels.append(f"{label} | {self.name}")
        self.get_image.cache_clear()  # clear cache
        return self.path, labels

    def get_dask_pyr(self) -> list[np.ndarray]:
        """Get dask representation of the pyramid."""
        return [self.get_image()]

    def get_channel_axis_and_n_channels(self, shape: tuple[int, ...] | None = None) -> tuple[int | None, int]:
        """Return channel axis and number of channels."""
        if shape is not None:
            return super().get_channel_axis_and_n_channels(shape)
        if len(self.data) == 1:
            return None, 1
        return 2, len(self.data)

    @lru_cache(maxsize=1)
    def get_image(self):
        """Return image as a stack."""
        if len(self.data) == 1:
            key = next(iter(self.data))
            return self.data[key]
        return np.dstack([self.data[key] for key in self.data])

    def get_channel(self, index: int, pyramid: int = 0, split_rgb: bool | None = None) -> np.ndarray:
        """Return channel."""
        image = self.get_channel_pyramid(index)[pyramid]
        return image

    def get_channel_pyramid(self, index: int) -> list[np.ndarray]:
        """Return channel pyramid."""
        name = self.channel_names[index]
        return [self.preprocessor("", self.data[name])]


class LazyCoordinateImageReader(BaseReader, CoordinateImagerMixin):  # type: ignore[misc]
    """Lazy coordinate image reader."""

    lazy = True

    def __init__(
        self,
        path: PathLike,
        x: np.ndarray,
        y: np.ndarray,
        lazy_wrapper: LazyImageWrapper,
        key: str | None = None,
        resolution: float = 1.0,
        channel_names: list[str] | None = None,
        auto_pyramid: bool | None = None,
    ):
        super().__init__(path, key, auto_pyramid=auto_pyramid)
        self.x = x
        self.y = y
        self.resolution = resolution
        self._is_rgb = False
        self._array_dtype = lazy_wrapper.dtype
        self._array_shape = (y.max(), x.max(), lazy_wrapper.shape[-1])
        self.lazy_wrapper = lazy_wrapper
        if channel_names:
            self._channel_names = channel_names
        set_dimensions(self)

    def get_dask_pyr(self) -> list[np.ndarray]:
        """Get dask representation of the pyramid."""
        return [self.get_image()]

    def get_image(self) -> np.ndarray:
        """Return image as a stack."""
        return self.lazy_wrapper.get_image()

    def get_channel_axis_and_n_channels(self, shape: tuple[int, ...] | None = None) -> tuple[int | None, int]:
        """Return channel axis and number of channels."""
        if shape:
            return super().get_channel_axis_and_n_channels(shape)
        return 2, len(self.channel_names)

    def get_channel_pyramid(self, index: int) -> list[np.ndarray]:
        """Return channel pyramid."""
        return [self.preprocessor("", self.lazy_wrapper[index])]
