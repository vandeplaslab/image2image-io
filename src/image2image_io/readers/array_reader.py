"""Numpy array wrapper."""

from __future__ import annotations

import numpy as np
from koyo.typing import PathLike

from image2image_io.readers._base_reader import BaseReader


class ArrayImageReader(BaseReader):
    """Reader for data that has defined coordinates."""

    is_fixed: bool = False

    def __init__(
        self,
        path: PathLike,
        array: np.ndarray,
        key: str | None = None,
        resolution: float = 1.0,
        auto_pyramid: bool | None = None,
        channel_names: list[str] | None = None,
    ):
        super().__init__(path, key, auto_pyramid=auto_pyramid)
        self.array = array
        self.resolution = resolution
        if channel_names is not None:
            self._channel_names = channel_names

    @property
    def channel_names(self) -> list[str]:
        """List of channel names."""
        if self._channel_names is not None:
            return self._channel_names
        if self.array.ndim == 2:
            return ["C0"]
        _, n_channels = self.get_channel_axis_and_n_channels()
        self._channel_names = [f"C{i}" for i in range(n_channels)]
        return self._channel_names

    @property
    def pyramid(self) -> list:
        """Pyramid."""
        return [self.array]
