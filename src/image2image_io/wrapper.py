from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger

from image2image_io.config import CONFIG
from image2image_io.readers._base_reader import BaseReader


class ImageWrapper:
    """Wrapper around image data."""

    data: dict[str, BaseReader]
    resolution: float = 1.0

    def __init__(self, reader: dict[str, BaseReader] | None = None):
        self.data = reader or {}

        resolution = [1.0]
        for _, _reader_or_array in self.data.items():
            if hasattr(_reader_or_array, "base_layer_pixel_res"):
                resolution.append(_reader_or_array.resolution)
        self.resolution = np.min(resolution)

    def __repr__(self) -> str:
        return f"ImageWrapper<{len(self.data)}>"

    def __getitem__(self, item: int | PathLike) -> BaseReader:
        if isinstance(item, int):
            return list(self.data.values())[item]
        return self.data[item]

    def add(self, reader: BaseReader) -> None:
        """Add data to wrapper."""
        self.data[reader.key] = reader
        logger.trace(f"Added '{reader.key}' to ImageWrapper.")

    def remove(self, key_or_reader: str | BaseReader) -> None:
        """Remove data from wrapper."""
        if hasattr(key_or_reader, "key"):
            key_or_reader = key_or_reader.key
        if key_or_reader in self.data:
            reader = self.data.pop(key_or_reader, None)
            del reader

    def remove_path(self, path: PathLike) -> list[str]:
        """Remove readers for specified path."""
        keys = self.get_key_for_path(path)
        for key in keys:
            self.remove(key)
        return keys

    def get_key_for_path(self, path: PathLike) -> list[str]:
        """Return key(s) for the specified path."""
        path = Path(path)
        keys = []
        for reader in self.reader_iter():
            if reader.path == path:
                keys.append(reader.key)
            elif reader.key == path.name:
                keys.append(reader.key)
        return keys

    def get_reader_for_path(self, path: PathLike) -> BaseReader:
        """Return reader for specified path."""
        keys = self.get_key_for_path(path)
        if len(keys) == 1:
            return self.data[keys[0]]
        raise ValueError(f"Reader for path '{path}' not found.")

    def get_reader_for_key(self, key: str) -> BaseReader:
        """Return reader for specified key."""
        try:
            return self.data[key]
        except KeyError:
            return None

    def is_loaded(self, path: PathLike) -> bool:
        """Check if the path is loaded."""
        from image2image_io.readers import sanitize_read_path

        path = Path(path)
        for reader in self.reader_iter():
            if reader.path == path or reader.path == sanitize_read_path(path, raise_on_error=False):
                return True
        return False

    def channel_names_for_names(self, names: ty.Sequence[str]) -> list[str]:
        """Return list of channel names for a given wrapper/dataset."""
        clean_names = []
        for name in names:
            if isinstance(name, Path):
                name = str(name.name)
            if "| " not in name:
                name = f"| {name}"
            clean_names.append(name)

        channel_names = []
        for channel_name in self.channel_names():
            for name in clean_names:
                if channel_name.endswith(name):
                    channel_names.append(channel_name)
        return channel_names

    def map_channel_to_index(self, dataset: str, channel_name: str) -> int:
        """Map channel name to index."""
        dataset_to_channel_map: dict[str, list[str]] = {}
        for name in self.channel_names():
            dataset, channel = name.split(" | ")
            dataset_to_channel_map.setdefault(dataset, []).append(channel)
        channels: list[str] = dataset_to_channel_map[dataset]
        return channels.index(channel_name)

    def channel_image_iter(self) -> ty.Generator[tuple[str, list[np.ndarray]], None, None]:
        """Iterator of channel name + image."""
        yield from zip(self.channel_names(), self.image_iter())

    def channel_image_for_channel_names_iter(
        self, channel_names: list[str] | None
    ) -> ty.Generator[tuple[str, list[np.ndarray] | None, BaseReader], None, None]:
        """Iterate of channel name + image for a specified list of channels."""
        if channel_names is None:
            channel_names = self.channel_names()
        for channel_name in channel_names:
            name, dataset = channel_name.split(" | ")
            reader = self.data[dataset]
            index = reader.channel_to_index(name)
            if reader.reader_type == "image":
                yield channel_name, reader.get_channel_pyramid(index), reader
            else:
                yield channel_name, None, reader

    def channel_image_reader_iter(self) -> ty.Generator[tuple[str, list[np.ndarray], BaseReader], None, None]:
        """Iterator of channel name + image."""
        for channel_name, (_, reader, image, _) in zip(self.channel_names(), self.reader_data_iter()):
            yield channel_name, image, reader

    def path_reader_iter(self) -> ty.Generator[tuple[Path, BaseReader], None, None]:
        """Iterator of a path + reader."""
        for reader in self.reader_iter():
            yield reader.path, reader

    def reader_channel_iter(self) -> ty.Generator[tuple[str, BaseReader, int], None, None]:
        """Iterator to add channels."""
        for reader_name, reader in self.data.items():
            yield from self.reader_channel_iter_for_reader(reader_name, reader)

    def reader_channel_iter_for_reader(
        self, reader_name: str, reader: BaseReader
    ) -> ty.Generator[tuple[str, BaseReader, int], None, None]:
        """Iterator to add channels."""
        if reader.reader_type == "shapes":
            yield reader_name, reader, 0
        elif reader.reader_type == "points":
            yield reader_name, reader, 0
        else:
            # spetial case for RGB images
            if reader.is_rgb and not CONFIG.split_rgb:
                yield reader_name, reader, None
            else:
                if reader.channel_names:
                    for index, _ in enumerate(reader.channel_names):
                        yield reader_name, reader, index
                else:
                    yield from self._reader_channel_iter(reader_name, reader)
                # for reader_name_, index in self._reader_channel_iter(reader_name, reader):
                #     # for reader_name_, _, _, index in self._reader_image_iter(reader_name, reader):
                #     yield reader_name_, reader, index

    @staticmethod
    def _reader_channel_iter(
        reader_name: str, reader: BaseReader
    ) -> ty.Generator[tuple[str, BaseReader, int], None, None]:
        """Iterator to add channels."""
        for channel_index in range(reader.n_channels):
            yield reader_name, reader, channel_index

    def reader_data_iter(
        self,
    ) -> ty.Generator[tuple[str, BaseReader, list[np.ndarray] | None, int], None, None]:
        """Iterator to add channels."""
        for reader_name, reader_or_array in self.data.items():
            if reader_or_array.reader_type == "shapes":
                yield reader_name, reader_or_array, None, 0
            elif reader_or_array.reader_type == "points":
                yield reader_name, reader_or_array, None, 0
            else:
                yield from self._reader_image_iter(reader_name, reader_or_array)

    def reader_image_iter(
        self,
    ) -> ty.Generator[tuple[str, BaseReader, list[np.ndarray], int], None, None]:
        """Iterator to add channels."""
        for reader_name, reader_or_array in self.data.items():
            yield from self._reader_image_iter(reader_name, reader_or_array)

    @staticmethod
    def _reader_image_iter(
        reader_name: str, reader: BaseReader
    ) -> ty.Generator[tuple[str, BaseReader, list[np.ndarray], int], None, None]:
        # image is a numpy array
        temp = reader.pyramid
        array = temp if isinstance(temp, list) else [temp]

        # get the shape of the 'largest image in pyramid'
        shape = reader.image_shape
        # get the number of dimensions, which determines how images are split into channels
        ndim = len(shape)
        channel_axis, n_channels = reader.get_channel_axis_and_n_channels()

        for channel_index in range(n_channels):
            # 2D image or is RGB image
            if channel_axis is None:
                yield reader_name, reader, array, channel_index
            # 3D image where the first axis corresponds to different channels
            elif channel_axis == 0:
                yield reader_name, reader, [a[channel_index] for a in array], channel_index
            # 3D image where the second axis corresponds to different channels
            elif channel_axis == 1:
                yield reader_name, reader, [a[:, channel_index] for a in array], channel_index
            # 3D image where the last axis corresponds to different channels
            elif channel_axis == 2:
                yield reader_name, reader, [a[..., channel_index] for a in array], channel_index
            else:
                raise ValueError(f"Cannot read image with {ndim} dimensions")

    def image_iter(self) -> ty.Iterator[list[np.ndarray]]:
        """Iterator to add channels."""
        for _, _, image, _ in self.reader_image_iter():
            yield image

    def reader_iter(self) -> ty.Iterator[BaseReader]:
        """Iterator of readers."""
        for _, reader_or_array in self.data.items():
            yield reader_or_array

    def dataset_names(self, reader_type: tuple[str, ...] = ("all",)) -> list[str]:
        """Return list of channel names."""
        names = []
        show_all = "all" in reader_type
        for key, reader in self.data.items():
            if show_all or reader.reader_type in reader_type:
                names.append(key)
        return names

    def channel_names(self) -> list[str]:
        """Return list of channel names."""
        names = []
        for key, reader, index in self.reader_channel_iter():
            channel_names = self.get_channel_names_for_reader(key, reader, index)
            names.extend(channel_names)
        return names

    @staticmethod
    def get_channel_names_for_reader(key: str, reader: BaseReader, index: int) -> list[str]:
        """Retrieve channel names for specified reader."""
        try:
            if reader.is_rgb and not CONFIG.split_rgb:
                channel_names = ["RGB"]
            else:
                channel_names = [reader.channel_names[index]]
        except (IndexError, NotImplementedError):
            channel_names = [f"C{index}"]
        channel_names = [f"{name} | {key}" for name in channel_names]
        return channel_names

    @property
    def min_resolution(self) -> float:
        """Return minimum resolution."""
        resolutions = [reader.resolution for reader in self.reader_iter()]
        if resolutions:
            return min(resolutions)
        return 1.0

    @staticmethod
    def get_affine(reader: BaseReader, moving_resolution: float) -> np.ndarray:
        """Get affine transformation for specified reader."""
        transform_data = reader.transform_data
        if not transform_data:
            raise ValueError("No transformation data found")
        affine = transform_data.compute(moving_resolution=moving_resolution, px=False)
        return np.asarray(affine.params)
