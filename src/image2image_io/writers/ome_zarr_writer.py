"""OME-Zarr writer utilities."""

from __future__ import annotations

import shutil
import typing as ty
from contextlib import contextmanager
from pathlib import Path

import cv2
import dask.array as da
import numpy as np
import zarr
from koyo.decorators import retry
from koyo.typing import PathLike
from loguru import logger
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image

from image2image_io.utils.utilities import ensure_dask_array

if ty.TYPE_CHECKING:
    from image2image_io.readers._base_reader import BaseReader
    from image2image_io.writers.tiff_writer import OmeTiffWriter, Transformer

logger = logger.bind(src="OME-Zarr")


def write_ome_zarr(
    path: PathLike,
    reader: BaseReader,
    as_uint8: bool | None = None,
    tile_size: int = 1024,
    channel_ids: list[int] | None = None,
    channel_names: list[str] | None = None,
    overwrite: bool = False,
    write_pyramid: bool = True,
    ome_name: str | None = None,
) -> Path:
    """Write a reader to an OME-Zarr store."""
    output_path = normalize_ome_zarr_path(path)
    if not _prepare_output_path(output_path, overwrite):
        return output_path

    channel_ids, channel_names = _prepare_channels(reader, channel_ids, channel_names)
    image, axes = _reader_to_dask_image(reader, channel_ids)
    if as_uint8 is None:
        as_uint8 = image.dtype == np.uint8
    if as_uint8:
        image = _convert_to_uint8(image, axes)

    y_size, x_size = _get_yx_shape(image, axes)
    max_layer = _get_max_layer((y_size, x_size), tile_size) if write_pyramid else 0
    chunks = _get_chunks(image.shape, axes, tile_size)
    coordinate_transformations = _get_coordinate_transformations(axes, reader.resolution, max_layer)
    omero = _get_omero_metadata(channel_names, reader.channel_colors)

    root = zarr.open_group(str(output_path), mode="w")
    write_image(
        image,
        root,
        scaler=Scaler(max_layer=max_layer, method="nearest"),
        chunks=chunks,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        omero=omero,
    )
    _set_multiscale_name(root, ome_name or output_path.name)
    logger.trace(f"Wrote OME-Zarr store to {output_path}")
    return output_path


def write_ome_zarr_from_array(
    path: PathLike,
    reader: BaseReader | None,
    array: np.ndarray,
    resolution: float | None = None,
    channel_names: list[str] | None = None,
    tile_size: int = 1024,
    as_uint8: bool | None = None,
    ome_name: str | None = None,
    write_pyramid: bool = True,
    overwrite: bool = False,
) -> Path:
    """Write a numpy array to an OME-Zarr store."""
    from image2image_io.readers.array_reader import ArrayImageReader

    if reader:
        resolution = resolution or reader.resolution
        channel_names = channel_names or reader.channel_names
    resolution = resolution or 1.0

    array_reader = ArrayImageReader("", array, resolution=resolution, channel_names=channel_names)
    return write_ome_zarr(
        path,
        array_reader,
        as_uint8=as_uint8,
        tile_size=tile_size,
        overwrite=overwrite,
        write_pyramid=write_pyramid,
        ome_name=ome_name,
    )


class OmeZarrWrapper:
    """OME-Zarr wrapper for incremental channel writes."""

    reader: BaseReader
    writer: OmeTiffWriter
    root: zarr.Group
    path: Path

    as_uint8: bool
    arrays: list[zarr.Array]
    axes: str
    channel_names: list[str]
    dtype: np.dtype
    n_channels: int
    output_path: Path
    pyramid_shapes: list[tuple[int, int]]
    resolution: float
    tmp_output_path: Path
    written_channels: set[int]
    write_pyramid: bool

    @property
    def is_rgb(self) -> bool:
        """Return whether the wrapped image is RGB."""
        return self.reader.is_rgb

    @contextmanager
    def write(
        self,
        channel_names: list[str],
        resolution: float,
        shape: tuple[int, ...],
        dtype: np.dtype,
        name: str,
        output_dir: Path | str = "",
        write_pyramid: bool = True,
        tile_size: int = 1024,
        as_uint8: bool | None = None,
        overwrite: bool = False,
        transformer: Transformer | None = None,
        ome_name: str | None = None,
    ) -> ty.Generator[OmeZarrWrapper, None, None]:
        """Initialize an OME-Zarr store and yield a channel appender."""
        from image2image_io.readers._base_reader import DummyReader
        from image2image_io.writers.tiff_writer import OmeTiffWriter

        self.reader = DummyReader(
            channel_names=channel_names,
            resolution=resolution,
            shape=shape,
            dtype=dtype,
        )
        self.writer = OmeTiffWriter(reader=self.reader, transformer=transformer)
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        self.resolution = resolution
        self.as_uint8 = dtype == np.uint8 if as_uint8 is None else as_uint8
        self.dtype = np.dtype(np.uint8 if self.as_uint8 else dtype)
        self.axes = "yxc" if self.reader.is_rgb else "cyx"
        self.output_path = normalize_ome_zarr_path(Path(output_dir) / name)
        self.tmp_output_path = self.output_path.parent / f"{self.output_path.name}.tmp"
        self.written_channels = set()

        if not _prepare_output_path(self.output_path, overwrite):
            msg = f"OME-Zarr store {self.output_path} already exists."
            raise FileExistsError(msg)
        _remove_output_path(self.tmp_output_path)
        self.tmp_output_path.parent.mkdir(parents=True, exist_ok=True)

        y_size, x_size = _get_wrapper_yx_shape(self.reader, transformer)
        max_layer = _get_max_layer((y_size, x_size), tile_size) if write_pyramid else 0
        self.write_pyramid = max_layer > 0
        self.pyramid_shapes = _get_pyramid_shapes((y_size, x_size), max_layer)
        self.root = zarr.open_group(str(self.tmp_output_path), mode="w")
        self.arrays = self._create_datasets(tile_size)
        self._write_metadata(ome_name or self.output_path.name)

        try:
            yield self
            self._validate_written_channels()
        except Exception:
            _remove_output_path(self.tmp_output_path)
            raise

        retry(lambda: self.tmp_output_path.rename(self.output_path), PermissionError)()
        self.path = self.output_path
        logger.trace(f"Renamed temporary OME-Zarr store to {self.output_path}")

    def add_channel(self, channel_index: int | list[int], channel_name: str | list[str], array: np.ndarray) -> None:
        """Add one channel or one RGB channel group to the store."""
        if self.is_rgb:
            if not isinstance(channel_index, list) or not isinstance(channel_name, list):
                msg = "RGB OME-Zarr writing requires channel indices and names as lists."
                raise TypeError(msg)
            self._add_rgb_channel(channel_index, array)
            return
        if not isinstance(channel_index, int):
            msg = "Multichannel OME-Zarr writing requires a single integer channel index."
            raise TypeError(msg)
        self._add_multichannel_channel(channel_index, array)

    def _create_datasets(self, tile_size: int) -> list[zarr.Array]:
        """Create empty zarr datasets for every pyramid level."""
        arrays = []
        for pyramid_index, (y_size, x_size) in enumerate(self.pyramid_shapes):
            shape = (y_size, x_size, self.n_channels) if self.axes == "yxc" else (self.n_channels, y_size, x_size)
            chunks = _get_chunks(shape, self.axes, tile_size)
            arrays.append(
                self.root.create_dataset(
                    str(pyramid_index),
                    shape=shape,
                    chunks=chunks,
                    dtype=self.dtype,
                    fill_value=0,
                )
            )
        return arrays

    def _write_metadata(self, name: str) -> None:
        """Write NGFF multiscale metadata for the pre-created datasets."""
        datasets = []
        transformations = _get_coordinate_transformations(self.axes, self.resolution, len(self.pyramid_shapes) - 1)
        for pyramid_index, transformation in enumerate(transformations):
            datasets.append({"path": str(pyramid_index), "coordinateTransformations": transformation})
        self.root.attrs["multiscales"] = [
            {
                "version": "0.4",
                "datasets": datasets,
                "name": name,
                "axes": _get_axes_metadata(self.axes),
                "omero": _get_omero_metadata(self.channel_names, None),
            }
        ]

    def _add_rgb_channel(self, channel_indices: list[int], array: np.ndarray) -> None:
        """Add an RGB image to the store."""
        if array.ndim != 3:
            msg = f"RGB array must be 3D, got shape {array.shape}."
            raise ValueError(msg)
        if len(channel_indices) != array.shape[2]:
            msg = "The number of RGB channel indices must match the array channel dimension."
            raise ValueError(msg)
        for channel_index in channel_indices:
            self._validate_channel_index(channel_index)
        for array_index, channel_index in enumerate(channel_indices):
            plane = self.writer._process_image(
                array[:, :, array_index],
                resolution=self.resolution,
                as_uint8=self.as_uint8,
            )
            self._write_plane(channel_index, plane)

    def _add_multichannel_channel(self, channel_index: int, array: np.ndarray) -> None:
        """Add a single channel to the store."""
        self._validate_channel_index(channel_index)
        plane = self.writer._process_image(array, resolution=self.resolution, as_uint8=self.as_uint8)
        self._write_plane(channel_index, plane)

    def _validate_channel_index(self, channel_index: int) -> None:
        """Validate a channel index before writing."""
        if channel_index < 0 or channel_index >= self.n_channels:
            msg = f"Channel index {channel_index} is outside the available channel range."
            raise IndexError(msg)
        if channel_index in self.written_channels:
            msg = f"Channel index {channel_index} has already been written."
            raise ValueError(msg)

    def _write_plane(self, channel_index: int, plane: np.ndarray) -> None:
        """Write a processed channel plane to all pyramid levels."""
        if plane.shape != self.pyramid_shapes[0]:
            msg = f"Channel plane shape {plane.shape} does not match expected shape {self.pyramid_shapes[0]}."
            raise ValueError(msg)
        for pyramid_index, (y_size, x_size) in enumerate(self.pyramid_shapes):
            if pyramid_index == 0:
                pyramid_plane = plane
            else:
                pyramid_plane = cv2.resize(plane, (x_size, y_size), interpolation=cv2.INTER_LINEAR)
            pyramid_plane = np.asarray(pyramid_plane, dtype=self.dtype)
            if self.axes == "yxc":
                self.arrays[pyramid_index][:, :, channel_index] = pyramid_plane
            else:
                self.arrays[pyramid_index][channel_index, :, :] = pyramid_plane
        self.written_channels.add(channel_index)

    def _validate_written_channels(self) -> None:
        """Validate that all channels were written before finalizing."""
        missing_channels = sorted(set(range(self.n_channels)) - self.written_channels)
        if missing_channels:
            msg = f"Cannot finalize OME-Zarr store; missing channel indices: {missing_channels}."
            raise ValueError(msg)


def normalize_ome_zarr_path(path: PathLike) -> Path:
    """Return a path ending in `.ome.zarr` or `.zarr`."""
    output_path = Path(path)
    if output_path.name.endswith((".ome.zarr", ".zarr")):
        return output_path
    return output_path.parent / f"{output_path.name}.ome.zarr"


def _prepare_output_path(path: Path, overwrite: bool) -> bool:
    """Prepare the output path for writing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return True
    if not overwrite:
        logger.warning("OME-Zarr store {} already exists, skipping...", path)
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return True


def _remove_output_path(path: Path) -> None:
    """Remove an existing output path."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _prepare_channels(
    reader: BaseReader, channel_ids: list[int] | None, channel_names: list[str] | None
) -> tuple[list[int], list[str]]:
    """Return validated channel ids and names."""
    if channel_ids is None:
        channel_ids = list(range(reader.n_channels))
    if not all(isinstance(channel_id, int) for channel_id in channel_ids):
        msg = "OME-Zarr writing only supports integer channel ids."
        raise ValueError(msg)
    if channel_names is None:
        channel_names = reader.channel_names
    if len(channel_ids) != len(channel_names):
        channel_names = [channel_names[channel_id] for channel_id in channel_ids]
    if len(channel_ids) != len(channel_names):
        msg = "The number of channel ids and channel names does not match."
        raise ValueError(msg)
    return channel_ids, channel_names


def _reader_to_dask_image(reader: BaseReader, channel_ids: list[int]) -> tuple[da.Array, str]:
    """Convert a reader to a dask image and NGFF axes string."""
    channel_axis, _n_channels = reader.get_channel_axis_and_n_channels(reader.shape)
    if channel_axis is None:
        if channel_ids != [0]:
            msg = "Single-channel OME-Zarr writing only accepts channel id 0."
            raise ValueError(msg)
        return ensure_dask_array(reader.pyramid[0]), "yx"

    if reader.is_rgb:
        channels = [ensure_dask_array(reader.get_channel(channel_id, split_rgb=True)) for channel_id in channel_ids]
        return da.stack(channels, axis=0), "cyx"

    image = ensure_dask_array(reader.pyramid[0])
    if channel_axis == 0:
        return image[channel_ids, :, :], "cyx"
    if channel_axis == 1:
        return da.moveaxis(image[:, channel_ids, :], 1, 0), "cyx"
    if channel_axis == 2:
        return da.moveaxis(image[:, :, channel_ids], 2, 0), "cyx"
    msg = f"Unsupported channel axis {channel_axis} for OME-Zarr writing."
    raise ValueError(msg)


def _convert_to_uint8(image: da.Array, axes: str) -> da.Array:
    """Scale image intensities to uint8."""
    if image.dtype == np.uint8:
        return image
    if axes == "cyx":
        return da.stack([_scale_plane_to_uint8(image[index]) for index in range(image.shape[0])], axis=0)
    return _scale_plane_to_uint8(image)


def _scale_plane_to_uint8(image: da.Array) -> da.Array:
    """Scale a single image plane to uint8."""
    image = image.astype(np.float32)
    image_min = image.min()
    image_max = image.max()
    scaled = da.where(image_max > image_min, (image - image_min) / (image_max - image_min) * 255, 0)
    return da.clip(scaled, 0, 255).astype(np.uint8)


def _get_yx_shape(image: da.Array, axes: str) -> tuple[int, int]:
    """Return y/x shape from an image and axes string."""
    return int(image.shape[axes.index("y")]), int(image.shape[axes.index("x")])


def _get_wrapper_yx_shape(reader: BaseReader, transformer: Transformer | None) -> tuple[int, int]:
    """Return the output y/x shape for an incremental wrapper."""
    if transformer is not None:
        x_size, y_size = transformer.output_size
        return y_size, x_size
    return reader.image_shape


def _get_pyramid_shapes(image_shape: tuple[int, int], max_layer: int) -> list[tuple[int, int]]:
    """Return y/x shapes for every pyramid level."""
    y_size, x_size = image_shape
    return [(max(1, y_size // (2**level)), max(1, x_size // (2**level))) for level in range(max_layer + 1)]


def _get_max_layer(image_shape: tuple[int, int], tile_size: int) -> int:
    """Return the number of downsampled pyramid levels to write."""
    if tile_size <= 0:
        return 0
    max_layer = 0
    min_size = min(image_shape)
    while min_size // (2 ** (max_layer + 1)) >= tile_size:
        max_layer += 1
    return max_layer


def _get_chunks(shape: tuple[int, ...], axes: str, tile_size: int) -> tuple[int, ...]:
    """Return chunks for OME-Zarr writing."""
    if tile_size <= 0:
        return shape
    chunks = []
    for axis_name, axis_size in zip(axes, shape):
        if axis_name == "c":
            chunks.append(1)
        else:
            chunks.append(min(int(axis_size), tile_size))
    return tuple(chunks)


def _get_coordinate_transformations(axes: str, resolution: float, max_layer: int) -> list[list[dict[str, ty.Any]]]:
    """Return NGFF scale coordinate transformations for each pyramid level."""
    transformations = []
    for pyramid_index in range(max_layer + 1):
        downscale = 2**pyramid_index
        scale = [1.0 if axis_name == "c" else float(resolution) * downscale for axis_name in axes]
        transformations.append([{"type": "scale", "scale": scale}])
    return transformations


def _get_axes_metadata(axes: str) -> list[dict[str, str]]:
    """Return NGFF axes metadata."""
    return [{"name": axis_name, "type": "channel" if axis_name == "c" else "space"} for axis_name in axes]


def _get_omero_metadata(channel_names: list[str], channel_colors: list[str] | None) -> dict[str, list[dict[str, str]]]:
    """Return OMERO channel metadata."""
    channels = []
    for index, channel_name in enumerate(channel_names):
        channel = {"label": channel_name}
        if channel_colors and index < len(channel_colors):
            channel["color"] = channel_colors[index].lstrip("#")
        channels.append(channel)
    return {"channels": channels}


def _set_multiscale_name(root: zarr.Group, name: str) -> None:
    """Set the multiscale display name after writing."""
    multiscales = root.attrs.get("multiscales", [])
    if not multiscales:
        return
    multiscales[0]["name"] = name
    root.attrs["multiscales"] = multiscales
