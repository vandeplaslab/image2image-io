"""OME-Zarr writer utilities."""

from __future__ import annotations

import shutil
import typing as ty
from pathlib import Path

import dask.array as da
import numpy as np
import zarr
from koyo.typing import PathLike
from loguru import logger
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image

from image2image_io.utils.utilities import ensure_dask_array

if ty.TYPE_CHECKING:
    from image2image_io.readers._base_reader import BaseReader

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
