"""OME-Zarr image reader."""

from __future__ import annotations

import typing as ty

import dask.array as da
import numpy as np
import zarr
from koyo.typing import PathLike
from loguru import logger

from image2image_io._zarr import Array, Group
from image2image_io.readers._base_reader import BaseReader

logger = logger.bind(src="OME-Zarr")


class OmeZarrImageReader(BaseReader):
    """Reader for OME-Zarr image stores."""

    reader = "ome-zarr"

    def __init__(
        self,
        path: PathLike,
        key: str | None = None,
        init_pyramid: bool | None = None,
        auto_pyramid: bool | None = None,
    ):
        super().__init__(path, key, auto_pyramid=auto_pyramid)
        store = zarr.open(str(self.path), mode="r")
        self.root: Group | None = None
        self.array: Array | None = None
        if isinstance(store, Group):
            self.root = store
        elif isinstance(store, Array):
            self.array = store
        else:
            msg = f"Unsupported zarr store at {self.path}"
            raise TypeError(msg)

        self._multiscale = self._get_multiscale_metadata()
        self._dataset_paths = self._get_dataset_paths()
        first_array = self._open_dask_array(self._dataset_paths[0])
        first_array = self._normalize_array(first_array)
        self.shape = first_array.shape
        self._array_dtype = first_array.dtype
        self._image_shape = self.get_image_shape_for_shape(self.shape)
        self.resolution = self._get_resolution()
        self._channel_names = self._get_channel_names()
        self._channel_colors = self._get_channel_colors()

        if init_pyramid:
            self._pyramid = self.pyramid

    def _get_root_attrs(self) -> dict[str, ty.Any]:
        """Return root zarr attributes."""
        if self.root is None:
            return {}
        return self.root.attrs.asdict()

    def _get_multiscale_metadata(self) -> dict[str, ty.Any]:
        """Return the primary OME-NGFF multiscale metadata."""
        multiscales = self._get_root_attrs().get("multiscales", [])
        if not multiscales:
            return {}
        if not isinstance(multiscales, list):
            msg = f"Invalid OME-Zarr multiscales metadata in {self.path}"
            raise TypeError(msg)
        return multiscales[0]

    def _get_dataset_paths(self) -> list[str]:
        """Return ordered zarr dataset paths for the image pyramid."""
        if self.array is not None:
            return [""]

        datasets = self._multiscale.get("datasets", [])
        if datasets:
            return [dataset["path"] for dataset in datasets]

        if self.root is None:
            msg = f"Could not find OME-Zarr root group in {self.path}"
            raise ValueError(msg)
        paths = sorted(self.root.array_keys(), key=_dataset_sort_key)
        if not paths:
            msg = f"Could not find image datasets in {self.path}"
            raise ValueError(msg)
        return paths

    def _open_dask_array(self, dataset_path: str) -> da.Array:
        """Open a zarr dataset as a dask array."""
        if self.array is not None:
            return da.from_zarr(self.array)
        if self.root is None:
            msg = f"Could not find OME-Zarr root group in {self.path}"
            raise ValueError(msg)
        return da.from_zarr(self.root[dataset_path])

    def _get_axes(self, array: da.Array) -> list[str]:
        """Return axis names for an array."""
        axes = self._multiscale.get("axes", [])
        if axes:
            axis_names = [axis["name"] if isinstance(axis, dict) else axis for axis in axes]
            if len(axis_names) != array.ndim:
                msg = (
                    f"OME-Zarr axes metadata length ({len(axis_names)}) does not match "
                    f"array dimensionality ({array.ndim}) in {self.path}"
                )
                raise ValueError(msg)
            return [str(axis).lower() for axis in axis_names]

        if array.ndim == 2:
            return ["y", "x"]
        if array.ndim == 3:
            return ["y", "x", "c"] if array.shape[-1] in (3, 4) else ["c", "y", "x"]
        msg = f"OME-Zarr axes metadata is required for {array.ndim}D arrays in {self.path}"
        raise ValueError(msg)

    def _normalize_array(self, array: da.Array) -> da.Array:
        """Normalize arrays to shapes supported by BaseReader."""
        axis_names = self._get_axes(array)
        array, axis_names = _squeeze_singleton_non_image_axes(array, axis_names, self.path)
        if "y" not in axis_names or "x" not in axis_names:
            msg = f"OME-Zarr datasets must include y and x axes in {self.path}"
            raise ValueError(msg)

        if "c" not in axis_names:
            return _move_axes(array, axis_names, ["y", "x"])

        channel_axis = axis_names.index("c")
        if channel_axis == len(axis_names) - 1 and array.shape[channel_axis] in (3, 4):
            target_axes = ["y", "x", "c"]
        else:
            target_axes = ["c", "y", "x"]
        return _move_axes(array, axis_names, target_axes)

    def _get_omero_metadata(self) -> dict[str, ty.Any]:
        """Return OMERO rendering metadata when present."""
        if "omero" in self._multiscale:
            return self._multiscale["omero"]
        return self._get_root_attrs().get("omero", {})

    def _get_channel_names(self) -> list[str]:
        """Return channel names from OMERO metadata or deterministic fallbacks."""
        _channel_axis, n_channels = self.get_channel_axis_and_n_channels(self.shape)
        channels = self._get_omero_metadata().get("channels", [])
        channel_names = [channel["label"] for channel in channels if channel.get("label")]
        if len(channel_names) >= n_channels:
            return channel_names[:n_channels]
        if self.is_rgb and n_channels in (3, 4):
            return ["R", "G", "B", "A"][:n_channels]
        return [f"C{index}" for index in range(n_channels)]

    def _get_channel_colors(self) -> list[str] | None:
        """Return channel colors from OMERO metadata."""
        channels = self._get_omero_metadata().get("channels", [])
        colors = [channel["color"] for channel in channels if channel.get("color")]
        return colors or None

    def _get_resolution(self) -> float:
        """Return the average y/x pixel size from NGFF scale metadata."""
        datasets = self._multiscale.get("datasets", [])
        if not datasets:
            return 1.0

        scale = None
        for transform in datasets[0].get("coordinateTransformations", []):
            if transform.get("type") == "scale":
                scale = transform.get("scale")
                break
        if scale is None:
            return 1.0

        first_array = self._open_dask_array(self._dataset_paths[0])
        axis_names = self._get_axes(first_array)
        if len(scale) != len(axis_names):
            return 1.0
        y_resolution = float(scale[axis_names.index("y")])
        x_resolution = float(scale[axis_names.index("x")])
        if not np.isclose(y_resolution, x_resolution):
            logger.warning(
                "OME-Zarr has anisotropic y/x pixel sizes ({}, {}); using their mean.",
                y_resolution,
                x_resolution,
            )
        return float(np.mean([y_resolution, x_resolution]))

    def get_dask_pyr(self) -> list[da.Array]:
        """Return the OME-Zarr pyramid as dask arrays."""
        return [self._normalize_array(self._open_dask_array(dataset_path)) for dataset_path in self._dataset_paths]


def _dataset_sort_key(path: str) -> tuple[int, int | str]:
    """Sort numeric pyramid dataset paths before non-numeric paths."""
    return (0, int(path)) if path.isdigit() else (1, path)


def _squeeze_singleton_non_image_axes(
    array: da.Array, axis_names: list[str], path: PathLike
) -> tuple[da.Array, list[str]]:
    """Remove singleton non-image axes and reject unsupported dimensions."""
    axis_names = list(axis_names)
    for axis_index in reversed(range(len(axis_names))):
        axis_name = axis_names[axis_index]
        if axis_name in {"c", "y", "x"}:
            continue
        if array.shape[axis_index] != 1:
            msg = (
                f"OME-Zarr axis '{axis_name}' has size {array.shape[axis_index]} in {path}; "
                "only singleton non-image axes are currently supported."
            )
            raise ValueError(msg)
        array = da.squeeze(array, axis=axis_index)
        axis_names.pop(axis_index)
    return array, axis_names


def _move_axes(array: da.Array, source_axes: list[str], target_axes: list[str]) -> da.Array:
    """Move dask array axes to the requested order."""
    source_indices = [source_axes.index(axis_name) for axis_name in target_axes]
    return da.moveaxis(array, source_indices, list(range(len(target_axes))))
