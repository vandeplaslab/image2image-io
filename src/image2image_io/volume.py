"""Chunked serial-section volumes, independent of the 2D reader interface."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import dask.array as da
import numpy as np
import zarr

TileReader = Callable[[int, int, int, int, int, int], np.ndarray]


@dataclass(frozen=True)
class Volume:
    """Lazy CZYX pyramid with physical ZYX calibration in micrometers."""

    path: Path
    levels: tuple[da.Array, ...]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    channel_names: tuple[str, ...]
    present: tuple[bool, ...]
    provenance: dict[str, Any]


def open_volume(path: str | Path) -> Volume:
    """Open a completed CZYX OME-Zarr 0.5 volume without loading its pixels."""
    path = Path(path)
    if path.name.endswith(".partial"):
        message = "Partial volumes cannot be opened as completed results."
        raise ValueError(message)
    root = zarr.open_group(str(path), mode="r")
    try:
        ome = root.attrs.get("ome", {})
        if ome.get("version") != "0.5":
            message = "Expected an OME-Zarr 0.5 volume."
            raise ValueError(message)
        multiscale = ome["multiscales"][0]
        if [axis["name"] for axis in multiscale["axes"]] != list("czyx"):
            message = "Serial volumes must have CZYX axes."
            raise ValueError(message)
        if any(axis.get("unit") != "micrometer" for axis in multiscale["axes"][1:]):
            message = "Serial volume spatial axes must use micrometers."
            raise ValueError(message)
        datasets = multiscale["datasets"]
        levels = tuple(da.from_zarr(root[item["path"]]) for item in datasets)
        transforms = datasets[0]["coordinateTransformations"]
        scale = next(item["scale"] for item in transforms if item["type"] == "scale")
        origin = next((item["translation"] for item in transforms if item["type"] == "translation"), [0] * 4)
        if (
            len(scale) != 4
            or len(origin) != 4
            or not np.all(np.isfinite(scale))
            or min(scale) <= 0
            or not np.all(np.isfinite(origin))
        ):
            message = "Volume calibration must contain finite CZYX values and positive spacing."
            raise ValueError(message)
        metadata = root.attrs.get("image2image", {})
        if metadata.get("state", "complete") != "complete":
            message = "Volume reconstruction is incomplete."
            raise ValueError(message)
        channels = tuple(item["label"] for item in ome["omero"]["channels"])
        present = tuple(metadata.get("section_present", [True] * levels[0].shape[1]))
        if (
            not levels
            or any(level.ndim != 4 or level.shape[:2] != levels[0].shape[:2] for level in levels)
            or len(channels) != levels[0].shape[0]
            or len(present) != levels[0].shape[1]
        ):
            message = "Volume metadata does not match its array dimensions."
            raise ValueError(message)
        return Volume(
            path, levels, tuple(scale[1:]), tuple(origin[1:]), channels, present, metadata.get("provenance", {})
        )
    except (KeyError, IndexError, StopIteration, TypeError) as exc:
        message = "Incomplete or malformed volume metadata."
        raise ValueError(message) from exc


def write_volume(
    path: str | Path,
    *,
    shape: tuple[int, int, int, int],
    dtype: np.dtype,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    channel_names: Sequence[str],
    present: Sequence[bool],
    read_tile: TileReader,
    provenance: dict[str, Any] | None = None,
    tile_size: int = 512,
    progress: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Stream tiles into a new volume, publishing it only after every level succeeds.

    ``read_tile(c, z, y, x, height, width)`` returns native-intensity pixels.
    Missing sections are zero-filled and explicitly identified in metadata.
    Existing outputs are never overwritten. Cancellation raises InterruptedError.
    """
    path = Path(path)
    dtype = np.dtype(dtype)
    if len(shape) != 4 or any(size <= 0 for size in shape) or tile_size <= 0:
        message = "CZYX dimensions and tile size must be positive."
        raise ValueError(message)
    if len(channel_names) != shape[0] or len(present) != shape[1]:
        message = "Channels and section presence must match CZYX dimensions."
        raise ValueError(message)
    if (
        len(spacing) != 3
        or len(origin) != 3
        or not np.all(np.isfinite(spacing))
        or min(spacing) <= 0
        or not np.all(np.isfinite(origin))
    ):
        message = "Physical spacing must be positive and calibration must be finite."
        raise ValueError(message)
    if dtype.kind not in "uif":
        message = "Volumes require real integer or floating point intensities."
        raise ValueError(message)
    if path.exists():
        message = f"Output already exists: {path}"
        raise FileExistsError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid4().hex}.partial")

    def check_cancelled() -> None:
        if cancelled and cancelled():
            message = "Volume reconstruction cancelled."
            raise InterruptedError(message)

    def create_level(root: zarr.Group, index: int, size: tuple[int, ...]) -> zarr.Array:
        return root.create_array(
            str(index),
            shape=size,
            dtype=dtype,
            chunks=(1, 1, min(tile_size, size[2]), min(tile_size, size[3])),
            fill_value=0,
            dimension_names=tuple("czyx"),
        )

    try:
        root = zarr.open_group(str(partial), mode="w", zarr_format=3)
        root.attrs["image2image"] = {"state": "writing"}
        current = create_level(root, 0, shape)
        for z, exists in enumerate(present):
            check_cancelled()
            if not exists:
                continue
            for c in range(shape[0]):
                for y in range(0, shape[2], tile_size):
                    for x in range(0, shape[3], tile_size):
                        check_cancelled()
                        height, width = min(tile_size, shape[2] - y), min(tile_size, shape[3] - x)
                        tile = read_tile(c, z, y, x, height, width)
                        if tile.shape != (height, width) or tile.dtype != dtype:
                            message = "Tile shape or dtype does not match the volume."
                            raise ValueError(message)  # noqa: TRY301
                        current[c, z, y : y + height, x : x + width] = tile
            if progress:
                progress(f"Reconstructed section {z + 1}/{shape[1]}")
        datasets = []
        level = 0
        while True:
            factor = 2**level
            datasets.append(
                {
                    "path": str(level),
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, spacing[0], spacing[1] * factor, spacing[2] * factor]},
                        {"type": "translation", "translation": [0.0, *origin]},
                    ],
                }
            )
            if max(current.shape[2:]) <= tile_size:
                break
            size = (*shape[:2], (current.shape[2] + 1) // 2, (current.shape[3] + 1) // 2)
            following = create_level(root, level + 1, size)
            for z, exists in enumerate(present):
                if not exists:
                    continue
                for c in range(shape[0]):
                    for y in range(0, size[2], tile_size):
                        for x in range(0, size[3], tile_size):
                            check_cancelled()
                            height, width = min(tile_size, size[2] - y), min(tile_size, size[3] - x)
                            block = np.asarray(current[c, z, y * 2 : (y + height) * 2, x * 2 : (x + width) * 2])
                            block = np.pad(block, ((0, block.shape[0] % 2), (0, block.shape[1] % 2)), mode="edge")
                            reduced = block.astype(np.float64).reshape(height, 2, width, 2).mean(axis=(1, 3))
                            following[c, z, y : y + height, x : x + width] = reduced.astype(dtype)
            current = following
            level += 1
            if progress:
                progress(f"Built pyramid level {level}")
        root.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                {
                    "name": path.name,
                    "axes": [
                        {"name": "c", "type": "channel"},
                        *[{"name": axis, "type": "space", "unit": "micrometer"} for axis in "zyx"],
                    ],
                    "datasets": datasets,
                }
            ],
            "omero": {"channels": [{"label": name} for name in channel_names]},
        }
        root.attrs["image2image"] = {
            "state": "complete",
            "section_present": list(present),
            "provenance": provenance or {},
        }
        check_cancelled()
        if path.exists():
            message = f"Output already exists: {path}"
            raise FileExistsError(message)  # noqa: TRY301
        partial.rename(path)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return path
