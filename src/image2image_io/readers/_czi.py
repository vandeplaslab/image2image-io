"""CZI file reader."""

from __future__ import annotations

import typing as ty
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from math import ceil, floor
from multiprocessing import cpu_count
from pathlib import Path
from xml.etree import ElementTree as ET

import dask.array as da
import numpy as np
import zarr
from czifile import CziFile as _CziFile
from czifile import DimensionEntryDV1, DirectoryEntryDV
from koyo.timer import MeasureTimer
from loguru import logger
from tifffile import create_output
from tqdm import tqdm

from image2image_io._zarr import TempStore
from image2image_io.config import CONFIG
from image2image_io.readers.utilities import compute_sub_res

logger = logger.bind(src="CZI")


class CziFile(_CziFile):
    """Sub-class of CziFile with added functionality to only read certain channels."""

    def asarray_alt(self, resize=True, order=0, out=None, max_workers=None):
        """Return image data from file(s) as numpy array.

        Parameters
        ----------
        resize : bool
            If True (default), resize sub/supersampled subblock data.
        order : int
            The order of spline interpolation used to resize sub/supersampled
            subblock data. Default is 0 (nearest neighbor).
        out : numpy.ndarray, str, or file-like object; optional
            Buffer where image data will be saved.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        max_workers : int
            Maximum number of threads to read and decode subblock data.
            By default up to half the CPU cores are used.

        """
        if out is None:
            out = create_output(out, self.shape, self.dtype)

        if max_workers is None:
            max_workers = cpu_count() // 2

        def func(directory_entry, resize=resize, order=order, start=self.start, out=out):
            """Read, decode, and copy subblock data."""
            subblock = directory_entry.data_segment()
            tile = subblock.data(resize=resize, order=order)
            index = tuple(slice(i - j, i - j + k) for i, j, k in zip(directory_entry.start, start, tile.shape))
            try:
                out[index] = tile
            except ValueError as e:
                warnings.warn(str(e), stacklevel=2)
            with pbar.get_lock():
                pbar.update()

        if max_workers > 1:
            self._fh.lock = True
            with tqdm(
                total=len(self.filtered_subblock_directory), desc="Reading subblocks", mininterval=0.5, unit="block"
            ) as pbar, ThreadPoolExecutor(max_workers) as executor:
                executor.map(func, self.filtered_subblock_directory)
            self._fh.lock = None
        else:
            with tqdm(
                total=len(self.filtered_subblock_directory), desc="Reading subblocks", mininterval=0.5, unit="block"
            ) as pbar:
                for directory_entry in self.filtered_subblock_directory:
                    func(directory_entry)

        if hasattr(out, "flush"):
            out.flush()
        return out

    def zarr_pyramidize_czi(self, zarr_fp: TempStore, pyramid: bool = True, tile_size: int = 1024) -> list:
        """Create a pyramidal zarr store from a CZI file."""
        dask_pyr = []
        root = zarr.open_group(zarr_fp, mode="a")
        root.attrs["axes_names"] = list(self.axes)
        root.attrs["orig_shape"] = list(self.shape)

        all_axes = list(self.axes)
        yx_dims = np.where(np.isin(all_axes, ["Y", "X"]) == 1)[0].tolist()
        yx_shape = np.array(self.shape[slice(yx_dims[0], yx_dims[1] + 1)])

        pyramid_seq = 0
        rgb_chunk = self.shape[-1] if self.shape[-1] > 2 else 1
        n_dims = len(all_axes) - 3
        chunking = (*[1] * n_dims, tile_size, tile_size, rgb_chunk)

        out_shape = list(self.shape)
        out_dtype = self.dtype

        kws = {}
        if hasattr(zarr, "ThreadSynchronizer"):
            kws["synchronizer"] = zarr.ThreadSynchronizer() if CONFIG.multicore else None
        out = root.create_dataset(
            str(pyramid_seq),
            shape=tuple(out_shape),
            chunks=chunking,
            dtype=out_dtype,
            overwrite=True,
            **kws,
        )

        with MeasureTimer() as timer:
            self.as_tzcyx0_array(out=out, max_workers=cpu_count() if CONFIG.multicore else 1)
            z_array = da.squeeze(da.from_zarr(zarr.open(zarr_fp)[str(pyramid_seq)]))
            dask_pyr.append(z_array)
        logger.trace(f"Loaded data in {timer()} to shape {z_array.shape}")

        if not pyramid:
            logger.trace("Pyramid creation disabled")
            return dask_pyr

        ds = 1
        while np.min(yx_shape) // 2**ds >= tile_size:
            ds += 1

        ds_levels = list(range(1, ds))
        logger.trace(f"Generating {ds_levels} down-sampled images")
        for ds_factor in ds_levels:
            with MeasureTimer() as timer:
                zres = TempStore()
                rgb_chunk = self.shape[-1] if self.shape[-1] > 2 else 1
                is_rgb = rgb_chunk > 1
                z_array_ds = compute_sub_res(z_array, ds_factor, tile_size, is_rgb, self.dtype)
                da.to_zarr(z_array_ds, zres, component="0")
                dask_pyr.append(da.squeeze(da.from_zarr(zres, component="0")))
            logger.trace(f"Down-sampled {ds_factor} in {timer()} to shape {z_array_ds.shape}")
        return dask_pyr

    @cached_property
    def pos_x_um(self) -> float:
        """Return the position of the image in micrometers."""
        return self.scale_x_um * min((dim_entry.start for dim_entry in self._iter_dim_entries("X")), default=0.0)

    @cached_property
    def pos_y_um(self) -> float:
        """Return the position of the image in micrometers."""
        return self.scale_y_um * min((dim_entry.start for dim_entry in self._iter_dim_entries("Y")), default=0.0)

    @cached_property
    def pos_z_um(self) -> float:
        """Return the position of the image in micrometers."""
        return self.scale_z_um * min((dim_entry.start for dim_entry in self._iter_dim_entries("Z")), default=0.0)

    @cached_property
    def pos_t_seconds(self) -> float:
        """Return the position of the image in seconds."""
        return self.scale_t_seconds * min((dim_entry.start for dim_entry in self._iter_dim_entries("T")), default=0.0)

    @cached_property
    def scale_x_um(self) -> float:
        """Return the scale of the image in micrometers."""
        return self._get_scale("X", multiplier=10.0**6)

    @cached_property
    def scale_y_um(self) -> float:
        """Return the scale of the image in micrometers."""
        return self._get_scale("Y", multiplier=10.0**6)

    @cached_property
    def scale_z_um(self) -> float:
        """Return the scale of the image in micrometers."""
        return self._get_scale("Z", multiplier=10.0**6)

    @cached_property
    def scale_t_seconds(self) -> float:
        """Return the scale of the image in seconds."""
        return self._get_scale("T")

    @cached_property
    def channel_names(self) -> list[str] | None:
        """Return the names of the channels."""
        if "C" in self.axes:
            channel_elements = self._metadata_xml.findall(".//Metadata/Information/Image/Dimensions/Channels/Channel")
            if len(channel_elements) == self.shape[self.axes.index("C")]:
                return [c.attrib.get("Name", c.attrib["Id"]) for c in channel_elements]
        return None

    @cached_property
    def is_rgb(self) -> bool:
        """Return True if the image is RGB."""
        return "0" in self.axes and self.shape[self.axes.index("0")] > 1

    def as_tzcyx0_array(self, *args, **kwargs) -> np.ndarray:
        """Return image data as numpy array with axes TZCYX0."""
        data = self.asarray_alt(*args, **kwargs)
        tzcyx0_axis_indices = []
        if "T" in self.axes:
            tzcyx0_axis_indices.append(self.axes.index("T"))
        else:
            tzcyx0_axis_indices.append(data.ndim)
            data = np.expand_dims(data, -1)
        if "Z" in self.axes:
            tzcyx0_axis_indices.append(self.axes.index("Z"))
        else:
            tzcyx0_axis_indices.append(data.ndim)
            data = np.expand_dims(data, -1)
        if "C" in self.axes:
            tzcyx0_axis_indices.append(self.axes.index("C"))
        else:
            tzcyx0_axis_indices.append(data.ndim)
            data = np.expand_dims(data, -1)
        tzcyx0_axis_indices.append(self.axes.index("Y"))
        tzcyx0_axis_indices.append(self.axes.index("X"))
        if "0" in self.axes:
            tzcyx0_axis_indices.append(self.axes.index("0"))
        else:
            tzcyx0_axis_indices.append(data.ndim)
            data = np.expand_dims(data, -1)
        for axis_index in range(len(self.axes)):
            if axis_index not in tzcyx0_axis_indices:
                tzcyx0_axis_indices.append(axis_index)
        if hasattr(data, "transpose"):
            data = data.transpose(tzcyx0_axis_indices)
        else:
            data = np.transpose(data, tzcyx0_axis_indices)
        data.shape = data.shape[:6]
        return data

    def _get_scale(self, dimension: str, multiplier: float = 1.0):
        scale_element = self._metadata_xml.find(f'.//Metadata/Scaling/Items/Distance[@Id="{dimension}"]/Value')
        if scale_element is not None:
            scale = float(scale_element.text)
            if scale > 0:
                return scale * multiplier
        return 1.0

    @cached_property
    def _metadata_xml(self) -> ET.Element:
        return ET.fromstring(self.metadata())

    def _iter_dim_entries(self, dimension: str) -> ty.Iterable[DimensionEntryDV1]:
        for dir_entry in self.filtered_subblock_directory:
            for dim_entry in dir_entry.dimension_entries:
                if dim_entry.dimension == dimension:
                    yield dim_entry


class CziSceneFile(CziFile):
    @staticmethod
    def get_num_scenes(path: str | Path, *args, **kwargs) -> int:
        """Get the number of scenes."""
        with _CziFile(path, *args, **kwargs) as czi_file:
            if "S" in czi_file.axes:
                return czi_file.shape[czi_file.axes.index("S")]
            return 1

    def __init__(self, path: str | Path, scene_index: int, *args: ty.Any, **kwargs: ty.Any):
        super().__init__(str(path), *args, **kwargs)
        self.scene_index = scene_index

    @cached_property
    def filtered_subblock_directory(self) -> list[DirectoryEntryDV]:
        dir_entries = super().filtered_subblock_directory
        return list(
            filter(
                lambda dir_entry: self._get_scene_index(dir_entry) == self.scene_index,
                dir_entries,
            )
        )

    @staticmethod
    def _get_scene_index(dir_entry: DirectoryEntryDV) -> int:
        scene_indices = {dim_entry.start for dim_entry in dir_entry.dimension_entries if dim_entry.dimension == "S"}
        if len(scene_indices) == 0:
            return 0
        assert len(scene_indices) == 1
        return scene_indices.pop()


def czi_tile_grayscale(rgb_image):
    """
    Convert RGB image data to greyscale.

    Parameters
    ----------
    rgb_image: np.ndarray
        image data

    Returns
    -------
    image:np.ndarray
        returns 8-bit greyscale image for 24-bit RGB image
    """
    result = (
        (rgb_image[..., 0] * 0.2125).astype(np.uint8)
        + (rgb_image[..., 1] * 0.7154).astype(np.uint8)
        + (rgb_image[..., 2] * 0.0721).astype(np.uint8)
    )

    return np.expand_dims(result, axis=-1)


def get_level_blocks(czi: CziFile) -> dict:
    """Get level blocks."""
    level_blocks: dict = {}
    for idx, sb in enumerate(czi.subblocks()):
        level = sb.shape[3] // sb.stored_shape[3] if sb.pyramid_type != 0 else 0

        if level not in level_blocks:
            level_blocks[level] = []
        level_blocks[level].append((idx, sb))
    return level_blocks


def _get_dimension_entry(data_segment, dimension: str) -> DimensionEntryDV1 | None:
    """Return the CZI dimension entry for a data segment."""
    for dimension_entry in data_segment.dimension_entries:
        if dimension_entry.dimension == dimension:
            return dimension_entry
    return None


def _get_subblock_downsample(czi: CziFile, directory_entry: DirectoryEntryDV) -> int:
    """Return the spatial downsampling factor represented by a CZI subblock."""
    y_axis = czi.axes.index("Y")
    x_axis = czi.axes.index("X")
    data_segment = directory_entry.data_segment()

    y_factor = data_segment.shape[y_axis] / data_segment.stored_shape[y_axis]
    x_factor = data_segment.shape[x_axis] / data_segment.stored_shape[x_axis]
    return max(1, round(max(y_factor, x_factor)))


def _get_first_dimension_start(czi: CziFile, dimension: str) -> int:
    """Return the first start index for a dimension in the filtered CZI subblocks."""
    starts: list[int] = []
    for directory_entry in czi.filtered_subblock_directory:
        data_segment = directory_entry.data_segment()
        dimension_entry = _get_dimension_entry(data_segment, dimension)
        if dimension_entry is not None:
            starts.append(int(dimension_entry.start))
    if starts:
        return min(starts)
    return int(czi.start[czi.axes.index(dimension)])


def _get_plane_indices(czi: CziFile, channel_index: int | None) -> dict[str, int]:
    """Return representative non-spatial dimension indices for thumbnail extraction."""
    plane_indices: dict[str, int] = {}
    for dimension in czi.axes:
        if dimension in {"Y", "X", "0"}:
            continue
        if dimension == "C":
            if channel_index is None:
                continue
            plane_indices[dimension] = _get_first_dimension_start(czi, dimension) + channel_index
        else:
            plane_indices[dimension] = _get_first_dimension_start(czi, dimension)
    return plane_indices


def _entry_contains_plane(data_segment, plane_indices: dict[str, int]) -> bool:
    """Return whether a CZI subblock intersects the requested non-spatial plane."""
    for dimension, index in plane_indices.items():
        dimension_entry = _get_dimension_entry(data_segment, dimension)
        if dimension_entry is None:
            continue
        start = int(dimension_entry.start)
        stop = start + int(dimension_entry.size)
        if not start <= index < stop:
            return False
    return True


def _get_spatial_preview_shape(czi: CziFile, max_size: int) -> tuple[tuple[int, int], tuple[float, float]]:
    """Return thumbnail shape and spatial scale for a bounded CZI preview."""
    y_size = int(czi.shape[czi.axes.index("Y")])
    x_size = int(czi.shape[czi.axes.index("X")])
    scale = max(y_size / max_size, x_size / max_size, 1.0)
    preview_shape = (max(1, int(y_size / scale)), max(1, int(x_size / scale)))
    preview_scale = (y_size / preview_shape[0], x_size / preview_shape[1])
    return preview_shape, preview_scale


def _get_czi_channel_count(czi: CziFile) -> int:
    """Return the number of non-RGB channels in a CZI file."""
    if "C" not in czi.axes:
        return 1
    return int(czi.shape[czi.axes.index("C")])


def _extract_spatial_tile(czi: CziFile, data_segment, channel_index: int | None) -> np.ndarray:
    """Read a spatial tile from a CZI subblock without expanding the full dataset."""
    data = data_segment.data(resize=False)
    is_rgb = czi.is_rgb
    tile_index: list[int | slice] = []
    for axis, dimension in enumerate(czi.axes):
        if dimension in {"Y", "X"}:
            tile_index.append(slice(None))
            continue
        if dimension == "0":
            tile_index.append(slice(None) if is_rgb else 0)
            continue
        if dimension == "C" and channel_index is not None:
            dimension_entry = _get_dimension_entry(data_segment, dimension)
            if dimension_entry is None:
                tile_index.append(0)
            else:
                channel_start = _get_first_dimension_start(czi, dimension)
                tile_index.append(channel_start + channel_index - int(dimension_entry.start))
            continue
        if dimension == "C" and int(data_segment.stored_shape[axis]) > 1:
            tile_index.append(slice(None))
            continue
        tile_index.append(0)

    tile = np.asarray(data[tuple(tile_index)])
    if not is_rgb and tile.ndim == 3 and tile.shape[-1] == 1:
        return tile[:, :, 0]
    return tile


def _resize_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a CZI tile to the requested YX shape."""
    from image2image_io.utils.utilities import resize

    if tile.shape[:2] == shape:
        return tile
    return resize(tile, shape).astype(tile.dtype)


def _read_czi_thumbnail_plane(
    czi: CziFile,
    pixel_spacing: tuple[int, int] | tuple[float, float],
    max_size: int,
    channel_index: int | None,
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    """Read a single CZI thumbnail plane directly from subblocks."""
    preview_shape, preview_scale = _get_spatial_preview_shape(czi, max_size)
    scale_y, scale_x = preview_scale
    thumbnail_spacing = (float(pixel_spacing[0] * scale_y), float(pixel_spacing[1] * scale_x))

    plane_indices = _get_plane_indices(czi, channel_index)
    level_entries: dict[int, list[DirectoryEntryDV]] = {}
    for directory_entry in czi.filtered_subblock_directory:
        data_segment = directory_entry.data_segment()
        if not _entry_contains_plane(data_segment, plane_indices):
            continue
        level = _get_subblock_downsample(czi, directory_entry)
        if level not in level_entries:
            level_entries[level] = []
        level_entries[level].append(directory_entry)

    if not level_entries:
        return None, None

    directory_entries = level_entries[max(level_entries)]
    is_rgb = czi.is_rgb
    rgb_size = int(czi.shape[czi.axes.index("0")]) if is_rgb else 0
    thumbnail_shape = (*preview_shape, rgb_size) if is_rgb else preview_shape
    thumbnail_array = np.zeros(thumbnail_shape, dtype=czi.dtype)

    y_origin = int(czi.start[czi.axes.index("Y")])
    x_origin = int(czi.start[czi.axes.index("X")])
    for directory_entry in directory_entries:
        data_segment = directory_entry.data_segment()
        y_entry = _get_dimension_entry(data_segment, "Y")
        x_entry = _get_dimension_entry(data_segment, "X")
        if y_entry is None or x_entry is None:
            continue

        y_start = int(y_entry.start) - y_origin
        x_start = int(x_entry.start) - x_origin
        y_stop = y_start + int(y_entry.size)
        x_stop = x_start + int(x_entry.size)

        out_y_start = max(0, floor(y_start / scale_y))
        out_x_start = max(0, floor(x_start / scale_x))
        out_y_stop = min(preview_shape[0], ceil(y_stop / scale_y))
        out_x_stop = min(preview_shape[1], ceil(x_stop / scale_x))
        if out_y_start >= out_y_stop or out_x_start >= out_x_stop:
            continue

        tile = _extract_spatial_tile(czi, data_segment, channel_index)
        tile_shape = (out_y_stop - out_y_start, out_x_stop - out_x_start)
        thumbnail_array[out_y_start:out_y_stop, out_x_start:out_x_stop] = _resize_tile(tile, tile_shape)

    return thumbnail_array, thumbnail_spacing


def get_czi_thumbnail(
    czi: CziFile,
    pixel_spacing: tuple[int, int] | tuple[float, float],
    max_size: int = 1024,
    channel_index: int | None = None,
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    """Get a thumbnail directly from CZI subblocks without building a full image pyramid."""
    if max_size <= 0:
        message = "max_size must be greater than zero."
        raise ValueError(message)

    if czi.is_rgb or channel_index is not None or _get_czi_channel_count(czi) == 1:
        return _read_czi_thumbnail_plane(czi, pixel_spacing, max_size, channel_index)

    channel_thumbnails: list[np.ndarray] = []
    thumbnail_spacing = None
    for channel_id in range(_get_czi_channel_count(czi)):
        thumbnail, thumbnail_spacing = _read_czi_thumbnail_plane(czi, pixel_spacing, max_size, channel_id)
        if thumbnail is None:
            return None, None
        channel_thumbnails.append(thumbnail)
    return np.stack(channel_thumbnails), thumbnail_spacing


def get_czi_channel_thumbnail(
    czi: CziFile,
    pixel_spacing: tuple[int, int] | tuple[float, float],
    channel_index: int,
    max_size: int = 1024,
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    """Get a single-channel thumbnail directly from CZI subblocks."""
    if channel_index < 0 or channel_index >= _get_czi_channel_count(czi):
        message = f"Channel index {channel_index} is out of range."
        raise ValueError(message)
    return get_czi_thumbnail(czi, pixel_spacing, max_size=max_size, channel_index=channel_index)


def get_n_scenes(path: str | Path) -> int:
    """Get the number of scenes in CZI file."""
    return CziSceneFile.get_num_scenes(path)
