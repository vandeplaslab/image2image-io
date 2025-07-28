"""Base image wrapper."""

from __future__ import annotations

import math
import typing as ty
import warnings
from pathlib import Path

import numpy as np
import zarr.storage
from koyo.typing import PathLike
from loguru import logger

from image2image_io.config import CONFIG
from image2image_io.enums import DEFAULT_TRANSFORM_NAME, DisplayType
from image2image_io.models.preprocess import NoopProcessor
from image2image_io.models.transform import TransformData
from image2image_io.utils.mask import (
    _apply_bbox,
    _apply_mask,
    _hash_bbox_or_polygon,
    _prepare_bbox,
    _prepare_bbox_mask,
    _prepare_polygon,
    _prepare_polygon_mask,
)
from image2image_io.utils.utilities import guess_rgb

if ty.TYPE_CHECKING:
    from image2image_io.writers.tiff_writer import Transformer

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

    _closed = False
    _pyramid = None
    _is_rgb: bool | None = None
    _array_dtype: np.dtype | None = None
    _array_shape: tuple[int, ...] | None = None
    _image_shape: tuple[int, int] | None = None
    _zstore: zarr.storage.TempStore | None = None
    _scene_ids: list[int] | None = None
    _resolution: float = 1.0
    _channel_names: list[str] | None = None
    _channel_colors: list[str] | None = None
    _channel_ids: list[int] | None = None

    auto_pyramid: bool | None = None
    reader: str = "base"
    reader_type: str = "image"
    display_type: DisplayType | None = None
    n_scenes: int = 1
    lazy: bool = False
    fh: ty.Any | None = None
    allow_extraction: bool = False

    def __init__(
        self, path: PathLike, key: str | None = None, reader_kws: dict | None = None, auto_pyramid: bool | None = None
    ):
        self.auto_pyramid = auto_pyramid
        # This is the direct path to the image
        self.path = Path(path)
        # This is the attribute we will use to identify the image
        self.key = key or self.path.name
        self.reader_kws = reader_kws or {}
        self.transform_data: TransformData = TransformData()
        self.transform_name = DEFAULT_TRANSFORM_NAME
        self.preprocessor = NoopProcessor()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}<{self.key!r}; {self.path!r}; RGB={self.is_rgb}; {self.shape!r}"
            f" @ {self.resolution:.4f}>"
        )

    def __del__(self):
        """Close the file handle."""
        self.close()

    def find_channel_name(self, name: str) -> int | None:
        """Find cycle by name or part of the name."""
        if name in self.channel_names:
            return self.channel_names.index(name)
        for i, n in enumerate(self.channel_names):
            if name in n:
                return i
        return None

    def print_info(self) -> None:
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
    def scene_index(self) -> int:
        """Return scene index."""
        if self.reader_kws and "scene_index" in self.reader_kws:
            return self.reader_kws["scene_index"]
        return 0

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
        if self._array_shape is None:
            self._array_shape = self.pyramid[0].shape  # TODO: this is wrong!
        return self._array_shape

    @shape.setter
    def shape(self, value: tuple | np.ndarray) -> None:
        """Set shape."""
        if self._array_shape is not None:
            warnings.warn("Shape is already set, overwriting it.", stacklevel=2)
        self._array_shape = tuple(value)

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

    def get_channel_names(self, split_rgb: bool | None = None) -> list[str]:
        """Get channel names."""
        split_rgb = split_rgb if split_rgb is not None else CONFIG.split_rgb
        if self.is_rgb:
            return ["R", "B", "B"] if split_rgb else ["RGB"]
        return self.channel_names

    @property
    def channel_ids(self) -> list[int]:
        """Return channel indices."""
        if self._channel_ids is None:
            channel_names = self._channel_names or self.channel_names
            self._channel_ids = list(range(len(channel_names)))
        return self._channel_ids

    @channel_ids.setter
    def channel_ids(self, value: list[int]) -> None:
        """Set channel indices."""
        if not isinstance(value, list):
            value = list(value)
        assert len(value) <= len(self._channel_names), (
            f"Too many channels selected: {len(value)} > {len(self._channel_names)}"
        )
        assert max(value) < len(self._channel_names), (
            f"Channel index out of range: {max(value)} > {len(self._channel_names)}"
        )
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
    def channel_colors(self) -> list[str] | None:
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
        if self._array_dtype is None:
            self._array_dtype = self.pyramid[0].dtype
        return self._array_dtype

    @property
    def is_rgb(self) -> bool:
        """Return whether image is RGB."""
        if self._is_rgb is None:
            self._is_rgb = guess_rgb(self.shape)
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
    def display_name(self) -> str:
        """Retrieve display name from the path."""
        return self.path.stem

    @property
    def clean_key(self) -> str:
        """Key without extras."""
        return self.key[0:-4]

    @property
    def clean_name(self) -> str:
        """Return name of the input path."""
        name = self.name
        for ext in [
            ".ome",
            ".tiff",
            ".tif",
            ".png",
            ".jpg",
            ".svs",
            ".czi",
            ".d",
            ".imzML",
            ".tsf",
            ".tdf",
            ".qptiff",
            ".raw",
            ".intermediate",
        ]:
            name = name.replace(ext, "")
        return name

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
        if self._zstore is not None and hasattr(self._zstore, "path") and self._zstore.path:
            zarr_path = Path(self._zstore.path)
            if zarr_path.exists():
                zarr.storage.atexit_rmtree(zarr_path)
                logger.trace(f"Removed temporary zarr store: {zarr_path}")
        self.fh = None
        self._pyramid = None
        # check_if_open(self.path)
        if not self._closed:
            logger.trace(f"Closed file handle '{self.path}'")
            self._closed = True

    @property
    def pyramid(self) -> list:
        """Pyramid."""
        if self._pyramid is None:
            self._pyramid = self.get_dask_pyr()
        return self._pyramid

    def get_dask_pyr(self) -> list[ty.Any]:
        """Get dask representation of the pyramid."""
        raise NotImplementedError("Must implement method")

    def get_thumbnail(self) -> tuple[np.ndarray, tuple[float, float]]:
        """Return thumbnail."""
        return self.pyramid[-1], self.scale_for_pyramid(-1)

    def crop_region(
        self, bbox_or_yx: np.ndarray | tuple[int, int, int, int]
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop region."""
        if isinstance(bbox_or_yx, tuple) and len(bbox_or_yx) == 4:
            return self.crop_bbox(*bbox_or_yx)
        return self.crop_polygon(bbox_or_yx)

    def crop_region_iter(
        self, bbox_or_yx: np.ndarray | tuple[int, int, int, int]
    ) -> ty.Generator[tuple[np.ndarray | None, tuple[int, int, int, int]], None, None]:
        """Crop region."""
        if isinstance(bbox_or_yx, tuple) and len(bbox_or_yx) == 4:
            yield from self.crop_bbox_iter(*bbox_or_yx)
        else:
            yield from self.crop_polygon_iter(bbox_or_yx)

    def crop_bbox(
        self,
        left: int,
        right: int,
        top: int,
        bottom: int,
        array: np.ndarray | None = None,
        multiply: bool = True,
        apply: bool = True,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop image."""
        # Get the array from the pyramid or use the provided one
        array = self.pyramid[0] if array is None else array
        # Prepare bounding box coordinates
        left, right, top, bottom = _prepare_bbox(left, right, top, bottom, self.inv_resolution, multiply=multiply)
        # Apply mask to the cropped array
        channel_axis, _ = self.get_channel_axis_and_n_channels()
        array_ = _apply_bbox(array, left, right, top, bottom, channel_axis)
        # Check whether an array is dask array - if so, we need to compute it
        if hasattr(array_, "compute") and apply:
            return array_.compute(), (left, right, top, bottom)
        return array_, (left, right, top, bottom)

    def crop_polygon(self, yx: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop the image using polygon, however, try to minimize memory usage."""
        # Prepare polygon coordinates
        xy, (left, right, top, bottom) = _prepare_polygon(yx, self.inv_resolution, self.image_shape, multiply=True)
        # Crop the array using bounding box
        array, _ = self.crop_bbox(left, right, top, bottom, multiply=False, apply=True)
        # Adjust polygon coordinates for the cropped region
        xy_cropped = xy - np.array([left, top])
        # Create a mask only for the cropped region
        cropped_shape = self.get_image_shape_for_shape(array.shape)
        mask = _prepare_polygon_mask(cropped_shape, xy_cropped)
        # Apply mask to the cropped array
        channel_axis, _ = self.get_channel_axis_and_n_channels()
        array_ = _apply_mask(array, mask, channel_axis)
        # If the array is a dask array, compute it
        if hasattr(array_, "compute"):
            return array_.compute(), (left, right, top, bottom)
        return array_, (left, right, top, bottom)

    def crop_polygon_iter(
        self, yx: np.ndarray
    ) -> ty.Generator[tuple[np.ndarray | None, tuple[int, int, int, int]], None, None]:
        """Polygon iterator."""
        # Prepare polygon coordinates
        xy, (left, right, top, bottom) = _prepare_polygon(yx, self.inv_resolution, self.image_shape, multiply=True)
        # Adjust polygon coordinates for the cropped region
        xy_cropped = xy - np.array([left, top])
        yield None, (left, right, top, bottom)
        mask = None
        # Iterate over each channel and apply the mask
        for channel_id in range(self.n_channels):
            array = self.get_channel(channel_id, split_rgb=True)
            array_, _ = self.crop_bbox(left, right, top, bottom, array=array, multiply=False, apply=True)
            # Create mask only for the cropped region
            if mask is None:
                # Create mask only for the cropped region
                cropped_shape = self.get_image_shape_for_shape(array_.shape)
                mask = _prepare_polygon_mask(cropped_shape, xy_cropped)
            # Apply mask
            array_ *= mask
            # If the array is a dask array, compute it
            if hasattr(array_, "compute"):
                yield array_.compute(), (left, right, top, bottom)
            else:
                yield array_, (left, right, top, bottom)

    def crop_polygon_mask(self, yx: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop image."""
        # Prepare polygon coordinates
        xy, (left, right, top, bottom) = _prepare_polygon(yx, self.inv_resolution, self.image_shape, multiply=True)
        mask = _prepare_polygon_mask(self.image_shape, xy)
        # Get the array
        array = self.pyramid[0]
        channel_axis, _ = self.get_channel_axis_and_n_channels()
        array_ = _apply_mask(array, mask, channel_axis)
        # If the array is a dask array, compute it
        if hasattr(array_, "compute"):
            array_ = array_.compute()
        return self.crop_bbox(left, right, top, bottom, array=array_, multiply=False)

    def crop_polygon_mask_iter(
        self, yx: np.ndarray
    ) -> ty.Generator[tuple[np.ndarray | None, tuple[int, int, int, int]], None, None]:
        """Crop image."""
        # Prepare polygon coordinates
        xy, (left, right, top, bottom) = _prepare_polygon(yx, self.inv_resolution, self.image_shape, multiply=True)
        mask = _prepare_polygon_mask(self.image_shape, xy)
        yield None, (left, right, top, bottom)
        # Get the array
        for channel_id in range(self.n_channels):
            array_ = self.get_channel(channel_id)
            array_[~mask] = 0
            # If the array is a dask array, compute it
            if hasattr(array_, "compute"):
                array_ = array_.compute()
            yield self.crop_bbox(left, right, top, bottom, array=array_, multiply=False)

    def mask_region(
        self, bbox_or_yx: np.ndarray | tuple[int, int, int, int]
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Crop region."""
        if isinstance(bbox_or_yx, tuple) and len(bbox_or_yx) == 4:
            return self.mask_bbox(*bbox_or_yx)
        return self.mask_polygon(bbox_or_yx)

    def mask_region_iter(
        self, bbox_or_yx: np.ndarray | tuple[int, int, int, int]
    ) -> ty.Generator[tuple[np.ndarray | None, str], None, None]:
        """Crop region."""
        if isinstance(bbox_or_yx, tuple) and len(bbox_or_yx) == 4:
            yield from self.mask_bbox_iter(*bbox_or_yx)
        else:
            yield from self.mask_polygon_iter(bbox_or_yx)

    def mask_bbox(
        self,
        left: int,
        right: int,
        top: int,
        bottom: int,
        array: np.ndarray | None = None,
        multiply: bool = True,
        apply: bool = True,
    ) -> tuple[np.ndarray, str]:
        """Create a mask for the bounding box."""
        # Get the array from the pyramid or use the provided one
        array = self.pyramid[0] if array is None else array
        # Prepare bounding box coordinates
        left, right, top, bottom = _prepare_bbox(left, right, top, bottom, self.inv_resolution, multiply=multiply)
        mask = _prepare_bbox_mask(self.image_shape, left, right, top, bottom)
        # Apply mask to the cropped array
        channel_axis, _ = self.get_channel_axis_and_n_channels()
        array_ = _apply_mask(array, mask, channel_axis)
        # Check whether an array is dask array - if so, we need to compute it
        if hasattr(array_, "compute") and apply:
            return array_.compute(), (left, right, top, bottom)
        return array_, (left, right, top, bottom)

    def crop_bbox_iter(
        self,
        left: int,
        right: int,
        top: int,
        bottom: int,
        multiply: bool = True,
        apply: bool = True,
    ) -> ty.Generator[tuple[np.ndarray | None, str], None, None]:
        """Crop the image by iterating over each channel."""
        # Prepare bounding box coordinates
        left, right, top, bottom = _prepare_bbox(left, right, top, bottom, self.inv_resolution, multiply=multiply)
        yield None, (left, right, top, bottom)
        # Iterate over each channel and apply the mask
        for channel_id in range(self.n_channels):
            array = self.get_channel(channel_id, split_rgb=True)
            array_ = array[top:bottom, left:right]
            # Check whether an array is dask array - if so, we need to compute it
            if hasattr(array_, "compute") and apply:
                yield array_.compute(), (left, right, top, bottom)
            else:
                yield array_, (left, right, top, bottom)

    def mask_bbox_iter(
        self,
        left: int,
        right: int,
        top: int,
        bottom: int,
        multiply: bool = True,
        apply: bool = True,
    ) -> ty.Generator[tuple[np.ndarray | None, str], None, None]:
        """Mask the image by iterating ove each channel."""
        # Prepare bounding box coordinates
        left, right, top, bottom = _prepare_bbox(left, right, top, bottom, self.inv_resolution, multiply=multiply)
        hash_str = _hash_bbox_or_polygon(left=left, right=right, top=top, bottom=bottom)
        yield None, hash_str
        # Create mask for the bounding box
        mask = _prepare_bbox_mask(self.image_shape, left, right, top, bottom)
        # Iterate over each channel and apply the mask
        for channel_id in range(self.n_channels):
            array_ = self.get_channel(channel_id, split_rgb=True)
            array_ = array_.copy()
            array_[~mask] = 0
            # Check whether an array is dask array - if so, we need to compute it
            if hasattr(array_, "compute") and apply:
                yield array_.compute(), hash_str
            else:
                yield array_, hash_str

    def mask_polygon(self, yx: np.ndarray) -> tuple[np.ndarray, str]:
        """Mask the image."""
        array = self.pyramid[0]
        # Prepare polygon coordinates
        xy, (left, right, top, bottom) = _prepare_polygon(yx, self.inv_resolution, self.image_shape, multiply=True)
        hash_str = _hash_bbox_or_polygon(yx=yx)
        # Create a mask only for the cropped region
        mask = _prepare_polygon_mask(self.image_shape, xy)
        # Apply mask to the cropped array
        channel_axis, _ = self.get_channel_axis_and_n_channels()
        array_ = _apply_mask(array, mask, channel_axis)
        # If the array is a dask array, compute it
        if hasattr(array_, "compute"):
            return array_.compute(), hash_str
        return array_, hash_str

    def mask_polygon_iter(self, yx: np.ndarray) -> ty.Generator[tuple[np.ndarray | None, str], None, None]:
        """Polygon iterator."""
        # Prepare polygon coordinates
        xy, (left, right, top, bottom) = _prepare_polygon(yx, self.inv_resolution, self.image_shape, multiply=True)
        hash_str = _hash_bbox_or_polygon(yx=yx)
        yield None, (left, right, top, bottom)
        mask = None
        # Iterate over each channel and apply the mask
        for channel_id in range(self.n_channels):
            array_ = self.get_channel(channel_id, split_rgb=True)
            # Create mask only for the cropped region
            if mask is None:
                # Create the mask only for the cropped region
                mask = _prepare_polygon_mask(self.image_shape, xy)
            # Apply mask
            array_ *= mask
            # If the array is a dask array, compute it
            if hasattr(array_, "compute"):
                yield array_.compute(), hash_str
            else:
                yield array_, hash_str

    def warp(self, array: np.ndarray, affine: np.ndarray | None = None) -> np.ndarray:
        """Warp array."""
        from image2image_io.utils.mask import transform_mask

        if affine is None:
            affine = self.transform_data.compute(yx=True, px=True).params
        return transform_mask(array, affine, self.image_shape)

    def get_channel_axis_and_n_channels(self, shape: tuple[int, ...] | None = None) -> tuple[int | None, int]:
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
            # elif np.argmin(shape) == 2:
            #     channel_axis = 2
            #     n_channels = shape[2]
            else:  # channels first
                channel_axis = 0
                n_channels = shape[0]
        else:
            raise ValueError(f"Array has unsupported shape: {shape}")
        return channel_axis, n_channels

    def get_image_shape_for_shape(self, shape: tuple | None = None) -> tuple[int, int]:
        """Return shape of an image for a given shape."""
        if shape is None:
            shape = self.shape
        channel_axis, n_channels = self.get_channel_axis_and_n_channels(shape)
        if channel_axis is None or (self.is_rgb and not CONFIG.split_rgb):
            return shape[:2]
        if channel_axis == 0:
            return shape[1:]
        elif channel_axis == 1:
            return shape[0], shape[2]
        elif channel_axis == 2:
            return shape[:2]
        raise ValueError(f"Array has unsupported shape: {shape}")

    def get_channel(self, index: int, pyramid: int = 0, split_rgb: bool | None = None) -> np.ndarray:
        """Return channel."""
        split_rgb = split_rgb if split_rgb is not None else CONFIG.split_rgb

        array: np.ndarray = self.pyramid[pyramid]
        channel_axis, n_channels = self.get_channel_axis_and_n_channels()
        # if channel_axis in [None, 2] or (self.is_rgb and (not CONFIG.split_rgb and not split_rgb)):
        #     return self.preprocessor("rgb", array)
        if channel_axis is None:
            return self.preprocessor("gray", array)
        if channel_axis == 0:
            return self.preprocessor(self.channel_names[index], array[index])
        elif channel_axis == 1:
            return self.preprocessor(self.channel_names[index], array[:, index])
        elif channel_axis == 2:
            if self.is_rgb:
                if CONFIG.split_rgb or split_rgb:
                    return self.preprocessor("RGB"[index], array[:, :, index])
                return self.preprocessor("RGB", array)
            return self.preprocessor(self.channel_names[index], array[:, :, index])
        raise ValueError(f"Array has unsupported shape: {array.shape}")

    def get_channel_pyramid(self, index: int) -> list[np.ndarray]:
        """Return channel pyramid."""
        array = self.pyramid
        channel_axis, n_channels = self.get_channel_axis_and_n_channels()
        if channel_axis is None or (self.is_rgb and not CONFIG.split_rgb):
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

    @property
    def n_in_pyramid_quick(self) -> int | None:
        """Return number of images in the pyramid, if pyramid is available."""
        if self._pyramid is None:
            return None
        return self.n_in_pyramid

    def scale_for_pyramid(self, pyramid: int = 0) -> tuple[float, float]:
        """Return scale for pyramid."""
        if pyramid < 0:
            pyramid = list(range(self.n_in_pyramid))[pyramid]
            # pyramid = range(self.n_in_pyramid)[pyramid] + 1
        resolution = self.resolution * 2**pyramid
        return resolution, resolution

    def to_ome_tiff(
        self,
        path: PathLike,
        as_uint8: bool = False,
        tile_size: int = 512,
        channel_ids: list[int] | None = None,
        channel_names: list[str] | None = None,
        transformer: Transformer | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write image as OME-TIFF."""
        from image2image_io.writers import write_ome_tiff_alt

        return write_ome_tiff_alt(
            path,
            self,
            as_uint8=as_uint8,
            channel_names=channel_names,
            channel_ids=channel_ids,
            tile_size=tile_size,
            overwrite=overwrite,
            transformer=transformer,
        )

    def to_writer(self):
        """Get instance of writer."""
        from image2image_io.writers.tiff_writer import OmeTiffWriter

        writer = OmeTiffWriter(self, transformer=None)
        return writer


class DummyReader(BaseReader):
    """Dummy reader."""

    def __init__(self, resolution: float, channel_names: list[str], shape: tuple[int, ...], dtype: np.dtype):
        super().__init__("", "", auto_pyramid=False)
        self._channel_names = channel_names
        self._resolution = resolution
        self._array_dtype = dtype
        self.shape = shape

    def pyramid(self) -> list[np.ndarray]:
        """Pyramid."""
        return []
