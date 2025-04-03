"""Init."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from koyo.secret import hash_obj
from koyo.system import IS_MAC
from koyo.typing import PathLike
from loguru import logger

from image2image_io.config import CONFIG
from image2image_io.exceptions import UnsupportedFileFormatError
from image2image_io.models.transform import TransformData
from image2image_io.readers._base_reader import BaseReader
from image2image_io.readers.array_reader import ArrayImageReader
from image2image_io.readers.coordinate_reader import CoordinateImageReader, LazyCoordinateImageReader
from image2image_io.readers.czi_reader import CziImageReader, CziSceneImageReader
from image2image_io.readers.shapes_reader import ShapesReader, is_txt_and_has_columns
from image2image_io.readers.tiff_reader import TiffImageReader
from image2image_io.utils.utilities import get_yx_coordinates_from_shape, reshape, reshape_batch
from image2image_io.wrapper import ImageWrapper

__all__ = [
    "ArrayImageReader",
    "BaseReader",
    "CoordinateImageReader",
    "CziImageReader",
    "CziSceneImageReader",
    "LazyCoordinateImageReader",
    "ShapesReader",
    "TiffImageReader",
    "get_key",
    "is_supported",
    "read_data",
    "sanitize_path",
    "sanitize_read_path",
]


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
TIFF_EXTENSIONS = [
    ".scn",
    ".ome.tiff",
    ".tif",
    ".tiff",
    ".svs",
    ".ndpi",
    ".qptiff",
    ".qptiff.raw",
    ".qptiff.intermediate",
]
CZI_EXTENSIONS = [".czi"]
BRUKER_EXTENSIONS = [".tsf", ".tdf", ".d"]
IMZML_EXTENSIONS = [".imzml", ".ibd"]
H5_EXTENSIONS = [".h5", ".hdf5"]
IMSPY_EXTENSIONS = [".data"]
NPY_EXTENSIONS = [".npy"]
NPZ_EXTENSIONS = [".npz"]
GEOJSON_EXTENSIONS = [".geojson", ".json"]
POINTS_EXTENSIONS = [".csv", ".txt", ".parquet"]
SUPPORTED_IMAGE_FORMATS = [
    *IMAGE_EXTENSIONS,
    *TIFF_EXTENSIONS,
    *CZI_EXTENSIONS,
]


def sanitize_path(path: PathLike) -> Path:
    """Sanitize a path, so it has a unified format across models."""
    path = Path(path).resolve()
    if path.is_file():
        if path.suffix in [".tsf", ".tdf"]:
            path = path.parent
        elif path.name == "dataset.metadata.h5":
            path = path.parent
    return path


def get_alternative_path(path: PathLike) -> Path:
    """Retrieve alternative name."""
    path = Path(path)
    if path.is_dir() and path.suffix == ".data":
        return path / "dataset.metadata.h5"
    return path


def sanitize_read_path(path: PathLike, raise_on_error: bool = True) -> Path | None:
    """Sanitize the path that can be read by the reader."""
    path = Path(path)
    if not path.exists() and raise_on_error:
        raise FileNotFoundError(f"File does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in (
        TIFF_EXTENSIONS
        + IMAGE_EXTENSIONS
        + CZI_EXTENSIONS
        + NPY_EXTENSIONS
        + NPZ_EXTENSIONS
        + BRUKER_EXTENSIONS
        + IMZML_EXTENSIONS
        + H5_EXTENSIONS
        + IMSPY_EXTENSIONS
        + GEOJSON_EXTENSIONS
        + POINTS_EXTENSIONS
        + [".tmp"]
    ):
        if raise_on_error:
            raise ValueError(f"Unsupported file format: {path.suffix} ({path})")
        return None
    if suffix in BRUKER_EXTENSIONS:
        if path.suffix == ".d":
            if (path / "analysis.tsf").exists():
                path = path / "analysis.tsf"
            else:
                path = path / "analysis.tdf"
    elif suffix in IMSPY_EXTENSIONS:
        path = path / "dataset.metadata.h5"
    return path


def read_data(
    path: PathLike,
    wrapper: ImageWrapper | None = None,
    is_fixed: bool = False,
    transform_data: TransformData | None = None,
    resolution: float | None = None,
    split_czi: bool | None = None,
    reader_kws: dict[str, ty.Any] | None = None,
) -> tuple[ImageWrapper, list[str], dict[Path, Path]]:
    """Read image data."""
    path = Path(path)
    path = sanitize_read_path(path)  # type: ignore[assignment]
    if not path:
        raise UnsupportedFileFormatError("Could not sanitize path - are you sure this file is supported?")

    readers: dict[str, BaseReader]
    name = path.name
    original_path = path
    scene_index = reader_kws.get("scene_index", None) if reader_kws else None
    path, readers = get_reader(path, split_czi=split_czi, scene_index=scene_index)

    # add transformation information if provided
    if transform_data is not None:
        for reader_ in readers.values():
            reader_.transform_data = transform_data
            reader_.transform_name = name
    # add resolution information if provided
    if resolution is not None:
        for reader_ in readers.values():
            reader_.resolution = resolution
    # specify whether the model is fixed
    if hasattr(readers, "is_fixed"):
        readers.is_fixed = is_fixed

    # initialize image wrapper, unless it's been provided
    if wrapper is None:
        wrapper = ImageWrapper(None)
    # add readers to the wrapper
    just_added = []
    for reader_ in readers.values():
        wrapper.add(reader_)
        just_added.append(reader_.key)
    return wrapper, just_added, {original_path: path}


def get_key(path: Path, scene_index: int | None = None) -> str:
    """Return representative key."""
    name = path.name
    hash_of_path = hash_obj(path, n_in_hash=3)
    if name in ["dataset.metadata.h5", "analysis.tsf", "analysis.tdf"]:
        name = path.parent.name
    if scene_index is not None:
        name = f"Scene={scene_index}; {name}"
    return f"{name}-{hash_of_path}"


def is_supported(path: PathLike, raise_on_error: bool = True) -> bool:
    """Check whether the file is supported."""
    path = Path(path)
    path = sanitize_read_path(path, raise_on_error)  # type: ignore[assignment]
    if not path:
        raise UnsupportedFileFormatError("Could not sanitize path - are you sure this file is supported?")
    return path is not None


def get_reader(
    path: Path,
    split_czi: bool | None = None,
    split_roi: bool | None = None,
    quick: bool = False,
    scene_index: int | None = None,
) -> tuple[Path, dict[str, BaseReader]]:
    """Get reader for the specified path."""
    path = Path(path)
    path = sanitize_read_path(path)  # type: ignore[assignment]
    if not path:
        raise UnsupportedFileFormatError("Could not sanitize path - are you sure this file is supported?")

    split_czi = split_czi if split_czi is not None else CONFIG.split_czi
    split_roi = split_roi if split_roi is not None else CONFIG.split_roi
    include_all = True

    if scene_index is not None:
        split_czi = split_roi = True
        include_all = False

    readers: dict[str, BaseReader]
    suffix = path.suffix.lower()
    if suffix in TIFF_EXTENSIONS:
        CONFIG.trace(f"Reading TIFF file: {path}")
        path, readers = _read_tiff(path)
    elif suffix in CZI_EXTENSIONS:
        if split_czi and _check_multi_scene_czi(path):
            logger.trace(f"Reading multi-scene CZI file: {path}")
            path, readers = _read_multi_scene_czi(path, scene_index=scene_index)
        else:
            CONFIG.trace(f"Reading single-scene CZI file: {path}")
            path, readers = _read_single_scene_czi(path)
    elif suffix in IMAGE_EXTENSIONS:
        CONFIG.trace(f"Reading image file: {path}")
        path, readers = _read_image(path)
    elif suffix in NPY_EXTENSIONS:
        CONFIG.trace(f"Reading NPY file: {path}")
        path, readers = _read_npy_coordinates(path)
    elif suffix in NPZ_EXTENSIONS:
        CONFIG.trace(f"Reading NPY file: {path}")
        path, readers = _read_npz_coordinates(path)
    elif suffix in BRUKER_EXTENSIONS:
        CONFIG.trace(f"Reading Bruker file: {path}")
        if IS_MAC or split_roi:
            path, readers = _read_tsf_tdf_coordinates(path, split_roi, scene_index=scene_index, include_all=include_all)
        else:
            path, readers = _read_tsf_tdf_reader(path)
    elif suffix in IMZML_EXTENSIONS:
        CONFIG.trace(f"Reading imzML file: {path}")
        path, readers = _read_imzml_reader(path)
    elif suffix in H5_EXTENSIONS + IMSPY_EXTENSIONS:
        CONFIG.trace(f"Reading HDF5 file: {path}")
        if path.suffix == ".data":
            path = path / "dataset.metadata.h5"
        if path.name.startswith("dataset.metadata"):
            path, readers = _read_metadata_h5_coordinates(path)
        elif _is_h5_mask(path):
            path, readers = _read_mask_h5(path)
        else:
            path, readers = _read_centroids_h5_coordinates_lazy(path)
    elif suffix in GEOJSON_EXTENSIONS + POINTS_EXTENSIONS:
        if (
            suffix in GEOJSON_EXTENSIONS
            or is_txt_and_has_columns(path, ["vertex_x", "vertex_y"], [("cell", "cell_id", "shape", "shape_name")])
            or is_txt_and_has_columns(
                path, ["x", "y"], [("cell", "cell_id", "shape", "shape_name")], (np.dtype("O"), np.dtype("S"))
            )
        ):
            CONFIG.trace(f"Reading shape file: {path}")
            path, readers = _read_shapes(path)
        else:
            CONFIG.trace(f"Reading points file: {path}")
            path, readers = _read_points(path)  # type: ignore
    else:
        raise UnsupportedFileFormatError(f"Unsupported file format: '{suffix}'")
    return path, readers


def get_simple_reader(
    path: PathLike,
    init_pyramid: bool = True,
    auto_pyramid: bool = True,
    quick: bool = False,
    quiet: bool = False,
    scene_index: int | None = None,
) -> BaseReader:
    """Get simple reader."""
    quiet_ = CONFIG.quiet
    CONFIG.quiet = quiet
    init_pyramid_ = CONFIG.init_pyramid
    CONFIG.init_pyramid = init_pyramid
    auto_pyramid_ = CONFIG.auto_pyramid
    CONFIG.auto_pyramid = auto_pyramid
    path, readers = get_reader(path, split_czi=False, quick=quick, scene_index=scene_index)
    CONFIG.init_pyramid = init_pyramid_
    CONFIG.auto_pyramid = auto_pyramid_
    CONFIG.quiet = quiet_
    return next(iter(readers.values()))


def _check_multi_scene_czi(path: PathLike) -> bool:
    """Check whether this is a multi-scene CZI file."""
    from image2image_io.readers._czi import CziSceneFile

    path = Path(path)
    return bool(CziSceneFile.get_num_scenes(path) > 1)


def _read_shapes(path: PathLike) -> tuple[Path, dict[str, ShapesReader]]:
    """Read GeoJSON file."""
    from image2image_io.readers.shapes_reader import ShapesReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: ShapesReader(path, key=key)}


def _read_points(path: PathLike) -> tuple[Path, dict[str, ShapesReader]]:
    """Read GeoJSON file."""
    from image2image_io.readers.points_reader import PointsReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: PointsReader(path, key=key)}


def _read_single_scene_czi(path: PathLike, **kwargs: ty.Any) -> tuple[Path, dict[str, CziImageReader]]:
    """Read CZI file."""
    from image2image_io.readers.czi_reader import CziImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: CziImageReader(path, key=key, **kwargs)}


def _read_multi_scene_czi(
    path: PathLike, scene_index: int | None = None
) -> tuple[Path, dict[str, CziSceneImageReader]]:
    """Read CZI file."""
    from image2image_io.readers._czi import CziSceneFile
    from image2image_io.readers.czi_reader import CziSceneImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    n = CziSceneFile.get_num_scenes(path)
    scenes = list(range(n))
    if scene_index is not None:
        if scene_index not in scenes:
            raise ValueError(f"Scene index {scene_index} not found in the file")
        scenes = [scene_index]

    CONFIG.trace(f"Found {n} scenes in CZI file: {path}")
    return path, {
        f"S{i}_{path.name}": CziSceneImageReader(path, scene_index=i, key=get_key(path, scene_index=i)) for i in scenes
    }


def _read_tiff(path: PathLike) -> tuple[Path, dict[str, TiffImageReader]]:
    """Read TIFF file."""
    from image2image_io.readers.tiff_reader import TiffImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: TiffImageReader(path, key=key)}


def _read_image(path: PathLike) -> tuple[Path, dict[str, ArrayImageReader]]:
    """Read image."""
    from PIL import Image
    from skimage.io import imread

    # disable decompression bomb protection
    Image.MAX_IMAGE_PIXELS = 30_000 * 30_000  # 30k x 30k pixels

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: ArrayImageReader(path, imread(path), key=key)}


def _read_npy_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read data from npy file."""
    from image2image_io.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    with open(path, "rb") as f:
        image = np.load(f)
    assert image.ndim == 2, "Only 2D images are supported"
    y, x = get_yx_coordinates_from_shape(image.shape)
    key = get_key(path)
    return path, {path.name: CoordinateImageReader(path, x, y, array_or_reader=image, key=key)}


def _read_npz_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read data from npy file."""
    from image2image_io.readers.array_reader import ArrayImageReader

    path = Path(path)
    with np.load(path) as f:
        if "labels" in f:
            labels = f["labels"]
        elif "ppm_labels" in f:
            labels = f["ppm_labels"]
        else:
            raise ValueError("Labels not found in the file")
        labels = labels.tolist()
        if "array" in f:
            image = f["array"]
        elif "image" in f:
            image = f["image"]
        else:
            raise ValueError("Image not found in the file")
    assert image.ndim == 3, "Only 3D images are supported"
    key = get_key(path)
    return path, {path.name: ArrayImageReader(path, array=image, key=key, channel_names=labels)}


def _get_resolution_from_metadata(path: PathLike) -> float:
    """Get resolution from metadata."""
    from koyo.json import read_json_data

    resolution = 1.0
    path = Path(path)
    if path.name != "metadata.json":
        path_ = path / "metadata.json"
        if not path_.exists():
            path_ = path.parent / "metadata.json"
        path = path_
    if path.exists():
        metadata = read_json_data(path)
        resolution = metadata["metadata.experimental"]["pixel_size"]
    return resolution


def _read_metadata_h5_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read coordinates from HDF5 file."""
    import h5py

    from image2image_io.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix in H5_EXTENSIONS, "Only .h5 files are supported"

    # read coordinates
    with h5py.File(path, "r") as f:
        try:
            yx = f["Dataset/Spatial/Coordinates/yx"][:]
            tic = f["Dataset/Spatial/Sum/y"][:]
        except KeyError:
            yx = f["Dataset/Spectral/Coordinates/yx"][:]
            tic = f["Dataset/Spectral/Sum/y"][:]

    y = yx[:, 0]
    x = yx[:, 1]
    resolution = _get_resolution_from_metadata(path)
    key = get_key(path)
    return path, {
        path.name: CoordinateImageReader(path, x, y, resolution=resolution, array_or_reader=reshape(x, y, tic), key=key)
    }


def _read_centroids_h5_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read centroids data from HDF5 file."""
    path = Path(path)
    assert path.suffix in H5_EXTENSIONS, "Only .h5 files are supported"

    metadata_file = path.parent / "dataset.metadata.h5"
    if not metadata_file.exists():
        data_dir = path.parent.parent.with_suffix(".data")
        metadata_file = data_dir / "dataset.metadata.h5"
    if metadata_file.exists():
        return _read_centroids_h5_coordinates_with_metadata(path, metadata_file)
    return _read_centroids_h5_coordinates_without_metadata(path)


def _is_h5_mask(path: PathLike) -> bool:
    """Check whether this is a mask file."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        return "Mask" in f


def _read_mask_h5(path: PathLike) -> tuple[Path, dict[str, ArrayImageReader]]:
    """Read centroids data from HDF5 file."""
    import h5py

    path = Path(path)
    assert path.suffix in H5_EXTENSIONS, "Only .h5 files are supported"

    # read coordinates
    with h5py.File(path, "r") as f:
        mask = f["Mask/mask"][:]
    resolution = 1.0
    if path.parent.name == "Masks":
        resolution = _get_resolution_from_metadata(path.parent.parent.with_suffix(".data"))
    key = get_key(path)
    return path, {path.name: ArrayImageReader(path, mask, resolution=resolution, key=key)}


def _read_centroids_h5_coordinates_lazy(path: PathLike) -> tuple[Path, dict[str, LazyCoordinateImageReader]]:
    """Read centroids data from HDF5 file."""
    path = Path(path)
    assert path.suffix in H5_EXTENSIONS, "Only .h5 files are supported"

    metadata_file = path.parent / "dataset.metadata.h5"
    if not metadata_file.exists():
        data_dir = path.parent.parent.with_suffix(".data")
        metadata_file = data_dir / "dataset.metadata.h5"
    if metadata_file.exists():
        return _read_centroids_h5_coordinates_with_metadata_lazy(path, metadata_file)
    return _read_centroids_h5_coordinates_without_metadata_lazy(path)


def _read_centroids_h5_coordinates_with_metadata(
    path: Path, metadata_file: Path
) -> tuple[Path, dict[str, CoordinateImageReader]]:
    import h5py

    from image2image_io.utils.utilities import format_mz

    assert metadata_file.exists(), f"File does not exist: {metadata_file}"
    _, reader_ = _read_metadata_h5_coordinates(metadata_file)
    key = next(iter(reader_.keys()))
    reader = reader_[key]

    x = reader.x
    y = reader.y

    with h5py.File(path, "r") as f:
        if "xs" in f["Array"]:
            mzs = f["Array"]["xs"][:]  # retrieve m/zs
            labels = [format_mz(mz) for mz in mzs]  # generate labels
        else:
            labels = f["Annotations/annotations/annotations"][:]
            labels = [label.decode() for label in labels]
        centroids = f["Array"]["array"][:]  # retrieve ion images
    centroids = reshape_batch(x, y, centroids)  # reshape images
    reader.data.update(dict(zip(labels, centroids)))
    return path, {path.name: reader}


def _read_centroids_h5_coordinates_with_metadata_lazy(
    path: Path, metadata_file: Path
) -> tuple[Path, dict[str, LazyCoordinateImageReader]]:
    import h5py

    from image2image_io.utils.lazy import LazyImageWrapper
    from image2image_io.utils.utilities import format_mz

    assert metadata_file.exists(), f"File does not exist: {metadata_file}"
    _, readers = _read_metadata_h5_coordinates(metadata_file)
    key = next(iter(readers.keys()))
    reader_ = readers[key]

    x = reader_.x
    y = reader_.y
    resolution = reader_.resolution

    with h5py.File(path, "r") as f:
        if "xs" in f["Array"]:
            mzs = f["Array"]["xs"][:]  # retrieve m/zs
            labels = [format_mz(mz) for mz in mzs]  # generate labels
        else:
            labels = f["Annotations/annotations/annotations"][:]
            labels = [label.decode() for label in labels]
    lazy_wrapper = LazyImageWrapper(path, "Array/array", labels, x, y)
    key = get_key(path)
    reader = LazyCoordinateImageReader(
        path, x, y, resolution=resolution, lazy_wrapper=lazy_wrapper, key=key, channel_names=labels
    )
    return path, {path.name: reader}


def _read_centroids_h5_coordinates_without_metadata(path: Path) -> tuple[Path, dict[str, CoordinateImageReader]]:
    import h5py

    from image2image_io.readers.coordinate_reader import CoordinateImageReader
    from image2image_io.utils.utilities import format_mz

    with h5py.File(path, "r") as f:
        # get coordinate metadata
        x = f["Misc/Spatial/x_coordinates"][:]
        y = f["Misc/Spatial/y_coordinates"][:]
        resolution = float(f["Misc/Spatial"].attrs["pixel_size"])
        if "xs" in f["Array"]:
            mzs = f["Array"]["xs"][:]  # retrieve m/zs
            labels = [format_mz(mz) for mz in mzs]  # generate labels
        else:
            labels = f["Annotations/annotations/annotations"][:]
            labels = [label.decode() for label in labels]
        centroids = f["Array"]["array"][:]  # retrieve ion images
        tic = np.random.randint(128, 255, len(x), dtype=np.uint8)
    tic = reshape(x, y, tic)
    key = get_key(path)
    reader = CoordinateImageReader(path, x, y, resolution=resolution, array_or_reader=tic, key=key)
    centroids = reshape_batch(x, y, centroids)  # reshape images
    reader.data.update(dict(zip(labels, centroids)))
    return path, {path.name: reader}


def _read_centroids_h5_coordinates_without_metadata_lazy(
    path: Path,
) -> tuple[Path, dict[str, LazyCoordinateImageReader]]:
    import h5py

    from image2image_io.readers.coordinate_reader import LazyCoordinateImageReader
    from image2image_io.utils.lazy import LazyImageWrapper
    from image2image_io.utils.utilities import format_mz

    with h5py.File(path, "r") as f:
        # get coordinate metadata
        x = f["Misc/Spatial/x_coordinates"][:]
        y = f["Misc/Spatial/y_coordinates"][:]
        resolution = float(f["Misc/Spatial"].attrs["pixel_size"])
        if "xs" in f["Array"]:
            mzs = f["Array"]["xs"][:]  # retrieve m/zs
            labels = [format_mz(mz) for mz in mzs]  # generate labels
        else:
            labels = f["Annotations/annotations/annotations"][:]
            labels = [label.decode() for label in labels]
    lazy_wrapper = LazyImageWrapper(path, "Array/array", labels, x, y)
    key = get_key(path)
    reader = LazyCoordinateImageReader(
        path, x, y, resolution=resolution, lazy_wrapper=lazy_wrapper, key=key, channel_names=labels
    )
    return path, {path.name: reader}


def _read_tsf_tdf_coordinates(
    path: PathLike,
    split_roi: bool = True,
    include_all: bool = True,
    scene_index: int | None = None,
) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read coordinates from TSF file."""
    import sqlite3

    from image2image_io.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix in BRUKER_EXTENSIONS, "Only .tsf and .tdf files are supported"

    if path.suffix == ".d":
        if (path / "analysis.tsf").exists():
            path = path / "analysis.tsf"
        else:
            path = path / "analysis.tdf"

    # get wrapper
    conn = sqlite3.connect(path)

    try:
        cursor = conn.execute("SELECT SpotSize FROM MaldiFrameLaserInfo")
        resolution = float(cursor.fetchone()[0])
    except sqlite3.OperationalError:
        resolution = 1.0

    # get coordinates
    cursor = conn.cursor()
    cursor.execute("SELECT Frame, RegionNumber, XIndexPos, YIndexPos FROM MaldiFrameInfo")
    frame_index_position = np.array(cursor.fetchall())

    # get tic
    cursor = conn.execute("SELECT SummedIntensities FROM Frames")
    tic = np.array(cursor.fetchall())
    tic = tic[:, 0]

    # generate reader(s)
    readers = {}
    roi = frame_index_position[:, 1]
    unq_roi = np.unique(roi)
    if len(unq_roi) == 1:
        split_roi = False
        scene_index = None
        include_all = True

    # check whether scene index is correct
    if scene_index is not None:
        if scene_index not in unq_roi:
            raise ValueError(f"Scene index {scene_index} not found in the file")
        include_all = False
        unq_roi = [scene_index]
        split_roi = True

    # if we are not on macOS and there is only 1 ROI, we might as well get the proper data reader
    if not IS_MAC and not split_roi:
        return _read_tsf_tdf_reader(path)

    if include_all or not split_roi:  # or len(unq_roi) == 1):
        x = frame_index_position[:, 2]
        x = x - np.min(x)  # minimized
        y = frame_index_position[:, 3]
        y = y - np.min(y)  # minimized
        key = get_key(path)
        readers[path.name] = CoordinateImageReader(
            path, x, y, resolution=resolution, array_or_reader=reshape(x, y, tic), key=key
        )

    if split_roi:
        for current_scene_index in unq_roi:
            mask = roi == current_scene_index
            x_ = frame_index_position[mask, 2]
            x_ = x_ - np.min(x_)
            y_ = frame_index_position[mask, 3]
            y_ = y_ - np.min(y_)
            tic_ = tic[mask]
            key = get_key(path, current_scene_index)
            readers[f"S{current_scene_index}_{path.name}"] = CoordinateImageReader(
                path,
                x_,
                y_,
                resolution=resolution,
                array_or_reader=reshape(x_, y_, tic_),
                key=key,
                reader_kws={"scene_index": current_scene_index},
            )
    return path.parent, readers


def _read_tsf_tdf_reader(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read coordinates from Bruker file."""
    import sqlite3

    from imzy import get_reader

    from image2image_io.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix in BRUKER_EXTENSIONS, "Only .tsf and .tdf files are supported"

    if path.suffix == ".d":
        if (path / "analysis.tsf").exists():
            path = path / "analysis.tsf"
        else:
            path = path / "analysis.tdf"

    conn = sqlite3.connect(path)
    try:
        cursor = conn.execute("SELECT SpotSize FROM MaldiFrameLaserInfo")
        resolution = float(cursor.fetchone()[0])
    except sqlite3.OperationalError:
        resolution = 1.0
    conn.close()

    # get wrapper
    path = path.parent
    reader = get_reader(path)
    x = reader.x_coordinates
    y = reader.y_coordinates
    key = get_key(path)
    return path, {path.name: CoordinateImageReader(path, x, y, resolution=resolution, array_or_reader=reader, key=key)}


def _read_imzml_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read coordinates from imzML file."""
    from imzy import get_reader

    from image2image_io.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix.lower() in IMZML_EXTENSIONS, "Only .imzML files are supported"
    if path.suffix.lower() == ".ibd":
        path = path.with_suffix(".imzML")

    # get wrapper
    reader = get_reader(path)
    x = reader.x_coordinates
    x = x - np.min(x)  # minimized
    y = reader.y_coordinates
    y = y - np.min(y)  # minimized
    tic = reader.get_tic()
    key = get_key(path)
    return path, {path.name: CoordinateImageReader(path, x, y, array_or_reader=reshape(x, y, tic), key=key)}


def _read_imzml_reader(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read coordinates from imzML file."""
    from imzy import get_reader

    from image2image_io.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix.lower() in IMZML_EXTENSIONS, "Only .imzML files are supported"
    if path.suffix.lower() == ".ibd":
        path = path.with_suffix(".imzML")

    # get wrapper
    reader = get_reader(path)
    x = reader.x_coordinates
    x = x - np.min(x)  # minimized
    y = reader.y_coordinates
    y = y - np.min(y)  # minimized
    key = get_key(path)
    return path, {path.name: CoordinateImageReader(path, x, y, array_or_reader=reader, key=key)}


def get_czi_metadata(path: PathLike) -> dict[str, dict[str, ty.Any]]:
    """Read CZI metadata."""
    from image2image_io.readers._czi import CziSceneFile

    path = Path(path)
    assert path.suffix == ".czi", "Only .czi files are supported"

    metadata = {}
    n = CziSceneFile.get_num_scenes(path)
    if n == 1:
        metadata[get_key(path)] = {"path": path, "scene_index": None}
    else:
        for scene_index in range(n):
            metadata[get_key(path, scene_index)] = {"path": path, "scene_index": scene_index}
    return metadata
