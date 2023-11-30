"""Generic wrapper."""
from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger

from image2image_reader.config import CONFIG
from image2image_reader.exceptions import UnsupportedFileFormatError
from image2image_reader.models.transform import TransformData
from image2image_reader.wrapper import ImageWrapper

if ty.TYPE_CHECKING:
    from image2image_reader.readers._base_reader import BaseReader
    from image2image_reader.readers.array_reader import ArrayImageReader
    from image2image_reader.readers.coordinate_reader import CoordinateImageReader
    from image2image_reader.readers.czi_reader import CziImageReader, CziSceneImageReader
    from image2image_reader.readers.geojson_reader import GeoJSONReader
    from image2image_reader.readers.tiff_reader import TiffImageReader

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
TIFF_EXTENSIONS = [".scn", ".ome.tiff", ".tif", ".tiff", ".svs", ".ndpi", ".qptiff"]
CZI_EXTENSIONS = [".czi"]
BRUKER_EXTENSIONS = [".tsf", ".tdf", ".d"]
IMZML_EXTENSIONS = [".imzml"]
H5_EXTENSIONS = [".h5", ".hdf5"]
IMSPY_EXTENSIONS = [".data"]
NPY_EXTENSIONS = [".npy"]
GEOJSON_EXTENSIONS = [".geojson", ".json"]


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


def sanitize_read_path(path: PathLike, raise_error: bool = True) -> Path | None:
    """Sanitize the path that can be read by the reader."""
    path = Path(path)
    if not path.exists():
        if raise_error:
            raise FileNotFoundError(f"File does not exist: {path}")
        return None
    suffix = path.suffix.lower()
    if suffix not in (
        TIFF_EXTENSIONS
        + IMAGE_EXTENSIONS
        + CZI_EXTENSIONS
        + NPY_EXTENSIONS
        + BRUKER_EXTENSIONS
        + IMZML_EXTENSIONS
        + H5_EXTENSIONS
        + IMSPY_EXTENSIONS
        + GEOJSON_EXTENSIONS
    ):
        if raise_error:
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
) -> tuple[ImageWrapper, list[str], dict[Path, Path]]:
    """Read image data."""
    path = Path(path)
    path = sanitize_read_path(path)  # type: ignore[assignment]
    if not path:
        raise UnsupportedFileFormatError("Could not sanitize path - are you sure this file is supported?")

    readers: dict[str, BaseReader]
    name = path.name
    original_path = path
    path, readers = get_reader(path, split_czi)

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
    if name == "dataset.metadata.h5":
        name = path.parent.name
    if scene_index is not None:
        name = f"Scene={scene_index}; {name}"
    return name


def get_reader(path: Path, split_czi: bool | None = None) -> tuple[Path, dict[str, BaseReader]]:
    """Get reader for the specified path."""
    path = Path(path)
    path = sanitize_read_path(path)  # type: ignore[assignment]
    if not path:
        raise UnsupportedFileFormatError("Could not sanitize path - are you sure this file is supported?")

    split_czi = split_czi if split_czi is not None else CONFIG.split_czi

    readers: dict[str, BaseReader]
    suffix = path.suffix.lower()
    if suffix in TIFF_EXTENSIONS:
        logger.trace(f"Reading TIFF file: {path}")
        path, readers = _read_tiff(path)  # type: ignore
    elif suffix in CZI_EXTENSIONS:
        if split_czi and _check_multi_scene_czi(path):
            logger.trace(f"Reading multi-scene CZI file: {path}")
            path, readers = _read_multi_scene_czi(path)  # type: ignore
        else:
            logger.trace(f"Reading single-scene CZI file: {path}")
            path, readers = _read_single_scene_czi(path)  # type: ignore
    elif suffix in IMAGE_EXTENSIONS:
        logger.trace(f"Reading image file: {path}")
        path, readers = _read_image(path)  # type: ignore
    elif suffix in NPY_EXTENSIONS:
        logger.trace(f"Reading NPY file: {path}")
        path, readers = _read_npy_coordinates(path)  # type: ignore
    elif suffix in BRUKER_EXTENSIONS:
        logger.trace(f"Reading Bruker file: {path}")
        path, readers = _read_tsf_tdf_reader(path)  # type: ignore
    elif suffix in IMZML_EXTENSIONS:
        logger.trace(f"Reading imzML file: {path}")
        path, readers = _read_imzml_reader(path)  # type: ignore
    elif suffix in H5_EXTENSIONS + IMSPY_EXTENSIONS:
        logger.trace(f"Reading HDF5 file: {path}")
        if path.suffix == ".data":
            path = path / "dataset.metadata.h5"
        if path.name.startswith("dataset.metadata"):
            path, readers = _read_metadata_h5_coordinates(path)  # type: ignore
        else:
            path, readers = _read_centroids_h5_coordinates(path)  # type: ignore
    elif suffix in GEOJSON_EXTENSIONS:
        logger.trace(f"Reading GeoJSON file: {path}")
        path, readers = _read_geojson(path)  # type: ignore
    else:
        raise UnsupportedFileFormatError(f"Unsupported file format: '{suffix}'")
    return path, readers


def _check_multi_scene_czi(path: PathLike) -> bool:
    """Check whether this is a multi-scene CZI file."""
    from image2image_reader.readers._czi import CziSceneFile

    path = Path(path)
    return bool(CziSceneFile.get_num_scenes(path) > 1)


def _read_geojson(path: PathLike) -> tuple[Path, dict[str, GeoJSONReader]]:
    """Read GeoJSON file."""
    from image2image_reader.readers.geojson_reader import GeoJSONReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: GeoJSONReader(path, key=key)}


def _read_single_scene_czi(path: PathLike, **kwargs: ty.Any) -> tuple[Path, dict[str, CziImageReader]]:
    """Read CZI file."""
    from image2image_reader.readers.czi_reader import CziImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: CziImageReader(path, key=key, **kwargs)}


def _read_multi_scene_czi(path: PathLike) -> tuple[Path, dict[str, CziSceneImageReader]]:
    """Read CZI file."""
    from image2image_reader.readers._czi import CziSceneFile
    from image2image_reader.readers.czi_reader import CziSceneImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    n = CziSceneFile.get_num_scenes(path)
    logger.trace(f"Found {n} scenes in CZI file: {path}")
    return path, {
        f"S{i}_{path.name}": CziSceneImageReader(path, scene_index=i, key=get_key(path, scene_index=i))
        for i in range(n)
    }


def _read_tiff(path: PathLike) -> tuple[Path, dict[str, TiffImageReader]]:
    """Read TIFF file."""
    from image2image_reader.readers.tiff_reader import TiffImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: TiffImageReader(path, key=key)}


def _read_image(path: PathLike) -> tuple[Path, dict[str, ArrayImageReader]]:
    """Read image."""
    from skimage.io import imread

    from image2image_reader.readers.array_reader import ArrayImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: ArrayImageReader(path, imread(path), key=key)}


def _read_npy_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read data from npz or npy file."""
    from image2image_reader.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    with open(path, "rb") as f:
        image = np.load(f)
    assert image.ndim == 2, "Only 2D images are supported"
    y, x = get_yx_coordinates_from_shape(image.shape)
    key = get_key(path)
    return path, {path.name: CoordinateImageReader(path, x, y, array_or_reader=image, key=key)}


def _read_metadata_h5_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read coordinates from HDF5 file."""
    import h5py
    from koyo.json import read_json_data

    from image2image_reader.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix in H5_EXTENSIONS, "Only .h5 files are supported"

    # read coordinates
    with h5py.File(path, "r") as f:
        try:
            yx = f["Dataset/Spectral/Coordinates/yx"][:]
            tic = f["Dataset/Spectral/Sum/y"][:]
        except KeyError:
            yx = f["Dataset/Spatial/Coordinates/yx"][:]
            tic = f["Dataset/Spatial/Sum/y"][:]
    y = yx[:, 0]
    x = yx[:, 1]
    # read pixel size (resolution)
    resolution = 1.0
    if (path.parent / "metadata.json").exists():
        metadata = read_json_data(path.parent / "metadata.json")
        resolution = metadata["metadata.experimental"]["pixel_size"]
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


def _read_centroids_h5_coordinates_with_metadata(
    path: Path, metadata_file: Path
) -> tuple[Path, dict[str, CoordinateImageReader]]:
    import h5py

    from image2image_reader.utils.utilities import format_mz

    assert metadata_file.exists(), f"File does not exist: {metadata_file}"
    _, reader_ = _read_metadata_h5_coordinates(metadata_file)
    key = next(iter(reader_.keys()))
    reader = reader_[key]

    x = reader.x
    y = reader.y

    with h5py.File(path, "r") as f:
        mzs = f["Array"]["xs"][:]  # retrieve m/zs
        centroids = f["Array"]["array"][:]  # retrieve ion images
    mzs = [format_mz(mz) for mz in mzs]  # generate labels
    centroids = reshape_batch(x, y, centroids)  # reshape images
    reader.data.update(dict(zip(mzs, centroids)))
    return path, {path.name: reader}


def _read_centroids_h5_coordinates_without_metadata_lazy(path: Path) -> tuple[Path, dict[str, CoordinateImageReader]]:
    import h5py

    from image2image_reader.readers.coordinate_reader import CoordinateImageReader
    from image2image_reader.utils.utilities import format_mz

    with h5py.File(path, "r") as f:
        # get coordinate metadata
        x = f["Misc/Spatial/x_coordinates"][:]
        y = f["Misc/Spatial/y_coordinates"][:]
        resolution = float(f["Misc/Spatial"].attrs["pixel_size"])
        mzs = f["Array"]["xs"][:]  # retrieve m/zs
        centroids = f["Array"]["array"][:]  # retrieve ion images
    tic = np.random.randint(128, 255, len(x), dtype=np.uint8)
    tic = reshape(x, y, tic)
    key = get_key(path)
    reader = CoordinateImageReader(path, x, y, resolution=resolution, array_or_reader=tic, key=key)
    mzs = [format_mz(mz) for mz in mzs]  # generate labels
    centroids = reshape_batch(x, y, centroids)  # reshape images
    reader.data.update(dict(zip(mzs, centroids)))
    return path, {path.name: reader}


def _read_centroids_h5_coordinates_without_metadata(path: Path) -> tuple[Path, dict[str, CoordinateImageReader]]:
    import h5py

    from image2image_reader.readers.coordinate_reader import CoordinateImageReader
    from image2image_reader.utils.utilities import format_mz

    with h5py.File(path, "r") as f:
        # get coordinate metadata
        x = f["Misc/Spatial/x_coordinates"][:]
        y = f["Misc/Spatial/y_coordinates"][:]
        resolution = float(f["Misc/Spatial"].attrs["pixel_size"])
        mzs = f["Array"]["xs"][:]  # retrieve m/zs
        centroids = f["Array"]["array"][:]  # retrieve ion images
    tic = np.random.randint(128, 255, len(x), dtype=np.uint8)
    tic = reshape(x, y, tic)
    key = get_key(path)
    reader = CoordinateImageReader(path, x, y, resolution=resolution, array_or_reader=tic, key=key)
    mzs = [format_mz(mz) for mz in mzs]  # generate labels
    centroids = reshape_batch(x, y, centroids)  # reshape images
    reader.data.update(dict(zip(mzs, centroids)))
    return path, {path.name: reader}


def _read_tsf_tdf_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read coordinates from TSF file."""
    import sqlite3

    from image2image_reader.readers.coordinate_reader import CoordinateImageReader

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
    x = frame_index_position[:, 2]
    x = x - np.min(x)  # minimized
    y = frame_index_position[:, 3]
    y = y - np.min(y)  # minimized

    # get tic
    cursor = conn.execute("SELECT SummedIntensities FROM Frames")
    tic = np.array(cursor.fetchall())
    tic = tic[:, 0]
    key = get_key(path)
    return path.parent, {
        path.name: CoordinateImageReader(path, x, y, resolution=resolution, array_or_reader=reshape(x, y, tic), key=key)
    }


def _read_tsf_tdf_reader(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read coordinates from Bruker file."""
    import sqlite3

    from imzy import get_reader

    from image2image_reader.readers.coordinate_reader import CoordinateImageReader

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

    from image2image_reader.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix.lower() in IMZML_EXTENSIONS, "Only .imzML files are supported"

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

    from image2image_reader.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix.lower() in IMZML_EXTENSIONS, "Only .imzML files are supported"

    # get wrapper
    reader = get_reader(path)
    x = reader.x_coordinates
    x = x - np.min(x)  # minimized
    y = reader.y_coordinates
    y = y - np.min(y)  # minimized
    key = get_key(path)
    return path, {path.name: CoordinateImageReader(path, x, y, array_or_reader=reader, key=key)}


def reshape(x: np.ndarray, y: np.ndarray, array: np.ndarray, fill_value: float = 0) -> np.ndarray:
    """Reshape array."""
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    shape = (ymax - ymin + 1, xmax - xmin + 1)
    dtype = np.float32 if np.isnan(fill_value) else array.dtype
    new_array = np.full(shape, fill_value=fill_value, dtype=dtype)
    new_array[y - ymin, x - xmin] = array
    return new_array


def reshape_batch(x: np.ndarray, y: np.ndarray, array: np.ndarray, fill_value: float = 0) -> np.ndarray:
    """Batch reshaping of images."""
    if array.ndim != 2:
        raise ValueError("Expected 2-D array.")
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    y = y - ymin
    x = x - xmin
    n = array.shape[1]
    shape = (n, ymax - ymin + 1, xmax - xmin + 1)
    dtype = np.float32 if np.isnan(fill_value) else array.dtype
    im = np.full(shape, fill_value=fill_value, dtype=dtype)
    for i in range(n):
        im[i, y, x] = array[:, i]
    return im


def get_yx_coordinates_from_shape(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Get coordinates from image shape."""
    _y, _x = np.indices(shape)
    yx_coordinates = np.c_[np.ravel(_y), np.ravel(_x)]
    return yx_coordinates[:, 0], yx_coordinates[:, 1]
