"""Writer functions."""
from __future__ import annotations

import typing as ty
from pathlib import Path

import numba
import numpy as np
from koyo.system import IS_MAC_ARM
from koyo.typing import PathLike
from loguru import logger

from image2image_io.utils.utilities import (
    get_dtype_for_array,
    get_flat_shape_of_image,
    get_shape_of_image,
    reshape_fortran,
)

if ty.TYPE_CHECKING:
    from image2image_io.readers._base_reader import BaseReader


def czis_to_ome_tiff(
    paths: ty.Iterable[PathLike], output_dir: PathLike | None = None
) -> ty.Generator[tuple[str, int, int, int, int], None, None]:
    """Convert multiple CZI images to OME-TIFF."""
    from image2image_io.readers._czi import CziSceneFile

    # calculate true total number of scenes
    total_n_scenes = 0
    paths_ = []
    for path_ in paths:
        path_ = Path(path_)
        if path_.is_dir():
            for path__ in path_.glob("**/*.czi"):
                total_n_scenes += CziSceneFile.get_num_scenes(path__)
                paths_.append(path__)
        else:
            total_n_scenes += CziSceneFile.get_num_scenes(path_)
            paths_.append(path_)

    current = 0
    for path_ in paths_:
        path_ = Path(path_)
        for key, current_file_scene, total_file_scenes in czi_to_ome_tiff(path_, output_dir):
            yield key, current_file_scene, total_file_scenes, current, total_n_scenes
            current += 1


def czi_to_ome_tiff(
    path: PathLike, output_dir: PathLike | None = None
) -> ty.Generator[tuple[str, int, int], None, None]:
    """Convert Czi image to OME-TIFF."""
    from image2image_io._reader import get_key
    from image2image_io.readers._czi import CziSceneFile
    from image2image_io.readers.czi_reader import CziSceneImageReader

    path = Path(path)
    key = get_key(path)
    if output_dir is None:
        output_dir = path.parent
    output_dir = Path(output_dir)
    try:
        n = CziSceneFile.get_num_scenes(path)
        yield key, 0, n
    except Exception as e:
        logger.error(f"Could not read Czi file {path} - {e}")
        return

    # iterate over each scene in the czi file
    for scene_index in range(n):
        filename = path.name.replace(".czi", "") + (f"_scene={scene_index:02d}" if n > 1 else "")
        output_path = output_dir / filename
        # skip if the output file already exists
        if output_path.with_suffix(".ome.tiff").exists():
            logger.info(f"Skipping {output_path} - already exists")
            yield key, scene_index + 1, n
            continue

        # read the scene
        reader = CziSceneImageReader(path, scene_index=scene_index, auto_pyramid=False, init_pyramid=True)
        write_ome_tiff_alt(output_path, reader)
        yield key, scene_index + 1, n


def write_ome_tiff_from_array(path: PathLike, reader: BaseReader, array: np.ndarray) -> Path:
    """Write OME-TIFF by also specifying an array."""
    from image2image_io.readers.array_reader import ArrayImageReader
    from image2image_io.writers.tiff_writer import OmeTiffWriter

    if array.ndim == 2:
        array = np.atleast_3d(array)

    array_reader = ArrayImageReader("", array, resolution=reader.resolution, channel_names=reader.channel_names)

    path = Path(path)
    filename = path.name.replace(".ome.tiff", "")
    writer = OmeTiffWriter(array_reader)
    output_path = writer.write_image_by_plane(filename, path.parent, write_pyramid=True)
    return output_path


def write_ome_tiff_alt(path: PathLike, reader: BaseReader) -> Path:
    """Write OME-TIFF."""
    from image2image_io.writers.tiff_writer import OmeTiffWriter

    path = Path(path)
    filename = path.name.replace(".ome.tiff", "")
    writer = OmeTiffWriter(reader)
    output_path = writer.write_image_by_plane(filename, path.parent, write_pyramid=True)
    return output_path


def images_to_fusion(
    paths: ty.Iterable[PathLike], output_dir: PathLike | None = None
) -> ty.Generator[tuple[str, int, int, str], None, None]:
    """Convert multiple images to Fusion."""
    for path_ in paths:
        path_ = Path(path_)
        yield from image_to_fusion(path_, output_dir)


def image_to_fusion(
    path: PathLike, output_dir: PathLike | None = None
) -> ty.Generator[tuple[str, int, int, str], None, None]:
    """Convert image of any type to Fusion format."""
    from image2image_io._reader import get_reader

    path = Path(path)
    path, readers = get_reader(path)
    for reader in readers.values():
        key = reader.key
        if output_dir is None:
            output_dir = path.parent
        output_dir = Path(output_dir)
        filename = output_dir / path.stem.replace(".ome", "")
        yield from write_reader_to_txt(reader, filename, key=key)
        logger.trace(f"Exported Fusion CSV for {path}")
        if not filename.with_suffix(".xm").exists():
            write_reader_to_xml(reader, filename)
            logger.trace(f"Exported XML metadata for {path}")
        del reader


def write_reader_to_xml(reader: BaseReader, filename: PathLike) -> None:
    """Get filename."""
    from xml.dom.minidom import parseString

    from dicttoxml import dicttoxml

    _, _, image_shape = get_shape_of_image(reader.shape)
    shape = get_flat_shape_of_image(reader.shape)

    meta = {
        "modality": "microscopy",
        "data_label": reader.path.stem,
        "nr_spatial_dims": 2,
        "spatial_grid_size": f"{image_shape[0]} {image_shape[1]}",
        "nr_spatial_grid_elems": image_shape[0] * image_shape[1],
        "spatial_resolution_um": reader.resolution,
        "nr_obs": shape[0],
        "nr_vars": shape[1],
    }
    xml = dicttoxml(meta, custom_root="data_source", attr_type=False)

    # filename = xml_output_path.parent / (reader.path.stem + ".xml")
    filename = Path(filename).with_suffix(".xml")
    with open(filename, "w") as f:
        f.write(parseString(xml).toprettyxml())
    logger.debug(f"Saved XML file to {filename}")


def write_reader_to_txt(
    reader: BaseReader, filename: PathLike, update_freq: int = 100_000, key: str = ""
) -> ty.Generator[tuple[str, int, int, str], None, None]:
    """Write image data to text."""
    filename = Path(filename).with_suffix(".txt")
    if filename.exists():
        yield key, 0, 0, ""

    # convert to appropriate format
    array, shape = reader.flat_array()
    array = _insert_indices(array, shape)

    if reader.n_channels == 3 and reader.dtype == np.uint8:
        yield from write_rgb_to_txt(filename, array, update_freq=update_freq, key=key)
    elif np.issubdtype(reader.dtype, np.integer):
        yield from write_int_to_txt(filename, array, reader.channel_names, update_freq=update_freq, key=key)
    else:
        yield from write_float_to_txt(filename, array, reader.channel_names, update_freq=update_freq, key=key)


def float_format_to_row(row: np.ndarray) -> str:
    """Format string to row."""
    return ",".join([",".join([str(int(v)) for v in row[0:2]]), ",".join([f"{v:.2f}" for v in row[2:]]), "\n"])


@numba.njit(nogil=True, cache=True)  # type: ignore[misc]
def int_format_to_row(row: np.ndarray) -> str:
    """Format string to row."""
    return ",".join([str(v) for v in row]) + ",\n"


def _insert_indices(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    n = array.shape[1]
    y, x = np.indices(shape)
    y = y.ravel(order="F")
    x = x.ravel(order="F")
    dtype = get_dtype_for_array(array)
    res = np.zeros((x.size, n + 2), dtype=dtype)
    res[:, 0] = y + 1
    res[:, 1] = x + 1
    res[:, 2:] = reshape_fortran(array, (-1, n))
    return res


def write_rgb_to_txt(
    path: PathLike, array: np.ndarray, update_freq: int = 100_000, key: str = ""
) -> ty.Generator[tuple[str, int, int, str], None, None]:
    """Write IMS data to text."""
    columns = ["row", "col", "Red (R)", "Green (G)", "Blue (B)"]
    fmt_func = int_format_to_row.py_func if IS_MAC_ARM else int_format_to_row
    yield from _write_txt(path, columns, array, fmt_func, update_freq=update_freq, key=key)


def write_int_to_txt(
    path: PathLike, array: np.ndarray, channel_names: list[str], update_freq: int = 100_000, key: str = ""
) -> ty.Generator[tuple[str, int, int, str], None, None]:
    """Write IMS data to text."""
    columns = ["row", "col", *channel_names]
    fmt_func = int_format_to_row.py_func if IS_MAC_ARM else int_format_to_row
    yield from _write_txt(path, columns, array, fmt_func, update_freq=update_freq, key=key)


def write_float_to_txt(
    path: PathLike, array: np.ndarray, channel_names: list[str], update_freq: int = 100_000, key: str = ""
) -> ty.Generator[tuple[str, int, int, str], None, None]:
    """Write IMS data to text."""
    columns = ["row", "col", *channel_names]
    yield from _write_txt(path, columns, array, float_format_to_row, update_freq=update_freq, key=key)


def _write_txt(
    path: PathLike,
    columns: list[str],
    array: np.ndarray,
    str_func: ty.Callable,
    update_freq: int = 100_000,
    key: str = "",
) -> ty.Generator[tuple[str, int, int, str], None, None]:
    """Write data to csv file."""
    from tqdm import tqdm

    path = Path(path)
    assert path.suffix == ".txt", "Path must have .txt extension."

    n = array.shape[0]
    logger.debug(
        f"Exporting array with {array.shape[0]:,} observations, {array.shape[1]:,} features and {array.dtype} data"
        f" type to '{path}'"
    )
    with open(path, "w", newline="\n", encoding="cp1252") as f:
        f.write(",".join(columns) + ",\n")
        with tqdm(array, total=n, mininterval=1, desc="Exporting to CSV...") as pbar:
            for i, row in enumerate(pbar):
                f.write(str_func(row))
                if i % update_freq == 0 and i != 0:
                    d = pbar.format_dict
                    if d["rate"]:
                        eta = pbar.format_interval((d["total"] - d["n"]) / d["rate"])
                    else:
                        eta = ""
                    yield key, i, n, eta
        yield key, n, n, ""
