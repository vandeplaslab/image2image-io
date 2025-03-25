"""Writer functions."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numba
import numpy as np
from koyo.system import IS_MAC_ARM
from koyo.typing import PathLike
from loguru import logger
from tqdm import tqdm

from image2image_io.utils.utilities import (
    get_dtype_for_array,
    get_flat_shape_of_image,
    get_shape_of_image,
    reshape_fortran,
)
from image2image_io.writers.merge_tiff_writer import MergeImages, MergeOmeTiffWriter
from image2image_io.writers.tiff_writer import OmeTiffWrapper, OmeTiffWriter, Transformer

if ty.TYPE_CHECKING:
    from image2image_io.readers._base_reader import BaseReader

MetadataScene = dict[str, ty.Union[str, list[ty.Union[int, str]]]]
MetadataReader = dict[int, MetadataScene]
MetadataDict = dict[Path, MetadataReader]
ExportProgress = tuple[str, int, int, int, bool]


__all__ = [
    "MergeOmeTiffWriter",
    "OmeTiffWriter",
    "czi_to_ome_tiff",
    "czis_to_ome_tiff",
    "image_to_fusion",
    "image_to_ome_tiff",
    "images_to_fusion",
    "images_to_ome_tiff",
]


def get_total_n_scenes(paths: ty.Iterable[PathLike]) -> tuple[int, list[Path]]:
    """Get total number of scenes in all files."""
    from image2image_io.readers._czi import CziSceneFile

    total_n_scenes = 0
    paths_ = []
    for path_ in paths:
        path_ = Path(path_)
        if path_.is_dir():
            for path__ in path_.glob("**/*.czi"):
                if path__.suffix == ".czi":
                    total_n_scenes += CziSceneFile.get_num_scenes(path__)
                else:
                    total_n_scenes += 1
                paths_.append(path__)
        else:
            if path_.suffix == ".czi":
                total_n_scenes += CziSceneFile.get_num_scenes(path_)
            else:
                total_n_scenes += 1
            paths_.append(path_)
    return total_n_scenes, paths_


def merge_images(
    name: str,
    paths: ty.Iterable[PathLike],
    output_dir: PathLike,
    as_uint8: bool | None = None,
    tile_size: int = 512,
    metadata: MetadataDict | None = None,
    overwrite: bool = False,
) -> None:
    """Merge multiple images to OME-TIFF."""
    from image2image_io.readers import get_simple_reader

    name = name.replace(".ome.tiff", "")
    output_dir = Path(output_dir)
    filename = output_dir / f"{name}.ome.tiff"
    if filename.exists() and not overwrite:
        logger.info(f"Skipping {filename} - already exists")
        return

    pixel_sizes = []
    channel_names = []
    channel_ids = []
    reader_names = []
    for path_ in paths:
        reader = get_simple_reader(path_, init_pyramid=False, auto_pyramid=False)
        reader_metadata: MetadataReader
        if metadata:
            reader_metadata = metadata.get(path_, None)
            if not reader_metadata:
                reader_metadata = {
                    0: {
                        "name": reader.clean_name,
                        "channel_names": reader.channel_names,
                        "channel_ids": reader.channel_ids,
                    }
                }
                logger.warning(f"Metadata not found for {path_}. Using default metadata.")
        else:
            reader_metadata = {
                0: {"name": reader.clean_name, "channel_names": reader.channel_names, "channel_ids": reader.channel_ids}
            }
            logger.trace(f"Metadata not specified for {path_}. Using default metadata.")

        scene_metadata: dict[str, list[int | str]]
        if reader_metadata:
            scene_metadata = reader_metadata.get(0, None)
            if not scene_metadata:
                scene_metadata = {"channel_ids": reader.channel_ids, "channel_names": reader.channel_names}
                logger.warning("Metadata not found for 0. Using default metadata.")
        else:
            scene_metadata = {
                "name": reader.clean_name,
                "channel_ids": reader.channel_ids,
                "channel_names": reader.channel_names,
            }
            logger.trace("Metadata not specified for 0. Using default metadata.")

        pixel_sizes.append(reader.resolution)
        reader_names.append(scene_metadata.get("name", reader.clean_name))
        channel_names.append(scene_metadata.get("channel_names", reader.channel_names))
        channel_ids.append(scene_metadata.get("channel_ids", reader.channel_ids))

    writer = MergeOmeTiffWriter(
        MergeImages(
            paths,
            pixel_sizes,
            channel_names=channel_names,
            channel_ids=channel_ids,
        )
    )
    writer.merge_write_image_by_plane(
        name,
        reader_names,
        output_dir=output_dir,
        as_uint8=as_uint8,
        tile_size=tile_size,
    )


def images_to_ome_tiff(
    paths: ty.Iterable[PathLike],
    output_dir: PathLike | None = None,
    as_uint8: bool | None = None,
    tile_size: int = 512,
    metadata: MetadataDict | None = None,
    path_to_scene: dict[Path, list[int]] | None = None,
    extras: dict[Path, dict[str, int | float | None]] | None = None,
    overwrite: bool = False,
) -> ty.Generator[ExportProgress, None, None]:
    """Convert multiple images to OME-TIFF."""
    output_dir = Path(output_dir) if output_dir else None

    # get total number of scenes
    total_n_scenes, paths = get_total_n_scenes(paths)
    current_total_scene = 0
    for _current, path_ in enumerate(tqdm(paths, desc="Converting to OME-TIFF...", total=len(paths))):
        path_ = Path(path_)
        reader_metadata = metadata.get(path_, None) if metadata else None
        reader_scenes = path_to_scene.get(path_, None) if path_to_scene else None
        if path_.suffix == ".czi":
            try:
                for key, current_file_scene, total_file_scenes, increment_by, is_exported in czi_to_ome_tiff(
                    path_,
                    output_dir=output_dir,
                    as_uint8=as_uint8,
                    tile_size=tile_size,
                    metadata=reader_metadata,
                    reader_scenes=reader_scenes,
                    overwrite=overwrite,
                ):
                    yield key, current_file_scene, total_file_scenes, current_total_scene, total_n_scenes, is_exported
                    current_total_scene += increment_by
            except (ValueError, TypeError, OSError) as err:
                logger.error(f"Could not read Czi file {path_} ({err})")
                logger.exception(err)
                continue
        else:
            try:
                for key, current_file_scene, total_file_scenes, increment_by, is_exported in image_to_ome_tiff(
                    path_,
                    output_dir=output_dir,
                    as_uint8=as_uint8,
                    tile_size=tile_size,
                    metadata=reader_metadata,
                    overwrite=overwrite,
                ):
                    yield key, current_file_scene, total_file_scenes, current_total_scene, total_n_scenes, is_exported
                    current_total_scene += increment_by
            except (ValueError, TypeError, OSError) as err:
                logger.error(f"Could not read {path_.suffix} file {path_} ({err})")
                logger.exception(err)
                continue


def image_to_ome_tiff(
    path: PathLike,
    output_dir: PathLike | None = None,
    as_uint8: bool | None = None,
    tile_size: int = 512,
    suffix: str = "",
    metadata: MetadataReader | None = None,
    transformer: Transformer | None = None,
    resolution: float | None = None,
    overwrite: bool = False,
) -> ty.Generator[ExportProgress, None, None]:
    """Convert image of any type to OME-TIFF.

    Returns
    -------
    ExportProgress
        Tuple containing the key, current scene, total scenes, current total scene, and if the scene was exported
    """
    from image2image_io.readers import get_key, get_simple_reader

    path = Path(path)

    if output_dir is None:
        output_dir = path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    key = get_key(path)
    filename = path.name.replace(".ome.tiff", "").replace(path.suffix, "")
    if suffix:
        filename += suffix
    output_path = output_dir / filename
    yield key, 1, 1, 1, False

    # skip if the output file already exists
    if output_path.with_suffix(".ome.tiff").exists() and not overwrite:
        logger.info(f"Skipping {output_path} - already exists")
        yield key, 1, 1, 1, True
        return

    # read the scene
    reader = get_simple_reader(path, auto_pyramid=False, init_pyramid=False)
    if resolution:
        reader.resolution = resolution

    scene_metadata: dict[str, list[int | str]]
    if metadata:
        scene_metadata = metadata.get(path, None)
        if not scene_metadata:
            scene_metadata = {"channel_ids": reader.channel_ids, "channel_names": reader.channel_names}
            logger.warning(f"Metadata not found for {path}. Using default metadata.")
    else:
        scene_metadata = {"channel_ids": reader.channel_ids, "channel_names": reader.channel_names}
        logger.trace(f"Metadata not specified for {path}. Using default metadata.")

    if scene_metadata:
        assert "channel_ids" in scene_metadata, "Channel IDs must be specified in metadata."
        assert "channel_names" in scene_metadata, "Channel names must be specified in metadata."
    if not scene_metadata["channel_ids"]:
        yield key, 1, 1, 1, True
    else:
        write_ome_tiff_alt(
            output_path,
            reader,
            as_uint8=as_uint8,
            tile_size=tile_size,
            transformer=transformer,
            **scene_metadata,
            overwrite=overwrite,
        )
        yield key, 1, 1, 1, True


def czis_to_ome_tiff(
    paths: ty.Iterable[PathLike],
    output_dir: PathLike | None = None,
    as_uint8: bool | None = None,
    tile_size: int = 512,
    metadata: MetadataDict | None = None,
) -> ty.Generator[ExportProgress, None, None]:
    """Convert multiple CZI images to OME-TIFF."""
    # calculate true total number of scenes
    total_n_scenes, paths_ = get_total_n_scenes(paths)

    current_total_scene = 0
    for path_ in paths_:
        path_ = Path(path_)
        reader_metadata = metadata.get(path_, None) if metadata else None
        try:
            for key, current_file_scene, total_file_scenes, increment_by, is_exported in czi_to_ome_tiff(
                path_, output_dir, as_uint8, tile_size, reader_metadata
            ):
                yield key, current_file_scene, total_file_scenes, current_total_scene, total_n_scenes, is_exported
                current_total_scene += increment_by
        except (ValueError, TypeError, OSError):
            logger.error(f"Could not read Czi file {path_}")
            continue


def czi_to_ome_tiff(
    path: PathLike,
    output_dir: PathLike | None = None,
    as_uint8: bool | None = None,
    tile_size: int = 512,
    metadata: MetadataReader | None = None,
    reader_scenes: list[int] | None = None,
    scenes: list[int] | None = None,
    overwrite: bool = False,
) -> ty.Generator[ExportProgress, None, None]:
    """Convert Czi image to OME-TIFF."""
    from image2image_io.readers import get_key
    from image2image_io.readers._czi import CziSceneFile
    from image2image_io.readers.czi_reader import CziSceneImageReader

    path = Path(path)
    if output_dir is None:
        output_dir = path.parent
    output_dir = Path(output_dir)
    try:
        n = CziSceneFile.get_num_scenes(path)
        has_scenes = n != 1
    except Exception as e:
        logger.error(f"Could not read Czi file {path} - {e}")
        return
    if scenes is None:
        scenes = list(range(n))
    if reader_scenes is not None:
        scenes = [scene for scene in scenes if scene in reader_scenes]

    assert min(scenes) >= 0, "Scene index must be greater than or equal to 0."
    assert max(scenes) <= n, "Scene index must be less than the total number of scenes in the file."
    scene_key = get_key(path, scenes[0] if has_scenes else None)
    yield scene_key, 0, len(scenes), 0, False

    if metadata is None:
        metadata = {}
        logger.trace("Metadata not specified. Setting to empty.")

    # iterate over each scene in the czi file
    for scene_index in scenes:
        scene_key = get_key(path, scene_index if has_scenes else None)
        logger.debug(f"Converting scene {scene_index + 1}/{n} from {path}...")
        filename = path.name.replace(".czi", "") + (f"_scene={scene_index:02d}" if n > 1 else "")
        output_path = output_dir / filename
        # skip if the output file already exists
        if output_path.with_suffix(".ome.tiff").exists() and not overwrite:
            logger.info(f"Skipping {output_path} - already exists")
            yield scene_key, scene_index + 1, n, 1, True
            continue

        # read the scene
        reader = CziSceneImageReader(path, scene_index=scene_index, auto_pyramid=False, init_pyramid=False)
        path = reader.path

        scene_metadata: dict[str, list[int | str]]
        if metadata:
            scene_metadata = metadata.get(scene_index, None)
            if not scene_metadata:
                scene_metadata = {"channel_ids": reader.channel_ids, "channel_names": reader.channel_names}
                logger.warning(
                    f"Metadata not found for {path} {scene_index}. Using default metadata. "
                    f"Available keys: {metadata.keys()}"
                )
        else:
            scene_metadata = {"channel_ids": reader.channel_ids, "channel_names": reader.channel_names}
            logger.trace(f"Metadata not specified for {path} {scene_index}. Using default metadata.")

        if scene_metadata:
            assert "channel_ids" in scene_metadata, "Channel IDs must be specified in metadata."
            assert "channel_names" in scene_metadata, "Channel names must be specified in metadata."

        # skip if there are no channel IDs
        if not scene_metadata["channel_ids"]:
            yield scene_key, scene_index + 1, n, 1, True
        else:
            write_ome_tiff_alt(output_path, reader, as_uint8=as_uint8, tile_size=tile_size, **scene_metadata)
            yield scene_key, scene_index + 1, n, 1, True


def write_ome_tiff_from_array(
    path: PathLike,
    reader: BaseReader | None,
    array: np.ndarray,
    resolution: float | None = None,
    channel_names: list[str] | None = None,
    compression: str | None = "default",
    tile_size: int = 512,
    as_uint8: bool | None = None,
    ome_name: str | None = None,
) -> Path:
    """Write OME-TIFF by also specifying an array."""
    from image2image_io.readers.array_reader import ArrayImageReader
    from image2image_io.writers.tiff_writer import OmeTiffWriter

    if array.ndim == 2:
        array = array[np.newaxis, ...]

    if reader:
        resolution = resolution or reader.resolution
        channel_names = channel_names or reader.channel_names
    resolution = resolution or 1.0

    array_reader = ArrayImageReader("", array, resolution=resolution, channel_names=channel_names)

    path = Path(path)
    filename = path.name.replace(".ome.tiff", "")
    writer = OmeTiffWriter(array_reader)
    output_path = writer.write_image_by_plane(
        filename,
        path.parent,
        write_pyramid=True,
        compression=compression,
        tile_size=tile_size,
        as_uint8=as_uint8,
        ome_name=ome_name,
    )
    return output_path


def write_ome_tiff_alt(
    path: PathLike,
    reader: BaseReader,
    as_uint8: bool | None = None,
    tile_size: int = 512,
    channel_ids: list[int] | None = None,
    channel_names: list[str] | None = None,
    transformer: Transformer | None = None,
    overwrite: bool = False,
    write: bool = True,
    ome_name: str | None = None,
) -> Path:
    """Write OME-TIFF."""
    from image2image_io.writers.tiff_writer import OmeTiffWriter

    path = Path(path)
    filename = path.name.replace(".ome.tiff", "")
    writer = OmeTiffWriter(reader, transformer=transformer)
    if write:
        output_path = writer.write_image_by_plane(
            filename,
            path.parent,
            write_pyramid=tile_size > 0,
            as_uint8=as_uint8,
            channel_ids=channel_ids,
            channel_names=channel_names,
            tile_size=tile_size,
            overwrite=overwrite,
            ome_name=ome_name,
        )
        return output_path
    return writer


def images_to_fusion(
    paths: ty.Iterable[PathLike],
    output_dir: PathLike | None = None,
    metadata: MetadataDict | None = None,
) -> ty.Generator[tuple[str, int, int, str], None, None]:
    """Convert multiple images to Fusion."""
    for path_ in paths:
        path_ = Path(path_)
        reader_metadata = metadata.get(path_, None) if metadata else None
        if reader_metadata:
            reader_metadata = reader_metadata.get(0, None)
        yield from image_to_fusion(path_, output_dir, reader_metadata)


def image_to_fusion(
    path: PathLike,
    output_dir: PathLike | None = None,
    metadata: MetadataReader | None = None,
) -> ty.Generator[tuple[str, int, int, str], None, None]:
    """Convert image of any type to Fusion format."""
    from image2image_io.readers import get_reader

    path = Path(path)
    path, readers = get_reader(path)
    for reader in readers.values():
        key = reader.key
        if output_dir is None:
            output_dir = path.parent
        output_dir = Path(output_dir)
        filename = output_dir / path.stem.replace(".ome", "")
        yield from write_reader_to_txt(reader, filename, metadata=metadata, key=key)
        logger.trace(f"Exported Fusion CSV for {path}")
        if not filename.with_suffix(".xml").exists():
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
    reader: BaseReader,
    filename: PathLike,
    metadata: MetadataReader | None = None,
    update_freq: int = 100_000,
    key: str = "",
) -> ty.Generator[tuple[str, int, int, str], None, None]:
    """Write image data to text."""
    filename = Path(filename).with_suffix(".txt")
    if filename.exists():
        yield key, 0, 0, ""

    # convert to appropriate format
    array, shape = reader.flat_array()
    array = _insert_indices(array, shape)

    if reader.is_rgb and reader.dtype == np.uint8:
        channel_names = ["Red (R)", "Green (G)", "Blue (B)"]
        yield from write_rgb_to_txt(filename, array, channel_names, update_freq=update_freq, key=key)
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
    path: PathLike, array: np.ndarray, channel_names: list[str], update_freq: int = 100_000, key: str = ""
) -> ty.Generator[tuple[str, int, int, str], None, None]:
    """Write IMS data to text."""
    columns = ["row", "col", *channel_names]
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
