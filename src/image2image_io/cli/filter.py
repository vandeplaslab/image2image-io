"""Filter microscopy image channels."""

from __future__ import annotations

import csv
import re
import typing as ty
from contextlib import ExitStack
from pathlib import Path

import click
from koyo.click import cli_parse_paths_sort
from loguru import logger

from image2image_io.cli._common import fmt_, overwrite_, tile_size_

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader


def _parse_channel_ids(value: str) -> list[int]:
    channel_ids: list[int] = []
    for token in value.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", token)
        if match is None:
            msg = "Channel IDs must be nonnegative integers or inclusive ranges, e.g. 0,2,4-6."
            raise ValueError(msg)
        start = int(match[1])
        end = int(match[2]) if match[2] is not None else start
        if end < start:
            msg = "Channel ID ranges must be ascending."
            raise ValueError(msg)
        channel_ids.extend(range(start, end + 1))
    return channel_ids


def _read_channel_file(path: Path) -> tuple[list[int], list[str] | None, int]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".tsv", ".txt"}:
            msg = "Channel file must be CSV, TSV, or TXT."
            raise ValueError(msg)
        delimiter = "\t" if suffix == ".tsv" else ","
        if suffix == ".txt":
            header = stream.readline()
            stream.seek(0)
            if "," in header or "\t" in header:
                delimiter = csv.Sniffer().sniff(header, delimiters=",\t").delimiter
        table = csv.reader(stream, delimiter=delimiter, strict=True)
        columns = next(table, [])
        if "include" not in columns:
            msg = "Channel file must have an 'include' column."
            raise ValueError(msg)
        if len(set(columns)) != len(columns):
            msg = "Channel file has duplicate column names."
            raise ValueError(msg)
        channel_ids: list[int] = []
        channel_names: list[str] | None = [] if "channel_names" in columns else None
        n_rows = 0
        for index, values in enumerate(table):
            n_rows += 1
            row = dict(zip(columns, values))
            include = (row.get("include") or "").strip().lower()
            if len(values) != len(columns) or include not in {"true", "false", "1", "0"}:
                msg = f"Invalid include value or malformed row at channel {index}; use true/false or 1/0."
                raise ValueError(msg)
            if include in {"true", "1"}:
                channel_ids.append(index)
                if channel_names is not None:
                    name = (row.get("channel_names") or "").strip()
                    if not name:
                        msg = f"Missing channel_names value for included channel {index}."
                        raise ValueError(msg)
                    channel_names.append(name)
    return channel_ids, channel_names, n_rows


def _validate_image(reader: BaseReader) -> None:
    from image2image_io.readers import CziImageReader, CziSceneImageReader, TiffImageReader

    axes: str = ""
    shape = reader.shape
    if isinstance(reader, TiffImageReader):
        series = reader.fh.series[reader.series_index]
        axes, shape = series.axes, series.shape
    elif isinstance(reader, (CziImageReader, CziSceneImageReader)):
        axes, shape = reader.fh.axes, reader.fh.shape
    if any(axis in {"Z", "T"} and size > 1 for axis, size in zip(axes, shape)):
        msg = "Only 2D channel images are supported; Z stacks and time series cannot be filtered."
        raise ValueError(msg)
    packed_rgb = any(axis == "S" and size > 1 for axis, size in zip(axes, shape))
    if reader.is_rgb or (isinstance(reader, TiffImageReader) and packed_rgb):
        msg = "Packed RGB images are not supported; supply separate microscopy channels."
        raise ValueError(msg)
    if len(reader.shape) not in {2, 3}:
        msg = "Only 2D channel images are supported."
        raise ValueError(msg)


@click.command("filter", context_settings={"help_option_names": ["-h", "--help"]})
@overwrite_
@tile_size_
@fmt_
@click.option(
    "-u/-U",
    "--as_uint8/--no_as_uint8",
    "--uint8/--no_uint8",
    default=None,
    help="Scale each retained channel to uint8; otherwise preserve the input dtype.",
)
@click.option(
    "-i",
    "--image",
    "-p",
    "--path",
    required=True,
    multiple=True,
    type=click.UNPROCESSED,
    callback=cli_parse_paths_sort,
    help="Input image path(s) or glob patterns.",
)
@click.option(
    "-o",
    "--output_dir",
    required=True,
    type=click.Path(file_okay=False, resolve_path=True),
    help="Directory for filtered images.",
)
@click.option("-C", "--channel_ids", help="Zero-based channel IDs, e.g. 0,2,4-6, in output order.")
@click.option("--channel_names", multiple=True, help="Repeat once per selected ID to rename channels.")
@click.option(
    "--channel_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV/TSV/TXT with one row per input channel and a boolean include column.",
)
def filter_channels(
    image: list[str],
    output_dir: str,
    channel_ids: str | None,
    channel_names: tuple[str, ...],
    channel_file: Path | None,
    fmt: str,
    tile_size: str,
    as_uint8: bool | None,
    overwrite: bool,
) -> None:
    """Keep selected microscopy channels and optionally rename them."""
    try:
        filter_runner(
            image,
            output_dir,
            channel_ids=_parse_channel_ids(channel_ids) if channel_ids is not None else None,
            channel_names=list(channel_names) if channel_names else None,
            channel_file=channel_file,
            fmt=fmt,
            tile_size=int(tile_size),
            as_uint8=as_uint8,
            overwrite=overwrite,
        )
    except (ValueError, TypeError, OSError, RuntimeError, csv.Error) as exc:
        raise click.ClickException(str(exc)) from exc


def filter_runner(
    paths: ty.Sequence[str | Path],
    output_dir: str | Path,
    channel_ids: list[int] | None = None,
    channel_names: list[str] | None = None,
    channel_file: str | Path | None = None,
    fmt: str = "ome-tiff",
    tile_size: int = 1024,
    as_uint8: bool | None = None,
    overwrite: bool = False,
) -> None:
    """Export selected 2D microscopy channels using one selection for all inputs.

    Supply either zero-based channel IDs (with optional corresponding names), or
    a CSV/TSV/TXT channel file with an include column. Validate the whole batch
    before writing. Outputs have an _filtered suffix; existing outputs are skipped
    unless overwrite is enabled. Input readers are closed even when export fails.
    """
    from image2image_io.config import CONFIG
    from image2image_io.readers import (
        CZI_EXTENSIONS,
        IMAGE_EXTENSIONS,
        OME_ZARR_EXTENSIONS,
        TIFF_EXTENSIONS,
        ArrayImageReader,
        CziImageReader,
        CziSceneImageReader,
        get_simple_reader,
    )
    from image2image_io.readers._czi import CziSceneFile
    from image2image_io.writers import write_ome_tiff_alt, write_ome_zarr

    if (channel_ids is None) == (channel_file is None):
        msg = "Supply exactly one of --channel_ids or --channel_file."
        raise ValueError(msg)
    if channel_file is not None and channel_names is not None:
        msg = "--channel_names requires --channel_ids; use the table's channel_names column for renaming."
        raise ValueError(msg)
    n_rows: int | None = None
    if channel_file is not None:
        channel_ids, channel_names, n_rows = _read_channel_file(Path(channel_file))
    if not channel_ids:
        msg = "Select at least one channel."
        raise ValueError(msg)
    if any(type(index) is not int or index < 0 for index in channel_ids):
        msg = "Channel IDs must be nonnegative integers."
        raise ValueError(msg)
    if len(set(channel_ids)) != len(channel_ids):
        msg = "Duplicate channel IDs are not allowed."
        raise ValueError(msg)
    if channel_names is not None and (
        len(channel_names) != len(channel_ids) or any(not name.strip() for name in channel_names)
    ):
        msg = "Supply one nonempty channel name per selected ID."
        raise ValueError(msg)
    if not paths:
        msg = "No input images matched the supplied paths."
        raise ValueError(msg)
    if fmt not in {"ome-tiff", "ome-zarr"}:
        msg = "Output format must be ome-tiff or ome-zarr."
        raise ValueError(msg)

    input_paths = [Path(path).resolve() for path in paths]
    output_dir = Path(output_dir).resolve()
    destinations: set[Path] = set()
    exports: list[tuple[BaseReader, Path, list[str]]] = []
    with ExitStack() as stack:
        # get_simple_reader changes these globals; restore them even if opening fails.
        for key in ("init_pyramid", "auto_pyramid", "quiet"):
            stack.callback(setattr, CONFIG, key, getattr(CONFIG, key))
        CONFIG.init_pyramid = CONFIG.auto_pyramid = False
        for path in input_paths:
            if path.suffix.lower() not in TIFF_EXTENSIONS + CZI_EXTENSIONS + IMAGE_EXTENSIONS + OME_ZARR_EXTENSIONS:
                msg = f"{path}: Expected a microscopy image (TIFF, CZI, OME-Zarr, PNG, or JPEG)."
                raise ValueError(msg)
            n_scenes = CziSceneFile.get_num_scenes(path) if path.suffix.lower() == ".czi" else 1
            for scene_index in range(n_scenes):
                try:
                    reader = get_simple_reader(
                        path,
                        init_pyramid=False,
                        auto_pyramid=False,
                        scene_index=scene_index if n_scenes > 1 else None,
                    )
                    stack.callback(reader.close)
                    _validate_image(reader)
                except (ValueError, TypeError, OSError, RuntimeError) as exc:
                    msg = f"{path}: {exc}"
                    raise ValueError(msg) from exc
                if n_rows is not None and n_rows != reader.n_channels:
                    msg = f"{path}: Channel file has {n_rows} rows; image has {reader.n_channels} channels."
                    raise ValueError(msg)
                if max(channel_ids) >= reader.n_channels:
                    msg = f"{path}: Channel ID {max(channel_ids)} is out of range for {reader.n_channels} channels."
                    raise ValueError(msg)
                names = channel_names if channel_names is not None else [reader.channel_names[i] for i in channel_ids]
                stem = path.stem
                if stem.lower().endswith(".ome"):
                    stem = stem[:-4]
                if n_scenes > 1:
                    stem += f"_scene={scene_index:02d}"
                if fmt == "ome-tiff":
                    # Match the TIFF writer's filename normalization when checking collisions.
                    stem = stem.replace(".ome", "").replace(".tiff", "").replace(".tif", "")
                destination = output_dir / f"{stem}_filtered.{'ome.tiff' if fmt == 'ome-tiff' else 'ome.zarr'}"
                if destination in destinations:
                    msg = f"Multiple inputs would write to {destination}."
                    raise ValueError(msg)
                if destination.resolve() in input_paths:
                    msg = f"Output {destination} would overwrite an input image."
                    raise ValueError(msg)
                destinations.add(destination)
                exports.append((reader, destination, names))

        output_dir.mkdir(parents=True, exist_ok=True)
        writer = write_ome_zarr if fmt == "ome-zarr" else write_ome_tiff_alt
        for reader, destination, names in exports:
            if destination.exists() and not overwrite:
                logger.info(f"Skipping {destination} - already exists")
                continue
            try:
                export_reader = reader
                if isinstance(reader, (CziImageReader, CziSceneImageReader)):
                    # CZI pyramid loading squeezes singleton axes, including C.
                    export_reader = ArrayImageReader(
                        reader.path,
                        reader.pyramid[0].reshape(reader.shape),
                        resolution=reader.resolution,
                        channel_names=reader.channel_names,
                    )
                writer(
                    destination,
                    export_reader,
                    channel_ids=channel_ids,
                    channel_names=names,
                    tile_size=tile_size,
                    as_uint8=as_uint8,
                    overwrite=overwrite,
                )
            except (ValueError, TypeError, OSError, RuntimeError) as exc:
                msg = f"{reader.path} -> {destination}: {exc}"
                raise ValueError(msg) from exc
