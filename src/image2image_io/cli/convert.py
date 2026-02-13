"""Utilities."""

from __future__ import annotations

import click
from koyo.click import Parameter, cli_parse_paths_sort, print_parameters

from image2image_io.cli._common import ALLOW_EXTRA_ARGS, as_uint8_, fmt_, overwrite_, tile_size_


@overwrite_
@as_uint8_
@tile_size_
@fmt_
@click.option(
    "-o",
    "--output_dir",
    help="Path to the output directory where images should be saved to.",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
    show_default=True,
    required=True,
)
@click.option(
    "-i",
    "--image",
    help="Path(s) of images to be converted.",
    type=click.UNPROCESSED,
    show_default=True,
    required=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
@click.command("convert", context_settings=ALLOW_EXTRA_ARGS)
def convert(image: list[str], output_dir: str, fmt: str, tile_size: str, as_uint8: bool, overwrite: bool) -> None:
    """Convert images to pyramidal OME-TIFF."""
    convert_runner(image, output_dir, fmt, tile_size, as_uint8, overwrite)


def convert_runner(
    paths: list[str],
    output_dir: str,
    fmt: str = "ome-tiff",
    tile_size: int | str = 1024,
    as_uint8: bool | None = None,
    overwrite: bool = False,
) -> None:
    """Convert images to pyramidal OME-TIFF."""
    from image2image_io.writers import images_to_ome_tiff

    print_parameters(
        Parameter("Images", "-i/--image", paths),
        Parameter("Output directory", "-o/--output_dir", output_dir),
        Parameter("Output format", "-f/--fmt", fmt),
        Parameter("Tile size", "-t/--tile_size", tile_size),
        Parameter("Downcast to uint8", "-u/--as_uint8", as_uint8),
        Parameter("Overwrite", "-W/--overwrite", overwrite),
    )
    for _ in images_to_ome_tiff(paths, output_dir, tile_size=int(tile_size), as_uint8=as_uint8, overwrite=overwrite):
        pass
