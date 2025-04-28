"""Create thumbnail for image(s)."""

from __future__ import annotations

import click
from koyo.click import cli_parse_paths_sort
from loguru import logger
from tqdm import tqdm


@click.option(
    "-f/-F",
    "--first_only/--no_first_only",
    help="Export only the first channel from the stack.",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "-t/-T",
    "--with_title/--no_with_title",
    help="Export image with title.",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option(
    "-o",
    "--output_dir",
    help="Path to directory where OME-TIFF files will be saved.",
    default=".",  # cwd
    type=click.Path(exists=False, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
)
@click.option(
    "-i",
    "--input",
    "input_",
    help="Path to supported file(s).",
    type=click.UNPROCESSED,
    show_default=True,
    required=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
@click.command()
def thumbnail(input_: str, output_dir: str, with_title: bool, first_only: bool) -> None:
    """Create a thumbnail for image(s)."""
    from image2image_io.utils.utilities import write_thumbnail

    for path in tqdm(input_, desc="Creating thumbnails..."):
        write_thumbnail(path, output_dir, with_title=with_title, first_only=first_only)
        logger.info(f"Created thumbnail for '{path}'.")
