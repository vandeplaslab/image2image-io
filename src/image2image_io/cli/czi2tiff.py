"""Convert images from one format to OME-TIFF format."""

from __future__ import annotations

import click
from koyo.click import arg_split_int, arg_split_str
from loguru import logger


@click.option(
    "-N",
    "--channel_names",
    type=click.STRING,
    default=None,
    help="Specify a list of channel names to be used by the task. Names should be separated by commas. "
    "e.g. 'DAPI,CD3,CD20'",
    callback=arg_split_str,
    show_default=True,
    required=False,
)
@click.option(
    "-I",
    "--channel_ids",
    type=click.STRING,
    default=None,
    help="Specify a list of channel indices to be used by the task. Channel IDs should be separated by comma."
    "e.g. '1,2,4'.",
    callback=arg_split_int,
    show_default=True,
    required=False,
)
@click.option(
    "-u",
    "--as_uint8",
    is_flag=True,
    help="Convert to uint8. If not specified, the original data type will be used.",
    show_default=True,
    required=False,
)
@click.option(
    "-t",
    "--tile_size",
    help="Tile size.",
    type=click.Choice(["256", "512", "1024", "2048"], case_sensitive=False),
    default="512",
    show_default=True,
    required=False,
)
@click.option(
    "-s",
    "--scene",
    "scene_index",
    type=click.INT,
    help="Specify the scene to be processed. If not specified, all scenes will be processed.",
    default=None,
    show_default=True,
    required=True,
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
    help="Path to the CZI file that should be converted.",
    type=click.Path(exists=False, resolve_path=False, file_okay=True, dir_okay=False),
    show_default=True,
    required=True,
)
@click.command()
def czi2tiff(
    input_: str,
    output_dir: str,
    scene_index: int,
    tile_size: str,
    as_uint8: bool,
    channel_ids: list[int] | None,
    channel_names: list[str | None],
) -> None:
    """Convert CZI to OME-TIFF.

    This command converts a CZI to OME-TIFF while allowing the user to specify the scene, channel names, channel IDs and
    OME-TIFF tile size and output data type.
    """
    from image2image_io.writers import czi_to_ome_tiff

    metadata = None
    if channel_ids and channel_names:
        if len(channel_ids) != len(channel_names):
            raise ValueError("Number of channel IDs and channel names must be equal.")
        metadata = {scene_index: {"channel_ids": channel_ids, "channel_names": channel_names}}

    for key, scene_index, total, _ in czi_to_ome_tiff(
        input_, output_dir, as_uint8=as_uint8, tile_size=int(tile_size), metadata=metadata, scenes=[scene_index]
    ):
        logger.info(f"Converted {key} scene {scene_index}/{total}")


@click.option(
    "-i",
    "--input",
    "input_",
    help="Path to the CZI file that should be converted.",
    type=click.Path(exists=False, resolve_path=False, file_okay=True, dir_okay=False),
    show_default=True,
    required=True,
)
@click.command()
def cziinfo(input_: str) -> None:
    """Print information about the CZI file."""
    from image2image_io.config import CONFIG
    from image2image_io.readers import get_reader

    with CONFIG.temporary_overwrite(init_pyramid=False, auto_pyramid=False):
        path, readers = get_reader(input_, split_czi=True)
        print(f"File: {path!r}")
        print(f"Number of scenes: {len(readers)}")
        for index, reader in enumerate(readers.values()):
            print(f"Scene {index}")
            print(f"Image shape: {reader.image_shape}")
            print(f"Pixel size: {reader.resolution}")
            print(f"  Number of channels: {reader.n_channels}")
            for channel_index, channel_name in enumerate(reader.channel_names):
                print(f"    Channel {channel_index}: {channel_name}")
            print("-" * 80)
