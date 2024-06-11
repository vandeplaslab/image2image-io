"""Transform image."""

from __future__ import annotations

import click
from koyo.click import arg_split_int, arg_split_str
from koyo.utilities import is_installed
from loguru import logger

if not is_installed("image2image_reg"):
    transform = None
else:

    @click.option(
        "-W",
        "--overwrite",
        help="Overwrite existing data.",
        is_flag=True,
        default=None,
        show_default=True,
    )
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
        "tile_size",
        type=click.INT,
        help="Specify tile size of the pyramid.",
        default=512,
        show_default=True,
        required=True,
    )
    @click.option(
        "-T",
        "--transform",
        "transform_",
        help="Path to the i2r.json transformation file.",
        type=click.Path(exists=False, resolve_path=False, file_okay=True, dir_okay=False),
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
        help="Path to the OME-TIFF file that should be converted.",
        type=click.Path(exists=False, resolve_path=False, file_okay=True, dir_okay=False),
        show_default=True,
        required=True,
    )
    @click.command()
    def transform(
        input_: str,
        output_dir: str,
        transform_: str,
        tile_size: int,
        as_uint8: bool,
        channel_ids: list[int] | None,
        channel_names: list[str | None],
        overwrite: bool,
    ) -> None:
        """Transform OME-TIFF using i2r.json or i2reg."""
        from image2image_reg.models.transform_sequence import TransformSequence

        from image2image_io.writers import image_to_ome_tiff

        metadata = None
        if channel_ids and channel_names:
            if len(channel_ids) != len(channel_names):
                raise ValueError("Number of channel IDs and channel names must be equal.")
            metadata = {0: {"channel_ids": channel_ids, "channel_names": channel_names}}

        if ".i2r.json" in transform_:
            transform_seq = TransformSequence.from_i2r(transform_, input_)
        else:
            transform_seq = TransformSequence.from_path(transform_)
        for key, scene_index, total, _ in image_to_ome_tiff(
            input_,
            output_dir,
            as_uint8=as_uint8,
            tile_size=tile_size,
            metadata=metadata,
            transformer=transform_seq,
            overwrite=overwrite,
        ):
            logger.info(f"Transformed {key} {scene_index}/{total}")
