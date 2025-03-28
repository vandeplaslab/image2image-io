"""Transform image."""

from __future__ import annotations

import typing as ty

import click
from koyo.click import Parameter, arg_split_int, arg_split_str, cli_parse_paths_sort, print_parameters, warning_msg
from loguru import logger

from image2image_io.enums import MaskOutputFmt


@click.group()
def transform():
    """Transform command."""


@click.option(
    "-W",
    "--overwrite",
    help="Overwrite existing data.",
    is_flag=True,
    default=None,
    show_default=True,
)
@click.option("-s", "--scene_index", help="Scene index to transform.", type=click.INT, default=None, show_default=True)
@click.option(
    "-f",
    "--fmt",
    help="Output format of the mask.",
    type=click.Choice(ty.get_args(MaskOutputFmt), case_sensitive=False),
    multiple=True,
    default=("binary",),
)
@click.option(
    "-o",
    "--output_dir",
    help="Path to directory where mask files will be saved.",
    default=".",  # cwd
    type=click.Path(exists=False, resolve_path=True, file_okay=False, dir_okay=True),
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
    "-m",
    "--mask",
    "mask_",
    help="Path to the mask file(s) that should be transformed. Can be GeoJSON or text format.",
    type=click.Path(exists=False, resolve_path=False, file_okay=True, dir_okay=False),
    show_default=True,
    required=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
@click.option(
    "-i",
    "--image",
    "image_",
    help="Path to the image file that the mask is transformed to. This could be OME-TIFF or IMS file (e.g. .d)",
    type=click.Path(exists=False, resolve_path=False, file_okay=True, dir_okay=True),
    show_default=True,
    required=True,
)
@transform.command(name="mask")
def mask(
    image_: str,
    mask_: list[str],
    output_dir: str,
    transform_: str,
    fmt: list[str],
    scene_index: int | None,
    overwrite: bool,
) -> None:
    """Transform GeoJSON or point mask using image2image transformation matrix (i2r.json)."""
    from image2image_io.utils.mask import transform_masks

    if not any(ext in transform_ for ext in (".i2r.json", ".i2r.toml")):
        raise ValueError("Only i2r.json or i2r.toml files are supported for mask transformation.")

    transform_masks(image_, mask_, output_dir, fmt, transform_, scene_index=scene_index, overwrite=overwrite)


@click.option(
    "-W",
    "--overwrite",
    help="Overwrite existing data.",
    is_flag=True,
    default=None,
    show_default=True,
)
@click.option(
    "-o",
    "--output_dir",
    help="Path to directory where mask files will be saved.",
    default=".",  # cwd
    type=click.Path(exists=False, resolve_path=True, file_okay=False, dir_okay=True),
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
    "-f",
    "--files",
    help="Path to the mask file(s) that should be transformed. Can be GeoJSON or text format.",
    type=click.Path(exists=False, resolve_path=False, file_okay=True, dir_okay=False),
    show_default=True,
    required=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
@transform.command(name="attachment")
def attachment(
    files: list[str],
    output_dir: str,
    transform_: str,
    overwrite: bool,
) -> None:
    """Transform GeoJSON or point mask using image2image transformation matrix (i2r.json)."""
    from image2image_io.utils.mask import transform_shapes_or_points

    if not any(ext in transform_ for ext in (".i2r.json", ".i2r.toml")):
        raise ValueError("Only i2r.json or i2r.toml files are supported for mask transformation.")

    transform_shapes_or_points(files, output_dir, transform_, overwrite=overwrite)


@click.option(
    "-W",
    "--overwrite",
    help="Overwrite existing data.",
    is_flag=True,
    default=None,
    show_default=True,
)
@click.option(
    "-R",
    "--resolution",
    help="Image resolution - override in case it's not specified in the file.",
    type=click.FLOAT,
    default=None,
    show_default=True,
)
@click.option(
    "--inverse",
    is_flag=True,
    help="Transform the image using the inverse transformation matrix (from moving to fixed).",
    show_default=True,
    required=False,
    default=False,
)
@click.option(
    "-s",
    "--suffix",
    type=click.STRING,
    help="Specify the suffix that goes at the end of the filename.",
    default="_transformed",
    show_default=True,
    required=True,
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
    help="Tile size.",
    type=click.Choice(["256", "512", "1024", "2048"], case_sensitive=False),
    default="512",
    show_default=True,
    required=False,
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
    "-T",
    "--transform",
    "transform_",
    help="Path to the i2r.json transformation file.",
    type=click.Path(exists=False, resolve_path=False, file_okay=True, dir_okay=False),
    show_default=True,
    required=True,
)
@click.option(
    "-i",
    "--image",
    "image_",
    help="Path to the OME-TIFF file that should be converted.",
    type=click.Path(exists=False, resolve_path=False, file_okay=True, dir_okay=False),
    show_default=True,
    required=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
@transform.command(name="image")
def image(
    image_: list[str],
    output_dir: str,
    transform_: str,
    tile_size: int,
    as_uint8: bool,
    channel_ids: list[int] | None,
    channel_names: list[str | None],
    suffix: str = "_transformed",
    inverse: bool = True,
    resolution: float | None = None,
    overwrite: bool = False,
) -> None:
    """Transform OME-TIFF using image2image transformation matrix (i2r.json) or elastix (i2reg) transformation."""
    from image2image_io.writers import image_to_ome_tiff

    metadata = None
    if channel_ids and channel_names:
        if len(channel_ids) != len(channel_names):
            raise ValueError("Number of channel IDs and channel names must be equal.")
        metadata = {0: {"channel_ids": channel_ids, "channel_names": channel_names}}
    if len(image_) > 1 and metadata:
        warning_msg(
            "Multiple image files were specified wih a single set of metadata - it will be repeated for each image."
        )

    if any(ext in transform_ for ext in (".i2r.json", ".i2r.toml")):
        from image2image_io.utils.warp import ImageWarper

        transform_seq = ImageWarper(transform_, inv=inverse)
    else:
        try:
            from image2image_reg.elastix.transform_sequence import TransformSequence

            transform_seq = TransformSequence.from_path(transform_)
        except ImportError:
            raise ImportError("Please install image2image-reg to use i2reg transformations.")

    print_parameters(
        Parameter("Name", "-i/--image", image_),
        Parameter("Output directory", "-o/--output_dir", output_dir),
        Parameter("Channel ids", "-C/--channel_ids", channel_ids),
        Parameter("Write images as uint8", "--as_uint8/--no_as_uint8", as_uint8),
        Parameter("Overwrite", "-W/--overwrite", overwrite),
    )

    for image_path in image_:
        for key, scene_index, total, _, _ in image_to_ome_tiff(
            image_path,
            output_dir,
            as_uint8=as_uint8,
            tile_size=int(tile_size),
            metadata=metadata,
            transformer=transform_seq,
            overwrite=overwrite,
            suffix=suffix,
            resolution=resolution,
        ):
            logger.info(f"Transformed {key} {scene_index}/{total}")
