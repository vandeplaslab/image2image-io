import click


# noinspection PyUnusedLocal
def arg_split_bbox(ctx, param, value):
    """Split arguments."""
    if value is None:
        return None
    args = [int(arg.strip()) for arg in value.split(",")]
    assert len(args) == 4, "Bounding box must have 4 values"
    return args


ALLOW_EXTRA_ARGS = {"help_option_names": ["-h", "--help"], "ignore_unknown_options": True, "allow_extra_args": True}
overwrite_ = click.option(
    "-W",
    "--overwrite",
    help="Overwrite existing data.",
    is_flag=True,
    default=False,
    show_default=True,
)
as_uint8_ = click.option(
    "-u/-U",
    "--as_uint8/--no_as_uint8",
    help="Downcast the image data format to uint8 which will substantially reduce the size of the files (unless it's"
    " already in uint8...).",
    is_flag=True,
    default=None,
    show_default=True,
)
fmt_ = click.option(
    "-f",
    "--fmt",
    help="Output format.",
    type=click.Choice(["ome-tiff"], case_sensitive=False),
    default="ome-tiff",
    show_default=True,
    required=False,
)
