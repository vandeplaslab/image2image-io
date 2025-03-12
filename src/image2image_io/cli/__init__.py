"""Main CLI."""

import click
import koyo.compat
from loguru import logger

from image2image_io import __version__
from image2image_io.cli.convert import convert
from image2image_io.cli.czi2tiff import czi2tiff
from image2image_io.cli.merge import merge
from image2image_io.cli.thumbnail import thumbnail
from image2image_io.cli.transform import transform

LOG_FMT = "[<level>{level: <8}</level>][{time:YYYY-MM-DD HH:mm:ss:SSS}][{extra[src]}] {message}"
COLOR_LOG_FMT = (
    "<green>[<level>{level: <8}</level>]</green>"
    "<cyan>[{time:YYYY-MM-DD HH:mm:ss:SSS}]</cyan>"
    "<red>[{process}]</red>"
    "<blue>[{extra[src]}]</blue>"
    " {message}"
)


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120,
        "ignore_unknown_options": True,
    }
)
@click.option(
    "--dev",
    help="Flat to indicate that CLI should run in development mode and catch all errors.",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--no_color",
    help="Flag to enable colored while doing tasks.",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.version_option(__version__)
@click.option(
    "--verbose",
    "-v",
    "verbosity",
    default=1,
    count=True,
    help="Verbose output. This is additive flag so `-vvv` will print `INFO` messages and -vvvv will print `DEBUG`"
    " information.",
)
@click.option("--quiet", "-q", "verbosity", flag_value=0, help="Minimal output")
@click.option("--debug", "verbosity", flag_value=0.5, help="Maximum output")
def cli(verbosity: int, no_color: bool, dev: bool) -> None:
    """image2image-io CLI."""
    from koyo.hooks import install_debugger_hook, uninstall_debugger_hook
    from koyo.logging import get_loguru_config, set_loguru_env, set_loguru_log

    verbosity = 2 - int(verbosity)  # default is INFO
    if verbosity < 0:
        verbosity = 0
    if dev:
        install_debugger_hook()
        verbosity = 0
    elif dev:
        uninstall_debugger_hook()
    level = verbosity * 10
    level, fmt, colorize, enqueue = get_loguru_config(level, no_color=no_color)
    set_loguru_env(fmt, level, colorize, enqueue)
    set_loguru_log(
        level=level.upper(),
        no_color=no_color,
        logger=logger,
        fmt=LOG_FMT if no_color else COLOR_LOG_FMT,
    )
    logger.configure(extra={"src": "CLI"})
    logger.enable("image2image_io")
    logger.enable("koyo")
    logger.debug(f"Activated logger with level '{level}'.")


# register commands
cli.add_command(czi2tiff)
cli.add_command(thumbnail)
cli.add_command(merge)
cli.add_command(convert)
if transform:
    cli.add_command(transform)


def main():
    """Execute the "imimspy" command line program."""
    cli.main(windows_expand_args=False)


if __name__ == "__main__":
    cli.main(windows_expand_args=False)
