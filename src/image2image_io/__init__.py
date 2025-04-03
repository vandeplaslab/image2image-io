"""Various image readers."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("image2image-io")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Lukasz G. Migas"
__email__ = "lukas.migas@yahoo.com"
