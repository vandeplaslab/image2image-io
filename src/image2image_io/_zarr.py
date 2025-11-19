"""Zarr storage."""

from zarr import Array

try:
    from zarr.hierarchy import Group
except (AttributeError, ModuleNotFoundError):
    from zarr.core import Group
