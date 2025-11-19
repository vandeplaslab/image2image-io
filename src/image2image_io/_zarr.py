"""Zarr storage."""

from zarr import Array

try:
    from zarr.hierarchy import Group
except (AttributeError, ModuleNotFoundError):
    try:
        from zarr.core.group import Group
    except (AttributeError, ModuleNotFoundError):
        from zarr.group import Group
