"""Zarr storage."""

from zarr.core import Array

try:
    from zarr.hierarchy import Group
except AttributeError:
    from zarr.core import Group
