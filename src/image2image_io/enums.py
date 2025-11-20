"""Enums."""

import typing as ty
from enum import Enum

DEFAULT_TRANSFORM_NAME: str = "Identity matrix"

TIME_FORMAT = "%d/%m/%Y-%H:%M:%S:%f"

MaskOutputFmt = ty.Literal["hdf5", "binary", "geojson"]
WriterMode = ty.Literal["sitk", "ome-zarr", "ome-tiff", "ome-tiff-by-plane", "ome-tiff-by-tile"]
DisplayType = ty.Literal["points", "polygon", "path", "path or polygon"]


class ViewType(str, Enum):
    """View type."""

    RANDOM = "random"
    OVERLAY = "overlay"
