"""Enums."""

import typing as ty
from enum import Enum

import numpy as np

DEFAULT_TRANSFORM_NAME: str = "Identity matrix"

TIME_FORMAT = "%d/%m/%Y-%H:%M:%S:%f"

MaskOutputFmt = ty.Literal["hdf5", "binary", "geojson"]
WriterMode = ty.Literal["sitk", "ome-zarr", "ome-tiff", "ome-tiff-by-plane", "ome-tiff-by-tile"]
DisplayType = ty.Literal["points", "polygon", "path", "path or polygon"]


class ViewType(str, Enum):
    """View type."""

    RANDOM = "random"
    OVERLAY = "overlay"


SITK_TO_NP_DTYPE = {
    0: np.int8,
    1: np.uint8,
    2: np.int16,
    3: np.uint16,
    4: np.int32,
    5: np.uint32,
    6: np.int64,
    7: np.uint64,
    8: np.float32,
    9: np.float64,
    10: np.complex64,
    11: np.complex64,
    12: np.int8,
    13: np.uint8,
    14: np.int16,
    15: np.int16,
    16: np.int32,
    17: np.int32,
    18: np.int64,
    19: np.int64,
    20: np.float32,
    21: np.float64,
    22: np.uint8,
    23: np.uint16,
    24: np.uint32,
    25: np.uint64,
}
