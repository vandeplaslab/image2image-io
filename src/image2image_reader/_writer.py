from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike

if ty.TYPE_CHECKING:
    from image2image_reader.readers._base_reader import BaseReader


def write_ome_tiff(path: PathLike, array: np.ndarray, reader: BaseReader) -> Path:
    """Write OME-TIFF."""
    from wsireg.reg_images import NumpyRegImage
    from wsireg.writers.ome_tiff_writer import OmeTiffWriter

    if array.ndim == 2:
        array = np.atleast_3d(array)

    reg = NumpyRegImage(
        array,
        reader.resolution,
        channel_names=reader.channel_names,
    )

    path = Path(path)
    writer = OmeTiffWriter(reg)
    filename = Path(writer.write_image_by_plane(path.stem, path.parent, write_pyramid=True))
    return filename
