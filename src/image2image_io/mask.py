from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from loguru import logger

from image2image_io.enums import WriterFileMode


def export_mask_regions(
    path: Path,
    output_dir: Path | None,
    regions: list[tuple[int, int, int, int] | np.ndarray],
    tile_size: int = 1024,
    as_uint8: bool = True,
    multiply: bool = True,
    fmt: WriterFileMode = "ome-tiff",
) -> ty.Generator[tuple[Path, int, int], None, None]:
    """Export mask images."""
    from image2image_io.readers import get_simple_reader
    from image2image_io.writers import OmeTiffWrapper, OmeZarrWrapper

    path = Path(path)
    reader = get_simple_reader(path)
    output_dir = Path(output_dir) if output_dir else None
    output_dir_ = output_dir

    n = len(regions)
    ext = ".ome.tiff" if fmt == "ome-tiff" else ".ome.zarr"
    for current, polygon_or_bbox in enumerate(regions, start=1):
        if output_dir_ is None:
            output_dir = path.parent

        output_path, dtype, hash_str, shape, rgb = None, None, None, None, []
        wrapper = OmeTiffWrapper() if fmt == "ome-tiff" else OmeZarrWrapper()
        for _, hash_str in reader.mask_region_iter(polygon_or_bbox, multiply=multiply):
            output_path = output_dir / f"{path.stem}_{hash_str}".replace(".ome", "")
            dtype = reader.dtype
            shape = reader.shape
            break

        if dtype is None or hash_str is None or shape is None or output_path is None:
            logger.warning(f"Skipping {path} as it has no data.")
            continue
        if output_path.with_suffix(ext).exists():
            logger.warning(f"Skipping {output_path} as it already exists.")
            yield output_path, current, n
        else:
            with wrapper.write(
                channel_names=reader.channel_names,
                resolution=reader.resolution,
                dtype=dtype,
                shape=shape,
                name=output_path.name,
                output_dir=output_dir,
                tile_size=tile_size,
                as_uint8=as_uint8,
            ):
                channel_index = 0
                for channel, _ in reader.mask_region_iter(polygon_or_bbox, multiply=multiply):
                    if channel is None:
                        continue
                    if reader.is_rgb:
                        rgb.append(channel)
                    else:
                        wrapper.add_channel(channel_index, reader.channel_names[channel_index], channel)
                        channel_index += 1
                if rgb:
                    wrapper.add_channel([0, 1, 2], ["R", "G", "B"], np.dstack(rgb))
            yield wrapper.path, current, n
