"""Create mask(s) for image(s) based on shapes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from shapely import Polygon
from tqdm import tqdm

from image2image_io.enums import TIME_FORMAT, MaskOutputFmt

logger = logger.bind(src="Mask")


def mask_to_polygon(mask: np.ndarray, epsilon: float = 1) -> np.ndarray:
    """Convert mask to polygon."""
    mask = np.asarray(mask, dtype=np.uint8)
    # ensure mask is between 0-255
    if mask.max() == 1:
        mask = mask * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate contours to polygons and get the coordinates
    polygons = [cv2.approxPolyDP(contour, epsilon=epsilon, closed=True) for contour in contours]
    polygons = np.asarray(polygons)
    return polygons


def shapes_to_polygons(
    shapes: list[np.ndarray], with_index: bool = False, inv_pixel_size: float = 1.0, silent: bool = True
) -> list[Polygon] | list[tuple[Polygon, int]]:
    """Convert shapes to polygons."""
    polygons = []
    for index, shape in enumerate(
        tqdm(shapes, desc="Drawing shapes", leave=False, miniters=1000, disable=silent), start=1
    ):
        if isinstance(shape, dict):
            shape = shape["array"]
        yx = shape * inv_pixel_size
        if with_index:
            polygons.append((Polygon(yx), index))
        else:
            polygons.append(Polygon(yx))
    return polygons


def polygons_to_mask(polygons: list[Polygon], output_shape: tuple[int, int]) -> np.ndarray:
    """Convert polygons to mask."""
    from rasterio.features import rasterize

    mask: np.ndarray = rasterize(polygons, out_shape=output_shape)
    return mask


def transform_mask(mask: np.ndarray, transform: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Transform mask."""
    from scipy.ndimage import affine_transform
    from skimage.transform import AffineTransform

    tform = AffineTransform(matrix=transform)
    transformed_mask: np.ndarray = affine_transform(
        mask, tform.params, output_shape=output_shape, mode="constant", order=0
    )
    return transformed_mask


def write_masks_as_hdf5(
    path: PathLike,
    name: str,
    mask: np.ndarray,
    shapes_data: dict[str, np.ndarray | str],
    display_name: str | None = None,
    creation_date: datetime | None = None,
    color: np.ndarray | None = None,
    metadata: dict[str, np.ndarray] | None = None,
) -> None:
    """Write masks to file."""
    import h5py

    # need the following attributes to be valid
    assert mask.max() <= 1, "Expected binary mask."
    assert mask.ndim == 2, "Expected 2D mask."
    # turn into a binary mask
    mask = mask.astype(bool)
    shape = mask.shape

    if display_name is None:
        display_name = name
    creation_date = creation_date or datetime.now()
    if isinstance(creation_date, datetime):
        creation_date = creation_date.strftime(TIME_FORMAT)
    if color is None:
        color = np.asarray([1.0, 0.0, 0.0, 1.0])
    if shapes_data is None:
        shapes_data = {}
    if shapes_data:
        assert "shape_data" in shapes_data, "Expected 'shape_data' key in shapes_data."
        assert "shape_types" in shapes_data, "Expected 'shape_types' key in shapes_data."

    # write to file
    with h5py.File(path, "w") as f:
        # write attributes first
        grp = f.create_group("Mask")
        grp.attrs["shape"] = tuple(shape)
        grp.attrs["name"] = name
        grp.attrs["display_name"] = display_name
        grp.attrs["creation_date"] = creation_date
        # write mask data
        grp.create_dataset("mask", data=mask)
        grp.create_dataset("color", data=color)
        # write shapes data
        if shapes_data:
            for index, (data, shape_type) in enumerate(zip(shapes_data["shape_data"], shapes_data["shape_types"])):
                grp = f.create_group(f"Mask/Shapes/{index}")
                grp.attrs["shape_type"] = shape_type
                grp.create_dataset("data", data=data)
        # add extra data
        if metadata:
            grp = f.create_group("Mask/Metadata")
            for meta_name, meta_data in metadata.items():
                if meta_data.ndim != mask.ndim or meta_data.shape != mask.shape:
                    logger.warning(
                        f"Skipped metadata key={meta_name} as it had wrong shape or dimension. {meta_data.shape}"
                    )
                    continue
                grp.create_dataset(f"{meta_name}", data=meta_data)


def write_masks_as_geojson(path: PathLike, shapes_data: dict[str, np.ndarray | str], display_name: str) -> None:
    """Write masks to GeoJSON."""
    from koyo.json import write_json_data
    from shapely import Polygon, to_geojson

    polygons = []
    for index, array in enumerate(shapes_data["shape_data"]):
        polygons.append(
            {
                "type": "Feature",
                "id": index,
                "geometry": to_geojson(Polygon(array)),
                "properties": {"classification": {"name": display_name}},
            }
        )
    write_json_data(path, polygons)


def write_masks_as_image(path: PathLike, mask: np.ndarray) -> None:
    """Write masks to image."""
    from PIL import Image

    # need the following attributes to be valid
    assert mask.max() <= 1, "Expected binary mask."
    assert mask.ndim == 2, "Expected 2D mask."
    # turn into a binary mask
    mask = mask.astype(bool)

    # write to file
    img = Image.fromarray(mask)
    img.save(path)


def is_polygon_valid(x: np.ndarray, y: np.ndarray | None = None) -> bool:
    """Check whether the polygons are valid."""
    p = Polygon(x) if y is None else Polygon((x, y))
    return p.is_valid and p.is_simple  # type: ignore[no-any-return]


def remove_invalid(
    shape_data: list[dict[str, np.ndarray | str]],
) -> tuple[list[int], list[dict[str, np.ndarray | str]]]:
    """Remove invalid polygons."""
    keep, valid_shapes = [], []
    for index, shape in enumerate(shape_data):
        if is_polygon_valid(shape["array"], None):
            valid_shapes.append(shape)
            keep.append(index)
    logger.warning(f"Removed {len(shape_data) - len(valid_shapes)} invalid polygons.")
    return keep, valid_shapes


def transform_masks(
    transform_to: PathLike,
    masks: list[PathLike],
    output_dir: PathLike,
    fmt: str | MaskOutputFmt | list[str] | list[MaskOutputFmt],
    config_path: PathLike,
    scene_index: int | None = None,
    overwrite: bool = False,
) -> None:
    """Transform and export masks."""
    from image2image_io.readers import ShapesReader, get_simple_reader
    from image2image_io.utils.warp import get_affine_from_config

    if isinstance(fmt, str):
        fmt = [fmt]
    if not fmt:
        raise ValueError("No output format specified.")

    affine, mask_shape, pixel_size = get_affine_from_config(config_path, yx=True, px=True, inv=False)
    mask_inv_pixel_size = 1 / pixel_size
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (3, 3):
        raise ValueError("Expected 3x3 affine matrix.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # image that masks are transformed to
    image_reader = get_simple_reader(transform_to, init_pyramid=False, auto_pyramid=False, scene_index=scene_index)
    # mask_shape = image_reader.image_shape
    # mask_inv_pixel_size = image_reader.inv_resolution
    logger.trace(f"Initial mask shape {mask_shape} with {1 / mask_inv_pixel_size:.4f} resolution.")

    with_index = any(f in ["hdf5"] for f in fmt)
    with_shapes = any(f in ["hdf5", "geojson"] for f in fmt)
    for mask in tqdm(masks, desc="Exporting masks..."):
        with MeasureTimer() as timer:
            mask_reader = ShapesReader(mask)
            display_name = mask_reader.display_name or mask_reader.path.stem
            logger.trace(f"Reading mask {display_name} in {timer()}")

            mask_indexed = None
            mask = mask_reader.to_mask(mask_shape, inv_pixel_size=mask_inv_pixel_size, silent=False)
            if with_index:
                mask_indexed = mask_reader.to_mask(
                    mask_shape, inv_pixel_size=mask_inv_pixel_size, with_index=True, silent=False
                )
            shapes = None
            if with_shapes:
                _, shapes = mask_reader.to_shapes()
            # masks must be transformed to the image shape - sometimes that might involve warping if affine matrix
            # is specified
            transformed_mask = image_reader.warp(mask, affine=affine)
            transformed_mask_indexed = None
            if mask_indexed is not None:
                transformed_mask_indexed = image_reader.warp(mask_indexed, affine=affine)
            logger.trace(f"Transformed mask {display_name} in {timer(since_last=True)}")

            for fmt_ in fmt:
                extension = {"hdf5": "h5", "binary": "png", "geojson": "geojson"}[fmt_]
                output_path = output_dir / f"{display_name}_ds={image_reader.path.stem}.{extension}"
                if fmt_ == "hdf5":
                    write_masks_as_hdf5(
                        output_path,
                        display_name,
                        transformed_mask,
                        shapes,
                        display_name,
                        metadata={"polygon_index": transformed_mask_indexed},
                    )
                elif fmt_ == "binary":
                    write_masks_as_image(output_path, transformed_mask)
                elif fmt_ == "geojson":
                    write_masks_as_geojson(output_path, shapes, display_name)
                else:
                    raise ValueError(f"Unsupported format '{fmt_}'")
                logger.info(f"Exported {output_path} in {timer(since_last=True)}")


def transform_shapes_or_points(
    files: list[PathLike],
    output_dir: PathLike,
    config_path: PathLike,
    overwrite: bool = False,
):
    """Transform and export shapes or points."""
    from image2image_io.readers import get_simple_reader
    from image2image_io.utils.warp import ImageWarper

    transform_seq = ImageWarper(config_path, inv=False)
    for file in tqdm(files, desc="Exporting masks..."):
        with MeasureTimer() as timer:
            reader = get_simple_reader(file)
            display_name = reader.display_name
            logger.trace(f"Reading shape/point {display_name} in {timer()}")
            if reader.reader_type == "points":
                x, y = reader.x, reader.y
                transform_seq.transform_points(x, y)
