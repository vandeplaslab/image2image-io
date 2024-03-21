"""GeoJSON reader for image2image."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike

from image2image_io.readers._base_reader import BaseReader
from image2image_io.readers.geojson_utils import get_int_dtype, read_geojson, shape_reader


def is_txt_and_has_columns(path: PathLike, columns: list[str]) -> bool:
    """Check if a text file has the required columns."""
    import pandas as pd

    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path, nrows=1)
    elif path.suffix == ".txt":
        df = pd.read_csv(path, delimiter="\t", nrows=1)
    elif path.suffix == ".tsv":
        df = pd.read_csv(path, delimiter="\t", nrows=1)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Invalid file extension: {path.suffix}")
    return all(col in df.columns for col in columns)


def read_shapes(path: PathLike) -> tuple:
    """Read shapes."""
    import pandas as pd

    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".txt":
        df = pd.read_csv(path, delimiter="\t")
    elif path.suffix == ".tsv":
        df = pd.read_csv(path, delimiter="\t")
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Invalid file extension: {path.suffix}")
    for col in ["vertex_x", "vertex_y", "cell"]:
        if col not in df.columns:
            raise ValueError(f"Missing required columns: {col}. Available columns: {df.columns}")

    shapes_geojson, shape_data = [], []
    for group, indices in df.groupby("cell").groups.items():
        dff = df.iloc[indices]
        shape_data.append(
            {
                "array": np.c_[dff["vertex_x"].values, dff["vertex_y"].values].astype(np.float32),
                "shape_type": "polygon",
                "shape_name": group,
            }
        )
        shapes_geojson.append({"geometry": {"type": "Polygon"}, "properties": {"classification": {"name": group}}})
    return shapes_geojson, shape_data


def read_data(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Read data."""
    if path.suffix in [".json", ".geojson"]:
        geojson_data, shape_data = read_geojson(path)
    else:
        geojson_data, shape_data = read_shapes(path)
    return geojson_data, shape_data


class ShapesReader(BaseReader):
    """GeoJSON reader for image2image."""

    reader_type = "shapes"
    _channel_names: list[str]

    def __init__(
        self,
        path: PathLike,
        key: str | None = None,
        auto_pyramid: bool | None = None,
    ):
        super().__init__(path, key=key, auto_pyramid=auto_pyramid)
        self.geojson_data, self.shape_data = read_data(self.path)
        self._channel_names = [self.path.stem]

    @property
    def display_name(self) -> str:
        """Retrieve display name from the path."""
        return self.path.stem

    def remove_invalid(self) -> None:
        """Remove invalid shapes."""
        from image2image_io.utils.mask import remove_invalid

        keep, self.shape_data = remove_invalid(self.shape_data)
        self.geojson_data = [self.geojson_data[i] for i in keep]

    def to_mask(self, output_shape: tuple[int, int], with_index: bool = False) -> np.ndarray:
        """Convert to mask."""
        from image2image_io.utils.mask import polygons_to_mask, shapes_to_polygons

        polygons = shapes_to_polygons(self.shape_data, with_index=with_index)
        mask = polygons_to_mask(polygons, output_shape)
        return mask

    def to_mask_alt(self, output_size: tuple[int, int], with_index: bool = False) -> np.ndarray:
        """
        Draw a binary or label mask using shape data.

        Parameters
        ----------
        output_size: tuple of int
            Size of the mask in tuple(x,y)
        with_index: bool
            Whether to write each mask instance as a label (1-n_shapes) or to write all as binary (255)

        Returns
        -------
        mask: np.ndarray
            Drawn mask at set output size

        """
        import cv2

        dtype = np.uint8
        if with_index:
            dtype = get_int_dtype(len(self.shape_data))  # type: ignore[assignment]
        mask = np.zeros(output_size[::-1], dtype=dtype)
        shapes = self.shape_data
        for idx, sh in enumerate(shapes):
            mask = cv2.fillPoly(
                mask,
                pts=[sh["array"].astype(np.int32)],
                color=idx + 1 if with_index else np.iinfo(dtype).max,
            )
        return mask

    def to_shapes(self) -> tuple[str, dict[str, np.ndarray | str]]:
        """Convert to shapes that can be exported to Shapes layer."""
        _, shape_types, shape_names, shape_arrays, *_ = self.parse_data()
        if len(shape_types) > 1_000:
            shape_types = ["path"] * len(shape_types)
        else:
            shape_types = [s.lower() for s in shape_types]  # expected polygon not Polygon
        return shape_names[0], {"shape_types": shape_types, "shape_data": shape_arrays}

    def parse_data(self) -> tuple:
        """Parse data."""
        shape_data = self.shape_data
        shapes_geojson, shapes = shape_reader(shape_data)

        n_shapes = len(shapes_geojson)
        shape_types = [sh["geometry"]["type"] for sh in shapes_geojson]
        shape_names = [sh["properties"]["classification"]["name"] for sh in shapes_geojson]
        shape_arrays = [s["array"][:, [1, 0]] for s in self.shape_data]
        shape_props = {"name": shape_names}
        shape_text = {
            "text": "{name}",
            "color": "white",
            "anchor": "center",
            "size": 12,
            "visible": False,
        }
        return n_shapes, shape_types, shape_names, shape_arrays, shape_props, shape_text

    def to_shapes_kwargs(self, **kwargs: ty.Any) -> dict:
        """Return data so it's compatible with Shapes layer."""
        *_, shape_arrays, shape_props, shape_text = self.parse_data()
        kws = {
            "data": shape_arrays,
            "properties": shape_props,
            "text": shape_text,
            "shape_type": "polygon" if len(shape_arrays) < 1_000 else "path",
            "scale": self.scale,
            "affine": self.transform,
        }
        kws.update(kwargs)
        return kws

    def get_dask_pyr(self) -> list[ty.Any]:
        """Get dask representation of the pyramid."""
        raise NotImplementedError("Must implement method")

    def to_csv(self, path: PathLike) -> str:
        """Export data as CSV file."""
        raise NotImplementedError("Must implement method")
