"""GeoJSON reader for image2image."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
import pandas as pd
from koyo.rand import temporary_seed
from koyo.typing import PathLike
from loguru import logger
from tqdm import tqdm

from image2image_io.config import CONFIG
from image2image_io.readers._base_reader import BaseReader
from image2image_io.readers.geojson_utils import get_int_dtype, read_geojson, shape_reader
from image2image_io.readers.utilities import check_df_columns, get_column_name

if ty.TYPE_CHECKING:
    from shapely.geometry import Polygon


PATH_IF_COUNT = 1_000


def is_txt_and_has_columns(
    path: PathLike, required: list[str], either: list[tuple[str, ...]], either_dtype: tuple[np.dtype, ...] | None = None
) -> bool:
    """Check if a text file has the required columns."""
    import pandas as pd

    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path, nrows=1)
    elif path.suffix in [".txt", ".tsv"]:
        temp = pd.read_csv(path, delimiter="\t", nrows=1)
        sep = "\t" if len(temp.columns) > 1 else " "
        df = pd.read_csv(path, delimiter=sep, nrows=1)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Invalid file extension: {path.suffix}")
    return check_df_columns(df, required, either, either_dtype)


def get_shape_columns(path: PathLike) -> tuple[str, str, str | None]:
    """Get columns."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path, nrows=1)
    elif path.suffix in [".txt", ".tsv"]:
        temp = pd.read_csv(path, delimiter="\t", nrows=1)
        sep = "\t" if len(temp.columns) > 1 else " "
        df = pd.read_csv(path, delimiter=sep, nrows=1)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Invalid file extension: {path.suffix}")
    x_key = get_column_name(df, ["vertex_x", "x", "x:x", "x_location", "x_centroid"])
    y_key = get_column_name(df, ["vertex_y", "y", "y:y", "y_location", "y_centroid"])
    group_by = get_column_name(df, ["cell", "cell_id", "shape", "shape_name"])
    return x_key, y_key, group_by


def read_shapes(path: PathLike) -> tuple:
    """Read shapes."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in [".txt", ".tsv"]:
        temp = pd.read_csv(path, delimiter="\t", nrows=1)
        sep = "\t" if len(temp.columns) > 1 else " "
        df = pd.read_csv(path, delimiter=sep)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Invalid file extension: {path.suffix}")
    return read_shapes_from_df(df)


def read_shapes_from_df(df: pd.DataFrame) -> tuple:
    """Read shapes from DataFrame."""
    x_key = get_column_name(df, ["vertex_x", "x"])
    y_key = get_column_name(df, ["vertex_y", "y"])
    group_by = get_column_name(df, ["cell", "cell_id", "shape", "shape_name"])
    is_points = x_key == "x"
    shapes_geojson, shape_data = [], []
    for group, indices in df.groupby(group_by).groups.items():
        dff = df.iloc[indices]
        shape_data.append(
            {
                "array": np.c_[dff[x_key].values, dff[y_key].values].astype(np.float32),
                "shape_type": "polygon",
                "shape_name": group,
            }
        )
        shapes_geojson.append({"geometry": {"type": "Polygon"}, "properties": {"classification": {"name": group}}})
    return shapes_geojson, shape_data, is_points


def napari_to_shapes_data(name: str, data: list[np.ndarray], shapes: list[str]) -> dict:
    """Convert napari shapes to data."""
    shape_data = []
    for array, _shape_type in zip(data, shapes):
        shape_data.append(
            {
                "array": array[:, [1, 0]],
                "shape_type": "polygon",  # shape_type,
                "shape_name": name,
            }
        )
    return shape_data


def read_data(path: Path) -> dict[str, dict[str, np.ndarray | str], bool]:
    """Read data."""
    if path.suffix.lower() in [".json", ".geojson"]:
        geojson_data, shape_data, is_points = read_geojson(path)
    else:
        geojson_data, shape_data, is_points = read_shapes(path)
    return geojson_data, shape_data, is_points


class ShapesReader(BaseReader):
    """GeoJSON reader for image2image."""

    _is_rgb: bool = False
    reader_type = "shapes"
    _channel_names: list[str]
    _array_dtype = np.dtype("float32")

    def __init__(self, path: PathLike, key: str | None = None, auto_pyramid: bool | None = None, init: bool = True):
        super().__init__(path, key=key, auto_pyramid=auto_pyramid)
        if not init:
            self.geojson_data, self.shape_data = [], []
            return
        self.geojson_data, self.shape_data, is_points = read_data(self.path)
        self.display_type = "points" if is_points else None
        self._channel_names = [self.path.stem]

    @classmethod
    def create(cls, name: str = "", channel_names: list[str] | None = None):
        """Create empty instance."""
        obj = cls(name, key=name, init=False)
        if channel_names:
            obj._channel_names = channel_names
        return obj

    def remove_invalid(self) -> None:
        """Remove invalid shapes."""
        from image2image_io.utils.mask import remove_invalid

        keep, self.shape_data = remove_invalid(self.shape_data)
        self.geojson_data = [self.geojson_data[i] for i in keep]

    def to_mask(
        self, output_shape: tuple[int, int], with_index: bool = False, inv_pixel_size: float = 1.0, silent: bool = True
    ) -> np.ndarray:
        """Convert to mask.

        It's possible that GeoJSON/shapes data is in the physical units, so we need to convert it to pixels. In order to
        do so, it's necessary to multiply the coordinates by 1/pixel_size.
        """
        from image2image_io.utils.mask import polygons_to_mask, shapes_to_polygons

        # if native resolution is 1 (not set) and the specified pixel_size is not 1, then we need to multiply the
        if self.resolution != 1.0 and inv_pixel_size != 1.0:
            inv_pixel_size = 1.0

        polygons = shapes_to_polygons(
            self.shape_data, with_index=with_index, inv_pixel_size=inv_pixel_size, silent=silent
        )
        mask = polygons_to_mask(polygons, output_shape)
        return mask

    def to_mask_alt(
        self, output_size: tuple[int, int], with_index: bool = False, inv_pixel_size: float = 1.0, silent: bool = True
    ) -> np.ndarray:
        """
        Draw a binary or label mask using shape data.

        Parameters
        ----------
        output_size: tuple of int
            Size of the mask in tuple(x,y)
        with_index: bool
            Whether to write each mask instance as a label (1-n_shapes) or to write all as binary (255)
        inv_pixel_size: float
            Inverse pixel size for the mask
        silent: bool
            Whether to display progress bar

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
        for idx, sh in enumerate(tqdm(shapes, desc="Drawing shapes", leave=False, miniters=1000, disable=silent)):
            shape = sh["array"]
            yx = shape * inv_pixel_size
            mask = cv2.fillPoly(
                mask,
                pts=[yx.astype(np.int32)],
                color=idx + 1 if with_index else np.iinfo(dtype).max,
            )
        return mask

    def parse_data(self) -> tuple:
        """Parse data."""
        shape_data = self.shape_data
        shapes_geojson, _, _ = shape_reader(shape_data)
        n_shapes = len(shapes_geojson)
        if n_shapes > 10_000 and CONFIG.subsample and CONFIG.subsample_ratio < 1.0:
            n_subsample = int(CONFIG.subsample_ratio * n_shapes)
            logger.trace(f"Subsampling to {n_subsample:,} shapes.")
            with temporary_seed(CONFIG.subsample_random_seed, skip_if_negative_one=True):
                indices = np.random.choice(n_shapes, n_subsample, replace=False)
            shapes_geojson = [shapes_geojson[i] for i in indices]
            shape_data = [shape_data[i] for i in indices]

        n_shapes = len(shapes_geojson)
        shape_types = [sh["geometry"]["type"] for sh in shapes_geojson]
        shape_names = [sh["properties"]["classification"]["name"] for sh in shapes_geojson]
        shape_arrays = [s["array"][:, [1, 0]] for s in shape_data]  # expect y, x
        shape_props = {"name": shape_names}
        shape_text = {
            "string": "{name}",
            "color": "white",
            "anchor": "center",
            "size": 12,
            "visible": False,
        }
        return n_shapes, shape_types, shape_names, shape_arrays, shape_props, shape_text

    def to_shapes(self) -> tuple[str, dict[str, np.ndarray | str]]:
        """Convert to shapes that can be exported to Shapes layer."""
        _, shape_types, shape_names, shape_arrays, *_ = self.parse_data()
        if len(shape_types) > PATH_IF_COUNT:
            shape_types = ["path"] * len(shape_types)
        else:
            shape_types = [s.lower() for s in shape_types]  # expected polygon not Polygon
        return shape_names[0], {"shape_types": shape_types, "shape_data": shape_arrays}

    def to_shapes_kwargs(self, edge_color: str = "cyan", **kwargs: ty.Any) -> dict:
        """Return data so it's compatible with Shapes layer."""
        n_shapes, _, _, shape_arrays, shape_props, shape_text = self.parse_data()
        if CONFIG.shape_display == "polygon or path":
            shape_type = "polygon" if n_shapes < PATH_IF_COUNT else "path"
        else:
            shape_type = CONFIG.shape_display

        kws = {
            "data": [(shape, shape_type) for shape in shape_arrays],
            "properties": shape_props,
            "text": shape_text,
            "scale": self.scale,
            "affine": self.transform,
            "edge_width": 10,
            "edge_color": edge_color,
        }
        kws.update(kwargs)
        return kws

    def to_points_kwargs(self, face_color: str, **kwargs: ty.Any) -> dict:
        """Return data so it's compatible with Shapes layer."""
        df = _convert_geojson_to_df(self.shape_data)
        x = df["x"].values
        y = df["y"].values
        n = len(x)
        size = 15
        if n > 5_000:
            size = 5
        elif n > 50_000:
            size = 1

        kws = {
            "data": np.c_[y, x],
            "scale": self.scale,
            "affine": self.transform,
            "face_color": face_color,
            "size": size,
        }
        kws.update(kwargs)
        return kws

    def get_dask_pyr(self) -> list[ty.Any]:
        """Get dask representation of the pyramid."""
        raise NotImplementedError("Must implement method")

    def to_shapely(self) -> list[Polygon]:
        """Convert to shapely."""
        from shapely.geometry import Polygon

        return [Polygon(s["array"]) for s in self.shape_data]

    def to_table(self):
        """Convert to table."""
        data = []
        # iterate over shapes
        for i, shape in enumerate(self.shape_data):
            name = shape["shape_name"] + f"-{i}"
            # create numpy array with x, y, shape-name columns
            data.append(np.c_[shape["array"], np.full(shape["array"].shape[0], name)])
        # concatenate all arrays
        data = np.concatenate(data)
        # create DataFrame
        df = pd.DataFrame(data, columns=["x", "y", "shape"])
        df = df.astype({"x": np.float32, "y": np.float32})
        return df

    def to_csv(self, path: PathLike, as_px: bool = False) -> Path:
        """Export data as CSV file."""
        df = self.to_table()
        df.to_csv(path, index=False)
        return Path(path)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape of data."""
        if hasattr(self, "shape_data"):
            return (len(self.shape_data),)
        return (0,)


def _convert_geojson_to_df(shape_data: list[dict]) -> pd.DataFrame:
    """Convert GeoJSON data so that it can be transformed back to GeoJSON."""
    # types: pt = Point; pg = Polygon; mp = MultiPolygon
    data = []  # columns: x, y, unique_index, inner, outer, type
    for feature in shape_data:
        data.append(feature["array"])
    if data:
        data = np.concatenate(data)
    df = pd.DataFrame(data, columns=["x", "y"])
    return df
