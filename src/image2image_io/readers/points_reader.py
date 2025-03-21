"""GeoJSON reader for image2image."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
import pandas as pd
from koyo.rand import temporary_seed
from koyo.typing import PathLike
from loguru import logger

from image2image_io.config import CONFIG
from image2image_io.readers._base_reader import BaseReader
from image2image_io.readers.utilities import get_column_name


def read_points(path: PathLike, return_df: bool = False) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | pd.DataFrame:
    """Read points from CSV/parquet file."""
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
    if return_df:
        return df
    return read_points_from_df(df)


def read_points_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Read points from DataFrame."""
    x_key = get_column_name(df, ["x", "x_location", "x_centroid", "x:x"])
    y_key = get_column_name(df, ["y", "y_location", "y_centroid", "y:y"])
    if x_key not in df.columns or y_key not in df.columns:
        raise ValueError(f"Invalid columns: {df.columns}")
    x = df[x_key].values
    y = df[y_key].values
    return x, y, df.drop(columns=[x_key, y_key])


def get_channel_names_from_df(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Return list of column names. This is very specific and might break under certain circumstances."""
    filter_col = get_column_name(df, ["feature_name", "name"])
    if not filter_col:
        return "", list(df.columns)
    return filter_col, list(df[filter_col].unique())


class PointsReader(BaseReader):
    """GeoJSON reader for image2image."""

    reader_type = "points"
    _channel_names: list[str]
    _is_rgb: bool = False
    _array_dtype = np.dtype("float32")

    def __init__(
        self,
        path: PathLike,
        key: str | None = None,
        auto_pyramid: bool | None = None,
    ):
        super().__init__(path, key=key, auto_pyramid=auto_pyramid)

        self.x, self.y, self.df = read_points(self.path)  # type: ignore[var-annotated]
        # self.filter_column, self._channel_names = get_channel_names_from_df(self.df)
        self._channel_names = list(self.df.columns)

    def to_points(self) -> tuple[str, dict[str, np.ndarray | str]]:
        """Convert to shapes that can be exported to Shapes layer."""
        x, y, df = self.parse_data()
        return "points", {"data": np.c_[y, x], "properties": df}

    def parse_data(self) -> tuple:
        """Parse data."""
        x, y, df = self.x, self.y, self.df
        n_points = len(x)
        if n_points > 1_000_000 and CONFIG.subsample and CONFIG.subsample_ratio < 1.0:
            n_subsample = int(CONFIG.subsample_ratio * n_points)
            logger.trace(f"Subsampling to {n_subsample:,} points.")
            with temporary_seed(CONFIG.subsample_random_seed, skip_if_negative_one=True):
                indices = np.random.choice(n_points, n_subsample, replace=False)
            x = x[indices]
            y = y[indices]
            df = df.iloc[indices]
        return x, y, df

    def to_points_kwargs(self, face_color: str, **kwargs: ty.Any) -> dict:
        """Return data so it's compatible with Shapes layer."""
        x, y, df = self.parse_data()
        n = len(x)

        kws = {
            "data": np.c_[y, x],
            "scale": self.scale,
            "affine": self.transform,
            "features": df,
            "face_color": face_color,
            "size": 5 if n < 50_000 else 1,
        }
        kws.update(kwargs)
        return kws

    def get_dask_pyr(self) -> list[ty.Any]:
        """Get dask representation of the pyramid."""
        raise NotImplementedError("Must implement method")

    def to_csv(self, path: PathLike) -> str:
        """Export data as CSV file."""
        raise NotImplementedError("Must implement method")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape of data."""
        return (len(self.x),)
