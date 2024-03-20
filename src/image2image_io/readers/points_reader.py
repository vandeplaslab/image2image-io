"""GeoJSON reader for image2image."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
import pandas as pd
from koyo.typing import PathLike

from image2image_io.readers._base_reader import BaseReader


def read_points(path: PathLike) -> pd.DataFrame:
    """Read points from CSV/parquet file."""
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

    x = df["x"].values
    y = df["y"].values
    return x, y, df.drop(columns=["x", "y"])


class PointsReader(BaseReader):
    """GeoJSON reader for image2image."""

    reader_type = "points"
    _channel_names: list[str]

    def __init__(
        self,
        path: PathLike,
        key: str | None = None,
        auto_pyramid: bool | None = None,
    ):
        super().__init__(path, key=key, auto_pyramid=auto_pyramid)

        self.x, self.y, self.df = read_points(self.path)
        self._channel_names = list(self.df.columns)

    @property
    def display_name(self) -> str:
        """Retrieve display name from the path."""
        return self.path.stem

    def to_points(self) -> tuple[str, dict[str, np.ndarray | str]]:
        """Convert to shapes that can be exported to Shapes layer."""
        x, y, df = self.parse_data()
        return "points", {"data": np.c_[y, x], "properties": df}

    def parse_data(self) -> tuple:
        """Parse data."""
        return self.x, self.y, self.df

    def to_points_kwargs(self, channel_name: str, **kwargs: ty.Any) -> dict:
        """Return data so it's compatible with Shapes layer."""
        x, y, df = self.parse_data()
        kws = {
            "data": np.c_[y, x],
            "scale": self.scale,
            "affine": self.transform,
            "features": df,
            "face_color": channel_name,
        }
        kws.update(kwargs)
        return kws

    def get_dask_pyr(self) -> list[ty.Any]:
        """Get dask representation of the pyramid."""
        raise NotImplementedError("Must implement method")

    def to_csv(self, path: PathLike) -> str:
        """Export data as CSV file."""
        raise NotImplementedError("Must implement method")
