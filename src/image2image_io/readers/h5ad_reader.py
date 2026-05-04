"""AnnData H5AD reader."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from koyo.image import reshape_array_from_coordinates
from koyo.rand import temporary_seed
from koyo.typing import PathLike
from loguru import logger
from scipy import sparse

from image2image_io.config import CONFIG
from image2image_io.readers._base_reader import BaseReader

if ty.TYPE_CHECKING:
    from scipy.sparse import spmatrix


def _frame_to_features(df: pl.DataFrame) -> dict[str, list[ty.Any]]:
    """Convert a Polars DataFrame to a features-compatible mapping."""
    return df.to_dict(as_series=False)


def _decode_array(values: np.ndarray) -> list[ty.Any]:
    """Decode an HDF5 string-like array."""
    array = np.asarray(values)
    decoded = array.tolist()
    if not isinstance(decoded, list):
        decoded = [decoded]
    return [value.decode("utf-8") if isinstance(value, bytes) else value for value in decoded]


def _read_dataframe_group(group: h5py.Group) -> pl.DataFrame:
    """Read a minimal AnnData dataframe group into Polars."""
    columns: dict[str, list[ty.Any]] = {}
    column_order = _decode_array(group.attrs.get("column-order", np.array([], dtype="S")))
    index_key = group.attrs.get("_index", "_index")
    if isinstance(index_key, bytes):
        index_key = index_key.decode("utf-8")

    keys = [key for key in column_order if key in group]
    keys.extend(key for key in group.keys() if key not in keys and key != index_key)
    for key in keys:
        value = group[key]
        if isinstance(value, h5py.Group) and value.attrs.get("encoding-type") == b"categorical":
            categories = np.asarray(value["categories"][:])
            codes = np.asarray(value["codes"][:], dtype=int)
            decoded_categories = _decode_array(categories)
            columns[key] = [decoded_categories[code] if code >= 0 else None for code in codes]
        elif isinstance(value, h5py.Dataset):
            columns[key] = _decode_array(value[:])
    return pl.DataFrame(columns)


def _read_var_names(group: h5py.Group) -> list[str]:
    """Read variable names from an AnnData var group."""
    index_key = group.attrs.get("_index", "_index")
    if isinstance(index_key, bytes):
        index_key = index_key.decode("utf-8")
    if index_key not in group:
        raise ValueError("AnnData 'var' group is missing the variable index.")
    return [str(value) for value in _decode_array(group[index_key][:])]


def _read_matrix(node: h5py.Dataset | h5py.Group) -> np.ndarray | spmatrix:
    """Read a dense or sparse AnnData matrix."""
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[:])

    encoding_type = node.attrs.get("encoding-type", b"")
    if isinstance(encoding_type, bytes):
        encoding_type = encoding_type.decode("utf-8")
    shape = tuple(int(value) for value in node.attrs["shape"])
    data = np.asarray(node["data"][:])
    indices = np.asarray(node["indices"][:])
    indptr = np.asarray(node["indptr"][:])
    if encoding_type == "csr_matrix":
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
    if encoding_type == "csc_matrix":
        return sparse.csc_matrix((data, indices, indptr), shape=shape)
    raise ValueError(f"Unsupported AnnData matrix encoding: {encoding_type}")


def _read_spatial_coordinates(handle: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """Read AnnData spatial coordinates."""
    if "obsm" in handle and "spatial" in handle["obsm"]:
        spatial = np.asarray(handle["obsm"]["spatial"][:], dtype=np.float32)
        if spatial.ndim != 2 or spatial.shape[1] < 2:
            raise ValueError("AnnData 'obsm/spatial' must be a 2D array with at least two columns.")
        return spatial[:, 0], spatial[:, 1]

    if "obs" not in handle:
        raise ValueError("AnnData file is missing both 'obsm/spatial' and 'obs' coordinates.")

    obs = _read_dataframe_group(handle["obs"])
    x_key = next((key for key in ["x", "x_location", "x_centroid", "array_col"] if key in obs.columns), None)
    y_key = next((key for key in ["y", "y_location", "y_centroid", "array_row"] if key in obs.columns), None)
    if x_key is None or y_key is None:
        raise ValueError("Could not find spatial coordinates in AnnData 'obs' or 'obsm/spatial'.")
    return obs[x_key].cast(pl.Float32).to_numpy(), obs[y_key].cast(pl.Float32).to_numpy()


def read_h5ad(path: PathLike) -> tuple[np.ndarray, np.ndarray, pl.DataFrame, list[str], np.ndarray | spmatrix]:
    """Read an H5AD file."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        x, y = _read_spatial_coordinates(handle)
        obs = _read_dataframe_group(handle["obs"]) if "obs" in handle else pl.DataFrame()
        var_names = _read_var_names(handle["var"])
        matrix = _read_matrix(handle["X"])
    return x, y, obs, var_names, matrix


class H5ADReader(BaseReader):
    """Reader for AnnData H5AD files."""

    reader_type = "image"
    _is_rgb: bool = False
    _array_dtype = np.dtype("float32")
    allow_extraction = True

    def __init__(self, path: PathLike, key: str | None = None, auto_pyramid: bool | None = None):
        super().__init__(path, key=key, auto_pyramid=auto_pyramid)
        self.x, self.y, self.df, self._channel_names, self.matrix = read_h5ad(self.path)
        self._grid_x = np.rint(self.x - np.min(self.x) + 1).astype(np.int32)
        self._grid_y = np.rint(self.y - np.min(self.y) + 1).astype(np.int32)
        self._coordinates = np.c_[self._grid_x, self._grid_y]
        self._image_shape = (int(np.max(self._grid_y)), int(np.max(self._grid_x)))
        self._array_shape = (*self._image_shape, len(self._channel_names))
        self._pyramid = None

    def _extract_feature(self, name: str) -> np.ndarray:
        """Extract a single AnnData variable as a dense vector."""
        index = self.channel_names.index(name)
        column = self.matrix[:, index]
        if sparse.issparse(column):
            column = column.toarray()
        array = np.asarray(column).reshape(-1)
        return array.astype(np.float32, copy=False)

    def _reshape_feature(self, name: str) -> np.ndarray:
        """Reshape a single AnnData variable into image space."""
        return reshape_array_from_coordinates(
            self._extract_feature(name),
            self.image_shape,
            self._coordinates,
            fill_value=np.nan,
            offset=1,
        )

    def extract(self, feature_names: str | list[str]) -> tuple[Path, list[str]]:
        """Extract variables into the points feature table."""
        feature_names = [feature_names] if isinstance(feature_names, str) else list(feature_names)
        labels = []
        for name in feature_names:
            if name not in self.df.columns:
                if name not in self.channel_names:
                    raise ValueError(f"Feature '{name}' not found in AnnData variables.")
                self.df = self.df.with_columns(pl.Series(name, self._extract_feature(name)))
            labels.append(f"{name} | {self.name}")
        return self.path, labels

    def parse_data(self) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
        """Parse data."""
        x, y, df = self.x, self.y, self.df
        n_points = len(x)
        if n_points > 1_000_000 and CONFIG.subsample and CONFIG.subsample_ratio < 1.0:
            n_subsample = int(CONFIG.subsample_ratio * n_points)
            logger.trace(f"Subsampling to {n_subsample:,} AnnData points.")
            with temporary_seed(CONFIG.subsample_random_seed, skip_if_negative_one=True):
                indices = np.random.default_rng().choice(n_points, n_subsample, replace=False)
            x = x[indices]
            y = y[indices]
            df = df[indices]
        return x, y, df

    def get_image(self) -> np.ndarray:
        """Return the full AnnData image stack."""
        if self._pyramid is None:
            self._pyramid = [np.dstack([self._reshape_feature(name) for name in self.channel_names])]
        return self._pyramid[0]

    def get_dask_pyr(self) -> list[np.ndarray]:
        """Get image pyramid."""
        return [self.get_image()]

    def get_channel_axis_and_n_channels(self, shape: tuple[int, ...] | None = None) -> tuple[int | None, int]:
        """Return channel axis and number of channels."""
        return 2, len(self.channel_names)

    def get_channel_pyramid(self, index: int) -> list[np.ndarray]:
        """Return a single reshaped AnnData variable."""
        return [self._reshape_feature(self.channel_names[index])]

    def to_points(self) -> tuple[str, dict[str, ty.Any]]:
        """Convert to point-layer compatible data."""
        x, y, df = self.parse_data()
        return "points", {"data": np.c_[y, x], "properties": _frame_to_features(df)}

    def to_points_kwargs(self, face_color: str, **kwargs: ty.Any) -> dict[str, ty.Any]:
        """Return data compatible with napari points."""
        x, y, df = self.parse_data()
        n = len(x)
        kws = {
            "data": np.c_[y, x],
            "scale": self.scale,
            "affine": self.transform,
            "features": _frame_to_features(df),
            "face_color": face_color,
            "size": 5 if n < 50_000 else 1,
        }
        kws.update(kwargs)
        return kws

    def to_csv(self, path: PathLike) -> str:
        """Export data as CSV file."""
        self.df.write_csv(path)
        return str(path)
