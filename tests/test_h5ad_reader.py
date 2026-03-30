"""Tests for the H5AD AnnData reader."""

from __future__ import annotations

import h5py
import numpy as np
import polars as pl

from image2image_io.readers import get_simple_reader, sanitize_read_path
from image2image_io.readers.h5ad_reader import read_h5ad


def _write_string_dataset(group: h5py.Group, name: str, values: list[str]) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype="S"))


def _write_minimal_h5ad(path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=np.asarray([[1, 0], [0, 2], [3, 4]], dtype=np.float32))

        obs = handle.create_group("obs")
        obs.attrs["_index"] = "_index"
        obs.attrs["column-order"] = np.asarray(["sample"], dtype="S")
        _write_string_dataset(obs, "_index", ["cell1", "cell2", "cell3"])
        _write_string_dataset(obs, "sample", ["a", "b", "c"])

        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        _write_string_dataset(var, "_index", ["gene_a", "gene_b"])

        obsm = handle.create_group("obsm")
        obsm.create_dataset("spatial", data=np.asarray([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]], dtype=np.float32))


def test_read_h5ad(tmp_path):
    path = tmp_path / "test.h5ad"
    _write_minimal_h5ad(path)

    x, y, obs, var_names, matrix = read_h5ad(path)

    np.testing.assert_array_equal(x, np.array([10.0, 11.0, 12.0], dtype=np.float32))
    np.testing.assert_array_equal(y, np.array([20.0, 21.0, 22.0], dtype=np.float32))
    assert isinstance(obs, pl.DataFrame)
    assert obs.to_dict(as_series=False) == {"sample": ["a", "b", "c"]}
    assert var_names == ["gene_a", "gene_b"]
    np.testing.assert_array_equal(matrix, np.asarray([[1, 0], [0, 2], [3, 4]], dtype=np.float32))


def test_get_simple_reader_h5ad_and_extract(tmp_path):
    path = tmp_path / "test.h5ad"
    _write_minimal_h5ad(path)

    reader = get_simple_reader(path)

    assert reader.reader_type == "image"
    assert reader.allow_extraction
    assert reader.channel_names == ["gene_a", "gene_b"]
    assert sanitize_read_path(path) == path
    assert reader.shape == (3, 3, 2)

    np.testing.assert_allclose(
        reader.get_channel(0),
        np.array(
            [
                [1.0, np.nan, np.nan],
                [np.nan, 0.0, np.nan],
                [np.nan, np.nan, 3.0],
            ],
            dtype=np.float32,
        ),
        equal_nan=True,
    )

    _, labels = reader.extract("gene_a")
    assert labels == [f"gene_a | {path.name}"]

    kind, payload = reader.to_points()
    assert kind == "points"
    np.testing.assert_array_equal(payload["data"], np.array([[20.0, 10.0], [21.0, 11.0], [22.0, 12.0]], dtype=np.float32))
    assert payload["properties"]["sample"] == ["a", "b", "c"]
    assert payload["properties"]["gene_a"] == [1.0, 0.0, 3.0]
