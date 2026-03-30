"""Test shapes reader and utilities."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from image2image_io.readers.shapes_reader import (
    ShapesReader,
    _convert_geojson_to_df,
    get_shape_columns,
    is_txt_and_has_columns,
    napari_to_shapes_data,
    read_shapes,
    read_shapes_from_df,
)


@pytest.mark.parametrize("columns", [["vertex_x", "vertex_y", "cell"], ["vertex_x", "vertex_y", "cell_id"]])
def test_read_points_from_df(columns):
    x_col, y_col, cell_col = columns
    df = pl.DataFrame({x_col: [1, 2, 3], y_col: [4, 5, 6], cell_col: ["a", "a", "a"]})
    shapes_geojson, shapes_data, is_points = read_shapes_from_df(df)
    assert len(shapes_data) == 1, "Expected to read shapes from DataFrame."
    assert not is_points, "Expected to read shapes from DataFrame."

    df = pl.DataFrame({x_col: [1, 2, 3], y_col: [4, 5, 6], cell_col: ["a", "b", "c"]})
    _shapes_geojson, shapes_data, is_points = read_shapes_from_df(df)
    assert len(shapes_data) == 3, "Expected to read shapes from DataFrame."
    assert not is_points, "Expected to read shapes from DataFrame."


def test_shapes_reader_to_table_and_csv(tmp_path):
    reader = ShapesReader.create("shapes")
    reader.shape_data = [
        {"array": np.array([[1, 2], [3, 4]], dtype=np.float32), "shape_type": "polygon", "shape_name": "a"},
        {"array": np.array([[5, 6]], dtype=np.float32), "shape_type": "point", "shape_name": "b"},
    ]

    df = reader.to_table()

    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["x", "y", "shape"]
    assert df.shape == (3, 3)
    assert df["x"].dtype == pl.Float32
    assert df["y"].dtype == pl.Float32

    output = tmp_path / "shapes.csv"
    result = reader.to_csv(output)
    assert result == output
    assert Path(output).exists()


def test_shapes_reader_to_points_kwargs_uses_polars_frame():
    reader = ShapesReader.create("points")
    reader.shape_data = [
        {"array": np.array([[1, 2], [3, 4]], dtype=np.float32), "shape_type": "point", "shape_name": "a"},
    ]

    kwargs = reader.to_points_kwargs(face_color="red")

    np.testing.assert_array_equal(kwargs["data"], np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32))
    assert kwargs["face_color"] == "red"


def test_shapes_helpers_with_space_delimited_txt(tmp_path):
    path = tmp_path / "shapes.txt"
    path.write_text("vertex_x vertex_y cell\n1 4 a\n2 5 a\n3 6 b\n")

    assert is_txt_and_has_columns(path, ["vertex_x", "vertex_y"], [("cell", "cell_id")])
    assert get_shape_columns(path) == ("vertex_x", "vertex_y", "cell")

    shapes_geojson, shapes_data, is_points = read_shapes(path)
    assert len(shapes_geojson) == 2
    assert len(shapes_data) == 2
    assert not is_points


def test_read_shapes_from_parquet_and_point_mode(tmp_path):
    path = tmp_path / "shapes.parquet"
    pl.DataFrame({"x": [1, 2], "y": [3, 4], "shape_name": ["a", "a"]}).write_parquet(path)

    _geojson, shape_data, is_points = read_shapes(path)

    assert is_points
    assert len(shape_data) == 1
    np.testing.assert_array_equal(shape_data[0]["array"], np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32))


def test_napari_to_shapes_data_swaps_axes():
    data = [np.array([[10, 20], [30, 40]], dtype=np.float32)]

    result = napari_to_shapes_data("roi", data, ["polygon"])

    np.testing.assert_array_equal(result[0]["array"], np.array([[20, 10], [40, 30]], dtype=np.float32))
    assert result[0]["shape_name"] == "roi"


def test_convert_geojson_to_df_empty():
    df = _convert_geojson_to_df([])
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["x", "y"]
    assert df.shape == (0, 2)


def test_shapes_reader_to_table_empty():
    reader = ShapesReader.create("empty")
    reader.shape_data = []

    df = reader.to_table()

    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["x", "y", "shape"]
    assert df.shape == (0, 3)


@pytest.mark.parametrize(("n_points", "expected_size"), [(2, 15), (6_000, 5), (60_000, 1)])
def test_shapes_reader_to_points_kwargs_size_thresholds(n_points, expected_size):
    reader = ShapesReader.create("points")
    coords = np.column_stack([np.arange(n_points, dtype=np.float32), np.arange(n_points, dtype=np.float32)])
    reader.shape_data = [{"array": coords, "shape_type": "point", "shape_name": "a"}]

    kwargs = reader.to_points_kwargs(face_color="red")

    assert kwargs["size"] == expected_size
