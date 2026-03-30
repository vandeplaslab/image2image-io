"""Test points reader and utilities."""

import numpy as np
import polars as pl
import pytest

from image2image_io.readers.points_reader import get_channel_names_from_df, read_points, read_points_from_df


@pytest.mark.parametrize("columns", [["x", "y"], ["x_location", "y_location"], ["x", "y_location"]])
def test_read_points_from_df(columns):
    x_col, y_col = columns
    df = pl.DataFrame({x_col: [1, 2, 3], y_col: [4, 5, 6]})
    x, _y, _ = read_points_from_df(df)
    assert len(x) == 3, "Expected to read points from DataFrame."


def test_read_points_from_csv_and_return_df(tmp_path):
    path = tmp_path / "points.csv"
    pl.DataFrame({"x": [1, 2], "y": [3, 4], "label": ["a", "b"]}).write_csv(path)

    x, y, df = read_points(path)

    np.testing.assert_array_equal(x, np.array([1, 2]))
    np.testing.assert_array_equal(y, np.array([3, 4]))
    assert df.to_dict(as_series=False) == {"label": ["a", "b"]}
    assert read_points(path, return_df=True).columns == ["x", "y", "label"]


def test_read_points_from_space_delimited_txt(tmp_path):
    path = tmp_path / "points.txt"
    path.write_text("x y label\n1 3 a\n2 4 b\n")

    x, y, df = read_points(path)

    np.testing.assert_array_equal(x, np.array([1, 2]))
    np.testing.assert_array_equal(y, np.array([3, 4]))
    assert df.columns == ["label"]


def test_read_points_from_parquet(tmp_path):
    path = tmp_path / "points.parquet"
    pl.DataFrame({"x_location": [1, 2], "y_location": [3, 4], "value": [5, 6]}).write_parquet(path)

    x, y, df = read_points(path)

    np.testing.assert_array_equal(x, np.array([1, 2]))
    np.testing.assert_array_equal(y, np.array([3, 4]))
    assert df.to_dict(as_series=False) == {"value": [5, 6]}


def test_get_channel_names_from_df():
    filter_col, channel_names = get_channel_names_from_df(pl.DataFrame({"feature_name": ["a", "b", "a"], "x": [1, 2, 3]}))
    assert filter_col == "feature_name"
    assert set(channel_names) == {"a", "b"}

    filter_col, channel_names = get_channel_names_from_df(pl.DataFrame({"x": [1], "y": [2]}))
    assert filter_col == ""
    assert channel_names == ["x", "y"]
