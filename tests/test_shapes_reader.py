"""Test shapes reader and utilities."""

import pandas as pd
import pytest

from image2image_io.readers.shapes_reader import read_shapes_from_df


@pytest.mark.parametrize("columns", [["vertex_x", "vertex_y", "cell"], ["vertex_x", "vertex_y", "cell_id"]])
def test_read_points_from_df(columns):
    x_col, y_col, cell_col = columns
    df = pd.DataFrame({x_col: [1, 2, 3], y_col: [4, 5, 6], cell_col: ["a", "a", "a"]})
    shapes_geojson, shapes_data, is_points = read_shapes_from_df(df)
    assert len(shapes_data) == 1, "Expected to read shapes from DataFrame."
    assert not is_points, "Expected to read shapes from DataFrame."

    df = pd.DataFrame({x_col: [1, 2, 3], y_col: [4, 5, 6], cell_col: ["a", "b", "c"]})
    shapes_geojson, shapes_data, is_points = read_shapes_from_df(df)
    assert len(shapes_data) == 3, "Expected to read shapes from DataFrame."
    assert not is_points, "Expected to read shapes from DataFrame."
