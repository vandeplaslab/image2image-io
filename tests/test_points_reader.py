"""Test points reader and utilities."""

import pandas as pd
import pytest

from image2image_io.readers.points_reader import read_points_from_df


@pytest.mark.parametrize("columns", [["x", "y"], ["x_location", "y_location"], ["x", "y_location"]])
def test_read_points_from_df(columns):
    x_col, y_col = columns
    df = pd.DataFrame({x_col: [1, 2, 3], y_col: [4, 5, 6]})
    x, y, _ = read_points_from_df(df)
    assert len(x) == 3, "Expected to read points from DataFrame."
