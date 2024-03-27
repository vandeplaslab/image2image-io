"""Test utilities."""

import pandas as pd
import pytest

from image2image_io.readers.utilities import check_df_columns


@pytest.mark.parametrize("columns", [["x", "y"], ["x_location", "y_location"], ["x", "y_location"]])
def test_if_points(columns):
    x_col, y_col = columns
    df = pd.DataFrame({x_col: [1, 2, 3], y_col: [4, 5, 6]})
    assert check_df_columns(df, [x_col, y_col]), "Expected columns to be valid."


@pytest.mark.parametrize(
    "columns",
    [
        ["x", "y", ("cell", "cell_id")],
        ["vertex_x", "vertex_y", ("cell", "cell_id")],
        ["x", "vertex_y", ("cell", "cell_id")],
    ],
)
def test_if_shapes(columns):
    x_col, y_col, either = columns
    df = pd.DataFrame({x_col: [1, 2, 3], y_col: [4, 5, 6], either[0]: ["a", "b", "c"]})
    assert check_df_columns(df, [x_col, y_col], [either]), "Expected columns to be valid."

    df = pd.DataFrame({x_col: [1, 2, 3], y_col: [4, 5, 6], either[1]: ["a", "b", "c"]})
    assert check_df_columns(df, [x_col, y_col], [either]), "Expected columns to be valid."
