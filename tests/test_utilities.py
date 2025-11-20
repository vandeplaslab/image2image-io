"""Test utilities."""

import dask.array as da
import pandas as pd
import pytest

from image2image_io.utils.utilities import check_df_columns, ensure_dask_array, sort_pyramid


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


def test_sort_pyramid_ordered():
    pyramid = [
        da.array([1, 2, 3]),
        da.array([1, 2]),
        da.array([1]),
    ]
    sorted_pyramid = sort_pyramid(pyramid)
    assert len(sorted_pyramid) == 3, "Pyramid should have 3 levels."
    assert sorted_pyramid[0].shape[0] == 3, "First level should have shape 3."
    assert sorted_pyramid[1].shape[0] == 2, "Second level should have shape 2."
    assert sorted_pyramid[2].shape[0] == 1, "Third level should have shape 1."


def test_sort_pyramid_unordered():
    pyramid = [
        da.array([1, 2]),
        da.array([1]),
        da.array([1, 2, 3]),
    ]
    sorted_pyramid = sort_pyramid(pyramid)
    assert len(sorted_pyramid) == 3, "Pyramid should have 3 levels."
    assert sorted_pyramid[0].shape[0] == 3, "First level should have shape 3."
    assert sorted_pyramid[1].shape[0] == 2, "Second level should have shape 2."
    assert sorted_pyramid[2].shape[0] == 1, "Third level should have shape 1."


def test_ensure_dask_array_numpy():
    import numpy as np

    array = np.array([[1, 2, 3], [4, 5, 6]])
    dask_array = ensure_dask_array(array)
    assert isinstance(dask_array, da.Array), "Output should be a Dask array."
    assert dask_array.shape == array.shape, "Shapes should match."
