"""Test utilities."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest

from image2image_io.utils.utilities import (
    check_df_columns,
    ensure_dask_array,
    format_mz,
    get_shape_of_image,
    resize,
    sort_pyramid,
)


def test_resize():
    array_2d = np.random.random((10, 15))
    resized_2d = resize(array_2d, (20, 30))
    assert resized_2d.shape == (20, 30), f"Expected shape (20, 20), got {resized_2d.shape}"

    array_3d = np.random.random((10, 10, 3))
    resized_3d = resize(array_3d, (20, 20))
    assert resized_3d.shape == (20, 20, 3), f"Expected shape (20, 20, 3), got {resized_3d.shape}"

    array_multichannel = np.random.random((5, 10, 10))
    resized_multichannel = resize(array_multichannel, (20, 20))
    assert resized_multichannel.shape == (5, 20, 20), f"Expected shape (5, 20, 20), got {resized_multichannel.shape}"


def test_get_shape_of_image():
    array_2d = np.random.random((10, 15))
    n_channels, channel_axis, shape = get_shape_of_image(array_2d)
    assert n_channels == 1, f"Expected 1 channel, got {n_channels}"
    assert channel_axis is None, f"Expected channel_axis None, got {channel_axis}"
    assert shape == (10, 15), f"Expected shape (10, 15), got {shape}"

    array_3d = np.random.random((10, 15, 3))
    n_channels, channel_axis, shape = get_shape_of_image(array_3d)
    assert n_channels == 3, f"Expected 3 channels, got {n_channels}"
    assert channel_axis == 2, f"Expected channel_axis 2, got {channel_axis}"
    assert shape == (10, 15), f"Expected shape (10, 15), got {shape}"

    array_multichannel = np.random.random((5, 10, 10))
    n_channels, channel_axis, shape = get_shape_of_image(array_multichannel)
    assert n_channels == 5, f"Expected 5 channels, got {n_channels}"
    assert channel_axis == 0, f"Expected channel_axis 0, got {channel_axis}"
    assert shape == (10, 10), f"Expected shape (10, 10), got {shape}"


def test_format_mz():
    assert format_mz(100) == "m/z 100.000"
    assert format_mz(100.123456) == "m/z 100.123"
    assert format_mz(100.1) == "m/z 100.100"
    assert format_mz(100.9999) == "m/z 101.000"


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
