"""Test combine utilities."""

import numpy as np
import pytest

from image2image_io.combine import reduce


@pytest.mark.parametrize("reduce_func", ["mean", "max", "sum"])
def test_reduce_rgb(reduce_func):
    arrays = [np.random.randint(0, 255, (100, 100, 3)), np.random.randint(0, 255, (100, 100, 3))]
    result = reduce(arrays, reduce_func=reduce_func)
    assert result.shape == (100, 100, 3), "Result shape should be (100, 100, 3)"


@pytest.mark.parametrize("reduce_func", ["mean", "max", "sum"])
def test_reduce_grayscale(reduce_func):
    arrays = [np.random.randint(0, 255, (100, 100)), np.random.randint(0, 255, (100, 100))]
    result = reduce(arrays, reduce_func=reduce_func)
    assert result.shape == (100, 100), "Result shape should be (100, 100)"
