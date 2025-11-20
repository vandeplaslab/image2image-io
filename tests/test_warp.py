"""Warp image with image2image transformation matrix."""

import numpy as np
import pytest

from image2image_io.utils._test import get_test_files
from image2image_io.utils.warp import centered_transform, get_affine_from_config, warp


def test_centered_transform():
    image_shape = (100, 200)
    centered_matrix = centered_transform(image_shape, 1.0, 0.0)
    assert centered_matrix.shape == (3, 3), "Centered matrix should be 3x3."


@pytest.mark.parametrize("inv", [True, False])
@pytest.mark.parametrize("px", [True, False])
@pytest.mark.parametrize("yx", [True, False])
@pytest.mark.parametrize("path", get_test_files("transform/*.json"))
def test_get_affine_from_config(path, yx, px, inv):
    affine_matrix, fixed_image_shape, fixed_pixel_size_um, moving_image_shape, moving_pixel_size_um = (
        get_affine_from_config(path, yx=yx, px=px, inv=inv)
    )
    assert affine_matrix.shape == (3, 3), "Affine matrix should be 3x3."
    assert isinstance(fixed_image_shape, tuple), "Fixed image shape should be a tuple."
    assert isinstance(moving_image_shape, tuple), "Moving image shape should be a tuple."
    assert isinstance(fixed_pixel_size_um, float), "Fixed pixel size should be a float."
    assert isinstance(moving_pixel_size_um, float), "Moving pixel size should be a float."


def test_warp_rgb():
    image = np.random.random((100, 100, 3))
    transform_matrix = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
    warped = warp(transform_matrix, (150, 150), image)
    assert image.ndim == warped.ndim == 3, "Input and warped images should be 3D."
    assert warped.shape == (150, 150, 3), "Warped image should have correct shape."


def test_warp_2d():
    image = np.random.random((100, 109))
    transform_matrix = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
    warped = warp(transform_matrix, (150, 150), image)
    assert image.ndim == warped.ndim == 2, "Input and warped images should be 2D."
    assert warped.shape == (150, 150), "Warped image should have correct shape."


def test_warp_3d():
    image = np.random.random((4, 100, 109))
    transform_matrix = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
    warped = warp(transform_matrix, (150, 150), image)
    assert image.ndim == warped.ndim == 3, "Input and warped images should be 3D."
    assert warped.shape == (4, 150, 150), "Warped image should have correct shape."


def test_warp_4d():
    image = np.random.random((4, 5, 5, 1))
    transform_matrix = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]])
    with pytest.raises(ValueError):
        warp(transform_matrix, (150, 150), image)
