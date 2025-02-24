"""Test reader."""

import numpy as np
import pytest

from image2image_io.readers import ArrayImageReader, get_simple_reader
from image2image_io.utils._test import get_test_files


@pytest.mark.parametrize("path", get_test_files("*.tiff"))
def test_get_simple_reader_tiff(path):
    reader = get_simple_reader(path)
    assert not reader.is_rgb, "Reader should not be rgb"
    assert reader.n_channels == 1, "Reader should have 1 channel"
    assert reader.reader_type == "image", "Reader should be image"


@pytest.mark.parametrize("path", get_test_files("*.png"))
def test_get_simple_reader_png(path):
    reader = get_simple_reader(path)
    assert reader.is_rgb, "Reader should not be rgb"
    assert reader.n_channels == 4, "Reader should have 1 channel"
    assert reader.reader_type == "image", "Reader should be image"


@pytest.mark.parametrize("path", get_test_files("*.jpg"))
def test_get_simple_reader_jpg(path):
    reader = get_simple_reader(path)
    assert reader.is_rgb, "Reader should not be rgb"
    assert reader.n_channels == 4, "Reader should have 1 channel"
    assert reader.reader_type == "image", "Reader should be image"


@pytest.mark.parametrize("path", get_test_files("*.imzML") + get_test_files("*.ibd"))
def test_get_simple_reader_imzml(path):
    reader = get_simple_reader(path)
    assert not reader.is_rgb, "Reader should not be rgb"
    assert reader.allow_extraction, "Reader should allow extraction"


@pytest.mark.parametrize("path", get_test_files("*.geojson"))
def test_get_simple_reader_geojson(path):
    reader = get_simple_reader(path)
    assert reader.reader_type == "shapes", "Reader should be shapes"


def test_crop_bbox_rgb(tmp_path):
    array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    reader = ArrayImageReader(tmp_path, array)
    assert reader.is_rgb, "Array should be rgb"

    cropped, _ = reader.crop_bbox(0, 512, 0, 512)
    assert cropped.shape == (512, 512, 3), "Cropped shape should be (512, 512, 3)"

    for array, _ in reader.crop_bbox_iter(0, 512, 0, 512):
        if array is None:
            continue
        assert array.shape == (512, 512), "Cropped shape should be (512, 512)"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_crop_bbox_multichannel(tmp_path, dtype):
    array = np.random.random((7, 1024, 1024)).astype(dtype)
    reader = ArrayImageReader(tmp_path, array)
    assert not reader.is_rgb, "Array should not be rgb"

    cropped, _ = reader.crop_bbox(0, 512, 0, 512)
    assert cropped.shape == (7, 512, 512), "Cropped shape should be (512, 512, 3)"

    for array, _ in reader.crop_bbox_iter(0, 512, 0, 512):
        if array is None:
            continue
        assert array.shape == (512, 512), "Cropped shape should be (512, 512)"


def test_crop_polygon_rgb(tmp_path):
    array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    reader = ArrayImageReader(tmp_path, array)
    assert reader.is_rgb, "Array should be rgb"

    cropped, bbox = reader.crop_polygon([(0, 0), (512, 0), (512, 512), (0, 512)])
    assert cropped.shape == (512, 512, 3), "Cropped shape should be (512, 512, 3)"

    for array, _bbox in reader.crop_polygon_iter([(0, 0), (512, 0), (512, 512), (0, 512)]):
        if array is None:
            continue
        assert array.shape == (512, 512), "Cropped shape should be (512, 512)"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_crop_polygon_multichannel(tmp_path, dtype):
    array = np.random.random((7, 1024, 1024)).astype(dtype)
    reader = ArrayImageReader(tmp_path, array)
    assert not reader.is_rgb, "Array should not be rgb"

    cropped, bbox = reader.crop_polygon([(0, 0), (512, 0), (512, 512), (0, 512)])
    assert cropped.shape == (7, 512, 512), "Cropped shape should be (512, 512, 3)"

    for array, _bbox in reader.crop_polygon_iter([(0, 0), (512, 0), (512, 512), (0, 512)]):
        if array is None:
            continue
        assert array.shape == (512, 512), "Cropped shape should be (512, 512)"
