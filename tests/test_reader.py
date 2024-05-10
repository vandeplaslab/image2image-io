"""Test reader."""

import pytest

from image2image_io.readers import get_simple_reader
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
    assert reader.n_channels == 1, "Reader should have 1 channel"
    assert reader.reader_type == "image", "Reader should be image"


@pytest.mark.parametrize("path", get_test_files("*.jpg"))
def test_get_simple_reader_jpg(path):
    reader = get_simple_reader(path)
    assert reader.is_rgb, "Reader should not be rgb"
    assert reader.n_channels == 1, "Reader should have 1 channel"
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
