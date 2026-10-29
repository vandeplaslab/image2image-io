"""Test reader."""
from __future__ import annotations

from pathlib import Path
from time import sleep
from types import SimpleNamespace

import imzy
import numpy as np
import pytest

from image2image_io.readers import ArrayImageReader, get_simple_reader, is_supported
from image2image_io.readers._czi import CziFile
from image2image_io.readers.czi_reader import CziImageReader, CziSceneImageReader
from image2image_io.readers.utilities import get_tifffile_info
from image2image_io.utils._test import get_test_file, get_test_files


class _RaceyArray:
    """Array stand-in that exposes unsynchronised read-modify-write losses."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.values = np.zeros(shape, dtype=np.uint8)

    def __setitem__(self, index: tuple[slice, ...], value: np.ndarray) -> None:
        values = self.values.copy()
        sleep(0.01)
        values[index] = value
        self.values = values


class _FakeCziSubblock:
    """Minimal decoded CZI subblock for concurrent-write tests."""

    def __init__(self, tile: np.ndarray) -> None:
        self._tile = tile

    def data(self, *, resize: bool, order: int) -> np.ndarray:
        return self._tile


class _FakeCziDirectoryEntry:
    """Minimal CZI directory entry for concurrent-write tests."""

    def __init__(self, start: tuple[int, int], tile: np.ndarray) -> None:
        self.start = start
        self._subblock = _FakeCziSubblock(tile)

    def data_segment(self) -> _FakeCziSubblock:
        return self._subblock


def test_get_tifffile_info_legacy_zarr_store(monkeypatch):
    """Test tifffile's legacy mapping store with the Zarr 3 runtime."""
    import json

    values = np.arange(12, dtype="<i4").reshape(3, 4)
    mapping = {
        ".zarray": json.dumps(
            {
                "zarr_format": 2,
                "shape": [3, 4],
                "chunks": [3, 4],
                "dtype": "<i4",
                "compressor": None,
                "fill_value": 0,
                "order": "C",
                "filters": None,
            }
        ).encode(),
        ".zattrs": b"{}",
        "0.0": values,
    }
    monkeypatch.setattr("image2image_io.readers.utilities.imread", lambda *args, **kwargs: mapping)

    shape, normalized_shape, dtype, series_index = get_tifffile_info("unused.tiff", series_index=0)

    assert tuple(shape) == (3, 4)
    assert tuple(normalized_shape) == (1, 3, 4)
    assert dtype == values.dtype
    assert series_index == 0


def test_czi_array_writes_are_serialized_for_shared_zarr_chunks() -> None:
    """Concurrent CZI tile writes must not lose updates within one Zarr chunk."""
    tiles = [np.full((1, 8), value, dtype=np.uint8) for value in range(1, 9)]
    czi = SimpleNamespace(
        shape=(8, 8),
        dtype=np.dtype(np.uint8),
        start=(0, 0),
        _fh=SimpleNamespace(lock=None),
        filtered_subblock_directory=[
            _FakeCziDirectoryEntry((index, 0), tile) for index, tile in enumerate(tiles)
        ],
    )
    out = _RaceyArray((8, 8))

    CziFile.asarray_alt(czi, out=out, max_workers=8)

    np.testing.assert_array_equal(out.values, np.vstack(tiles))


@pytest.mark.parametrize("path", get_test_files("*.tiff"))
def test_get_simple_reader_tiff(path):
    reader = get_simple_reader(path)
    assert not reader.is_rgb, "Reader should not be rgb"
    assert reader.n_channels == 1, "Reader should have 1 channel"
    assert reader.reader_type == "image", "Reader should be image"
    assert reader.pyramid is not None, "Reader should have pyramid"
    assert reader.n_in_pyramid == 1, "Reader should have 1 pyramid"
    thumbnail, scale = reader.get_thumbnail(100)
    assert isinstance(thumbnail, np.ndarray), "thumbnail should be np.ndarray"
    assert all(s > 0 for s in scale), "scale should be greater than 0"

    if reader._zstore:
        zarr_path = Path(reader._zstore.path)
        assert zarr_path.exists(), "Zarr path should exist"
        reader.close()
        assert not zarr_path.exists(), "Zarr path should be removed after close"


# @pytest.mark.xfail(reason="Needs to be fixed on zarr-v3")
@pytest.mark.parametrize("path", get_test_files("2d-image*.czi"))
def test_get_simple_reader_czi_2d(path):
    reader = get_simple_reader(path)
    assert not reader.is_rgb, "Reader should not be rgb"
    assert reader.n_channels == 1, "Reader should have 1 channel"
    assert reader.reader_type == "image", "Reader should be image"
    assert reader.pyramid is not None, "Reader should have pyramid"
    assert reader.n_in_pyramid == 1, "Reader should have 1 pyramid"
    thumbnail, scale = reader.get_thumbnail(100)
    assert isinstance(thumbnail, np.ndarray), "thumbnail should be np.ndarray"
    assert all(s > 0 for s in scale), "scale should be greater than 0"

    if reader._zstore:
        zarr_path = Path(reader._zstore.path)
        assert zarr_path.exists(), "Zarr path should exist"
        reader.close()
        assert not zarr_path.exists(), "Zarr path should be removed after close"


# @pytest.mark.parametrize("path", get_test_files("multichannel*.czi"))
# def test_get_simple_reader_czi_multichannel(path):
#     reader = get_simple_reader(path)
#     assert not reader.is_rgb, "Reader should not be rgb"
#     assert reader.n_channels == 1, "Reader should have more than 1 channel"
#     assert reader.reader_type == "image", "Reader should be image"


@pytest.mark.parametrize("path", get_test_files("*.png"))
def test_get_simple_reader_png(path):
    reader = get_simple_reader(path)
    assert reader.is_rgb, "Reader should not be rgb"
    assert reader.n_channels == 4, "Reader should have 1 channel"
    assert reader.reader_type == "image", "Reader should be image"
    thumbnail, scale = reader.get_thumbnail(100)
    assert isinstance(thumbnail, np.ndarray), "thumbnail should be np.ndarray"
    assert all(s > 0 for s in scale), "scale should be greater than 0"


@pytest.mark.parametrize("path", get_test_files("*.jpg"))
def test_get_simple_reader_jpg(path):
    reader = get_simple_reader(path)
    assert reader.is_rgb, "Reader should not be rgb"
    assert reader.n_channels == 4, "Reader should have 1 channel"
    assert reader.reader_type == "image", "Reader should be image"
    thumbnail, scale = reader.get_thumbnail(100)
    assert isinstance(thumbnail, np.ndarray), "thumbnail should be np.ndarray"
    assert all(s > 0 for s in scale), "scale should be greater than 0"


@pytest.mark.parametrize("path", get_test_files("*.imzML") + get_test_files("*.ibd"))
def test_get_simple_reader_imzml(path):
    reader = get_simple_reader(path)
    assert not reader.is_rgb, "Reader should not be rgb"
    assert reader.allow_extraction, "Reader should allow extraction"


requires_scils = pytest.mark.skipif(
    not hasattr(imzy, "SCiLSReader"), reason="Installed version of imzy does not support SCiLS files."
)


@requires_scils
@pytest.mark.parametrize("path", get_test_files("scils/*.sbd") + get_test_files("scils/*.slx"))
def test_get_simple_reader_scils(path):
    """Both the .sbd and .slx files should open the same dataset."""
    reader = get_simple_reader(path)
    assert not reader.is_rgb, "Reader should not be rgb"
    assert reader.allow_extraction, "Reader should allow extraction"
    assert reader.path.suffix == ".sbd", "Reader should always refer to the .sbd file"
    assert reader.resolution == 10.0
    assert reader.image_shape == (20, 30)
    assert reader.xmin == 0 and reader.ymin == 0

    path, labels = reader.extract(np.asarray([450.0, 500.0]), ppm=2000)
    assert len(labels) == 2
    assert reader.channel_names[1:] == [label.split(" | ")[0] for label in labels]
    image = reader.get_channel(1)
    assert image.shape == reader.image_shape
    assert image.sum() > 0
    assert reader.get_image().shape == (*reader.image_shape, 3)


def test_is_supported_scils():
    assert is_supported(get_test_file("scils/synthetic.sbd"))
    assert is_supported(get_test_file("scils/synthetic.slx"))


@pytest.mark.parametrize("path", get_test_files("*.geojson"))
def test_get_simple_reader_geojson(path):
    reader = get_simple_reader(path)
    assert reader.reader_type == "shapes", "Reader should be shapes"


def test_thumbnail(tmp_path):
    array = np.random.default_rng().integers(0, 255, (1000, 1000, 3), dtype=np.uint8)
    reader = ArrayImageReader(tmp_path, array, resolution=1.0)
    assert reader.is_rgb, "Array should be rgb"
    thumbnail, scale = reader.get_thumbnail(100)
    assert thumbnail.shape == (100, 100, 3), "Thumbnail shape should be (100, 100, 3)"
    assert all(s == 10.0 for s in scale), f"Scale should be (10.0, 10.0) not {scale}"

    thumbnail, scale = reader.get_thumbnail(2000)
    assert thumbnail.shape == (1000, 1000, 3), "Thumbnail shape should be (1000, 1000, 3)"
    assert all(s == 1.0 for s in scale), f"Scale should be (1.0, 1.0) not {scale}"


@pytest.mark.parametrize("path", get_test_files("*.czi"))
def test_czi_thumbnail_without_pyramid(path):
    reader = get_simple_reader(path, init_pyramid=False, auto_pyramid=False)
    thumbnail, scale = reader.get_thumbnail(128)
    assert isinstance(thumbnail, np.ndarray), "thumbnail should be np.ndarray"
    assert max(thumbnail.shape) <= 128, "thumbnail should fit within requested max size"
    assert all(s > 0 for s in scale), "scale should be greater than 0"
    assert reader._pyramid is None, "thumbnail should not initialize pyramid"
    reader.close()


def test_write_czi_thumbnail_without_pyramid(tmp_path, monkeypatch):
    from image2image_io.utils.utilities import write_thumbnail

    def fail_pyramid_init(self):
        pytest.fail("CZI thumbnail export should not initialize pyramid")

    monkeypatch.setattr(CziImageReader, "get_dask_pyr", fail_pyramid_init)
    monkeypatch.setattr(CziSceneImageReader, "get_dask_pyr", fail_pyramid_init)

    write_thumbnail(get_test_files("2d-image-small.czi")[0], tmp_path, with_title=False, first_only=True)
    thumbnails = list(tmp_path.glob("*thumbnail.jpg"))
    assert len(thumbnails) == 1, "Expected one thumbnail image"


def test_crop_bbox_rgb(tmp_path):
    array = np.random.default_rng().integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
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
    array = np.random.default_rng().integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
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
