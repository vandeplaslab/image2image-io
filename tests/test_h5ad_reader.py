"""Tests for the imspy AnnData H5AD reader."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from image2image_io.readers import H5ADReader, get_simple_reader, is_supported, read_data, sanitize_read_path
from image2image_io.readers.h5ad_reader import H5ADFormatError


def _write_imspy_h5ad(
    path: Path,
    *,
    matrix: np.ndarray | None = None,
    coordinates: np.ndarray | None = None,
    channel_names: tuple[str, ...] = ("100.0 Da", "200.0 Da"),
    image_shape: tuple[int, int] | None = (5, 6),
    resolution: float | None = 10.0,
) -> None:
    matrix = np.asarray([[1, 10], [2, 20], [3, 30]], dtype=np.float32) if matrix is None else matrix
    coordinates = np.asarray([[1, 2], [3, 4], [0, 5]], dtype=np.int64) if coordinates is None else coordinates
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=matrix, compression="gzip")

        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.asarray(channel_names, dtype=h5py.string_dtype("utf-8")))

        obsm = handle.create_group("obsm")
        obsm.create_dataset("spatial", data=coordinates)
        obsm.create_dataset("ignored-mask", data=np.ones(len(coordinates), dtype=np.uint8))

        if image_shape is not None or resolution is not None:
            imspy = handle.create_group("uns/spatial/imspy")
            if image_shape is not None:
                metadata = imspy.create_group("metadata")
                metadata.create_dataset("height", data=image_shape[0])
                metadata.create_dataset("width", data=image_shape[1])
            if resolution is not None:
                scalefactors = imspy.create_group("scalefactors")
                scalefactors.create_dataset("pixel_size_um", data=resolution)


def test_h5ad_reader_routes_and_preserves_imspy_spatial_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "centroids.h5ad"
    _write_imspy_h5ad(path)

    reader = get_simple_reader(path)

    assert isinstance(reader, H5ADReader)
    assert is_supported(path)
    assert sanitize_read_path(path) == path
    assert reader.reader_type == "image"
    assert reader.lazy
    assert not reader.allow_extraction
    assert reader.channel_names == ["100.0 Da", "200.0 Da"]
    assert reader.shape == (5, 6, 2)
    assert reader.image_shape == (5, 6)
    assert reader.resolution == 10.0
    assert reader.dtype == np.dtype("float32")
    assert isinstance(reader._matrix, h5py.Dataset)
    assert reader._pyramid is None

    monkeypatch.setattr(reader, "get_image", lambda: pytest.fail("Full image stack should not be loaded"))
    channel = reader.get_channel(1)

    expected = np.full((5, 6), np.nan, dtype=np.float32)
    expected[1, 2] = 10
    expected[3, 4] = 20
    expected[0, 5] = 30
    np.testing.assert_allclose(channel, expected, equal_nan=True)
    assert reader._pyramid is None
    reader.close()


def test_h5ad_reader_falls_back_to_coordinate_extents_and_unit_resolution(tmp_path: Path) -> None:
    path = tmp_path / "no-metadata.h5ad"
    _write_imspy_h5ad(path, image_shape=None, resolution=None)

    reader = H5ADReader(path)

    assert reader.image_shape == (4, 6)
    assert reader.resolution == 1.0
    reader.close()


def test_read_data_accepts_h5ad(tmp_path: Path) -> None:
    path = tmp_path / "centroids.h5ad"
    _write_imspy_h5ad(path)

    wrapper, added, path_map = read_data(path)
    reader = wrapper[0]

    assert isinstance(reader, H5ADReader)
    assert added == [reader.key]
    assert path_map == {path: path}
    reader.close()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_x", "missing the 'X'"),
        ("observation_count", "observation count"),
        ("non_integral", "integral pixel indices"),
        ("out_of_bounds", "outside the image shape"),
    ],
)
def test_h5ad_reader_rejects_invalid_schema(tmp_path: Path, mutation: str, message: str) -> None:
    path = tmp_path / "invalid.h5ad"
    coordinates = np.asarray([[1, 2], [3, 4], [0, 5]], dtype=np.float64)
    _write_imspy_h5ad(path, coordinates=coordinates)
    with h5py.File(path, "r+") as handle:
        if mutation == "missing_x":
            del handle["X"]
        elif mutation == "observation_count":
            spatial = np.asarray(handle["obsm/spatial"][:2])
            del handle["obsm/spatial"]
            handle["obsm"].create_dataset("spatial", data=spatial)
        elif mutation == "non_integral":
            handle["obsm/spatial"][0, 0] = 1.5
        else:
            handle["obsm/spatial"][0, 0] = 10

    with pytest.raises(H5ADFormatError, match=message):
        H5ADReader(path)
