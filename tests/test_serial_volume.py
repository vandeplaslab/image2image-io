"""Physical and transactional behavior of serial volumes."""

# ruff: noqa: INP001

from pathlib import Path

import numpy as np
import pytest
import zarr

from image2image_io.volume import open_volume, write_volume


def test_volume_roundtrip(tmp_path: Path) -> None:
    pixels = np.arange(2 * 3 * 13 * 15, dtype=np.uint16).reshape(2, 3, 13, 15)
    reads = []

    def read(c: int, z: int, y: int, x: int, h: int, w: int) -> np.ndarray:
        reads.append((z, h, w))
        return pixels[c, z, y : y + h, x : x + w]

    path = write_volume(
        tmp_path / "stack.ome.zarr",
        shape=pixels.shape,
        dtype=pixels.dtype,
        spacing=(7.5, 0.5, 0.5),
        origin=(0, -2, 3),
        channel_names=["a", "b"],
        present=[True, False, True],
        read_tile=read,
        tile_size=4,
    )
    volume = open_volume(path)
    expected = pixels.copy()
    expected[:, 1] = 0
    np.testing.assert_array_equal(volume.levels[0].compute(), expected)
    assert volume.spacing == (7.5, 0.5, 0.5)
    assert volume.origin == (0, -2, 3)
    assert volume.present == (True, False, True)
    assert volume.channel_names == ("a", "b")
    assert all(z != 1 and h <= 4 and w <= 4 for z, h, w in reads)
    assert [level.shape for level in volume.levels] == [(2, 3, 13, 15), (2, 3, 7, 8), (2, 3, 4, 4)]
    root = zarr.open_group(path, mode="r")
    assert root.metadata.zarr_format == 3
    assert root["0"].metadata.dimension_names == tuple("czyx")
    datasets = root.attrs["ome"]["multiscales"][0]["datasets"]
    assert datasets[2]["coordinateTransformations"][0]["scale"] == [1, 7.5, 2, 2]
    np.testing.assert_array_equal(volume.levels[1][:, 1].compute(), 0)
    assert volume.levels[1][0, 0, 0, 0].compute() == pixels[0, 0, :2, :2].mean().astype(np.uint16)


@pytest.mark.parametrize("failure", ["cancel", "read"])
def test_incomplete_store_is_not_published(tmp_path: Path, failure: str) -> None:
    calls = 0

    def read(c: int, z: int, y: int, x: int, h: int, w: int) -> np.ndarray:
        nonlocal calls
        calls += 1
        if failure == "read" and calls == 2:
            message = "bad source"
            raise RuntimeError(message)
        return np.ones((h, w), dtype=np.float32)

    path = tmp_path / "stack.ome.zarr"
    with pytest.raises((RuntimeError, InterruptedError)):
        write_volume(
            path,
            shape=(1, 2, 8, 8),
            dtype=np.dtype("float32"),
            spacing=(1, 1, 1),
            origin=(0, 0, 0),
            channel_names=["a"],
            present=[True, True],
            read_tile=read,
            tile_size=4,
            cancelled=lambda: failure == "cancel" and calls > 0,
        )
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_existing_output_preserved(tmp_path: Path) -> None:
    path = tmp_path / "existing.ome.zarr"
    path.mkdir()
    marker = path / "marker"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        write_volume(
            path,
            shape=(1, 2, 2, 2),
            dtype=np.dtype("uint16"),
            spacing=(1, 1, 1),
            origin=(0, 0, 0),
            channel_names=["a"],
            present=[True, True],
            read_tile=lambda *args: None,
        )
    assert marker.read_text() == "original"


@pytest.mark.parametrize("kind", ["partial", "writing", "malformed", "calibration"])
def test_open_rejects_incomplete_output(tmp_path: Path, kind: str) -> None:
    path = tmp_path / (".stack.partial" if kind == "partial" else "stack.ome.zarr")
    write_volume(
        path,
        shape=(1, 1, 2, 2),
        dtype=np.dtype("uint8"),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        channel_names=["a"],
        present=[True],
        read_tile=lambda *args: np.ones((2, 2), dtype=np.uint8),
    )
    root = zarr.open_group(path, mode="a")
    if kind == "malformed":
        root.attrs["ome"] = {"version": "0.5", "multiscales": []}
    elif kind == "calibration":
        ome = root.attrs["ome"]
        ome["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] = [1, 0, 1, 1]
        root.attrs["ome"] = ome
    elif kind == "writing":
        root.attrs["image2image"] = {"state": "writing"}
    with pytest.raises(ValueError):
        open_volume(path)
