"""Physical and transactional behavior of serial volumes."""

# ruff: noqa: INP001

import math
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.abc.buffer import Buffer
from zarr.storage import LocalStore

from image2image_io.volume import open_volume, write_volume


@pytest.mark.parametrize("sharding", [False, True])
@pytest.mark.parametrize("dot_separator", [False, True])
def test_volume_roundtrip(tmp_path: Path, sharding: bool, dot_separator: bool) -> None:
    pixels = np.arange(2 * 3 * 37 * 19, dtype=np.uint16).reshape(2, 3, 37, 19)
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
        sharding=sharding,
        dot_separator=dot_separator,
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
    assert [level.shape for level in volume.levels] == [
        (2, 3, 37, 19),
        (2, 3, 19, 10),
        (2, 3, 10, 5),
        (2, 3, 5, 3),
        (2, 3, 3, 2),
    ]
    root = zarr.open_group(path, mode="r")
    assert root.metadata.zarr_format == 3
    assert root["0"].metadata.dimension_names == tuple("czyx")
    for index in range(len(volume.levels)):
        level_path = path / str(index)
        assert (level_path / ("c.0.0.0.0" if dot_separator else "c/0/0/0/0")).is_file()
        if dot_separator:
            assert all(child.is_file() for child in level_path.iterdir())
        array = root[str(index)]
        assert (array.shards is not None) == sharding
        storage_shape = array.shards or array.chunks
        expected_files = 4 * math.ceil(array.shape[2] / storage_shape[2]) * math.ceil(array.shape[3] / storage_shape[3])
        files = [child for child in level_path.rglob("*") if child.is_file() and child.name != "zarr.json"]
        assert len(files) == expected_files
        if sharding:
            assert math.prod(array.shards) * pixels.dtype.itemsize <= 16 * 1024**2
            assert all(shard % chunk == 0 for shard, chunk in zip(array.shards, array.chunks, strict=True))
    datasets = root.attrs["ome"]["multiscales"][0]["datasets"]
    assert datasets[2]["coordinateTransformations"][0]["scale"] == [1, 7.5, 2, 2]
    np.testing.assert_array_equal(volume.levels[1][:, 1].compute(), 0)
    assert volume.levels[1][0, 0, 0, 0].compute() == pixels[0, 0, :2, :2].mean().astype(np.uint16)
    for level in volume.levels:
        np.testing.assert_array_equal(level.compute(), expected)
        padded = np.pad(expected, ((0, 0), (0, 0), (0, expected.shape[2] % 2), (0, expected.shape[3] % 2)), mode="edge")
        expected = padded.astype(np.float64).reshape(2, 3, padded.shape[2] // 2, 2, padded.shape[3] // 2, 2)
        expected = expected.mean(axis=(3, 5)).astype(pixels.dtype)


@pytest.mark.parametrize("failure", ["cancel", "read"])
@pytest.mark.parametrize("sharding", [False, True])
def test_incomplete_store_is_not_published(tmp_path: Path, failure: str, sharding: bool) -> None:
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
            sharding=sharding,
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


@pytest.mark.parametrize("dot_separator", [False, True])
def test_each_shard_is_written_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dot_separator: bool) -> None:
    writes: list[str] = []
    original = LocalStore.set

    async def record_write(store: LocalStore, key: str, value: Buffer) -> None:
        if not key.endswith("zarr.json"):
            writes.append(key)
        await original(store, key, value)

    monkeypatch.setattr(LocalStore, "set", record_write)
    write_volume(
        tmp_path / "stack.ome.zarr",
        shape=(1, 1, 37, 19),
        dtype=np.dtype("uint16"),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        channel_names=["a"],
        present=[True],
        read_tile=lambda c, z, y, x, h, w: np.ones((h, w), dtype=np.uint16),
        tile_size=4,
        sharding=True,
        dot_separator=dot_separator,
    )
    assert writes
    assert len(writes) == len(set(writes))


@pytest.mark.parametrize("dtype", ["uint16", "float64"])
def test_shards_respect_buffer_target(tmp_path: Path, dtype: str) -> None:
    path = write_volume(
        tmp_path / "stack.ome.zarr",
        shape=(1, 1, 4096, 4096),
        dtype=np.dtype(dtype),
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        channel_names=["a"],
        present=[False],
        read_tile=lambda *args: pytest.fail("Missing sections must not be read"),
        tile_size=2048,
        sharding=True,
    )
    root = zarr.open_group(path, mode="r")
    for _, level in root.arrays():
        chunk_bytes = math.prod(level.chunks) * np.dtype(dtype).itemsize
        assert math.prod(level.shards) * np.dtype(dtype).itemsize <= max(16 * 1024**2, chunk_bytes)
