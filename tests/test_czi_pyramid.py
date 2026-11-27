"""Tests for the lazy internal-pyramid CZI reader."""

from __future__ import annotations

import threading
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from image2image_io._zarr import TempStore
from image2image_io.readers._czi import CziFile
from image2image_io.readers._czi_pyramid import CziLevelArray, CziTileIndex, get_czi_lazy_pyramid

TEST_DATA = Path(__file__).parent / "_test_data"


class _Segment:
    def __init__(self, tile: np.ndarray, log: list) -> None:
        self._tile = tile
        self._log = log

    def data(self, resize: bool = True) -> np.ndarray:
        assert not resize, "pyramid reader must not resize subblocks"
        self._log.append(self._tile.shape)
        return self._tile


class _Entry:
    def __init__(self, dims: list, tile: np.ndarray, log: list) -> None:
        self.dimension_entries = dims
        self._segment = _Segment(tile, log)

    def data_segment(self) -> _Segment:
        return self._segment


def _dim(name: str, start: int, size: int, stored: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(dimension=name, start=start, size=size, stored_size=stored or size)


def _make_czi(
    image: np.ndarray,
    factors: tuple[int, ...] = (1, 2),
    tile: int = 64,
    rgb: bool = False,
    skip_level: int | None = None,
    extra_axis: bool = False,
) -> tuple[SimpleNamespace, list]:
    """Create a fake CZI with tiled subblocks for each level. `image` is (C, Y, X) or (Y, X, 3) if rgb."""
    log: list = []
    n_channels = 1 if rgb else image.shape[0]
    y_size, x_size = (image.shape[:2]) if rgb else image.shape[1:]
    entries = []
    for factor in factors:
        if factor == skip_level:
            continue
        level = image[::factor, ::factor] if rgb else image[:, ::factor, ::factor]
        level_y, level_x = level.shape[:2] if rgb else level.shape[1:]
        for channel in range(n_channels):
            for y in range(0, level_y, tile):
                for x in range(0, level_x, tile):
                    height, width = min(tile, level_y - y), min(tile, level_x - x)
                    plane = level if rgb else level[channel]
                    data = plane[y : y + height, x : x + width]
                    dims = [
                        _dim("Y", y * factor, height * factor, height),
                        _dim("X", x * factor, width * factor, width),
                        _dim("M", 0, 1),
                    ]
                    if not rgb:
                        dims.append(_dim("C", channel, 1))
                    entries.append(_Entry(dims, data, log))
    if rgb:
        axes, shape = "YX0", (y_size, x_size, image.shape[2])
    else:
        axes, shape = "CYX0", (n_channels, y_size, x_size, 1)
    if extra_axis:
        axes, shape = "T" + axes, (2, *shape)
    czi = SimpleNamespace(
        axes=axes,
        shape=shape,
        start=(0,) * len(shape),
        is_rgb=rgb,
        dtype=image.dtype,
        filtered_subblock_directory=entries,
        _fh=SimpleNamespace(lock=None),
    )
    return czi, log


@pytest.fixture
def multichannel() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 60000, (3, 200, 300), dtype=np.uint16)


def test_index_levels(multichannel):
    czi, _ = _make_czi(multichannel, factors=(1, 2, 4))
    index = CziTileIndex(czi)
    assert index.supported
    assert index.levels == [1, 2, 4]
    assert index.n_channels == 3
    assert index.shape_for_level(4) == (50, 75)


def test_lazy_levels_match_source(multichannel):
    czi, _ = _make_czi(multichannel, factors=(1, 2, 4))
    pyramid = get_czi_lazy_pyramid(czi)
    assert pyramid is not None
    assert [p.shape for p in pyramid] == [(3, 200, 300), (3, 100, 150), (3, 50, 75)]
    for array, factor in zip(pyramid, (1, 2, 4)):
        np.testing.assert_array_equal(array.compute(), multichannel[:, ::factor, ::factor])


def test_nothing_is_read_until_computed(multichannel):
    czi, log = _make_czi(multichannel)
    pyramid = get_czi_lazy_pyramid(czi)
    assert pyramid is not None
    assert log == []
    assert czi._fh.lock is True, "reads must be serialised for dask threads"


def test_region_read_only_touches_intersecting_tiles(multichannel):
    czi, log = _make_czi(multichannel, tile=64)
    level = CziLevelArray(CziTileIndex(czi), 1)
    region = level[1, 60:70, 120:130]  # straddles four 64 px tiles of channel 1
    np.testing.assert_array_equal(region, multichannel[1, 60:70, 120:130])
    assert len(log) == 4


def test_integer_and_open_slices(multichannel):
    czi, _ = _make_czi(multichannel)
    level = CziLevelArray(CziTileIndex(czi), 2)
    expected = multichannel[:, ::2, ::2]
    np.testing.assert_array_equal(level[2], expected[2])
    np.testing.assert_array_equal(level[..., 10:20], expected[..., 10:20])
    np.testing.assert_array_equal(level[:, 5, :], expected[:, 5, :])
    with pytest.raises(IndexError):
        level[::2]


def test_single_channel_is_2d():
    image = np.random.default_rng(1).integers(0, 255, (1, 150, 130), dtype=np.uint8)
    czi, _ = _make_czi(image)
    pyramid = get_czi_lazy_pyramid(czi)
    assert pyramid is not None
    assert pyramid[0].shape == (150, 130)
    np.testing.assert_array_equal(pyramid[0].compute(), image[0])
    np.testing.assert_array_equal(pyramid[1].compute(), image[0, ::2, ::2])


def test_rgb():
    image = np.random.default_rng(2).integers(0, 255, (150, 130, 3), dtype=np.uint8)
    czi, _ = _make_czi(image, rgb=True)
    pyramid = get_czi_lazy_pyramid(czi)
    assert pyramid is not None
    assert pyramid[0].shape == (150, 130, 3)
    np.testing.assert_array_equal(pyramid[0].compute(), image)
    np.testing.assert_array_equal(pyramid[1].compute(), image[::2, ::2])
    np.testing.assert_array_equal(pyramid[0][:, :, 1].compute(), image[:, :, 1])


def test_non_power_of_two_factor(multichannel):
    czi, _ = _make_czi(multichannel, factors=(1, 3))
    pyramid = get_czi_lazy_pyramid(czi)
    assert pyramid is not None
    assert pyramid[1].shape == (3, ceil(200 / 3), 100)
    np.testing.assert_array_equal(pyramid[1].compute(), multichannel[:, ::3, ::3])


def test_auto_pyramid_false_returns_base_level_only(multichannel):
    czi, _ = _make_czi(multichannel)
    pyramid = get_czi_lazy_pyramid(czi, auto_pyramid=False)
    assert pyramid is not None
    assert len(pyramid) == 1


def test_incomplete_level_is_dropped(multichannel):
    czi, _ = _make_czi(multichannel, factors=(1, 2, 4))

    def keep(entry: _Entry) -> bool:
        y_dim = entry.dimension_entries[0]
        is_factor_2 = y_dim.size == 2 * y_dim.stored_size
        return not (is_factor_2 and y_dim.start > 0)  # keep only the top row of factor-2 tiles

    czi.filtered_subblock_directory = [entry for entry in czi.filtered_subblock_directory if keep(entry)]
    index = CziTileIndex(czi)
    assert index.levels == [1, 4]


def _mean_by_2(array: np.ndarray) -> np.ndarray:
    """Reference 2x2 mean over the last two axes, trimming odd edges (same as `da.coarsen` + astype)."""
    y, x = array.shape[-2] // 2 * 2, array.shape[-1] // 2 * 2
    array = array[..., :y, :x]
    out = array.reshape(*array.shape[:-2], y // 2, 2, x // 2, 2).mean(axis=(-3, -1))
    return out.astype(array.dtype)


def test_no_internal_pyramid_without_auto_pyramid_returns_none(multichannel):
    """Eager reading decodes each tile only once, so it stays the better choice when no levels are needed."""
    czi, _ = _make_czi(multichannel, factors=(1,))
    assert get_czi_lazy_pyramid(czi, auto_pyramid=False) is None


def test_no_internal_pyramid_small_image_is_lazy_base_only(multichannel):
    czi, log = _make_czi(multichannel, factors=(1,))
    stores: list = []
    pyramid = get_czi_lazy_pyramid(czi, store_factory=lambda: stores.append(1))
    assert pyramid is not None
    assert len(pyramid) == 1
    assert not stores, "no store is needed when there are no coarse levels"
    assert log == []
    np.testing.assert_array_equal(pyramid[0].compute(), multichannel)


def test_no_internal_pyramid_builds_coarse_levels(multichannel):
    czi, log = _make_czi(multichannel, factors=(1,), tile=64)
    stores: list[TempStore] = []

    def factory() -> TempStore:
        stores.append(TempStore())
        return stores[-1]

    try:
        pyramid = get_czi_lazy_pyramid(czi, store_factory=factory, tile_size=16)
        assert pyramid is not None
        assert len(stores) == 1
        expected = multichannel
        np.testing.assert_array_equal(pyramid[0].compute(), expected)
        assert len(pyramid) == 4  # 200x300 -> 100x150 -> 50x75 -> 25x37
        for level in pyramid[1:]:
            expected = _mean_by_2(expected)
            assert level.shape == expected.shape
            np.testing.assert_array_equal(level.compute(), expected)
    finally:
        for store in stores:
            store.cleanup()


def test_no_internal_pyramid_builds_coarse_levels_rgb():
    image = np.random.default_rng(3).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    czi, _ = _make_czi(image, factors=(1,), rgb=True)
    store = TempStore()
    try:
        pyramid = get_czi_lazy_pyramid(czi, store_factory=lambda: store, tile_size=64)
        assert pyramid is not None
        assert [p.shape for p in pyramid] == [(200, 300, 3), (100, 150, 3)]
        expected = np.moveaxis(_mean_by_2(np.moveaxis(image, -1, 0)), 0, -1)
        np.testing.assert_array_equal(pyramid[1].compute(), expected)
    finally:
        store.cleanup()


def test_no_internal_pyramid_single_channel_2d():
    image = np.random.default_rng(4).integers(0, 255, (1, 200, 300), dtype=np.uint8)
    czi, _ = _make_czi(image, factors=(1,))
    store = TempStore()
    try:
        pyramid = get_czi_lazy_pyramid(czi, store_factory=lambda: store, tile_size=64)
        assert pyramid is not None
        assert [p.shape for p in pyramid] == [(200, 300), (100, 150)]
        np.testing.assert_array_equal(pyramid[1].compute(), _mean_by_2(image[0]))
    finally:
        store.cleanup()


def test_unsupported_layout_returns_none(multichannel):
    czi, _ = _make_czi(multichannel, extra_axis=True)
    assert not CziTileIndex(czi).supported
    assert get_czi_lazy_pyramid(czi) is None


@pytest.mark.parametrize("path", sorted(TEST_DATA.glob("*.czi")))
def test_bundled_czi_without_pyramid_uses_eager_path_when_no_levels_needed(path):
    """Bundled files have no internal pyramid, so without `auto_pyramid` they stay on the eager path."""
    czi = CziFile(str(path))
    try:
        assert get_czi_lazy_pyramid(czi, auto_pyramid=False) is None
    finally:
        czi.close()


def test_lazy_base_matches_eager_read_on_real_file():
    czi = CziFile(str(TEST_DATA / "2d-image.czi"))
    try:
        pyramid = get_czi_lazy_pyramid(czi)
        assert pyramid is not None
        eager = np.squeeze(czi.as_tzcyx0_array(max_workers=1))
        np.testing.assert_array_equal(pyramid[0].compute(), eager)
    finally:
        czi.close()


@pytest.mark.parametrize("reader_cls", ["CziImageReader", "CziSceneImageReader"])
def test_readers_use_lazy_pyramid_when_available(monkeypatch, reader_cls):
    """Readers must prefer the lazy internal pyramid and then not create a temporary zarr store."""
    import dask.array as da

    from image2image_io.readers import czi_reader

    path = TEST_DATA / "2d-image.czi"
    pyramid = [da.zeros((1920, 1920), dtype=np.uint16), da.zeros((960, 960), dtype=np.uint16)]
    monkeypatch.setattr(czi_reader, "get_czi_lazy_pyramid", lambda *_args, **_kwargs: pyramid)

    reader = getattr(czi_reader, reader_cls)(path, init_pyramid=True)
    try:
        assert reader.n_in_pyramid == 2
        assert reader._zstore is None
    finally:
        reader.close()


def test_background_build_returns_full_structure_and_same_values(multichannel):
    czi, _ = _make_czi(multichannel, factors=(1,), tile=64)
    sync_store, bg_store = TempStore(), TempStore()
    try:
        expected = get_czi_lazy_pyramid(czi, store_factory=lambda: sync_store, tile_size=16)
        pyramid = get_czi_lazy_pyramid(czi, store_factory=lambda: bg_store, tile_size=16, background=True)
        assert pyramid is not None
        assert expected is not None
        assert [p.shape for p in pyramid] == [p.shape for p in expected]
        assert [p.chunksize for p in pyramid] == [p.chunksize for p in expected]
        # reads wait for the background thread, so this cannot see partially-written levels
        for got, want in zip(pyramid, expected):
            np.testing.assert_array_equal(got.compute(), want.compute())
    finally:
        sync_store.cleanup()
        bg_store.cleanup()


def test_background_build_failure_raises_instead_of_hanging(multichannel):
    from concurrent.futures import ThreadPoolExecutor

    czi, _ = _make_czi(multichannel, factors=(1,), tile=64)

    def fail(*_args, **_kwargs):
        message = "corrupt subblock"
        raise ValueError(message)

    for entry in czi.filtered_subblock_directory:
        entry._segment.data = fail

    store = TempStore()
    try:
        pyramid = get_czi_lazy_pyramid(czi, store_factory=lambda: store, tile_size=16, background=True)
        assert pyramid is not None
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(pyramid[1].compute)
            with pytest.raises(RuntimeError, match="could not be built"):
                future.result(timeout=60)
    finally:
        store.cleanup()


def test_reader_requests_background_build(monkeypatch):
    from image2image_io.readers import czi_reader

    seen: dict = {}

    def fake(*_args, **kwargs):
        seen.update(kwargs)
        return  # fall back to the eager path, only the arguments matter here

    monkeypatch.setattr(czi_reader, "get_czi_lazy_pyramid", fake)
    reader = czi_reader.CziImageReader(TEST_DATA / "2d-image.czi", init_pyramid=True)
    try:
        assert seen["background"] is True
    finally:
        reader.close()


def test_background_build_does_not_block_the_caller(multichannel):
    """The pyramid must be returned while tiles are still being decoded (guards against dask probing the array)."""
    from concurrent.futures import ThreadPoolExecutor

    czi, _ = _make_czi(multichannel, factors=(1,), tile=64)
    gate = threading.Event()
    for entry in czi.filtered_subblock_directory:
        original = entry._segment.data

        def gated(*args, _original=original, **kwargs):
            gate.wait(60)
            return _original(*args, **kwargs)

        entry._segment.data = gated

    store = TempStore()
    try:
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(
                get_czi_lazy_pyramid, czi, store_factory=lambda: store, tile_size=16, background=True
            )
            returned = threading.Event()
            future.add_done_callback(lambda _: returned.set())
            returned_while_blocked = returned.wait(20)  # tiles cannot be decoded until the gate opens
            gate.set()  # always open it, so a regression fails below instead of hanging
            pyramid = future.result(timeout=60)
        assert returned_while_blocked, "get_czi_lazy_pyramid waited for the background build"
        assert pyramid is not None
        assert len(pyramid) == 4
        np.testing.assert_array_equal(pyramid[1].compute(), _mean_by_2(multichannel))
        pyramid[-1].compute()  # wait for the whole build before removing the store
    finally:
        gate.set()
        store.cleanup()


def _reference_levels(image: np.ndarray, n_levels: int) -> list[np.ndarray]:
    levels = [image]
    for _ in range(n_levels):
        levels.append(_mean_by_2(levels[-1]))
    return levels


def test_banded_build_matches_reference_with_odd_trailing_band(monkeypatch):
    """Several bands (the last one odd-sized) must give the same levels as reducing the whole image."""
    from image2image_io.readers import _czi_pyramid

    image = np.random.default_rng(5).integers(0, 60000, (3, 203, 301), dtype=np.uint16)
    czi, _ = _make_czi(image, factors=(1,), tile=64)
    monkeypatch.setattr(_czi_pyramid, "_BAND_BYTES", 16 * 301 * 2)  # 16 rows per band
    store = TempStore()
    try:
        pyramid = get_czi_lazy_pyramid(czi, store_factory=lambda: store, tile_size=16)
        assert pyramid is not None
        expected = _reference_levels(image, len(pyramid) - 1)
        assert len(pyramid) == 4  # 203x301 -> 101x150 -> 50x75 -> 25x37
        for got, want in zip(pyramid, expected):
            assert got.shape == want.shape
            np.testing.assert_array_equal(got.compute(), want)
    finally:
        store.cleanup()


def test_banded_build_rgb(monkeypatch):
    from image2image_io.readers import _czi_pyramid

    image = np.random.default_rng(6).integers(0, 255, (203, 301, 3), dtype=np.uint8)
    czi, _ = _make_czi(image, factors=(1,), rgb=True, tile=64)
    monkeypatch.setattr(_czi_pyramid, "_BAND_BYTES", 16 * 301 * 3)
    store = TempStore()
    try:
        pyramid = get_czi_lazy_pyramid(czi, store_factory=lambda: store, tile_size=32)
        assert pyramid is not None
        # the reference reduces the last two axes, so put the samples first
        expected = _reference_levels(np.moveaxis(image, -1, 0), len(pyramid) - 1)
        assert len(pyramid) > 1
        for got, want in zip(pyramid, expected):
            np.testing.assert_array_equal(got.compute(), np.moveaxis(want, 0, -1))
    finally:
        store.cleanup()


def test_banded_build_decodes_each_tile_once_when_bands_align_with_tiles(monkeypatch):
    """Dask chunks that do not line up with tiles decode tiles repeatedly. Bands that do must not."""
    from image2image_io.readers import _czi_pyramid

    image = np.random.default_rng(7).integers(0, 60000, (2, 256, 300), dtype=np.uint16)
    czi, log = _make_czi(image, factors=(1,), tile=64)
    n_tiles = len(czi.filtered_subblock_directory)
    monkeypatch.setattr(_czi_pyramid, "_BAND_BYTES", 128 * 300 * 2)  # 128 rows = two rows of tiles
    store = TempStore()
    try:
        pyramid = get_czi_lazy_pyramid(czi, store_factory=lambda: store, tile_size=32)
        assert pyramid is not None
        pyramid[-1].compute()
        assert len(log) == n_tiles
    finally:
        store.cleanup()


def test_parallel_decode_keeps_last_tile_on_top():
    """Overlapping tiles are decoded in parallel but must be pasted in mosaic order."""
    log: list = []
    entries = []
    for i in range(12):
        tile = np.full((40, 40), i + 1, dtype=np.uint8)
        dims = [_dim("Y", 0, 40), _dim("X", i * 10, 40), _dim("M", i, 1)]
        entries.append(_Entry(dims, tile, log))
    czi = SimpleNamespace(
        axes="YX0",
        shape=(40, 150, 1),
        start=(0, 0, 0),
        is_rgb=False,
        dtype=np.dtype(np.uint8),
        filtered_subblock_directory=entries,
        _fh=SimpleNamespace(lock=None),
    )
    index = CziTileIndex(czi)
    sequential = CziLevelArray(index, 1, workers=1)[:, :]
    parallel = CziLevelArray(index, 1, workers=8)[:, :]
    np.testing.assert_array_equal(parallel, sequential)
    # the last tile that covers a column wins
    for x in (0, 35, 45, 85, 149):
        assert parallel[0, x] == max(i + 1 for i in range(12) if i * 10 <= x < i * 10 + 40)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.uint32, np.float32])
@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("shape", [(100, 90), (101, 91), (1, 7), (5, 1), (64, 33, 3)])
def test_mean_by_2_matches_reference(dtype, workers, shape):
    from image2image_io.readers._czi_pyramid import _mean_by_2 as fast

    rng = np.random.default_rng(8)
    if np.issubdtype(dtype, np.floating):
        plane = rng.random(shape, dtype=np.float32) * 1000
    else:
        info = np.iinfo(dtype)
        plane = rng.integers(info.min, info.max, shape, dtype=dtype, endpoint=True)
    got = fast(plane, workers)
    # reference: np.mean over blocks, as in the eager reader (axes are the first two, so move them last)
    moved = np.moveaxis(plane, (0, 1), (-2, -1))
    want = np.moveaxis(_mean_by_2(moved), (-2, -1), (0, 1))
    assert got.dtype == plane.dtype
    assert got.shape == want.shape
    if np.issubdtype(dtype, np.floating):
        # the reference accumulates float32 in float32, the implementation in float64 (up to 1 ulp apart)
        np.testing.assert_allclose(got, want, rtol=1e-6)
    else:
        np.testing.assert_array_equal(got, want)


def test_mean_by_2_threaded_large_plane():
    """Large planes are split over threads, the seams must not show."""
    from image2image_io.readers._czi_pyramid import _mean_by_2 as fast

    plane = np.random.default_rng(9).integers(0, 65535, (2049, 2100), dtype=np.uint16)
    np.testing.assert_array_equal(fast(plane, 7), fast(plane, 1))


def test_build_lock_is_first_come_first_served():
    import time

    from image2image_io.readers._czi_pyramid import _FifoLock

    lock = _FifoLock()
    order: list[int] = []
    threads = []

    def worker(i: int) -> None:
        with lock.turn():
            order.append(i)

    with lock.turn():  # hold the lock while the others queue up
        for i in range(6):
            thread = threading.Thread(target=worker, args=(i,))
            thread.start()
            threads.append(thread)
            time.sleep(0.05)  # let it take its ticket before the next one arrives
    for thread in threads:
        thread.join(timeout=10)
    assert order == list(range(6))


def test_build_lock_is_released_after_an_error():
    from image2image_io.readers._czi_pyramid import _FifoLock

    lock = _FifoLock()
    with pytest.raises(ValueError, match="boom"), lock.turn():
        message = "boom"
        raise ValueError(message)
    with lock.turn():  # would deadlock if the failed turn was never released
        pass


def test_reserved_turns_are_served_in_reservation_order():
    """Builds must run in the order the readers were created, whatever order their threads start running."""
    import time

    from image2image_io.readers._czi_pyramid import _FifoLock

    lock = _FifoLock()
    tickets = [lock.reserve() for _ in range(5)]
    order: list[int] = []

    def worker(ticket: int) -> None:
        with lock.turn(ticket):
            order.append(ticket)

    threads = []
    for ticket in reversed(tickets):  # start the threads in the opposite order
        thread = threading.Thread(target=worker, args=(ticket,))
        thread.start()
        threads.append(thread)
        time.sleep(0.03)
    for thread in threads:
        thread.join(timeout=10)
    assert order == tickets


def test_abandoned_ticket_does_not_block_the_queue():
    from image2image_io.readers._czi_pyramid import _FifoLock

    lock = _FifoLock()
    first, second, third = lock.reserve(), lock.reserve(), lock.reserve()
    lock.abandon(second)  # abandoned while still waiting behind `first`
    with lock.turn(first):
        pass
    with lock.turn(third):  # would deadlock if `second` still blocked the queue
        pass
    lock.abandon(lock.reserve())  # abandoned while it is the one being served
    with lock.turn():
        pass


def test_failed_build_releases_its_turn(monkeypatch):
    """A build that fails must not block the builds queued behind it."""
    from image2image_io.readers import _czi_pyramid

    czi, _ = _make_czi(np.zeros((1, 100, 100), dtype=np.uint16), factors=(1,), tile=64)
    index = CziTileIndex(czi)

    def broken(*_args, **_kwargs):
        message = "no space left"
        raise OSError(message)

    monkeypatch.setattr(_czi_pyramid, "_build_levels", broken)
    ticket = _czi_pyramid._BUILD_LOCK.reserve()
    with pytest.raises(OSError, match="no space"):
        _czi_pyramid._build_coarse_levels(index, None, 16, ticket=ticket)
    with _czi_pyramid._BUILD_LOCK.turn():  # deadlocks if the failed build kept its turn
        pass
