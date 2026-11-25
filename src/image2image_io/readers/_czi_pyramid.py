"""Lazy, pyramid-aware access to CZI subblocks.

CZI files created by ZEN usually store reduced-resolution subblocks next to the full-resolution ones.
Here the subblock directory is indexed once (no pixel data is read) and each internal level is exposed
as a lazy dask array whose chunks are decoded from the matching subblocks on demand.

Files without internal levels get a lazy full-resolution level and, optionally, coarser levels that are built
in a single decoding pass.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import threading
import typing as ty
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import ceil

import dask.array as da
import numpy as np

if ty.TYPE_CHECKING:
    from image2image_io.readers._czi import CziFile

CHUNK_SIZE = 2048  # same as the TIFF reader
# memory used by one band of rows (of one channel) while building the coarse levels
_BAND_BYTES = 512 * 1024**2
# the level factor is computed from `size / stored_size`; allow for rounding of the stored size
_FACTOR_TOLERANCE = 0.05


@dataclass
class _LevelTiles:
    """Tiles that belong to a single pyramid level, in file (mosaic) order."""

    y: np.ndarray  # tile origin along Y in level pixels
    x: np.ndarray  # tile origin along X in level pixels
    h: np.ndarray  # stored tile height
    w: np.ndarray  # stored tile width
    channel: np.ndarray  # channel index
    entries: list[ty.Any]  # czifile directory entries


class CziTileIndex:
    """Index of the subblock directory of a CZI file, grouped by pyramid level.

    The index is built from directory entries only, so no data segment is touched. When the file
    cannot be represented as a lazy pyramid (e.g. T/Z stacks, several samples per channel, anisotropic
    levels) `supported` is False and the caller should use another strategy.

    Parameters
    ----------
    czi : CziFile
        Open CZI file. For multi-scene files, pass the scene-filtered `CziSceneFile`.
    """

    def __init__(self, czi: CziFile) -> None:
        self.czi = czi
        self.supported: bool = False
        self.levels: list[int] = []
        self.n_channels: int = 1
        self.n_samples: int = 1
        self.is_rgb: bool = False
        self.full_shape: tuple[int, int] = (0, 0)
        self.tiles: dict[int, _LevelTiles] = {}
        self._build()

    def _build(self) -> None:
        czi = self.czi
        axes = czi.axes
        shape, start = czi.shape, czi.start
        if "Y" not in axes or "X" not in axes:
            return
        # everything besides channels and spatial dimensions has to be a singleton
        for idx, axis in enumerate(axes):
            if axis not in "YX0C" and shape[idx] != 1:
                return
        self.is_rgb = bool(czi.is_rgb)
        self.n_samples = int(shape[axes.index("0")]) if "0" in axes else 1
        self.n_channels = int(shape[axes.index("C")]) if "C" in axes else 1
        if self.is_rgb and self.n_channels > 1:
            return
        y_origin, x_origin = int(start[axes.index("Y")]), int(start[axes.index("X")])
        c_origin = int(start[axes.index("C")]) if "C" in axes else 0
        self.full_shape = (int(shape[axes.index("Y")]), int(shape[axes.index("X")]))

        raw: dict[int, list[tuple[int, int, int, int, int, ty.Any]]] = {}
        for entry in czi.filtered_subblock_directory:
            dims = {dim.dimension: dim for dim in entry.dimension_entries}
            y_dim, x_dim = dims.get("Y"), dims.get("X")
            if y_dim is None or x_dim is None:
                return
            for name, dim in dims.items():
                if name not in "YXM" and (dim.size != 1 or dim.stored_size != 1):
                    return
            factor_y, factor_x = y_dim.size / y_dim.stored_size, x_dim.size / x_dim.stored_size
            factor = round(factor_y)
            if factor < 1 or abs(factor_y - factor) > _FACTOR_TOLERANCE * factor:
                return
            if abs(factor_x - factor) > _FACTOR_TOLERANCE * factor:
                return
            channel = int(dims["C"].start) - c_origin if "C" in dims else 0
            raw.setdefault(factor, []).append(
                (
                    round((y_dim.start - y_origin) / factor),
                    round((x_dim.start - x_origin) / factor),
                    int(y_dim.stored_size),
                    int(x_dim.stored_size),
                    channel,
                    entry,
                )
            )
        if 1 not in raw:
            return

        tiles = {factor: self._to_tiles(rows) for factor, rows in raw.items()}
        channels_at_base = set(tiles[1].channel.tolist())
        self.tiles = {1: tiles[1]}
        for factor in sorted(f for f in tiles if f > 1):
            if self._is_complete(tiles[factor], factor, channels_at_base):
                self.tiles[factor] = tiles[factor]
        self.levels = sorted(self.tiles)
        self.supported = True

    @staticmethod
    def _to_tiles(rows: list[tuple[int, int, int, int, int, ty.Any]]) -> _LevelTiles:
        y, x, h, w, channel, entries = zip(*rows)
        return _LevelTiles(
            y=np.asarray(y), x=np.asarray(x), h=np.asarray(h), w=np.asarray(w), channel=np.asarray(channel),
            entries=list(entries),
        )  # fmt: skip

    def _is_complete(self, tiles: _LevelTiles, factor: int, channels: set[int]) -> bool:
        """Check that a level spans the whole image for every channel (bounding box check)."""
        if set(tiles.channel.tolist()) != channels:
            return False
        level_shape = self.shape_for_level(factor)
        tol_y, tol_x = max(2, int(0.01 * level_shape[0])), max(2, int(0.01 * level_shape[1]))
        return bool(
            tiles.y.min() <= tol_y
            and tiles.x.min() <= tol_x
            and (tiles.y + tiles.h).max() >= level_shape[0] - tol_y
            and (tiles.x + tiles.w).max() >= level_shape[1] - tol_x
        )

    def shape_for_level(self, factor: int) -> tuple[int, int]:
        """Return the (Y, X) shape of a pyramid level."""
        return ceil(self.full_shape[0] / factor), ceil(self.full_shape[1] / factor)


class CziLevelArray:
    """Array-like view of a single pyramid level that decodes subblocks on access.

    The layout matches the squeezed output of the eager reader: `(Y, X)` for grayscale, `(C, Y, X)` for
    multi-channel and `(Y, X, 3)` for RGB images. Where tiles overlap, the last tile in mosaic order wins.
    """

    def __init__(self, index: CziTileIndex, factor: int, workers: int = 1) -> None:
        self._index = index
        self._tiles = index.tiles[factor]
        self.factor = factor
        self.workers = workers  # threads used to decode the tiles of a single read
        self.dtype = np.dtype(index.czi.dtype)
        y_size, x_size = index.shape_for_level(factor)
        if index.is_rgb:
            self._axes = "YXS"
            self.shape = (y_size, x_size, index.n_samples)
        elif index.n_channels > 1:
            self._axes = "CYX"
            self.shape = (index.n_channels, y_size, x_size)
        else:
            self._axes = "YX"
            self.shape = (y_size, x_size)
        self.ndim = len(self.shape)

    def chunks(self, chunk_size: int = CHUNK_SIZE) -> tuple[int, ...]:
        """Return dask chunk sizes."""
        sizes = {"C": 1, "Y": chunk_size, "X": chunk_size, "S": self._index.n_samples}
        return tuple(min(sizes[axis], size) for axis, size in zip(self._axes, self.shape))

    def __getitem__(self, key: ty.Any) -> np.ndarray:
        """Read a region. Supports integers and unit-step slices."""
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            pos = key.index(Ellipsis)
            key = key[:pos] + (slice(None),) * (self.ndim - len(key) + 1) + key[pos + 1 :]
        key = key + (slice(None),) * (self.ndim - len(key))
        if len(key) != self.ndim:
            message = f"Too many indices for array with {self.ndim} dimensions."
            raise IndexError(message)

        ranges: dict[str, tuple[int, int]] = {}
        drop: list[int] = []
        for dim, (axis, size, k) in enumerate(zip(self._axes, self.shape, key)):
            if isinstance(k, slice):
                start, stop, step = k.indices(size)
                if step != 1:
                    message = "Only unit-step slices are supported."
                    raise IndexError(message)
                ranges[axis] = (start, max(start, stop))
            else:
                idx = int(k) + size if int(k) < 0 else int(k)
                if not 0 <= idx < size:
                    message = f"Index {k} is out of bounds for axis {dim} with size {size}."
                    raise IndexError(message)
                ranges[axis] = (idx, idx + 1)
                drop.append(dim)
        out = self._read(ranges)
        return out[tuple(0 if dim in drop else slice(None) for dim in range(self.ndim))] if drop else out

    def _read(self, ranges: dict[str, tuple[int, int]]) -> np.ndarray:
        """Read the requested region in the layout of this array."""
        c0, c1 = ranges.get("C", (0, 1))
        y0, y1 = ranges["Y"]
        x0, x1 = ranges["X"]
        s0, s1 = ranges.get("S", (0, 1))
        n_samples = self._index.n_samples if "S" in ranges else 1
        buffer = np.zeros((c1 - c0, y1 - y0, x1 - x0, n_samples), dtype=self.dtype)

        tiles = self._tiles
        if buffer.size:
            # tiles that intersect the region and belong to the requested channels
            mask = (
                (tiles.y < y1)
                & (tiles.y + tiles.h > y0)
                & (tiles.x < x1)
                & (tiles.x + tiles.w > x0)
                & (tiles.channel >= c0)
                & (tiles.channel < c1)
            )
            selected = np.flatnonzero(mask)

            def decode(idx: int) -> np.ndarray:
                return np.asarray(tiles.entries[idx].data_segment().data(resize=False))

            def paste(idx: int, data: np.ndarray) -> None:
                ty0, tx0 = int(tiles.y[idx]), int(tiles.x[idx])
                height, width = int(tiles.h[idx]), int(tiles.w[idx])
                data = data.reshape(height, width, n_samples if n_samples > 1 else -1)
                oy0, oy1 = max(ty0, y0), min(ty0 + height, y1)
                ox0, ox1 = max(tx0, x0), min(tx0 + width, x1)
                buffer[int(tiles.channel[idx]) - c0, oy0 - y0 : oy1 - y0, ox0 - x0 : ox1 - x0] = data[
                    oy0 - ty0 : oy1 - ty0, ox0 - tx0 : ox1 - tx0
                ]

            if self.workers > 1 and len(selected) > 1:
                # decode in parallel but paste in mosaic order, so that the last tile still wins
                with ThreadPoolExecutor(self.workers) as pool:
                    for idx, data in zip(selected, pool.map(decode, selected)):
                        paste(int(idx), data)
            else:
                for idx in selected:
                    paste(int(idx), decode(int(idx)))

        if "S" in self._axes:  # RGB, there is always exactly one channel
            return buffer[0, :, :, s0:s1]
        buffer = buffer[..., 0]
        return buffer if "C" in self._axes else buffer[0]


def _coarse_shape(shape: tuple[int, ...], is_rgb: bool) -> tuple[int, ...]:
    """Return the shape after one 2x reduction of Y and X (odd edges are trimmed)."""
    if is_rgb or len(shape) == 2:  # (Y, X) or (Y, X, S)
        return (shape[0] // 2, shape[1] // 2, *shape[2:])
    return (shape[0], shape[1] // 2, shape[2] // 2)  # (C, Y, X)


def _coarse_chunks(shape: tuple[int, ...], is_rgb: bool, tile_size: int) -> tuple[int, ...]:
    """Return the chunks of a coarse level."""
    if is_rgb:
        chunks: tuple[int, ...] = (tile_size, tile_size, shape[2])
    elif len(shape) == 2:
        chunks = (tile_size, tile_size)
    else:
        chunks = (1, tile_size, tile_size)
    return tuple(min(chunk, size) for chunk, size in zip(chunks, shape))


def _mean_by_2(plane: np.ndarray, workers: int = 1) -> np.ndarray:
    """Mean-downsample the first two axes by 2, trimming odd edges.

    Gives the same values as `np.mean` over 2x2 blocks followed by `astype` (the eager reader's method), but sums
    strided slices, which is several times faster than a strided mean. Large planes are split into rows of blocks
    that are reduced by `workers` threads.
    """
    rows, cols = plane.shape[0] // 2 * 2, plane.shape[1] // 2 * 2
    out = np.empty((rows // 2, cols // 2, *plane.shape[2:]), dtype=plane.dtype)
    if out.size == 0:
        return out
    exact_integers = plane.dtype.kind == "u" and plane.dtype.itemsize <= 2

    def reduce(start: int, stop: int) -> None:
        view = plane[start:stop, :cols]
        # for 8/16-bit unsigned data the sum of four values is exact and >> 2 truncates like astype
        total = view[0::2, 0::2].astype(np.uint32 if exact_integers else np.float64)
        total += view[0::2, 1::2]
        total += view[1::2, 0::2]
        total += view[1::2, 1::2]
        if exact_integers:
            total >>= 2
        else:
            total *= 0.25
        np.copyto(out[start // 2 : stop // 2], total, casting="unsafe")

    if workers > 1 and plane.size >= 1 << 22:
        step = max(2, -(-rows // workers) // 2 * 2)
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda start: reduce(start, min(start + step, rows)), range(0, rows, step)))
    else:
        reduce(0, rows)
    return out


class _FifoLock:
    """Lock that hands out turns first come, first served (a plain `Lock` wakes waiters in any order)."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._tickets = itertools.count()
        self._serving = 0
        self._abandoned: set[int] = set()

    def reserve(self) -> int:
        """Reserve a place in the queue. It must be used with `turn` or given up with `abandon`."""
        with self._condition:
            return next(self._tickets)

    def abandon(self, ticket: int) -> None:
        """Give up a reserved place that will never be used, so that it does not block the others."""
        with self._condition:
            self._abandoned.add(ticket)
            self._advance()

    def _advance(self) -> None:
        # the caller holds the condition
        while self._serving in self._abandoned:
            self._abandoned.discard(self._serving)
            self._serving += 1
        self._condition.notify_all()

    @contextlib.contextmanager
    def turn(self, ticket: int | None = None) -> ty.Iterator[None]:
        """Wait for a turn (of a reserved ticket, or at the back of the queue) and hold it for the context."""
        if ticket is None:
            ticket = self.reserve()
        with self._condition:
            while ticket != self._serving:
                self._condition.wait()
        try:
            yield
        finally:
            with self._condition:
                self._serving += 1
                self._advance()


# coarse levels are built one file at a time (in order of request), since one build already keeps all cores busy
_BUILD_LOCK = _FifoLock()


class _PendingLevel:
    """Array-like for a coarse level that a background thread is still building.

    Reads wait until the level has been written. If the build failed, reads raise instead of waiting forever.
    """

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.ndim = len(shape)
        self._ready = threading.Event()
        self._array: ty.Any = None
        self._error: BaseException | None = None

    def set_result(self, array: ty.Any) -> None:
        """Mark the level as built."""
        self._array = array
        self._ready.set()

    def set_error(self, error: BaseException) -> None:
        """Mark the level as failed, unless it was already built."""
        if not self._ready.is_set():
            self._error = error
            self._ready.set()

    def __getitem__(self, key: ty.Any) -> np.ndarray:
        self._ready.wait()
        if self._array is None:
            message = "The pyramid level could not be built in the background."
            raise RuntimeError(message) from self._error
        return np.asarray(self._array[key])


def _n_coarse_levels(shape: tuple[int, ...], is_rgb: bool, tile_size: int) -> int:
    """Return the number of down-sampled levels, using the same rule as the eager reader."""
    yx = shape[:2] if is_rgb or len(shape) == 2 else shape[1:]
    n_levels = 1
    while min(yx) // 2**n_levels >= tile_size:
        n_levels += 1
    return n_levels - 1


def _build_coarse_levels(
    index: CziTileIndex,
    store: ty.Any,
    tile_size: int,
    on_level: ty.Callable[[int, ty.Any], None] | None = None,
    ticket: int | None = None,
) -> list[da.Array]:
    """Build the coarse levels once it is this build's turn (see `_build_levels`).

    `ticket` is a place reserved with `_BUILD_LOCK.reserve()`. Without one, the build queues up when called.
    """
    with _BUILD_LOCK.turn(ticket):
        return _build_levels(index, store, tile_size, on_level)


def _build_levels(
    index: CziTileIndex,
    store: ty.Any,
    tile_size: int,
    on_level: ty.Callable[[int, ty.Any], None] | None = None,
) -> list[da.Array]:
    """Build all down-sampled levels in a single pass over the tiles and persist them in a zarr store.

    The image is read in full-width bands of rows, so a tile is decoded about once (instead of once per dask
    chunk it touches) and the tiles of a band are decoded in parallel. Each band is reduced by 2 repeatedly and
    written to every level. Band heights are multiples of `2**n_levels`, so the result is the same as reducing
    the whole image. `on_level(level, zarr_array)` is called for every level once the pass is complete.
    """
    import zarr
    from koyo.timer import MeasureTimer
    from loguru import logger

    from image2image_io.config import CONFIG

    level0 = CziLevelArray(index, 1, workers=(os.cpu_count() or 1) if CONFIG.multicore else 1)
    n_levels = _n_coarse_levels(level0.shape, index.is_rgb, tile_size)
    root = zarr.open_group(store, mode="a", zarr_format=3)
    arrays = []
    shape = level0.shape
    for level in range(1, n_levels + 1):
        shape = _coarse_shape(shape, index.is_rgb)
        arrays.append(
            root.create_array(
                name=str(level),
                shape=shape,
                chunks=_coarse_chunks(shape, index.is_rgb, tile_size),
                dtype=level0.dtype,
                overwrite=True,
            )
        )

    multiple = 2**n_levels
    y_axis = level0._axes.index("Y")
    row_bytes = int(np.prod(level0.shape[y_axis + 1 :])) * level0.dtype.itemsize  # one row of one channel
    band_rows = max(multiple, int(_BAND_BYTES // row_bytes) // multiple * multiple)
    height = level0.shape[y_axis]
    channels = range(index.n_channels) if level0._axes == "CYX" else [None]

    with MeasureTimer() as timer:
        for channel in channels:
            for row0 in range(0, height, band_rows):
                row1 = min(row0 + band_rows, height)
                if channel is None:
                    band = level0[(slice(row0, row1),) + (slice(None),) * (level0.ndim - 1)]
                else:
                    band = level0[channel, row0:row1, :]
                for level, array in enumerate(arrays, start=1):
                    band = _mean_by_2(band, level0.workers)
                    if band.shape[0] == 0:
                        break
                    start = row0 // 2**level
                    target = (slice(start, start + band.shape[0]),) + (slice(None),) * (band.ndim - 1)
                    array[target if channel is None else (channel, *target)] = band
    logger.bind(src="CZI").trace(f"Built {n_levels} down-sampled level(s) in {timer()}")
    if on_level is not None:
        for level, array in enumerate(arrays, start=1):
            on_level(level, array)
    return [da.from_zarr(array) for array in arrays]


def _build_coarse_levels_in_background(
    base: da.Array, index: CziTileIndex, store: ty.Any, tile_size: int, token: str
) -> list[da.Array]:
    """Return placeholders for the coarse levels and fill them from a background thread."""
    import zarr  # noqa: F401  (fail here, not in the thread, if zarr is missing)
    from loguru import logger

    pending: list[_PendingLevel] = []
    shape = base.shape
    for _ in range(_n_coarse_levels(base.shape, index.is_rgb, tile_size)):
        shape = _coarse_shape(shape, index.is_rgb)
        pending.append(_PendingLevel(shape, base.dtype))

    arrays = [
        da.from_array(
            level,
            chunks=_coarse_chunks(level.shape, index.is_rgb, tile_size),
            name=f"czi-{token}-pending-{i}",
            # without it dask probes the array with a zero-size slice, which would wait for the build
            meta=np.empty((0,) * level.ndim, dtype=level.dtype),
        )
        for i, level in enumerate(pending, start=1)
    ]

    # Reserve the place in the queue here, not in the thread, so that builds start in the order the readers were
    # created (the one the user opened first is ready first) rather than in the order threads happen to run.
    ticket = _BUILD_LOCK.reserve()

    def run() -> None:
        try:
            _build_coarse_levels(
                index,
                store,
                tile_size,
                on_level=lambda level, array: pending[level - 1].set_result(array),
                ticket=ticket,
            )
        except BaseException as error:  # noqa: BLE001
            logger.bind(src="CZI").error(f"Failed to build pyramid in the background: {error}")
            for level in pending:
                level.set_error(error)

    try:
        threading.Thread(target=run, name="czi-pyramid", daemon=True).start()
    except BaseException:
        _BUILD_LOCK.abandon(ticket)  # the thread never ran, do not block the builds behind it
        raise
    return arrays


def get_czi_lazy_pyramid(
    czi: CziFile,
    auto_pyramid: bool = True,
    chunk_size: int = CHUNK_SIZE,
    store_factory: ty.Callable[[], ty.Any] | None = None,
    tile_size: int = 1024,
    background: bool = False,
) -> list[da.Array] | None:
    """Return a lazily-read pyramid of a CZI file as dask arrays (highest resolution first).

    Files that contain reduced-resolution subblocks expose them directly, nothing is read until needed.
    Files with only full-resolution subblocks get a lazy base level. If `auto_pyramid` is set, the coarser
    levels are built in one pass over the tiles and persisted in a temporary zarr store.

    Parameters
    ----------
    czi : CziFile
        Open CZI file (scene-filtered for multi-scene files).
    auto_pyramid : bool
        If False, only the full-resolution level is returned.
    chunk_size : int
        Chunk size along Y and X of the lazy levels.
    store_factory : Callable, optional
        Creates the zarr store for the coarse levels, only called when needed. Defaults to `TempStore`.
    tile_size : int
        Chunk size of the coarse levels. The pyramid is extended until the smaller image side drops below it.
    background : bool
        Build the coarse levels in a background thread and return immediately. Reading a coarse level waits
        until it has been built (and raises if the build failed). Otherwise, the call returns once built.

    Returns
    -------
    list[da.Array] | None
        Pyramid or None when the layout is not supported (or when there is no internal pyramid and
        `auto_pyramid` is False), in which case the caller should read the file eagerly.
    """
    index = CziTileIndex(czi)
    if not index.supported:
        return None
    has_internal_pyramid = len(index.levels) > 1
    if not has_internal_pyramid and not auto_pyramid:
        return None
    # dask reads from several threads. Reads are serialised by the file handle, decoding is not.
    czi._fh.lock = True
    factors = index.levels if auto_pyramid else index.levels[:1]
    token = uuid.uuid4().hex
    pyramid: list[da.Array] = []
    for factor in factors:
        level = CziLevelArray(index, factor)
        # explicit name, otherwise dask tries to tokenize (pickle) the open file
        pyramid.append(da.from_array(level, chunks=level.chunks(chunk_size), name=f"czi-{token}-{factor}"))
    if not has_internal_pyramid and _n_coarse_levels(pyramid[0].shape, index.is_rgb, tile_size) > 0:
        if store_factory is None:
            from image2image_io._zarr import TempStore

            store_factory = TempStore
        store = store_factory()
        if background:
            pyramid.extend(_build_coarse_levels_in_background(pyramid[0], index, store, tile_size, token))
        else:
            pyramid.extend(_build_coarse_levels(index, store, tile_size))
    return pyramid
