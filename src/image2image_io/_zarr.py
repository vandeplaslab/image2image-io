"""Zarr 3 storage helpers."""

from __future__ import annotations

import atexit
import glob
import os
import shutil
from collections.abc import AsyncIterator, Iterable, Mapping
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from zarr import Array, Group
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.storage import LocalStore


def atexit_rmtree(path: os.PathLike[str] | str) -> None:  # pragma: no cover
    """Ensure directory removal at interpreter exit."""
    path = os.fspath(path)
    if os.path.isdir(path):
        shutil.rmtree(path)


def atexit_rmglob(path: os.PathLike[str] | str) -> None:  # pragma: no cover
    """Ensure removal of files and directories matching a glob at exit."""
    for matched_path in glob.glob(os.fspath(path)):
        if os.path.isfile(matched_path):
            os.remove(matched_path)
        elif os.path.isdir(matched_path):
            shutil.rmtree(matched_path)


class TempStore(LocalStore):
    """Temporary Zarr store removed explicitly or at interpreter exit."""

    def __init__(self, root: Path | str | None = None, *, read_only: bool = False) -> None:
        self._path = Path(root) if root is not None else Path(mkdtemp())
        super().__init__(self._path, read_only=read_only)
        if root is None:
            atexit.register(atexit_rmtree, self._path)

    @property
    def path(self) -> Path:
        """Return the path to the temporary directory."""
        return self._path

    def cleanup(self) -> None:
        """Remove the temporary directory immediately."""
        atexit_rmtree(self._path)


class LegacyMappingStore(Store):
    """Expose a read-only Zarr 2 mapping through the Zarr 3 store API."""

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        super().__init__(read_only=True)
        self._mapping = mapping

    def __eq__(self, value: object) -> bool:
        return isinstance(value, LegacyMappingStore) and self._mapping is value._mapping

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Return a value from the wrapped mapping."""
        await self._ensure_open()
        try:
            value = self._mapping[key]
        except KeyError:
            return None

        if isinstance(value, str):
            data = value.encode()
        elif isinstance(value, bytes):
            data = value
        else:
            data = memoryview(value).tobytes()

        if isinstance(byte_range, RangeByteRequest):
            data = data[byte_range.start : byte_range.end]
        elif isinstance(byte_range, OffsetByteRequest):
            data = data[byte_range.offset :]
        elif isinstance(byte_range, SuffixByteRequest):
            data = data[-byte_range.suffix :] if byte_range.suffix else b""
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Return values for multiple key and byte-range pairs."""
        return [await self.get(key, prototype, byte_range) for key, byte_range in key_ranges]

    async def exists(self, key: str) -> bool:
        """Return whether the wrapped mapping contains a key."""
        return key in self._mapping

    @property
    def supports_writes(self) -> bool:
        """Return whether this store supports writes."""
        return False

    async def set(self, key: str, value: Buffer) -> None:
        """Reject writes to the legacy TIFF mapping."""
        raise NotImplementedError("Legacy mapping stores are read-only")

    @property
    def supports_deletes(self) -> bool:
        """Return whether this store supports deletes."""
        return False

    async def delete(self, key: str) -> None:
        """Reject deletes from the legacy TIFF mapping."""
        raise NotImplementedError("Legacy mapping stores are read-only")

    @property
    def supports_listing(self) -> bool:
        """Return whether this store supports key listing."""
        return False

    async def list(self) -> AsyncIterator[str]:
        """Reject key listing, which legacy TIFF mappings cannot provide."""
        raise NotImplementedError
        yield  # pragma: no cover

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """Reject prefix listing, which legacy TIFF mappings cannot provide."""
        raise NotImplementedError
        yield  # pragma: no cover

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """Reject directory listing, which legacy TIFF mappings cannot provide."""
        raise NotImplementedError
        yield  # pragma: no cover

    def close(self) -> None:
        """Close the wrapped mapping when it exposes a close method."""
        super().close()
        close = getattr(self._mapping, "close", None)
        if close is not None:
            close()


def ensure_zarr_store(store: Store | Mapping[str, Any]) -> Store:
    """Return a Zarr 3 store, adapting legacy mapping stores when needed."""
    if isinstance(store, Store):
        return store
    if isinstance(store, Mapping):
        return LegacyMappingStore(store)
    raise TypeError(f"Unsupported Zarr store type: {type(store).__name__}")  # noqa: TRY003


__all__ = [
    "Array",
    "Group",
    "LegacyMappingStore",
    "TempStore",
    "atexit_rmglob",
    "atexit_rmtree",
    "ensure_zarr_store",
]
