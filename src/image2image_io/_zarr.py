"""Zarr 3 storage helpers."""

from __future__ import annotations

import atexit
import glob
import os
import shutil
from pathlib import Path
from tempfile import mkdtemp

from zarr import Array, Group
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

    def __init__(self) -> None:
        self._path = Path(mkdtemp())
        super().__init__(self._path)
        atexit.register(atexit_rmtree, self._path)

    @property
    def path(self) -> Path:
        """Return the path to the temporary directory."""
        return self._path

    def cleanup(self) -> None:
        """Remove the temporary directory immediately."""
        atexit_rmtree(self._path)


__all__ = [
    "Array",
    "Group",
    "TempStore",
    "atexit_rmglob",
    "atexit_rmtree",
]
