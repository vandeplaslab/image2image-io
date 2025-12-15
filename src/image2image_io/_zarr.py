"""Zarr storage.

There are some differences between zarr version 2 and version 3, so this module handles the imports.
"""

from zarr import Array

try:
    # version 2
    from zarr.hierarchy import Group
    from zarr.storage import TempStore, atexit_rmglob, atexit_rmtree
except (AttributeError, ModuleNotFoundError):
    # version 3
    import glob
    import os
    import shutil
    from pathlib import Path
    from tempfile import TemporaryDirectory

    try:
        from zarr.core.group import Group
    except (AttributeError, ModuleNotFoundError):
        from zarr.group import Group

    from zarr.storage import LocalStore

    def atexit_rmtree(path, isdir=os.path.isdir, rmtree=shutil.rmtree):  # pragma: no cover
        """Ensure directory removal at interpreter exit."""
        if isdir(path):
            rmtree(path)

    def atexit_rmglob(
        path,
        glob=glob.glob,
        isdir=os.path.isdir,
        isfile=os.path.isfile,
        remove=os.remove,
        rmtree=shutil.rmtree,
    ):  # pragma: no cover
        """Ensure removal of multiple files at interpreter exit."""
        for p in glob(path):
            if isfile(p):
                remove(p)
            elif isdir(p):
                rmtree(p)

    class TempStore(LocalStore):
        """Temporary Zarr store that is removed at exit."""

        def __init__(self) -> None:
            super().__init__(TemporaryDirectory().name)

        @property
        def path(self) -> Path:
            """Return the path to the temporary directory."""
            return self.root


__all__ = [
    "Group",
    "Array",
    "TempStore",
    "atexit_rmglob",
    "atexit_rmtree",
]
