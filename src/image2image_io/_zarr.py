"""Zarr storage.

There are some differences between zarr version 2 and version 3, so this module handles the imports.
"""

from zarr import Array
from zarr.storage import atexit_rmtree

try:
    # version 2
    from zarr.hierarchy import Group
    from zarr.storage import TempStore, atexit_rmtree
except (AttributeError, ModuleNotFoundError):
    # version 3
    from pathlib import Path
    from tempfile import TemporaryDirectory

    try:
        from zarr.core.group import Group
    except (AttributeError, ModuleNotFoundError):
        from zarr.group import Group

    from zarr.storage import LocalStore

    class TempStore(LocalStore):
        """Temporary Zarr store that is removed at exit."""

        def __init__(self) -> None:
            path = TemporaryDirectory()
            super().__init__(path)

        @property
        def path(self) -> Path:
            """Return the path to temporary directory."""
            return self.root
