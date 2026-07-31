"""Test Zarr 3 helpers."""

import json

import numpy as np
import zarr

from image2image_io._zarr import LegacyMappingStore, TempStore, atexit_rmglob, ensure_zarr_store


class ClosableMapping(dict):
    """In-memory stand-in for tifffile's legacy Zarr mapping."""

    closed = False

    def close(self) -> None:
        """Record that the mapping was closed."""
        self.closed = True


def test_legacy_mapping_store() -> None:
    """Test reading a Zarr 2 mapping through the Zarr 3 store API."""
    values = np.arange(12, dtype="<i4").reshape(3, 4)
    mapping = ClosableMapping(
        {
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
    )

    store = ensure_zarr_store(mapping)
    array = zarr.open(store, mode="r")

    assert isinstance(store, LegacyMappingStore)
    np.testing.assert_array_equal(array[:], values)

    store.close()
    assert mapping.closed


def test_temp_store_lifetime() -> None:
    """Test that a temporary store remains usable until cleanup."""
    import zarr

    store = TempStore()
    path = store.path
    root = zarr.open_group(store=store, mode="w", zarr_format=3)
    array = root.create_array(name="values", shape=(2,), dtype="i4")
    array[:] = [1, 2]
    read_only_root = zarr.open_group(store=store, mode="r")

    assert path.exists()
    assert read_only_root["values"][:].tolist() == [1, 2]

    store.cleanup()
    assert not path.exists()


def test_atexit_rmglob() -> None:
    """Test atexit_rmglob."""
    # Create temporary files and directories
    import os
    import tempfile

    temp_dir = tempfile.TemporaryDirectory()
    dir_path = temp_dir.name

    file1 = os.path.join(dir_path, "tempfile1.txt")
    file2 = os.path.join(dir_path, "tempfile2.txt")
    sub_dir = os.path.join(dir_path, "subdir")
    os.mkdir(sub_dir)
    file3 = os.path.join(sub_dir, "tempfile3.txt")

    with open(file1, "w") as f:
        f.write("Temporary file 1")
    with open(file2, "w") as f:
        f.write("Temporary file 2")
    with open(file3, "w") as f:
        f.write("Temporary file 3")

    # Use atexit_rmglob to remove files and directories
    atexit_rmglob(os.path.join(dir_path, "*"))

    # Check that files and directories have been removed
    assert not os.path.exists(file1), "File1 was not removed"
    assert not os.path.exists(file2), "File2 was not removed"
    assert not os.path.exists(sub_dir), "Subdirectory was not removed"

    # Cleanup temporary directory
    temp_dir.cleanup()
