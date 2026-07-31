"""Test Zarr 3 helpers."""

from image2image_io._zarr import TempStore, atexit_rmglob


def test_temp_store_lifetime() -> None:
    """Test that a temporary store remains usable until cleanup."""
    import zarr

    store = TempStore()
    path = store.path
    array = zarr.create_array(store=store, shape=(2,), dtype="i4")
    array[:] = [1, 2]

    assert path.exists()
    assert array[:].tolist() == [1, 2]

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
