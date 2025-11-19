"""CLI Transform."""

import os

import pytest
from koyo.utilities import is_installed

from image2image_io.utils._test import get_test_file, get_test_files


def test_cli_entrypoint() -> None:
    """Test CLI entrypoint."""
    exit_status = os.system("i2io --help")
    assert exit_status == 0, "Exit status was not 0"


@pytest.mark.parametrize("input_key", ["-i", "--input"])
@pytest.mark.parametrize("with_title", ["-t", "-T"])
def test_thumbnail(tmp_path, input_key, with_title) -> None:
    tmp = tmp_path
    tiffs = get_test_files("*.ome.tiff")
    files = " ".join([f"{input_key} {tiff!s}" for tiff in tiffs])
    exit_status = os.system(f"i2io thumbnail {files} -o {tmp!s} {with_title}")
    assert exit_status == 0, "Exit status was not 0"


@pytest.mark.parametrize("as_uint8", ["-u", "-U"])
def test_merge(tmp_path, as_uint8) -> None:
    tmp = tmp_path
    tiffs = get_test_files("mask_pr*.ome.tiff")
    files = " ".join([f"-p {tiff!s}" for tiff in tiffs])
    exit_status = os.system(f"i2io merge -n test.ome.tiff {files} -o {tmp!s} {as_uint8}")
    assert exit_status == 0, "Exit status was not 0"


@pytest.mark.parametrize("file", get_test_files("*.czi"))
def test_cziinfo(file) -> None:
    exit_status = os.system(f"i2io cziinfo -i {file!s}")
    assert exit_status == 0, "Exit status was not 0"


@pytest.mark.xfail(reason="Needs to be fixed on zarr-v3")
@pytest.mark.parametrize("as_uint8", ["-u", "-U"])
@pytest.mark.parametrize("file", get_test_files("multichannel-image.czi"))
def test_czi2tiff(tmp_path, file, as_uint8) -> None:
    exit_status = os.system(f"i2io czi2tiff -i {file!s} -o {tmp_path!s} {as_uint8} --scene 0")
    assert exit_status == 0, "Exit status was not 0"


@pytest.mark.parametrize("transform", get_test_files("transform/*.i2r.json"))
def test_transform(tmp_path, transform) -> None:
    tmp = tmp_path
    tiff = get_test_file("transform/test-1.ome.tiff")
    exit_status = os.system(f"i2io transform image -i {tiff!s} -o {tmp!s} -T {transform}")
    assert exit_status == 0, "Exit status was not 0"
