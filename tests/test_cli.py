"""CLI Transform."""

import os

import pytest
from koyo.utilities import is_installed

from image2image_io.utils._test import get_test_file, get_test_files

has_reg = is_installed("image2image_reg")


def test_cli_entrypoint():
    """Test CLI entrypoint."""
    exit_status = os.system("i2io --help")
    assert exit_status == 0, "Exit status was not 0"


@pytest.mark.xfail(reason="needs to be resolved.")
@pytest.mark.parametrize("input_key", ["-i", "--input"])
@pytest.mark.parametrize("with_title", ["-t", "-T"])
def test_thumbnail(tmp_path, input_key, with_title):
    tmp = tmp_path
    tiffs = get_test_files("*.ome.tiff")
    files = " ".join([f"{input_key} {tiff!s}" for tiff in tiffs])
    exit_status = os.system(f"i2io thumbnail {files} -o {tmp!s} {with_title}")
    assert exit_status == 0, "Exit status was not 0"
    assert len(list(tmp.glob("*.jpg"))) > 0, "No thumbnail images"


@pytest.mark.xfail(reason="needs to be resolved.")
@pytest.mark.skipif(not has_reg, reason="image2image-reg is not installed.")
@pytest.mark.parametrize("transform", get_test_files("transform/*.i2r.json"))
def test_transform(tmp_path, transform):
    tmp = tmp_path
    tiff = get_test_file("transform/test-1.ome.tiff")
    exit_status = os.system(f"i2io transform image -i {tiff!s} -o {tmp!s} -T {transform}")
    assert exit_status == 0, "Exit status was not 0"
