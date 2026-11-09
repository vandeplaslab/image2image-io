# ruff: noqa: INP001
"""Channel filtering CLI tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from tifffile import imwrite

from image2image_io.cli import cli
from image2image_io.cli.filter import filter_channels, filter_runner
from image2image_io.readers import BaseReader, get_simple_reader
from image2image_io.utils._test import get_test_file


@pytest.fixture
def microscopy(tmp_path: Path) -> tuple[Path, np.ndarray]:
    array = np.arange(4 * 48 * 64, dtype=np.uint16).reshape(4, 48, 64)
    path = tmp_path / "sample.ome.tiff"
    imwrite(
        path,
        array,
        photometric="minisblack",
        metadata={
            "axes": "CYX",
            "Channel": {"Name": ["DAPI", "CD3", "CD4", "CD8"]},
            "PhysicalSizeX": 0.5,
            "PhysicalSizeY": 0.5,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        },
    )
    return path, array


def assert_output(path: Path, expected: np.ndarray, names: list[str], uint8: bool = False) -> None:
    reader = get_simple_reader(path, init_pyramid=False, auto_pyramid=False)
    try:
        assert reader.n_channels == len(names)
        assert reader.channel_names == names
        assert reader.resolution == 0.5
        assert reader.dtype == (np.uint8 if uint8 else expected.dtype)
        for index in range(len(names)):
            actual = np.asarray(reader.get_channel(index))
            assert actual.shape == expected[index].shape
            if uint8:
                plane = expected[index].astype(float)
                scaled = (plane - plane.min()) / (plane.max() - plane.min()) * 255
                np.testing.assert_allclose(actual, scaled, atol=1)
            else:
                np.testing.assert_array_equal(actual, expected[index])
    finally:
        reader.close()


@pytest.mark.parametrize("fmt", ["ome-tiff", "ome-zarr"])
@pytest.mark.parametrize("flag", [None, "-U", "--no_as_uint8", "--no_uint8", "-u", "--as_uint8", "--uint8"])
def test_filter_roundtrip(microscopy: tuple[Path, np.ndarray], tmp_path: Path, fmt: str, flag: str | None) -> None:
    source, array = microscopy
    args = ["-i", str(source), "-o", str(tmp_path / "output"), "-C", "3,0-1", "--fmt", fmt]
    if flag:
        args.append(flag)
    result = CliRunner().invoke(filter_channels, args)
    assert result.exit_code == 0, result.output
    extension = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    assert_output(
        tmp_path / "output" / f"sample_filtered.{extension}",
        array[[3, 0, 1]],
        ["CD8", "DAPI", "CD3"],
        uint8=flag in {"-u", "--as_uint8", "--uint8"},
    )


@pytest.mark.parametrize("fmt", ["ome-tiff", "ome-zarr"])
def test_filter_rename_single_channel(microscopy: tuple[Path, np.ndarray], tmp_path: Path, fmt: str) -> None:
    source, array = microscopy
    result = CliRunner().invoke(
        filter_channels,
        [
            "--path",
            str(source),
            "-o",
            str(tmp_path / "output"),
            "--channel_ids",
            "2",
            "--channel_names",
            "Renamed CD4",
            "--fmt",
            fmt,
        ],
    )
    assert result.exit_code == 0, result.output
    extension = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    assert_output(tmp_path / "output" / f"sample_filtered.{extension}", array[[2]], ["Renamed CD4"])


@pytest.mark.parametrize("suffix,delimiter", [("csv", ","), ("tsv", "\t"), ("txt", ","), ("txt", "\t")])
@pytest.mark.parametrize("rename", [True, False])
def test_filter_channel_file(
    microscopy: tuple[Path, np.ndarray],
    tmp_path: Path,
    suffix: str,
    delimiter: str,
    rename: bool,
) -> None:
    source, array = microscopy
    table = tmp_path / f"channels.{suffix}"
    rows = [["include", "channel_names"], ["TrUe", "Nucleus"], ["false", ""], ["0", ""], ["1", "T cell"]]
    table.write_text("\n".join(delimiter.join(row if rename else row[:1]) for row in rows), encoding="utf-8")
    result = CliRunner().invoke(
        filter_channels,
        [
            "-i",
            str(source),
            "-o",
            str(tmp_path / "output"),
            "--channel_file",
            str(table),
        ],
    )
    assert result.exit_code == 0, result.output
    assert_output(
        tmp_path / "output" / "sample_filtered.ome.tiff",
        array[[0, 3]],
        ["Nucleus", "T cell"] if rename else ["DAPI", "CD8"],
    )


@pytest.mark.parametrize(
    "args,message",
    [
        ([], "exactly one"),
        (["-C", ""], "nonnegative integers"),
        (["-C", "-1"], "nonnegative integers"),
        (["-C", "1.5"], "nonnegative integers"),
        (["-C", "1,abc"], "nonnegative integers"),
        (["-C", "0,"], "nonnegative integers"),
        (["-C", "1-2-3"], "nonnegative integers"),
        (["-C", "3-1"], "ascending"),
        (["-C", "1,1"], "Duplicate"),
        (["-C", "0-2,2"], "Duplicate"),
        (["-C", "4"], "out of range"),
        (["-C", "0,1", "--channel_names", "DAPI"], "one nonempty"),
        (["-C", "0", "--channel_names", " "], "one nonempty"),
        (["--channel_names", "DAPI"], "exactly one"),
    ],
)
def test_filter_invalid_ids(microscopy: tuple[Path, np.ndarray], tmp_path: Path, args: list[str], message: str) -> None:
    result = CliRunner().invoke(filter_channels, ["-i", str(microscopy[0]), "-o", str(tmp_path / "output"), *args])
    assert result.exit_code != 0
    assert message in result.output
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "table_text,message",
    [
        ("keep\ntrue\nfalse\nfalse\ntrue\n", "'include' column"),
        ("include\ntrue\nfalse\n", "2 rows"),
        ("include\nfalse\nfalse\nfalse\nfalse\n", "at least one"),
        ("include\nyes\nfalse\nfalse\nfalse\n", "Invalid include"),
        ("include\ntrue\n\nfalse\nfalse\n", "Invalid include"),
        ("include,other\n,x\nfalse,x\nfalse,x\nfalse,x\n", "Invalid include"),
        ("include\ntrue,extra\nfalse\nfalse\nfalse\n", "malformed row"),
        ("include,channel_names\ntrue,\nfalse,\nfalse,\nfalse,\n", "Missing channel_names"),
        ("include,include\ntrue,false\nfalse,false\nfalse,false\nfalse,false\n", "duplicate column"),
    ],
)
def test_filter_invalid_table(
    microscopy: tuple[Path, np.ndarray],
    tmp_path: Path,
    table_text: str,
    message: str,
) -> None:
    table = tmp_path / "channels.csv"
    table.write_text(table_text, encoding="utf-8")
    result = CliRunner().invoke(
        filter_channels,
        [
            "-i",
            str(microscopy[0]),
            "-o",
            str(tmp_path / "output"),
            "--channel_file",
            str(table),
        ],
    )
    assert result.exit_code != 0
    assert message in result.output
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("extra", [["-C", "0"], ["--channel_names", "DAPI"]])
def test_filter_conflicting_selectors(microscopy: tuple[Path, np.ndarray], tmp_path: Path, extra: list[str]) -> None:
    table = tmp_path / "channels.csv"
    table.write_text("include\ntrue\nfalse\nfalse\ntrue\n", encoding="utf-8")
    result = CliRunner().invoke(
        filter_channels,
        [
            "-i",
            str(microscopy[0]),
            "-o",
            str(tmp_path / "output"),
            "--channel_file",
            str(table),
            *extra,
        ],
    )
    assert result.exit_code != 0
    assert "exactly one" in result.output or "requires --channel_ids" in result.output


@pytest.mark.parametrize("fmt", ["ome-tiff", "ome-zarr"])
def test_filter_batch_and_overwrite(microscopy: tuple[Path, np.ndarray], tmp_path: Path, fmt: str) -> None:
    source, array = microscopy
    second = tmp_path / "second.ome.tif"
    second.write_bytes(source.read_bytes())
    args = ["-p", str(tmp_path / "*.ome.tif*"), "-o", str(tmp_path / "output"), "-C", "1", "--fmt", fmt]
    extension = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    output = tmp_path / "output" / f"sample_filtered.{extension}"
    result = CliRunner().invoke(filter_channels, args)
    assert result.exit_code == 0, result.output
    assert_output(output, array[[1]], ["CD3"])
    assert_output(tmp_path / "output" / f"second_filtered.{extension}", array[[1]], ["CD3"])
    result = CliRunner().invoke(filter_channels, [*args, "--channel_names", "Renamed"])
    assert result.exit_code == 0, result.output
    assert_output(output, array[[1]], ["CD3"])
    result = CliRunner().invoke(filter_channels, [*args, "--channel_names", "Renamed", "-W"])
    assert result.exit_code == 0, result.output
    assert_output(output, array[[1]], ["Renamed"])


def test_filter_output_collision(microscopy: tuple[Path, np.ndarray], tmp_path: Path) -> None:
    source, _ = microscopy
    second = tmp_path / "sample.tif"
    second.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="Multiple inputs"):
        filter_runner([source, second], tmp_path / "output", channel_ids=[0])
    assert not (tmp_path / "output").exists()


def test_filter_validates_whole_batch(microscopy: tuple[Path, np.ndarray], tmp_path: Path) -> None:
    source, array = microscopy
    second = tmp_path / "single.ome.tiff"
    imwrite(second, array[0], photometric="minisblack", metadata={"axes": "YX"})
    with pytest.raises(ValueError, match="out of range"):
        filter_runner([source, second], tmp_path / "output", channel_ids=[1])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("axes,shape", [("ZYX", (4, 48, 64)), ("TYX", (4, 48, 64)), ("ZCYX", (2, 4, 48, 64))])
def test_filter_rejects_stacks(tmp_path: Path, axes: str, shape: tuple[int, ...]) -> None:
    source = tmp_path / "stack.ome.tiff"
    imwrite(source, np.ones(shape, dtype=np.uint16), photometric="minisblack", metadata={"axes": axes})
    result = CliRunner().invoke(filter_channels, ["-i", str(source), "-o", str(tmp_path / "out"), "-C", "0"])
    assert result.exit_code != 0
    assert "Z stacks and time series" in result.output


def test_filter_rejects_rgb(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tiff"
    imwrite(source, np.ones((48, 64, 3), dtype=np.uint8), photometric="rgb")
    result = CliRunner().invoke(filter_channels, ["-i", str(source), "-o", str(tmp_path / "out"), "-C", "0"])
    assert result.exit_code != 0
    assert "Packed RGB" in result.output


def test_filter_closes_readers_after_export_failure(
    microscopy: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = get_simple_reader(microscopy[0], init_pyramid=False, auto_pyramid=False)
    monkeypatch.setattr("image2image_io.readers.get_simple_reader", lambda *args, **kwargs: reader)

    def fail(*args: object, **kwargs: object) -> None:
        msg = "export failed"
        raise RuntimeError(msg)

    monkeypatch.setattr("image2image_io.writers.write_ome_tiff_alt", fail)
    with pytest.raises(ValueError, match="export failed"):
        filter_runner([microscopy[0]], tmp_path / "output", channel_ids=[0])
    assert reader.fh is None


def test_filter_cli_registration() -> None:
    result = CliRunner().invoke(cli, ["filter", "--help"])
    assert result.exit_code == 0, result.output
    assert "--channel_ids" in result.output
    assert "--no_uint8" in result.output


def test_filter_does_not_overwrite_another_input(microscopy: tuple[Path, np.ndarray], tmp_path: Path) -> None:
    source, _ = microscopy
    second = tmp_path / "sample_filtered.ome.tiff"
    original = source.read_bytes()
    second.write_bytes(original)
    with pytest.raises(ValueError, match="overwrite an input"):
        filter_runner([source, second], tmp_path, channel_ids=[0], overwrite=True)
    assert second.read_bytes() == original


def test_filter_restores_config_on_open_failure(
    microscopy: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from image2image_io.config import CONFIG

    monkeypatch.setattr(CONFIG, "init_pyramid", True)
    monkeypatch.setattr(CONFIG, "auto_pyramid", True)
    monkeypatch.setattr(CONFIG, "quiet", True)
    missing = tmp_path / "missing.ome.tiff"
    with pytest.raises(ValueError):
        filter_runner([microscopy[0], missing], tmp_path / "output", channel_ids=[0])
    assert CONFIG.init_pyramid
    assert CONFIG.auto_pyramid
    assert CONFIG.quiet
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("fmt", ["ome-tiff", "ome-zarr"])
def test_filter_zarr_input(microscopy: tuple[Path, np.ndarray], tmp_path: Path, fmt: str) -> None:
    from image2image_io.writers import write_ome_zarr_from_array

    _, array = microscopy
    source = tmp_path / "source.ome.zarr"
    write_ome_zarr_from_array(source, None, array, resolution=0.5, channel_names=["DAPI", "CD3", "CD4", "CD8"])
    filter_runner([source], tmp_path / "output", channel_ids=[3, 1], fmt=fmt)
    extension = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    assert_output(tmp_path / "output" / f"source_filtered.{extension}", array[[3, 1]], ["CD8", "CD3"])


@pytest.mark.parametrize("fmt", ["ome-tiff", "ome-zarr"])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_filter_preserves_dtype(tmp_path: Path, fmt: str, dtype: type[np.generic]) -> None:
    source = tmp_path / "source.ome.tiff"
    array = np.arange(4 * 48 * 64).reshape(4, 48, 64).astype(dtype)
    imwrite(source, array, photometric="minisblack", metadata={"axes": "CYX"})
    filter_runner([source], tmp_path / "output", channel_ids=[2], fmt=fmt)
    extension = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    reader = get_simple_reader(tmp_path / "output" / f"source_filtered.{extension}")
    try:
        assert reader.dtype == dtype
        np.testing.assert_array_equal(np.asarray(reader.get_channel(0)), array[2])
    finally:
        reader.close()


@pytest.mark.parametrize("fmt", ["ome-tiff", "ome-zarr"])
def test_filter_czi(tmp_path: Path, fmt: str) -> None:
    source = get_test_file("2d-image.czi")
    filter_runner([source], tmp_path / "output", channel_ids=[0], channel_names=["Renamed"], fmt=fmt)
    extension = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    expected = get_simple_reader(source, init_pyramid=False, auto_pyramid=False)
    actual = get_simple_reader(tmp_path / "output" / f"2d-image_filtered.{extension}")
    try:
        assert actual.channel_names == ["Renamed"]
        assert actual.resolution == expected.resolution
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(np.asarray(actual.get_channel(0)), np.asarray(expected.pyramid[0]))
    finally:
        actual.close()
        expected.close()


def test_filter_czi_scenes(
    microscopy: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "scenes.czi"
    source.touch()
    readers: list[BaseReader] = []

    def open_scene(path: Path, **kwargs: object) -> BaseReader:
        assert kwargs["scene_index"] == len(readers)
        reader = get_simple_reader(microscopy[0], init_pyramid=False, auto_pyramid=False)
        readers.append(reader)
        return reader

    monkeypatch.setattr("image2image_io.readers._czi.CziSceneFile.get_num_scenes", lambda path: 2)
    monkeypatch.setattr("image2image_io.readers.get_simple_reader", open_scene)
    filter_runner([source], tmp_path / "output", channel_ids=[1])
    assert len(readers) == 2
    assert all(reader.fh is None for reader in readers)
    for scene in range(2):
        assert_output(tmp_path / "output" / f"scenes_scene={scene:02d}_filtered.ome.tiff", microscopy[1][[1]], ["CD3"])


@pytest.mark.parametrize("fmt", ["ome-tiff", "ome-zarr"])
def test_filter_pyramid(tmp_path: Path, fmt: str) -> None:
    source = tmp_path / "large.ome.tiff"
    array = np.ones((4, 1024, 1536), dtype=np.uint16)
    imwrite(source, array, photometric="minisblack", metadata={"axes": "CYX"})
    filter_runner([source], tmp_path / "output", channel_ids=[2, 0], fmt=fmt, tile_size=256)
    extension = "ome.tiff" if fmt == "ome-tiff" else "ome.zarr"
    reader = get_simple_reader(tmp_path / "output" / f"large_filtered.{extension}", init_pyramid=False)
    try:
        assert reader.n_in_pyramid > 1
        assert reader.n_channels == 2
        np.testing.assert_array_equal(np.asarray(reader.get_channel(0)), array[2])
    finally:
        reader.close()
