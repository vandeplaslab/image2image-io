# ruff: noqa: INP001
"""OME-Zarr reader and writer tests."""

import numpy as np
import zarr
from click.testing import CliRunner

from image2image_io.cli.convert import convert
from image2image_io.readers import get_simple_reader
from image2image_io.writers import OmeZarrWrapper, write_ome_tiff_from_array, write_ome_zarr_from_array


def test_write_and_read_ome_zarr_multichannel(tmp_path):
    """Test OME-Zarr writing and reader detection for multichannel images."""
    array = np.random.default_rng().integers(0, 255, (2, 64, 64), dtype=np.uint8)
    path = write_ome_zarr_from_array(
        tmp_path / "multi.ome.zarr",
        None,
        array,
        resolution=0.5,
        channel_names=["DAPI", "FITC"],
        tile_size=16,
    )

    reader = get_simple_reader(path)

    assert reader.n_channels == 2
    assert reader.channel_names == ["DAPI", "FITC"]
    assert reader.resolution == 0.5
    assert reader.n_in_pyramid == 3
    assert reader.get_channel(1).shape == (64, 64)


def test_write_ome_zarr_wrapper_multichannel(tmp_path):
    """Test incremental OME-Zarr writing for multichannel images."""
    array = np.random.default_rng().integers(0, 255, (2, 64, 64), dtype=np.uint8)
    channel_names = ["DAPI", "FITC"]
    wrapper = OmeZarrWrapper()

    with wrapper.write(
        channel_names=channel_names,
        resolution=0.5,
        shape=array.shape,
        dtype=array.dtype,
        name="wrapped",
        output_dir=tmp_path,
        tile_size=16,
    ):
        for channel_index, channel_name in enumerate(channel_names):
            wrapper.add_channel(channel_index, channel_name, array[channel_index])

    reader = get_simple_reader(wrapper.path)

    assert reader.n_channels == 2
    assert reader.channel_names == channel_names
    assert reader.resolution == 0.5
    assert reader.dtype == array.dtype
    assert reader.n_in_pyramid == 3
    np.testing.assert_array_equal(reader.get_channel(1).compute(), array[1])


def test_write_ome_zarr_wrapper_multichannel_as_uint8(tmp_path):
    """Test incremental OME-Zarr writing with uint8 conversion."""
    array = np.random.default_rng().random((2, 64, 64), dtype=np.float32)
    channel_names = ["C0", "C1"]
    wrapper = OmeZarrWrapper()

    with wrapper.write(
        channel_names=channel_names,
        resolution=0.25,
        shape=array.shape,
        dtype=array.dtype,
        name="wrapped-uint8",
        output_dir=tmp_path,
        tile_size=16,
        as_uint8=True,
    ):
        for channel_index, channel_name in enumerate(channel_names):
            wrapper.add_channel(channel_index, channel_name, array[channel_index])

    reader = get_simple_reader(wrapper.path)

    assert reader.n_channels == 2
    assert reader.channel_names == channel_names
    assert reader.resolution == 0.25
    assert reader.dtype == np.uint8
    assert reader.get_channel(0).shape == (64, 64)


def test_write_ome_zarr_wrapper_rgb(tmp_path):
    """Test incremental OME-Zarr writing for RGB images."""
    array = np.random.default_rng().integers(0, 255, (64, 64, 3), dtype=np.uint8)
    channel_names = ["R", "G", "B"]
    wrapper = OmeZarrWrapper()

    with wrapper.write(
        channel_names=channel_names,
        resolution=0.75,
        shape=array.shape,
        dtype=array.dtype,
        name="wrapped-rgb",
        output_dir=tmp_path,
        tile_size=16,
    ):
        wrapper.add_channel([0, 1, 2], channel_names, array)

    reader = get_simple_reader(wrapper.path)

    assert reader.is_rgb
    assert reader.n_channels == 3
    assert reader.channel_names == channel_names
    assert reader.resolution == 0.75
    assert reader.dtype == array.dtype
    assert reader.n_in_pyramid == 3
    np.testing.assert_array_equal(reader.get_channel(2, split_rgb=True).compute(), array[:, :, 2])


def test_read_ome_zarr_interleaved_rgb(tmp_path):
    """Test OME-Zarr reading for interleaved yxc RGB stores."""
    path = tmp_path / "rgb.ome.zarr"
    array = np.random.default_rng().integers(0, 255, (32, 32, 3), dtype=np.uint8)
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("0", data=array, chunks=(16, 16, 3))
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
                {"name": "c", "type": "channel"},
            ],
            "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [0.25, 0.25, 1.0]}]}],
            "omero": {"channels": [{"label": "R"}, {"label": "G"}, {"label": "B"}]},
        }
    ]

    reader = get_simple_reader(path)

    assert reader.is_rgb
    assert reader.n_channels == 3
    assert reader.channel_names == ["R", "G", "B"]
    assert reader.resolution == 0.25
    assert reader.get_channel(1, split_rgb=True).shape == (32, 32)


def test_convert_cli_writes_ome_zarr(tmp_path):
    """Test CLI conversion to OME-Zarr."""
    array = np.random.default_rng().integers(0, 255, (2, 32, 32), dtype=np.uint8)
    input_path = write_ome_tiff_from_array(
        tmp_path / "input.ome.tiff",
        None,
        array,
        channel_names=["C0", "C1"],
        tile_size=0,
    )

    result = CliRunner().invoke(
        convert,
        ["-i", str(input_path), "-o", str(tmp_path), "--fmt", "ome-zarr", "--tile_size", "256"],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "input.ome.zarr").exists()
