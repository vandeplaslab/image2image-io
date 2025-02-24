"""Writer tests."""

import numpy as np
import pytest

from image2image_io.models import MergeImages
from image2image_io.readers import ArrayImageReader, TiffImageReader
from image2image_io.writers import MergeOmeTiffWriter, OmeTiffWrapper, OmeTiffWriter


def test_write_rgb_array(tmp_path):
    array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    reader = ArrayImageReader(tmp_path, array)
    assert reader.is_rgb, "Array should be rgb"
    writer = OmeTiffWriter(reader)
    path = writer.write("test", tmp_path)
    assert path.exists(), "Path should exist"
    assert path.is_file(), "Path should be a file"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_write_array_dtype(tmp_path, dtype):
    array = np.random.random((3, 1024, 1024)).astype(dtype)
    reader = ArrayImageReader(tmp_path, array)
    assert reader.n_channels == 3, "Array should have 3 channels"
    assert not reader.is_rgb, "Array should not be rgb"
    writer = OmeTiffWriter(reader)
    path = writer.write("test", tmp_path)
    assert path.exists(), "Path should exist"
    assert path.is_file(), "Path should be a file"

    tiff = TiffImageReader(path)
    assert tiff.dtype == dtype, "Tiff should be float32"
    assert tiff.n_channels == 3, "Tiff should have 3 channels"


def test_write_float_array_as_uint8(tmp_path):
    array = np.random.random((3, 1024, 1024))
    reader = ArrayImageReader(tmp_path, array)
    assert reader.n_channels == 3, "Array should have 3 channels"
    assert not reader.is_rgb, "Array should not be rgb"
    writer = OmeTiffWriter(reader)
    path = writer.write("test", tmp_path, as_uint8=True)
    assert path.exists(), "Path should exist"
    assert path.is_file(), "Path should be a file"

    tiff = TiffImageReader(path)
    assert tiff.dtype == np.uint8, "Tiff should be uint8"
    assert tiff.n_channels == 3, "Tiff should have 3 channels"


def test_write_float_array_as_uint8_channel_ids(tmp_path):
    array = np.random.random((3, 1024, 1024))
    reader = ArrayImageReader(tmp_path, array)
    assert reader.n_channels == 3, "Array should have 3 channels"
    assert not reader.is_rgb, "Array should not be rgb"
    writer = OmeTiffWriter(reader)
    path = writer.write("test", tmp_path, as_uint8=True, channel_ids=(0, 2))
    assert path.exists(), "Path should exist"
    assert path.is_file(), "Path should be a file"

    tiff = TiffImageReader(path)
    assert tiff.dtype == np.uint8, "Tiff should be uint8"
    assert tiff.n_channels == 2, "Tiff should have 2 channels"


def test_write_lazy_multichannel(tmp_path):
    array = np.random.random((3, 1024, 1024))
    channel_names = ["C1", "C2", "C3"]
    resolution = 0.5

    wrapper = OmeTiffWrapper()
    with wrapper.write(
        channel_names=channel_names,
        resolution=resolution,
        dtype=array.dtype,
        shape=array.shape,
        name="test",
        output_dir=tmp_path,
    ):
        for i in range(array.shape[0]):
            wrapper.add_channel(i, channel_names[i], array[:, :, i])
    assert wrapper.path.exists(), "Path should exist"

    tiff = TiffImageReader(wrapper.path)
    assert tiff.dtype == array.dtype, "Tiff should be float32"
    assert tiff.n_channels == 3, "Tiff should have 3 channels"
    assert tiff.resolution == resolution, "Tiff should have resolution"
    assert tiff.channel_names == channel_names, "Tiff should have channel names"


def test_write_lazy_rgb(tmp_path):
    array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    channel_names = ["R", "G", "B"]
    resolution = 0.5

    wrapper = OmeTiffWrapper()
    with wrapper.write(
        channel_names=channel_names,
        resolution=resolution,
        dtype=array.dtype,
        shape=array.shape,
        name="test",
        output_dir=tmp_path,
    ):
        assert wrapper.reader.is_rgb, "Array should be rgb"
        assert wrapper.is_rgb, "Wrapper should be rgb"
        wrapper.add_channel([0, 1, 2], channel_names, array)
    assert wrapper.path.exists(), "Path should exist"

    tiff = TiffImageReader(wrapper.path)
    assert tiff.dtype == array.dtype, "Tiff should be float32"
    assert tiff.n_channels == 3, "Tiff should have 3 channels"
    assert tiff.resolution == resolution, "Tiff should have resolution"
    assert tiff.channel_names == channel_names, "Tiff should have channel names"
    assert tiff.is_rgb, "Tiff should be rgb"


def test_write_lazy_multichannel_as_uint8(tmp_path):
    array = np.random.random((3, 1024, 1024))
    channel_names = ["C0", "C1", "C2"]
    resolution = 0.5

    wrapper = OmeTiffWrapper()
    with wrapper.write(
        channel_names=channel_names,
        resolution=resolution,
        dtype=array.dtype,
        shape=array.shape,
        name="test",
        output_dir=tmp_path,
        as_uint8=True,
    ):
        for i in range(array.shape[0]):
            wrapper.add_channel(i, channel_names[i], array[i])
    assert wrapper.path.exists(), "Path should exist"

    tiff = TiffImageReader(wrapper.path)
    assert tiff.dtype == np.uint8, "Tiff should be uint8"
    assert tiff.n_channels == 3, "Tiff should have 3 channels"
    assert tiff.resolution == resolution, "Tiff should have resolution"
    assert tiff.channel_names == channel_names, "Tiff should have channel names"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_write_merge_dtype(tmp_path, dtype):
    array = np.random.random((3, 1024, 1024)).astype(dtype)
    reader = ArrayImageReader(tmp_path, array)

    merge = MergeImages([reader, reader], [1, 1])
    writer = MergeOmeTiffWriter(merge)
    path = writer.write("test", ["reader-1", "reader-2"], tmp_path)
    assert path.exists(), "Path should exist"

    tiff = TiffImageReader(path)
    assert tiff.dtype == dtype, "Tiff should be float32"
    assert tiff.n_channels == 6, "Tiff should have 6 channels"


def test_write_merge_as_uint8(tmp_path):
    array = np.random.random((3, 1024, 1024)).astype(np.float32)
    reader = ArrayImageReader(tmp_path, array)

    merge = MergeImages([reader, reader], [1, 1])
    writer = MergeOmeTiffWriter(merge)
    path = writer.write("test", ["reader-1", "reader-2"], tmp_path, as_uint8=True)
    assert path.exists(), "Path should exist"

    tiff = TiffImageReader(path)
    assert tiff.dtype == np.uint8, "Tiff should be uint8"
    assert tiff.n_channels == 6, "Tiff should have 6 channels"


def test_write_merge_as_uint8_channel_ids(tmp_path):
    array = np.random.random((3, 1024, 1024)).astype(np.float32)
    reader1 = ArrayImageReader(tmp_path, array)
    reader1._channel_names = ["C0", "C1", "C2"]

    reader2 = ArrayImageReader(tmp_path, array)
    reader2._channel_names = ["C3", "C4", "C5"]

    merge = MergeImages([reader1, reader2], [1, 1], channel_names=[reader1.channel_names, reader2.channel_names])
    writer = MergeOmeTiffWriter(merge)
    path = writer.write("test", ["reader-1", "reader-2"], tmp_path, as_uint8=True, channel_ids=[[0, 1], [0, 2]])
    assert path.exists(), "Path should exist"

    tiff = TiffImageReader(path)
    assert tiff.dtype == np.uint8, "Tiff should be uint8"
    assert tiff.n_channels == 4, "Tiff should have 4 channels"
    assert "C0" in tiff.channel_names[0], "Channel name should be C0"
    assert "C1" in tiff.channel_names[1], "Channel name should be C1"
    assert "C3" in tiff.channel_names[2], "Channel name should be C3"
    assert "C5" in tiff.channel_names[3], "Channel name should be C5"


def test_write_merge_as_uint8_channel_ids_not_all(tmp_path):
    array = np.random.random((3, 1024, 1024)).astype(np.float32)
    reader1 = ArrayImageReader(tmp_path, array)
    reader1._channel_names = ["C0", "C1", "C2"]

    reader2 = ArrayImageReader(tmp_path, array)
    reader2._channel_names = ["C3", "C4", "C5"]

    merge = MergeImages([reader1, reader2], [1, 1], channel_names=[reader1.channel_names, reader2.channel_names])
    writer = MergeOmeTiffWriter(merge)
    path = writer.write("test", ["reader-1", "reader-2"], tmp_path, as_uint8=True, channel_ids=[[0, 1], None])
    assert path.exists(), "Path should exist"

    tiff = TiffImageReader(path)
    assert tiff.dtype == np.uint8, "Tiff should be uint8"
    assert tiff.n_channels == 5, "Tiff should have 5 channels"
    assert "C0" in tiff.channel_names[0], "Channel name should be C0"
    assert "C1" in tiff.channel_names[1], "Channel name should be C1"
    assert "C3" in tiff.channel_names[2], "Channel name should be C3"
    assert "C4" in tiff.channel_names[3], "Channel name should be C4"
    assert "C5" in tiff.channel_names[4], "Channel name should be C5"


def test_write_merge_as_uint8_channel_ids_list_of_none(tmp_path):
    array = np.random.random((3, 1024, 1024)).astype(np.float32)
    reader1 = ArrayImageReader(tmp_path, array)
    reader1._channel_names = ["C0", "C1", "C2"]

    reader2 = ArrayImageReader(tmp_path, array)
    reader2._channel_names = ["C3", "C4", "C5"]

    merge = MergeImages([reader1, reader2], [1, 1], channel_names=[reader1.channel_names, reader2.channel_names])
    writer = MergeOmeTiffWriter(merge)
    path = writer.write("test", ["reader-1", "reader-2"], tmp_path, as_uint8=True, channel_ids=[None, None])
    assert path.exists(), "Path should exist"

    tiff = TiffImageReader(path)
    assert tiff.dtype == np.uint8, "Tiff should be uint8"
    assert tiff.n_channels == 6, "Tiff should have 6 channels"
