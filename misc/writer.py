"""Writer tests."""

import tempfile

import numpy as np

from image2image_io.readers import ArrayImageReader
from image2image_io.writers import OmeTiffWriter

tmp = tempfile.TemporaryDirectory()
tmp_path = tmp.name

array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
reader = ArrayImageReader(tmp_path, array)
assert reader.is_rgb, "Array should be rgb"
writer = OmeTiffWriter(reader)
path = writer.write("test", tmp_path)
assert path.exists(), "Path should exist"
assert path.is_file(), "Path should be a file"

tmp.cleanup()
