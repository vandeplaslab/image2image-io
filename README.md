# image2image-io

[![License](https://img.shields.io/pypi/l/image2image-io.svg?color=green)](https://github.com/vandeplaslab/image2image-io/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/image2image-io.svg?color=green)](https://pypi.org/project/image2image-io)
[![Python Version](https://img.shields.io/pypi/pyversions/image2image-io.svg?color=green)](https://python.org)
[![CI](https://github.com/vandeplaslab/image2image-io/actions/workflows/ci.yml/badge.svg)](https://github.com/vandeplaslab/image2image-io/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/vandeplaslab/image2image-io/branch/main/graph/badge.svg)](https://codecov.io/gh/vandeplaslab/image2image-io)

## Overview

This library provides reader/writer interface to several popular image formats. The main goal is to give a unified
interface to access the image data (e.g. for specific channel or pyramid level).

## Getting started

Currently supported formats:
- Standard image formats (.png, .jpg, .jpeg)
- TIFF (.tif, .tiff, .ome.tiff, .scn, .svs, .ndpi, .qptiff, .qptiff.raw, .qptiff.intermediate)
- CZI (.czi)
- Bruker (.tsf, .tdf, .d)
- ImzML (.imzML, .ibd)
- HDF5 (.h5, .hdf5) - specific schema required
- Numpy (.npy, .npz) - expects 2D or 3D numpy array
- GeoJSON (.geojson) - expects a dictionary with 'type' and 'features' keys (e.g. from QuPath or GeoPandas)
- Points (.csv, .txt, .parquet) - specific schema required

If you want to open an image, you can use the `get_simple_reader` function. This function will automatically detect the
image format and return the appropriate reader.

```python
from image2image_io.readers import get_simple_reader

# Path to your image
path_to_image = "path/to/image.ome.tiff"

# Get instance of the reader.
reader = get_simple_reader(
    path_to_image, 
    init_pyramid=True  # initialize the pyramid upon loading the image
)

# Retrieve the pyramid stack. In this case 'pyramid' is a list of numpy or dask arrays.
pyramid = reader.pyramid

# Retrieve the first channel of the first pyramid level
channel_first = reader.get_channel(0, 0)  # channel_id, pyramid_level

# Retrieve the first channel of the last pyramid level
channel_last = reader.get_channel(0, -1)  # channel_id, pyramid_level

# Writing to file is relatively easy
reader.to_ome_tiff(
    'path/to/output.ome.tiff',
    as_uint8=True,  # will convert the data to uint8
    tile_size=1024,  # tile size for the output image
    channel_ids=[0, 2],  # channel ids - specify which channels to write
    channel_names=['Channel 1', 'Channel 3'],  # channel names - specify the names of the channels
)
```

Writing a numpy array to an OME-TIFF file is also possible:
    
```python
import numpy as np
from image2image_io.writers import write_ome_tiff_from_array

# Create a numpy array
array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)  # RGB image  

# Write the numpy array to an OME-TIFF file
write_ome_tiff_from_array(
    "path/to/output.ome.tiff",
    None,
    array,
    tile_size=1024,  # tile size for the output image
    channel_names=["R", "G", "B"],  # for RGB images, this might now have any effect
    resolution=0.5,  # resolution of the image in microns
)
```

Merging multiple images is alsy fairly easy to do:

```python
from image2image_io.writers import merge_images

# Paths to the images
paths = ["path/to/image1.ome.tiff", "path/to/image2.ome.tiff"]

# Merge the images
merge_images(
    "output.ome.tiff",  # filename
    paths,  # list of paths to the images
    output_dir="path/to/output/dir",  # output directory for the final image
    tile_size=1024,  # tile size for the output image
    metadata={  # metadata for the output image - this specifies which channels to use in the export
        "path/to/image1.ome.tiff": {
            0: {
                "name": "image-1",
                "channel_ids": [0, 2],
                "channel_names": ["Channel 1", "Channel 3"],
                }
        },
        "path/to/image2.ome.tiff": {
            0: {
                "name": "image-2",
                "channel_ids": [0, 1],
                "channel_names": ["Channel 1", "Channel 2"],
                }
        }
    }
)
``` 

## Contributing

Contributions are always welcome. Please feel free to submit PRs with new features, bug fixes, or documentation improvements.

```bash
git clone https://github.com/vandeplaslab/image2image-io.git

pip install -e .[dev]
```