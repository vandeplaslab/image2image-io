"""OME-TIFF writer for MergeRegImage class."""
from __future__ import annotations

import typing as ty
from pathlib import Path
from typing import List

import cv2
import numpy as np
import SimpleITK as sitk
from koyo.timer import MeasureTimer
from loguru import logger
from tifffile import TiffWriter
from tqdm import tqdm, trange

from image2image_io.enums import SITK_TO_NP_DTYPE
from image2image_io.readers import BaseReader
from image2image_io.readers.merge import MergeImages
from image2image_io.readers.utilities import get_pyramid_info, prepare_ome_xml_str
from image2image_io.utils.utilities import format_channel_names


class Transformer(ty.Protocol):
    """Transformer protocol to transform image."""

    output_size: tuple[int, int]
    output_spacing: tuple[float, float]

    def __call__(self, image: sitk.Image) -> sitk.Image:
        """Transform image."""
        ...


class MergeOmeTiffWriter:
    """Class for writing multiple images into a single OME-TIFF."""

    x_size: int | None = None
    y_size: int | None = None
    y_spacing: int | float | None = None
    x_spacing: int | float | None = None
    tile_size: int = 512
    pyr_levels: list[tuple[int, int]] | None = None
    n_pyr_levels: int | None = None
    PhysicalSizeY: int | float | None = None
    PhysicalSizeX: int | float | None = None
    subifds: int | None = None
    compression: str = "deflate"

    def __init__(
        self,
        reader: MergeImages,
        transformers: list[Transformer] | None = None,
    ):
        """
        Class for writing multiple images wiuth and without transforms to a singel OME-TIFF.

        Parameters
        ----------
        reader: MergeImages
            MergeRegImage to be transformed
        reg_transform_seqs: List of RegTransformSeq or None
            Registration transformation sequences for each wsireg image to be merged

        Attibutes
        ---------
        x_size: int
            Size of the merged image after transformation in x
        y_size: int
            Size of the merged image after transformation in y
        y_spacing: float
            Pixel spacing in microns after transformation in y
        x_spacing: float
            Pixel spacing in microns after transformation in x
        tile_size: int
            Size of tiles to be written
        pyr_levels: list of tuples of int:
            Size of downsampled images in pyramid
        n_pyr_levels: int
            Number of downsamples in pyramid
        PhysicalSizeY: float
            physical size of image in micron for OME-TIFF in Y
        PhysicalSizeX: float
            physical size of image in micron for OME-TIFF in X
        subifds: int
            Number of sub-resolutions for pyramidal OME-TIFF
        compression: str
            tifffile string to pass to compression argument, defaults to "deflate" for minisblack
            and "jpeg" for RGB type images

        """
        self.reader = reader
        self.transformers = transformers

    def _check_transforms_and_readers(self, reader_names):
        """Make sure incoming data is kosher in dimensions."""
        if not isinstance(reader_names, list):
            raise ValueError("Require a list of image names for each image to merge")

        transformations = self.transformers
        if transformations is None:
            transformations = [None for i in range(len(self.reader.readers))]
        if len(transformations) != len(self.reader.readers):
            raise ValueError("The number of transforms does not match number of images")

    def _create_channel_names(self, sub_image_names):
        """Create channel names for merge data."""

        def _prepare_channel_names(name_: str, channel_names_: str) -> list[str]:
            return [f"{name_} - {c}" for c in channel_names_]

        self.reader.channel_names = [
            _prepare_channel_names(name, channel_names)
            for name, channel_names in zip(sub_image_names, self.reader.channel_names)
        ]
        self.reader.channel_names = [item for sublist in self.reader.channel_names for item in sublist]

    def _check_transforms_sizes_and_resolutions(self):
        """Check that all transforms as currently loaded output to the same size/resolution."""
        out_size = []
        out_spacing = []
        registered_transforms = self.transformers
        if registered_transforms is None:
            registered_transforms = [None for _ in range(len(self.reader.readers))]
        for reader, transform in zip(self.reader.readers, registered_transforms):
            if transform:
                out_size.append(transform.output_size)
                out_spacing.append(transform.output_spacing)
            else:
                out_size.append(reader.image_shape)
                out_spacing.append(reader.resolution)

        if not all(out_spacing):
            raise ValueError("All transforms output spacings and untransformed image spacings must match")
        if not all(out_size):
            raise ValueError("All transforms output sizes and untransformed image sizes must match")

    def _prepare_image_info(
        self,
        reader: BaseReader,
        image_name: str,
        dtype: np.dtype,
        transformer: Transformer | None = None,
        write_pyramid: bool = True,
        tile_size: int = 512,
        compression: str = "default",
    ):
        """Prepare OME-XML and other data needed for saving."""
        if transformer:
            self.x_size, self.y_size = transformer.output_size
            self.x_spacing, self.y_spacing = transformer.output_spacing
        else:
            self.y_size, self.x_size = reader.image_shape
            self.y_spacing, self.x_spacing = (
                reader.resolution,
                reader.resolution,
            )

        self.tile_size = tile_size
        # protect against too large tile size
        while self.y_size / self.tile_size <= 1 or self.x_size / self.tile_size <= 1:
            self.tile_size = self.tile_size // 2

        self.pyr_levels, _ = get_pyramid_info(self.y_size, self.x_size, reader.n_channels, self.tile_size)
        self.n_pyr_levels = len(self.pyr_levels)

        if transformer:
            self.PhysicalSizeY = self.y_spacing
            self.PhysicalSizeX = self.x_spacing
        else:
            self.PhysicalSizeY = reader.resolution
            self.PhysicalSizeX = reader.resolution

        channel_names = format_channel_names(self.reader.channel_names, self.reader.n_channels)
        self.omexml = prepare_ome_xml_str(
            self.y_size,
            self.x_size,
            len(channel_names),
            dtype,
            False,
            PhysicalSizeX=self.PhysicalSizeX,
            PhysicalSizeY=self.PhysicalSizeY,
            PhysicalSizeXUnit="µm",
            PhysicalSizeYUnit="µm",
            Name=image_name,
            Channel={"Name": channel_names},
        )

        self.subifds = self.n_pyr_levels - 1 if write_pyramid is True else None

        if compression == "default":
            logger.trace("using default compression")
            self.compression = "deflate"
        else:
            self.compression = compression

    def _get_merge_dtype(self):
        """Determine data type for merger. Will default to the largest
        dtype. If one image is np.uint8 and another np.uint16, the image at np.uint8
        will be cast to np.uint16.
        """
        dtype_max_size = [np.iinfo(r.im_dtype).max for r in self.reader.readers]

        merge_dtype_np = self.reader.readers[np.argmax(dtype_max_size)].dtype
        for k, v in SITK_TO_NP_DTYPE.items():
            if k < 12:
                if v == merge_dtype_np:
                    merge_dtype_sitk = k
        return merge_dtype_sitk, merge_dtype_np

    def merge_write_image_by_plane(
        self,
        name: str,
        reader_names: list[str],
        output_dir: Path | str = "",
        write_pyramid: bool = True,
        tile_size: int = 512,
        compression: str = "default",
    ) -> Path:
        """
         Write merged OME-TIFF image plane-by-plane to disk.
         RGB images will be de-interleaved with RGB channels written as separate planes.

        Parameters
        ----------
        name: str
             Name to be written WITHOUT extension for example if image_name = "cool_image" the file
             would be "cool_image.ome.tiff"
        reader_names: list of str
            Names added before each channel of a given image to distinguish it.
        output_dir: Path or str
             Directory where the image will be saved
        write_pyramid: bool
             Whether to write the OME-TIFF with sub-resolutions or not
        tile_size: int
             What size to write OME-TIFF tiles to disk
        compression: str
             tifffile string to pass to compression argument, defaults to "deflate" for minisblack
             and "jpeg" for RGB type images

        Returns
        -------
         output_file_name: str
             File path to the written OME-TIFF

        """
        merge_dtype_sitk, merge_dtype_np = self._get_merge_dtype()

        self._check_transforms_and_readers(reader_names)
        self._create_channel_names(reader_names)
        self._check_transforms_sizes_and_resolutions()

        output_file_name = str(Path(output_dir) / f"{name}.ome.tiff")
        logger.info(f"Saving to '{output_file_name}'")
        transformer = self.transformers[0] if self.transformers else None
        self._prepare_image_info(
            self.reader.readers[0],
            name,
            merge_dtype_np,
            transformer=transformer,
            write_pyramid=write_pyramid,
            tile_size=tile_size,
            compression=compression,
        )

        logger.trace(f"saving to {output_file_name}")
        with TiffWriter(output_file_name, bigtiff=True) as tif:
            options = {
                "tile": (tile_size, tile_size),
                "compression": self.compression,
                "photometric": "minisblack",
                "metadata": None,
            }
            logger.trace(f"TIFF options: {options}")

            for reader_index, reader in enumerate(tqdm(self.reader.readers, desc="writing sub-images")):
                merge_n_channels = reader.n_channels
                for channel_idx in trange(merge_n_channels, leave=False, desc=f"writing sub-image {reader_index}"):
                    image: np.ndarray = self.reader.readers[reader_index].get_channel(channel_idx)
                    image = np.squeeze(image)
                    image: sitk.Image = sitk.GetImageFromArray(image)
                    image.SetSpacing((reader.resolution, reader.resolution))

                    if image.GetPixelIDValue() != merge_dtype_sitk:
                        image = sitk.Cast(image, merge_dtype_sitk)

                    # transform
                    if self.transformers and self.transformers[reader_index]:
                        image = self.transformers[reader_index](image)

                    if isinstance(image, sitk.Image):
                        image = sitk.GetArrayFromImage(image)

                    # write OME-XML to the ImageDescription tag of the first page
                    description = self.omexml if channel_idx == 0 and reader_index == 0 else None

                    # write channel data
                    logger.trace(
                        f"Writing sub-image index {reader_index} : {reader_names[reader_index]} - "
                        f"channel index - {channel_idx} - shape: {image.shape}"
                    )
                    with MeasureTimer() as write_timer:
                        tif.write(image, subifds=self.subifds, description=description, **options)
                    logger.trace(
                        f"Wrote sub-image index {reader_index} : {reader_names[reader_index]} took {write_timer()}"
                    )

                    if write_pyramid:
                        for pyr_idx in range(1, self.n_pyr_levels):
                            with MeasureTimer() as write_timer:
                                resize_shape = (self.pyr_levels[pyr_idx][0], self.pyr_levels[pyr_idx][1])
                                image = cv2.resize(image, resize_shape, cv2.INTER_LINEAR)
                                logger.info(f"pyramid index {pyr_idx} : channel {channel_idx} shape: {image.shape}")
                                tif.write(image, **options, subfiletype=1)
                            logger.info(f"Wrote pyramid index {pyr_idx} took {write_timer()}")
            return Path(output_file_name)
