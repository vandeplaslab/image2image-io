"""Writer."""
from __future__ import annotations

import typing as ty
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from koyo.timer import MeasureTimer
from loguru import logger
from tifffile import TiffWriter
from tqdm import trange

from image2image_io.readers.utilities import get_pyramid_info, prepare_ome_xml_str

if ty.TYPE_CHECKING:
    from image2image_io.readers._base_reader import BaseReader


class Transformer(ty.Protocol):
    """Transformer protocol to transform image."""

    output_size: tuple[int, int]
    output_spacing: tuple[float, float]

    def __call__(self, image: sitk.Image) -> sitk.Image:
        """Transform image."""
        ...


class OmeTiffWriter:
    """OME-TIFF writer class."""

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

    def __init__(self, reader: BaseReader, transformer: Transformer | None = None):
        """
        Class for managing writing images to OME-TIFF.

        Attibutes
        ---------
        x_size: int
            Size of the output image after transformation in x
        y_size: int
            Size of the output image after transformation in y
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
        self.transformer = transformer

    def _prepare_image_info(
        self,
        image_name: str,
        write_pyramid: bool = True,
        tile_size: int = 512,
        compression: str | None = "default",
    ) -> None:
        """Get image info and OME-XML."""
        if self.transformer:
            self.x_size, self.y_size = self.transformer.output_size
            self.y_spacing, self.x_spacing = self.transformer.output_spacing
        else:
            self.y_size, self.x_size = self.reader.image_shape
            self.y_spacing, self.x_spacing = None, None

        self.tile_size = tile_size
        # protect against too large tile size
        while self.y_size / self.tile_size <= 1 or self.x_size / self.tile_size <= 1:
            self.tile_size = self.tile_size // 2

        self.pyr_levels, _ = get_pyramid_info(self.y_size, self.x_size, self.reader.n_channels, self.tile_size)
        self.n_pyr_levels = len(self.pyr_levels)

        if self.transformer:
            self.PhysicalSizeX = self.x_spacing
            self.PhysicalSizeY = self.y_spacing
        else:
            self.PhysicalSizeX = self.reader.resolution
            self.PhysicalSizeY = self.reader.resolution

        channel_names = self.reader.channel_names

        self.omexml = prepare_ome_xml_str(
            self.y_size,
            self.x_size,
            self.reader.n_channels,
            self.reader.dtype,
            self.reader.is_rgb,
            PhysicalSizeX=self.PhysicalSizeX,
            PhysicalSizeY=self.PhysicalSizeY,
            PhysicalSizeXUnit="µm",
            PhysicalSizeYUnit="µm",
            Name=image_name,
            Channel=None if self.reader.is_rgb else {"Name": channel_names},
        )

        self.subifds = self.n_pyr_levels - 1 if write_pyramid is True else None

        if compression == "default":
            self.compression = "jpeg" if self.reader.is_rgb else "deflate"
        else:
            self.compression = compression
        logger.info(f"Saving using {self.compression} compression")

    def write_image_by_plane(
        self,
        image_name: str,
        output_dir: Path | str = "",
        write_pyramid: bool = True,
        tile_size: int = 512,
        compression: str | None = "default",
    ) -> Path:
        """
        Write OME-TIFF image plane-by-plane to disk. WsiReg compatible RegImages all
        have methods to read an image channel-by-channel, thus each channel is read, transformed, and written to
        reduce memory during write.
        RGB images may run large memory footprints as they are interleaved before write, for RGB images,
        using the `OmeTiledTiffWriter` is recommended.

        Parameters
        ----------
        image_name: str
            Name to be written WITHOUT extension
            for example if image_name = "cool_image" the file
            would be "cool_image.ome.tiff"
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
        output_file_name = str(Path(output_dir) / f"{image_name}.ome.tiff")
        logger.info(f"Saving to '{output_file_name}'")
        self._prepare_image_info(
            image_name,
            write_pyramid=write_pyramid,
            tile_size=tile_size,
            compression=compression,
        )

        with TiffWriter(output_file_name, bigtiff=True) as tif:
            if self.reader.is_rgb:
                options = {
                    "tile": (self.tile_size, self.tile_size),
                    "compression": self.compression,
                    "photometric": "rgb",
                    "metadata": None,
                }
            else:
                options = {
                    "tile": (self.tile_size, self.tile_size),
                    "compression": self.compression,
                    "photometric": "rgb" if self.reader.is_rgb else "minisblack",
                    "metadata": None,
                }
            logger.trace(f"TIFF options: {options}")

            rgb_im_data = []
            for channel_idx in trange(self.reader.n_channels, desc="Writing channels..."):
                image: np.ndarray = self.reader.get_channel(channel_idx)
                image = np.squeeze(image)
                image: sitk.Image = sitk.GetImageFromArray(image)
                image.SetSpacing((self.reader.resolution, self.reader.resolution))

                # transform
                if self.transformer:
                    image = self.transformer(image)

                if self.reader.is_rgb:
                    rgb_im_data.append(image)
                else:
                    if isinstance(image, sitk.Image):
                        image = sitk.GetArrayFromImage(image)

                    # write OME-XML to the ImageDescription tag of the first page
                    description = self.omexml if channel_idx == 0 else None
                    # write channel data
                    logger.info(f"Writing channel {channel_idx} - shape: {image.shape}")
                    with MeasureTimer() as write_timer:
                        tif.write(image, subifds=self.subifds, description=description, **options)
                    logger.info(f"Writing channel {channel_idx} took {write_timer()}")

                    if write_pyramid:
                        with MeasureTimer() as write_timer:
                            for pyr_idx in range(1, self.n_pyr_levels):
                                resize_shape = (self.pyr_levels[pyr_idx][0], self.pyr_levels[pyr_idx][1])
                                image = cv2.resize(image, resize_shape, cv2.INTER_LINEAR)
                                logger.info(
                                    f"Writing pyramid index {pyr_idx} : channel {channel_idx} shape: {image.shape}"
                                )
                                tif.write(image, **options, subfiletype=1)
                            logger.info(
                                f"Wrote pyramid index {pyr_idx} : channel {channel_idx} took"
                                f" {write_timer(since_last=True)}"
                            )

            if self.reader.is_rgb:
                rgb_im_data = sitk.Compose(rgb_im_data)
                rgb_im_data = sitk.GetArrayFromImage(rgb_im_data)

                # write OME-XML to the ImageDescription tag of the first page
                description = self.omexml

                # write channel data
                tif.write(
                    rgb_im_data,
                    subifds=self.subifds,
                    description=description,
                    **options,
                )

                logger.info(f"RGB shape: {rgb_im_data.shape}")
                if write_pyramid:
                    logger.info("Writing pyramid...")
                    for pyr_idx in range(1, self.n_pyr_levels):
                        with MeasureTimer() as write_timer:
                            resize_shape = (self.pyr_levels[pyr_idx][0], self.pyr_levels[pyr_idx][1])
                            rgb_im_data = cv2.resize(rgb_im_data, resize_shape, cv2.INTER_LINEAR)
                            logger.info(f"pyramid index {pyr_idx} : shape: {resize_shape}")
                            tif.write(rgb_im_data, **options, subfiletype=1)
                        logger.info(f"Wrote pyramid index {pyr_idx} took {write_timer()}")
        return Path(output_file_name)

    def write(self, name: str, output_dir: Path) -> Path:
        """Write image."""
        return self.write_image_by_plane(name, output_dir)
