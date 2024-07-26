"""Writer."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from koyo.decorators import retry
from koyo.timer import MeasureTimer
from loguru import logger
from tifffile import TiffWriter
from tqdm import tqdm

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
    """OME-TIFF writer class.

    Attributes
    ----------
    x_size: int
        Size of the output image after transformation in x
    y_size: int
        Size of the output image after transformation in y
    tile_size: int
        Size of tiles to be written
    pyr_levels: list of tuples of int:
        Size of down-sampled images in pyramid
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

    x_size: int | None = None
    y_size: int | None = None
    tile_size: int = 512
    pyr_levels: list[tuple[int, int]]
    n_pyr_levels: int
    PhysicalSizeY: int | float
    PhysicalSizeX: int | float
    subifds: int | None = None
    compression: str | None = "default"

    def __init__(
        self,
        reader: BaseReader,
        transformer: Transformer | None = None,
        crop_mask: np.ndarray | None = None,
        crop_bbox: tuple[int, int, int, int] | None = None,
    ):
        """Class for managing writing images to OME-TIFF.

        Parameters
        ----------
        reader: BaseReader
            MergeRegImage to be transformed
        transformer: Transformer or None
            Registration transformation sequences for each wsireg image to be merged
        crop_mask : np.ndarray or None
            Crop mask to apply to images before writing.
        crop_bbox : tuple of ints or None
            Bounding box to crop images to before writing. Values should be x, y, width, height.
        """
        self.reader = reader
        self.transformer = transformer
        self.crop_mask = crop_mask
        self.crop_bbox = self._check_bbox(crop_bbox)
        if self.crop_mask is not None and self.crop_bbox is not None:
            raise ValueError("Cannot supply both crop_mask and crop_bbox")

    def _check_bbox(self, crop_bbox: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
        """Check bbox."""
        if crop_bbox is None:
            return None
        x, y, width, height = crop_bbox
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if self.transformer is not None:
            image_shape = self.transformer.output_size
        else:
            image_shape = self.reader.image_shape
        if x + width > image_shape[1]:
            width = image_shape[1] - x
        if y + height > image_shape[0]:
            height = image_shape[0] - y
        return x, y, width, height

    def _prepare_image_info(
        self,
        image_name: str,
        write_pyramid: bool = True,
        tile_size: int = 512,
        compression: str | None = "default",
        as_uint8: bool = False,
        channel_ids: list[int | tuple[int, ...]] | None = None,
        channel_names: list[str] | None = None,
    ) -> None:
        """Get image info and OME-XML."""
        if self.transformer:
            self.x_size, self.y_size = self.transformer.output_size
        elif self.crop_mask is not None:
            self.y_size, self.x_size = self.crop_mask.shape
        else:
            self.y_size, self.x_size = self.reader.image_shape

        if self.crop_bbox is not None:
            _, _, self.x_size, self.y_size = self.crop_bbox

        self.tile_size = tile_size
        if tile_size:
            # protect against too large tile size
            while self.y_size / self.tile_size <= 1 or self.x_size / self.tile_size <= 1:
                self.tile_size = self.tile_size // 2
            if self.tile_size < 256:
                self.tile_size = 256
            if self.x_size < self.tile_size or self.y_size < self.tile_size:
                self.tile_size = 0
        write_pyramid = self.tile_size > 0

        self.pyr_levels, _ = get_pyramid_info(self.y_size, self.x_size, self.reader.n_channels, self.tile_size)
        self.n_pyr_levels = len(self.pyr_levels)

        if self.transformer:
            self.PhysicalSizeX, self.PhysicalSizeY = self.transformer.output_spacing
        else:
            self.PhysicalSizeX = self.PhysicalSizeY = self.reader.resolution

        dtype: np.dtype[ty.Any] = np.uint8 if as_uint8 else self.reader.dtype  # type: ignore[assignment]
        if channel_names is None:
            channel_names = self.reader.channel_names
        if channel_ids is not None:
            if len(channel_ids) != len(channel_names):
                channel_names = [channel_names[i] for i in channel_ids]
        if len(channel_names) != len(channel_ids):  # type: ignore[arg-type]
            raise ValueError("The number of channel ids and channel names does not match.")

        self.omexml = prepare_ome_xml_str(
            self.y_size,
            self.x_size,
            len(channel_names),
            dtype,
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
        name: str,
        output_dir: Path | str = "",
        write_pyramid: bool = True,
        tile_size: int = 512,
        compression: str | None = "default",
        as_uint8: bool = False,
        channel_ids: list[int | tuple[int, ...]] | None = None,
        channel_names: list[str] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write OME-TIFF image plane-by-plane to disk.

        WsiReg compatible RegImages all have methods to read an image channel-by-channel, thus each channel is read,
        transformed, and written to reduce memory during file writing.

        RGB images may run large memory footprints as they are interleaved before write, for RGB images,
        using the `OmeTiledTiffWriter` is recommended.

        Parameters
        ----------
        name: str
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
        as_uint8: bool
            Whether to save the image in the 0-255 intensity range, substantially reducing file size at the cost of some
            intensity information.
        channel_ids: list of int
            Channel indices to write to OME-TIFF, if None, all channels are written
        channel_names: list of str
            Channel names.
        overwrite: bool
            Whether to overwrite the file if it already exists

        Returns
        -------
        output_file_name: str
            File path to the written OME-TIFF
        """
        # make sure user did not provide filename with OME-TIFF
        name = name.replace(".ome", "").replace(".tiff", "").replace(".tif", "")
        output_file_name = Path(output_dir) / f"{name}.ome.tiff"
        tmp_output_file_name = output_file_name.parent / f"{name}.ome.tiff.tmp"
        logger.info(f"Saving to '{output_file_name}'")
        logger.trace(f"Using transformer: {self.transformer}")

        if output_file_name.exists():
            if not overwrite:
                logger.warning(f"File {output_file_name} already exists, skipping...")
                return output_file_name
            try:
                output_file_name.unlink()
            except (PermissionError, FileNotFoundError, Exception) as e:
                raise PermissionError(f"Could not remove file {output_file_name} - {e}")

        if channel_ids and channel_names:
            if len(channel_ids) != len(channel_names):
                raise ValueError("The number of channel ids and channel names must match when being specified.")

        if channel_ids is None:
            channel_ids = list(range(self.reader.n_channels))
        if channel_names is None:
            channel_names = self.reader.channel_names

        if np.max(self.reader.image_shape) < 256:
            write_pyramid = False

        self._prepare_image_info(
            name,
            write_pyramid=write_pyramid,
            tile_size=tile_size,
            compression=compression,
            as_uint8=as_uint8,
            channel_ids=channel_ids,
            channel_names=channel_names,
        )

        if as_uint8:
            logger.trace(f"Writing image data in 0-255 range as uint8. Data type: {self.reader.dtype}")

        # some info about channels
        logger.trace(f"Writing channels ids: {channel_ids}")
        logger.trace(f"Writing channels: {channel_names}")

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
        if self.tile_size == 0 or np.max(self.reader.image_shape) < self.tile_size:
            options.pop("tile")

        logger.trace(f"TIFF options: {options}")
        # write OME-XML to the ImageDescription tag of the first page
        description = self.omexml

        reader = self.reader
        with TiffWriter(tmp_output_file_name, bigtiff=True) as tif, MeasureTimer() as main_timer:
            rgb_im_data: list[np.ndarray] = []
            for index, channel_index in enumerate(
                tqdm(
                    channel_ids,
                    desc="Writing channels..." if not reader.is_rgb else "Collecting channels...",
                )
            ):
                channel_name = channel_names[index]
                if channel_index not in channel_ids:
                    logger.trace(f"Skipping channel {channel_index}")
                    continue

                # if channel_index is int, then it's simple, otherwise, let's get maximum intensity projection
                if isinstance(channel_index, int):
                    image: sitk.Image = np.squeeze(
                        reader.get_channel(channel_index, split_rgb=True),  # type: ignore[assignment]
                    )
                else:
                    images: list[sitk.Image] = [
                        np.squeeze(reader.get_channel(i, split_rgb=True)) for i in channel_index
                    ]
                    images = np.dstack(images)  # type: ignore[assignment]
                    image = np.max(images, axis=2)  # type: ignore[assignment]

                # check whether we actually need to do any pre-processing
                if self.transformer or as_uint8:
                    # load data
                    image = sitk.GetImageFromArray(image)  # type: ignore[arg-type]
                    image.SetSpacing((reader.resolution, reader.resolution))  # type: ignore[no-untyped-call]

                    # transform
                    if self.transformer and callable(self.transformer):
                        with MeasureTimer() as timer:
                            image = self.transformer(image)
                        logger.trace(
                            f"Transformed image shape: {image.GetSize()[::-1]} in {timer()}",  # type: ignore[no-untyped-call]
                        )

                    # change dtype
                    if as_uint8:
                        image = sitk.RescaleIntensity(image, 0, 255)  # type: ignore[no-untyped-call]
                        image = sitk.Cast(image, sitk.sitkUInt8)  # type: ignore[no-untyped-call]

                    # convert to array if necessary
                    if isinstance(image, sitk.Image):
                        image: np.ndarray = sitk.GetArrayFromImage(image)  # type: ignore[no-redef]

                # ensure that the image is a numpy array and not e.g. dask/zarr array
                image = np.asarray(image)  # type: ignore[assignment]

                # if the image is RGB, let's accumulate the image data and write it at the end
                if reader.is_rgb:
                    rgb_im_data.append(image)  # type: ignore[arg-type]
                    continue

                # apply crop mask
                if self.crop_mask is not None:
                    image = self.crop_mask * image
                elif self.crop_bbox is not None:
                    x, y, width, height = self.crop_bbox
                    image = image[y : y + height, x : x + width]

                msg = f"Writing image for channel={channel_name} ({channel_index})"
                past_msg = msg.replace("Writing", "Wrote")
                # write channel data
                logger.trace(f"{msg} - {image.shape}...")  # type: ignore[attr-defined]
                with MeasureTimer() as write_timer:
                    tif.write(image, subifds=self.subifds, description=description, **options)
                    logger.trace(f"{past_msg} pyramid index 0 in {write_timer()}")
                    if write_pyramid:
                        for pyramid_index in range(1, self.n_pyr_levels):
                            resize_shape = (self.pyr_levels[pyramid_index][0], self.pyr_levels[pyramid_index][1])
                            image = cv2.resize(image, resize_shape, cv2.INTER_LINEAR)
                            logger.trace(
                                f"{msg} pyramid index {pyramid_index} - {image.shape}...",  # type: ignore[attr-defined]
                            )
                            tif.write(image, **options, subfiletype=1)
                            logger.trace(f"{past_msg} pyramid index {pyramid_index} in {write_timer(since_last=True)}")

            if reader.is_rgb and rgb_im_data:
                image: np.ndarray = np.dstack(rgb_im_data)  # type: ignore[no-redef]
                del rgb_im_data

                if self.crop_mask is not None:
                    image = np.atleast_3d(self.crop_mask) * image
                elif self.crop_bbox is not None:
                    x, y, width, height = self.crop_bbox
                    image = image[y : y + height, x : x + width, :]

                # write channel data
                msg = "Writing RGB image"
                past_msg = msg.replace("Writing", "Wrote")
                # write channel data
                logger.trace(f"{msg} - {image.shape}...")  # type: ignore[attr-defined]
                with MeasureTimer() as write_timer:
                    tif.write(image, subifds=self.subifds, description=description, **options)
                    logger.trace(f"{past_msg} pyramid index 0 in {write_timer()}")

                    if write_pyramid:
                        logger.info("Writing pyramid...")
                        for pyramid_index in range(1, self.n_pyr_levels):
                            resize_shape = (self.pyr_levels[pyramid_index][0], self.pyr_levels[pyramid_index][1])
                            image = cv2.resize(image, resize_shape, cv2.INTER_LINEAR)
                            logger.trace(
                                f"{msg} pyramid index {pyramid_index} - {image.shape}...",  # type: ignore[attr-defined]
                            )
                            tif.write(image, **options, subfiletype=1)
                            logger.trace(f"{past_msg} pyramid index {pyramid_index} in {write_timer(since_last=True)}")
        logger.trace(f"Exported OME-TIFF in {main_timer()}")
        # rename tmp file to output file
        retry(lambda: tmp_output_file_name.rename(output_file_name), PermissionError)()  # type: ignore[arg-type]
        logger.trace(f"Renamed tmp file to output file ({output_file_name})")
        return Path(output_file_name)

    def write(
        self,
        name: str,
        output_dir: Path,
        tile_size: int = 512,
        as_uint8: bool = False,
        channel_ids: list[int | tuple[int, ...]] | None = None,
        channel_names: list[str] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write image."""
        return self.write_image_by_plane(
            name,
            output_dir,
            tile_size=tile_size,
            channel_ids=channel_ids,
            as_uint8=as_uint8,
            channel_names=channel_names,
            overwrite=overwrite,
        )
