"""OME-TIFF writer for MergeRegImage class."""

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

from image2image_io.enums import SITK_TO_NP_DTYPE
from image2image_io.models.merge import MergeImages
from image2image_io.readers import BaseReader
from image2image_io.readers.utilities import get_pyramid_info, prepare_ome_xml_str
from image2image_io.utils.utilities import format_merge_channel_names


class Transformer(ty.Protocol):
    """Transformer protocol to transform image."""

    output_size: tuple[int, int]
    output_spacing: tuple[float, float]

    def __call__(self, image: sitk.Image) -> sitk.Image:
        """Transform image."""
        ...


class MergeOmeTiffWriter:
    """Class for writing multiple images into a single OME-TIFF.

    Attributes
    ----------
    x_size: int
        Size of the merged image after transformation in x
    y_size: int
        Size of the merged image after transformation in y
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
    compression: str = "deflate"

    def __init__(
        self,
        merge: MergeImages,
        transformers: list[Transformer | None] | None = None,
        crop_mask: np.ndarray | None = None,
        crop_bbox: tuple[int, int, int, int] | None = None,
    ):
        """
        Class for writing multiple images with and without transforms to a single OME-TIFF.

        Parameters
        ----------
        merge: MergeImages
            MergeRegImage to be transformed
        transformers: List of RegTransformSeq or None
            Registration transformation sequences for each wsireg image to be merged
        crop_mask : np.ndarray or None
            Crop mask to apply to images before writing.
        crop_bbox : tuple of ints or None
            Bounding box to crop images to before writing. Values should be x, y, width, height.
        """
        self.merge = merge
        self.transformers = transformers
        self.crop_mask = crop_mask
        self.crop_bbox = crop_bbox
        if self.crop_mask is not None and self.crop_bbox is not None:
            raise ValueError("Cannot supply both crop_mask and crop_bbox")

    @staticmethod
    def _check_bbox(
        crop_bbox: tuple[int, int, int, int] | None, transformer: Transformer | None, reader: BaseReader
    ) -> tuple[int, int, int, int] | None:
        """Check bbox."""
        if crop_bbox is None:
            return None
        x, y, width, height = crop_bbox
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if transformer is not None:
            image_shape = transformer.output_size
        else:
            image_shape = reader.image_shape
        if x + width > image_shape[1]:
            width = image_shape[1] - x
        if y + height > image_shape[0]:
            height = image_shape[0] - y
        return x, y, width, height

    def _check_transforms_and_readers(self) -> None:
        """Make sure incoming data is kosher in dimensions."""
        transformations = self.transformers
        if transformations is None:
            transformations = [None for _ in range(len(self.merge.readers))]
        if len(transformations) != len(self.merge.readers):
            raise ValueError("The number of transforms does not match number of images")

    def _create_channel_names(self, reader_names: list[str]) -> None:
        """Create channel names for merge data."""

        def _prepare_channel_names(reader_name: str, channel_names_: list[str]) -> list[str]:
            return [f"{channel_name} - {reader_name}" for channel_name in channel_names_]

        all_channel_names = []
        for channel_names in self.merge.channel_names:
            all_channel_names.extend(channel_names)

        # if len(set(all_channel_names)) != self.merge.n_channels:
        self.merge.channel_names = [
            _prepare_channel_names(name, channel_names)
            for name, channel_names in zip(reader_names, self.merge.channel_names)
        ]

    def _check_transforms_sizes_and_resolutions(self) -> None:
        """Check that all transforms as currently loaded output to the same size/resolution."""
        out_size = []
        out_spacing = []
        registered_transforms = self.transformers
        if registered_transforms is None:
            registered_transforms = [None for _ in range(len(self.merge.readers))]
        for reader, transform in zip(self.merge.readers, registered_transforms):
            if transform:
                out_size.append(transform.output_size)
                out_spacing.append(transform.output_spacing)
            else:
                out_size.append(reader.image_shape)
                out_spacing.append((reader.resolution, reader.resolution))

        if not all(out_spacing):
            raise ValueError("All transforms output spacings and untransformed image spacings must match")
        if not all(out_size):
            raise ValueError("All transforms output sizes and untransformed image sizes must match")

    def _check_channel_ids(self, channel_ids: list[list[int] | None] | None) -> list[list[int]]:
        """Check channel ids."""
        if channel_ids is None or not channel_ids:
            channel_ids = [reader.channel_ids for reader in self.merge.readers]
        if len(channel_ids) != len(self.merge.readers):
            raise ValueError(
                f"The number of channel_ids does not match number of images. Expected {len(self.merge.readers)} but"
                f" got {len(channel_ids)}"
            )
        channel_ids_ret = []
        for reader_index, channel_ids_ in enumerate(channel_ids):
            if channel_ids_ is None:
                channel_ids_ = list(range(0, self.merge.readers[reader_index].n_channels))
            if channel_ids_ is not None:
                if len(channel_ids_) == 0:
                    raise ValueError("Channel ids cannot be empty")
                elif max(channel_ids_) > self.merge.readers[reader_index].n_channels - 1:
                    raise ValueError("Channel ids cannot be larger than the number of channels in the image")
                channel_ids_ret.append(channel_ids_)
        return channel_ids_ret

    def _prepare_image_info(
        self,
        reader: BaseReader,
        image_name: str,
        dtype: np.dtype,
        channel_ids: list[list[int]] | None = None,
        transformer: Transformer | None = None,
        write_pyramid: bool = True,
        tile_size: int = 512,
        compression: str = "default",
    ) -> None:
        """Prepare OME-XML and other data needed for saving."""
        if transformer:
            self.x_size, self.y_size = transformer.output_size[::-1]
        elif self.crop_mask is not None:
            self.y_size, self.x_size = self.crop_mask.shape
        else:
            self.y_size, self.x_size = reader.image_shape

        self.crop_bbox = self._check_bbox(self.crop_bbox, transformer, reader)

        if self.crop_bbox is not None:
            _, _, self.x_size, self.y_size = self.crop_bbox

        self.tile_size = tile_size
        # protect against too large tile size
        while self.y_size / self.tile_size <= 1 or self.x_size / self.tile_size <= 1:
            self.tile_size = self.tile_size // 2

        if self.tile_size:
            self.pyr_levels, _ = get_pyramid_info(self.y_size, self.x_size, reader.n_channels, self.tile_size)
        else:
            self.pyr_levels = [(self.y_size, self.x_size)]
        self.n_pyr_levels = len(self.pyr_levels)

        if transformer:
            self.PhysicalSizeX, self.PhysicalSizeY = transformer.output_spacing
        else:
            self.PhysicalSizeX = self.PhysicalSizeY = reader.resolution

        channel_names = format_merge_channel_names(self.merge.channel_names, self.merge.n_channels, channel_ids)
        logger.trace(f"Exporting: {channel_names} for {channel_ids}")
        n_channels = len(channel_names)
        self.omexml = prepare_ome_xml_str(
            self.y_size,
            self.x_size,
            n_channels,
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

    def _get_merge_dtype(self, as_uint8: bool = False) -> tuple[int, np.dtype[ty.Any]]:
        """Determine the data type for merger - will default to the largest dtype.

        If one image is np.uint8 and another np.uint16, the image at np.uint8 will be cast to np.uint16.
        """
        dtype_max_size = [
            (np.finfo(r.dtype).max if np.issubdtype(r.dtype, np.floating) else np.iinfo(r.dtype).max)
            for r in self.merge.readers
        ]

        merge_dtype_np: np.dtype[ty.Any] = self.merge.readers[np.argmax(dtype_max_size)].dtype  # type: ignore[arg-type]
        merge_dtype_np = np.uint8 if as_uint8 else merge_dtype_np  # type: ignore[assignment]
        merge_dtype_sitk = 0
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
        as_uint8: bool | None = False,
        channel_ids: list[list[int | tuple[int, ...]] | None] | None = None,
    ) -> Path:
        """Write merged OME-TIFF image plane-by-plane to disk.

         RGB images will be de-interleaved with RGB channels written as separate planes.

        Parameters
        ----------
        name: str
             Name to be written WITHOUT the extension, for example, if image_name = "cool_image" the file
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
        as_uint8: bool
            Whether to save the image in the 0-255 intensity range, substantially reducing file size at the cost of some
            intensity information.
        channel_ids: list of list
            Channel indices to write to OME-TIFF, if None, all channels are written

        Returns
        -------
         output_file_name: str
             File path to the written OME-TIFF
        """
        merge_dtype_sitk, merge_dtype_np = self._get_merge_dtype(as_uint8=as_uint8)

        channel_ids_fixed: list[list[int]] = self._check_channel_ids(channel_ids)
        self._check_transforms_and_readers()
        self._create_channel_names(reader_names)
        self._check_transforms_sizes_and_resolutions()
        assert isinstance(channel_ids_fixed, list), "channel_ids must be a list of lists"

        # make sure user did not provide filename with OME-TIFF
        name = name.replace(".ome", "").replace(".tiff", "").replace(".tif", "")
        output_file_name = Path(output_dir) / f"{name}.ome.tiff"
        tmp_output_file_name = Path(output_dir) / f"{name}.ome.tiff.tmp"
        logger.info(f"Saving to '{output_file_name}'")
        transformer = self.transformers[0] if self.transformers else None
        self._prepare_image_info(
            self.merge.readers[0],
            name,
            merge_dtype_np,
            channel_ids=channel_ids_fixed,
            transformer=transformer,
            write_pyramid=write_pyramid,
            tile_size=tile_size,
            compression=compression,
        )

        if as_uint8:
            logger.trace("Writing image data in 0-255 range as uint8")
        logger.info(f"Writing channels: {channel_ids_fixed}")
        logger.info(f"Writing channel names: {self.merge.channel_names}")

        options = {
            "tile": (tile_size, tile_size),
            "compression": self.compression,
            "photometric": "minisblack",
            "metadata": None,
        }
        logger.trace(f"TIFF options: {options}")
        logger.trace(f"Pyramid levels: {self.pyr_levels} ({self.n_pyr_levels})")

        ome_set = False
        description = None
        with TiffWriter(tmp_output_file_name, bigtiff=True) as tif, MeasureTimer() as main_timer:
            for reader_index, reader in enumerate(tqdm(self.merge.readers, desc="Writing modality...")):
                channel_ids_ = channel_ids_fixed[reader_index]
                if channel_ids_ is None:
                    channel_ids_ = list(range(0, reader.n_channels))
                for channel_index in tqdm(
                    channel_ids_, leave=False, desc=f"Exporting images for '{reader_names[reader_index]}'"
                ):
                    channel_name = self.merge.channel_names[reader_index][channel_index]
                    if channel_index not in channel_ids_:
                        logger.trace(f"Skipped channel: {channel_name} ({channel_index})")
                        continue

                    # retrieve channel data
                    # if channel_index is int, then it's simple, otherwise, let's get maximum intensity projection
                    if isinstance(channel_index, int):
                        image: sitk.Image = np.squeeze(reader.get_channel(channel_index))  # type: ignore[assignment]
                    else:
                        images: list[sitk.Image] = [np.squeeze(reader.get_channel(i)) for i in channel_index]
                        images = np.dstack(images)  # type: ignore[assignment]
                        image = np.max(images, axis=2)  # type: ignore[assignment]
                    image: sitk.Image = sitk.GetImageFromArray(image)  # type: ignore[no-redef]
                    image.SetSpacing((reader.resolution, reader.resolution))  # type: ignore[attr-defined]

                    # transform
                    if self.transformers and callable(self.transformers[reader_index]):
                        with MeasureTimer() as timer:
                            image = self.transformers[reader_index](image)  # type: ignore[assignment,arg-type,misc]
                        logger.trace(
                            f"Transformed image shape: {image.GetSize()[::-1]} in {timer()}",  # type: ignore[attr-defined]
                        )

                    # change dtype
                    if as_uint8:
                        image = sitk.RescaleIntensity(image, 0, 255)  # type: ignore[no-untyped-call]
                        image = sitk.Cast(image, sitk.sitkUInt8)  # type: ignore[no-untyped-call]
                    elif image.GetPixelIDValue() != merge_dtype_sitk:  # type: ignore[attr-defined]
                        image = sitk.Cast(image, merge_dtype_sitk)  # type: ignore[no-untyped-call]

                    # make sure we have numpy array
                    if isinstance(image, sitk.Image):
                        image = sitk.GetArrayFromImage(image)

                    # apply crop mask
                    if self.crop_mask is not None:
                        image = self.crop_mask * image
                    elif self.crop_bbox is not None:
                        x, y, width, height = self.crop_bbox
                        image = image[y : y + height, x : x + width]

                    # write OME-XML to the ImageDescription tag of the first page
                    if not ome_set:
                        description = self.omexml
                        ome_set = True

                    # write channel data
                    msg = (
                        f"Writing image for reader={reader_names[reader_index]} ({reader_index})"
                        f" channel={channel_name} ({channel_index})"
                    )
                    past_msg = msg.replace("Writing", "Wrote")
                    logger.trace(f"{msg} - {image.shape}...")
                    with MeasureTimer() as write_timer:
                        tif.write(image, subifds=self.subifds, description=description, **options)
                        logger.trace(f"{past_msg} pyramid index 0 in {write_timer()}")
                        if write_pyramid:
                            for pyramid_index in range(1, self.n_pyr_levels):
                                resize_shape = (self.pyr_levels[pyramid_index][0], self.pyr_levels[pyramid_index][1])
                                image = cv2.resize(image, resize_shape, cv2.INTER_LINEAR)
                                logger.trace(f"{msg} pyramid index {pyramid_index} - {image.shape}...")
                                tif.write(image, **options, subfiletype=1)
                                logger.trace(
                                    f"{past_msg} pyramid index {pyramid_index} in {write_timer(since_last=True)}"
                                )
        logger.trace(f"Exported OME-TIFF in {main_timer()}")
        # rename temporary file to final file
        retry(lambda: tmp_output_file_name.rename(output_file_name), PermissionError)()  # type: ignore[arg-type]
        logger.trace(f"Renamed tmp file to output file ({output_file_name})")
        return Path(output_file_name)

    def write(
        self,
        name: str,
        reader_names: list[str],
        output_dir: Path,
        tile_size: int = 512,
        as_uint8: bool = False,
        channel_ids: list[list[int] | None] | None = None,
    ) -> Path:
        """Write image."""
        return self.merge_write_image_by_plane(
            name, reader_names, output_dir, tile_size=tile_size, as_uint8=as_uint8, channel_ids=channel_ids
        )
