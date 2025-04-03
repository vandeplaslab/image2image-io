"""Warp utilities."""

from __future__ import annotations

from pathlib import Path

import dask.array
import numpy as np
import SimpleITK as sitk
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from tqdm import trange


def get_affine_from_config(
    config_or_path: PathLike | dict, yx: bool = True, px: bool = True, inv: bool = False
) -> tuple[np.ndarray, tuple[int, int], float]:
    """Get affine transformation matrix from config."""
    from koyo.json import read_json_data
    from koyo.toml import read_toml_data

    if not isinstance(config_or_path, dict):
        config_or_path = Path(config_or_path)
        if not config_or_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_or_path}")
        if config_or_path.suffix == ".toml":
            config = read_toml_data(config_or_path)
        else:
            config = read_json_data(config_or_path)
    else:
        config = config_or_path

    # create matrix query
    key = "matrix"
    key += "_yx" if yx else "_xy"
    key += "_px" if px else "_um"
    key += "_inv" if inv else ""

    # get affine transformation
    affine_inv = config[key]
    affine_inv = np.asarray(affine_inv)
    # get output image shape
    fixed_paths = config["fixed_paths"]
    if not fixed_paths:
        raise ValueError("Fixed paths not found in config.")
    image_shapes = [tuple(t["image_shape"]) for t in fixed_paths]
    if len(set(image_shapes)) != 1:
        raise ValueError("Different image shapes found in config.")
    image_shape: tuple[int, int] = tuple(image_shapes[0])
    pixel_sizes_um = [float(t["pixel_size_um"]) for t in fixed_paths]
    if len(set(pixel_sizes_um)) != 1:
        raise ValueError("Different pixel sizes found in config.")
    pixel_size_um: float = pixel_sizes_um[0]
    return affine_inv, image_shape, pixel_size_um


def warp_path(config_path: PathLike, from_transform: PathLike) -> np.ndarray:
    """Warp image with image2image transformation matrix."""
    import cv2
    from scipy.ndimage import affine_transform

    from image2image_io.readers import get_simple_reader

    # load affine matrix
    affine_inv, output_shape, pixel_size = get_affine_from_config(config_path)

    if not Path(from_transform).exists():
        raise FileNotFoundError(f"File not found: {from_transform}")

    # load readers
    from_reader = get_simple_reader(from_transform)

    # due to a limitation in the opencv implementation, we need to use scipy if the image is too large
    use_cv2 = max(max(from_reader.image_shape), max(output_shape)) < 32767
    warped = []
    for channel in trange(from_reader.n_channels, desc="Warping images..."):
        img = from_reader.get_channel(channel)
        if use_cv2:
            if isinstance(img, dask.array.Array):
                img = img.T.compute()
            warp_img = cv2.warpAffine(img, np.linalg.inv(affine_inv)[:2, :], output_shape[::-1]).T
        else:
            warp_img = affine_transform(img, affine_inv, order=1, output_shape=output_shape)
        warped.append(warp_img)

    # stack image
    warped = np.dstack(warped)
    # ensure that RGB remains RGB but AF remain AF
    if warped.ndim == 3 and np.argmin(warped.shape) == 2 and not from_reader.is_rgb:
        warped = np.moveaxis(warped, 2, 0)
    return warped


def warp(affine_inv: np.ndarray, output_shape: tuple[int, int], image: np.ndarray, order: int = 1) -> np.ndarray:
    """Warp image."""
    import cv2
    from scipy.ndimage import affine_transform

    use_cv2 = max(max(image.shape), max(output_shape)) < 32767
    if use_cv2:
        logger.debug("Using cv2 for warping.")
        if isinstance(image, dask.array.Array):
            image = image.compute()
        warped = cv2.warpAffine(
            image.T,
            np.linalg.inv(affine_inv)[:2, :],
            output_shape,
            flags=cv2.INTER_NEAREST if order == 0 else cv2.INTER_LINEAR,
        ).T
    else:
        warped = affine_transform(image, affine_inv, order=order, output_shape=output_shape)
    return warped


class ImageWarper:
    """Image warper class."""

    def __init__(self, config_or_path: PathLike | dict, yx: bool = True, inv: bool = False):
        """Initialize."""
        self.config_or_path = Path(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
        self.affine_inv, self.output_size_yx, pixel_size = get_affine_from_config(self.config_or_path, yx=yx, inv=inv)
        self.output_size = self.output_size_yx[::-1]  # x, y
        self.output_spacing = (pixel_size, pixel_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<shape(xy)={self.output_size}; pixel_size(xy)={self.output_spacing}>"

    def __call__(self, image: np.ndarray | sitk.Image) -> np.ndarray | sitk.Image:
        """Warp image."""
        with MeasureTimer() as timer:
            is_sitk = isinstance(image, sitk.Image)
            if is_sitk:
                image = sitk.GetArrayFromImage(image)
            image = warp(self.affine_inv, self.output_size_yx, image)
            assert image.shape == self.output_size_yx, f"Image shape mismatch: {image.shape} != {self.output_size_yx}"
            if is_sitk:
                image = sitk.GetImageFromArray(image)
            logger.trace(f"Warped image in {timer()}.")
        return image
