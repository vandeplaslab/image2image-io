"""Warp utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from koyo.transform import warp_channel, arrange_warped
from loguru import logger
from tqdm import trange


def centered_transform(
    image_size: tuple[int, int],
    image_spacing: float,
    rotation_angle: float,
) -> np.ndarray:
    """Centered rotation transform."""
    angle = np.deg2rad(rotation_angle)

    sina = np.sin(angle)
    cosa = np.cos(angle)

    # build rot mat
    rot_mat = np.eye(3)
    rot_mat[0, 0] = cosa
    rot_mat[1, 1] = cosa
    rot_mat[1, 0] = sina
    rot_mat[0, 1] = -sina

    # recenter transform
    center_point = np.multiply(image_size, image_spacing) / 2
    center_point = np.append(center_point, 0)
    translation = center_point - np.dot(rot_mat, center_point)
    rot_mat[:2, 2] = translation[:2]
    return rot_mat


def get_affine_from_config(
    config_or_path: PathLike | dict, yx: bool = True, px: bool = True, inv: bool = False
) -> tuple[np.ndarray, tuple[int, int], float, tuple[int, int], float]:
    """Get affine transformation matrix from config."""
    from koyo.json import read_json_data
    from koyo.toml import read_toml_data

    if not isinstance(config_or_path, dict):
        config_or_path = Path(config_or_path)
        if not config_or_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_or_path}")
        config = read_toml_data(config_or_path) if config_or_path.suffix == ".toml" else read_json_data(config_or_path)
    else:
        config = config_or_path

    # create matrix query
    key = "matrix"
    key += "_yx" if yx else "_xy"
    key += "_px" if px else "_um"
    key += "_inv" if inv else ""

    # get affine transformation
    affine_matrix = config[key]
    affine_matrix = np.asarray(affine_matrix)

    def _get_paths_metadata(key: str) -> tuple[float, tuple[int, int]]:
        paths = config[key]
        if not paths:
            raise ValueError(f"{key} not found in config.")
        image_shapes = [tuple(t["image_shape"]) for t in paths]
        if len(set(image_shapes)) != 1:
            raise ValueError(f"Different image shapes found in config for {key}.")
        image_shape: tuple[int, int] = tuple(image_shapes[0])
        pixel_sizes_um = [float(t["pixel_size_um"]) for t in paths]
        if len(set(pixel_sizes_um)) != 1:
            raise ValueError(f"Different pixel sizes found in config for {key}.")
        pixel_size_um: float = pixel_sizes_um[0]
        return pixel_size_um, image_shape

    # get output image shape
    fixed_pixel_size_um, fixed_image_shape = _get_paths_metadata("fixed_paths")
    moving_pixel_size_um, moving_image_shape = _get_paths_metadata("moving_paths")
    return affine_matrix, fixed_image_shape, fixed_pixel_size_um, moving_image_shape, moving_pixel_size_um


def rescale_affine(affine: np.ndarray, scale: float, with_translation: bool = False) -> np.ndarray:
    """Rescale affine transformation matrix by a given scale factor.

    This assumes that the following affine matrix is in the form:
        [ a11 a12 tx ]
        [ a21 a22 ty ]
        [  0   0   1 ]
    where the top-left 2x2 submatrix represents scaling, rotation, and shearing,
    and the last column represents translation.
    """
    scaled_affine = np.copy(affine)
    if with_translation:
        scaled_affine[:2, :] *= scale
    else:
        scaled_affine[:2, :2] *= scale
    return scaled_affine


def affine_physical_to_pixel(
    affine_inv_micron: np.ndarray,
    input_pixel_size: float,
    output_pixel_size: float,
) -> np.ndarray:
    """Convert an inverse affine matrix from micron space to pixel index space.

    Applies:  T_pixel = S_out_inv · T_micron · S_in
    where S converts pixels → microns (diagonal scale matrix).
    """

    def _scale_matrix(px_size: float) -> np.ndarray:
        return np.array(
            [
                [px_size, 0, 0],
                [0, px_size, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    s_in = _scale_matrix(input_pixel_size)
    s_out_inv = _scale_matrix(1.0 / output_pixel_size)
    return s_out_inv @ affine_inv_micron @ s_in


def warp_path(config_path: PathLike, from_transform: PathLike, order: int = 1) -> np.ndarray:
    """Warp image with image2image transformation matrix."""
    from image2image_io.readers import get_simple_reader

    # load affine matrix
    affine_inv, fixed_image_shape, _fixed_pixel_size_um, _, _ = get_affine_from_config(config_path)

    if not Path(from_transform).exists():
        raise FileNotFoundError(f"File not found: {from_transform}")

    # load readers
    from_reader = get_simple_reader(from_transform)

    # due to a limitation in the opencv implementation, we need to use scipy if the image is too large
    warped = []
    for channel in trange(from_reader.n_channels, desc="Warping images..."):
        warped.append(warp_channel(from_reader.get_channel(channel), affine_inv, fixed_image_shape, order=order))
    return arrange_warped(warped, from_reader.is_rgb)


def warp_reader(affine_inv: np.ndarray, output_shape: tuple[int, int], reader, order: int = 1) -> np.ndarray:
    """Warp image with image2image transformation matrix."""
    # due to a limitation in the opencv implementation, we need to use scipy if the image is too large
    warped = []
    for channel in trange(reader.n_channels, desc="Warping images..."):
        warped.append(warp_channel(reader.get_channel(channel), affine_inv, output_shape, order=order))
    return arrange_warped(warped, reader.is_rgb)


def warp(affine_inv: np.ndarray, output_shape: tuple[int, int], image: np.ndarray, order: int = 1) -> np.ndarray:
    """Warp image."""
    from image2image_io.utils.utilities import guess_rgb

    if image.ndim == 2:
        return warp_channel(image, affine_inv, output_shape, order=order)
    if image.ndim == 3:
        warped = []
        is_rgb = guess_rgb(image.shape)
        if is_rgb:
            for c in range(image.shape[2]):
                warped.append(warp_channel(image[:, :, c], affine_inv, output_shape, order=order))
        else:
            for c in range(image.shape[0]):
                warped.append(warp_channel(image[c], affine_inv, output_shape, order=order))
        return arrange_warped(warped, is_rgb=is_rgb)
    raise ValueError(f"Unsupported image dimension: {image.ndim}")


def warp_points(yx: np.ndarray, fwd_affine: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 homogeneous affine matrix to yx pixel coordinates (forward mapping).

    Parameters
    ----------
    yx : (N, 2) ndarray of float — [[y0, x0], [y1, x1], ...]
    fwd_affine : (3, 3) homogeneous affine matrix in yx pixel space.

    Returns
    -------
    coords_out : (N, 2) ndarray of float
    """
    if fwd_affine.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) matrix, got {fwd_affine.shape}")
    if yx.ndim != 2 or yx.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) coords, got {yx.shape}")

    ones = np.ones((len(yx), 1), dtype=yx.dtype)
    coords_h = np.hstack([yx, ones])  # (N, 3)
    result_h = (fwd_affine @ coords_h.T).T  # (N, 3)
    return result_h[:, :2] / result_h[:, 2:3]  # dehomogenise


class ImageWarper:
    """Image warper class."""

    def __init__(self, config_or_path: PathLike | dict, yx: bool = True, inv: bool = False):
        """Initialize."""
        self.config_or_path = Path(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
        self.affine_inv, self.output_size_yx, pixel_size, _, _ = get_affine_from_config(
            self.config_or_path, yx=yx, inv=inv
        )
        self.output_size = self.output_size_yx[::-1]  # x, y
        self.output_spacing = (pixel_size, pixel_size)

    def __repr__(self) -> str:
        """Repr."""
        return f"{self.__class__.__name__}<shape(xy)={self.output_size}; pixel_size(xy)={self.output_spacing}>"

    def __call__(self, image: np.ndarray | sitk.Image) -> np.ndarray | sitk.Image:
        """Warp image."""
        with MeasureTimer() as timer:
            is_sitk = isinstance(image, sitk.Image)
            if is_sitk:
                image = sitk.GetArrayFromImage(image)
            image = warp_channel(image, self.affine_inv, self.output_size_yx)
            assert image.shape == self.output_size_yx, f"Image shape mismatch: {image.shape} != {self.output_size_yx}"
            if is_sitk:
                image = sitk.GetImageFromArray(image)
            logger.trace(f"Warped image in {timer()}.")
        return image
