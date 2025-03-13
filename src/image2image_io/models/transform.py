"""Transform."""

import typing as ty
from copy import deepcopy
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger
from pydantic import PrivateAttr
from skimage.transform import AffineTransform, ProjectiveTransform

from image2image_io.enums import DEFAULT_TRANSFORM_NAME
from image2image_io.models.base import BaseModel


class TransformData(BaseModel):
    """Transformation data."""

    # Transformation object
    name: str = DEFAULT_TRANSFORM_NAME
    _transform: ty.Optional[ProjectiveTransform] = PrivateAttr(None)
    # this value should never change
    fixed_resolution: float = 1.0
    # this value can change
    moving_resolution: float = 1.0
    # Arrays of fixed and moving points
    fixed_points: ty.Optional[np.ndarray] = None
    moving_points: ty.Optional[np.ndarray] = None
    # affine transformation matrix
    affine: ty.Optional[np.ndarray] = None
    # Type of transformation
    transformation_type: str = "affine"
    # Inverse
    is_inverse: bool = False

    def to_dict(self) -> ty.Dict:
        """Serialize data."""
        return {
            "fixed_points": self.fixed_points.tolist() if self.fixed_points is not None else [],
            "moving_points": self.moving_points.tolist() if self.moving_points is not None else [],
            "fixed_pixel_size_um": self.fixed_resolution,
            "moving_pixel_size_um": self.moving_resolution,
            "matrix_yx_um": self.compute().params.tolist(),
            "matrix_yx_px": self.compute(px=True).params.tolist(),
        }

    @property
    def transform(self) -> ProjectiveTransform:
        """Retrieve the transformation object."""
        if self._transform is None:
            if self.fixed_points is None or self.moving_points is None:
                if self.affine is None:
                    self.affine = np.eye(3)
                self._transform = AffineTransform(matrix=self.affine)
            else:
                self._transform = self.compute()
        return self._transform

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Transform coordinates."""
        if self.transform is None:
            raise ValueError("No transformation found.")
        return self.transform(coords)  # type: ignore[no-any-return]

    def inverse(self, coords: np.ndarray) -> np.ndarray:
        """Inverse transformation of coordinates."""
        if self.transform is None:
            raise ValueError("No transformation found.")
        return self.transform.inverse(coords)  # type: ignore[no-any-return]

    @property
    def matrix(self) -> np.ndarray:
        """Retrieve the transformation array."""
        if self.transform is None:
            raise ValueError("No transformation found.")
        return self.transform.params  # type: ignore[no-any-return]

    def compute(
        self, moving_resolution: ty.Optional[float] = None, yx: bool = True, px: bool = False
    ) -> ProjectiveTransform:
        """Compute transformation matrix."""
        from image2image_io.utils.utilities import compute_transform

        if moving_resolution is None:
            moving_resolution = self.moving_resolution

        moving_points = self.moving_points
        fixed_points = self.fixed_points
        if moving_points is None or len(moving_points) == 0 or fixed_points is None or len(fixed_points) == 0:
            if self.affine is None:
                self.affine = np.eye(3)
                logger.trace("Transform has not been specified - using identity matrix.")
            affine = self.affine
            return AffineTransform(matrix=affine)

        # swap yx to xy
        if not yx:
            moving_points = moving_points[:, ::-1]
            fixed_points = fixed_points[:, ::-1]
        # convert to pixels
        if not px:
            moving_points = moving_points * moving_resolution
            fixed_points = fixed_points * self.fixed_resolution
        if self.is_inverse:
            moving_points, fixed_points = fixed_points, moving_points

        # compute transform
        transform = compute_transform(
            moving_points,  # source
            fixed_points,  # destination
            self.transformation_type,
        )
        self.moving_resolution = moving_resolution
        return transform

    @classmethod
    def from_array(cls, matrix: np.ndarray) -> "TransformData":
        """Load from array."""
        matrix = np.asarray(matrix)
        assert matrix.shape == (3, 3), "Expected (3, 3) matrix"
        return TransformData(affine=matrix)


class TransformModel(BaseModel):
    """Model containing transformation data."""

    transforms: ty.Optional[ty.Dict[Path, TransformData]] = None

    class Config:
        """Config."""

        arbitrary_types_allowed = True

    @property
    def name_to_path_map(self) -> ty.Dict[ty.Union[str, Path], Path]:
        """Returns dictionary that maps transform name to path."""
        if self.transforms is None:
            return {}

        mapping: ty.Dict[PathLike, Path] = {}
        for name in self.transforms:
            if isinstance(name, str):
                name = Path(name)
            mapping[name.name] = name
            mapping[Path(name.name)] = name
            mapping[name] = name
        return mapping

    @property
    def transform_names(self) -> ty.List[str]:
        """Return list of transform names."""
        return [Path(t).name for t in self.transforms] if self.transforms else []

    def add_transform(
        self, name_or_path: PathLike, transform_data: TransformData, with_inverse: bool = True, silent: bool = False
    ) -> None:
        """Add transformation matrix."""
        if self.transforms is None:
            self.transforms = {}

        path = Path(name_or_path)
        self.transforms[path] = transform_data
        if not silent:
            logger.info(f"Added '{path.name}' to list of transformations")
        if with_inverse:
            path = Path(name_or_path).parent / (path.name + " (inverse)")
            transform_data = deepcopy(transform_data)
            transform_data.is_inverse = True
            self.transforms[path] = transform_data
            if not silent:
                logger.info(f"Added '{path.name}' to list of transformations")

    def remove_transform(self, name_or_path: PathLike) -> None:
        """Remove transformation matrix."""
        if self.transforms is None:
            return

        def _remove_transform(path_of_transform: Path) -> None:
            if path_of_transform in self.transforms:
                del self.transforms[path_of_transform]
                logger.info(f"Removed '{path_of_transform.name}' from list of transformations")
            else:
                for path in self.transforms:
                    if path.name == path_of_transform.name:
                        del self.transforms[path]
                        logger.info(f"Removed '{path.name}' from list of transformations")
                        break

        name_or_path = Path(name_or_path)
        _remove_transform(name_or_path)
        inv_name_or_path = name_or_path.parent / (name_or_path.name + " (inverse)")
        _remove_transform(inv_name_or_path)

    def get_matrix(self, name_or_path: PathLike) -> ty.Optional[TransformData]:
        """Get transformation matrix."""
        if self.transforms is None:
            return None

        name_or_path = Path(name_or_path)
        name_or_path = self.name_to_path_map.get(name_or_path, None)  # type: ignore
        if name_or_path is None:
            return None
        if name_or_path in self.transforms:
            return self.transforms[name_or_path]
        return None

    def get_info(self) -> str:
        info = ""
        transform = self.compute()
        if hasattr(transform, "scale"):
            scale = transform.scale
            scale = (scale, scale) if isinstance(scale, float) else scale
            info += f"\nScale: {scale[0]:.3f}, {scale[1]:.3f}"
        if hasattr(transform, "translation"):
            translation = transform.translation
            translation = (translation, translation) if isinstance(translation, float) else translation
            info += f"\nTranslation: {translation[0]:.3f}, {translation[1]:.3f}"
        if hasattr(transform, "rotation"):
            radians = transform.rotation
            degrees = radians * 180 / 3.141592653589793
            info += f"\nRotation: {radians:.3f} ({degrees:.3f}°)"
        return info
