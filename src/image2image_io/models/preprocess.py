"""Normalizer model class."""

import numpy as np
from pydantic import field_validator

from image2image_io.models.base import BaseModel


class PreProcessor(BaseModel):
    name: str = ""

    def __call__(self, channel_name: str, image: np.ndarray) -> np.ndarray:
        """Normalize image."""
        raise NotImplementedError("Must implement method")


class NoopProcessor(PreProcessor):
    """A container class for no operation."""

    name: str = "noop"

    def __call__(self, channel_name: str, image: np.ndarray) -> np.ndarray:
        """Normalize image."""
        return image


class IMSNormalizer(PreProcessor):
    """A container class to normalize images."""

    # Attributes
    name: str = ""
    array: np.ndarray

    # noinspection PyMethodParameters
    @field_validator("array")
    def _validate_array(cls, array: np.ndarray) -> np.ndarray:
        """Validate array."""
        if not isinstance(array, np.ndarray):
            raise ValueError("Array must be a numpy array.")
        if not array.ndim == 2:
            raise ValueError("Array must be 2D.")
        return array

    def __call__(self, channel_name: str, image: np.ndarray) -> np.ndarray:
        """Normalize image."""
        if image.ndim != 2:
            raise ValueError("Image must be 2D.")
        if image.shape != self.array.shape:
            raise ValueError("Image shape does not match array")
        return image / self.array  # type: ignore[no-any-return]
