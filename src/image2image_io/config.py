"""Configuration to override few parameters."""

import typing as ty
from contextlib import contextmanager

from koyo.config import BaseConfig
from pydantic import Field, validator

from image2image_io.enums import ViewType
from image2image_io.utils._appdirs import USER_CONFIG_DIR


# noinspection PyMethodParameters
class Config(BaseConfig):
    """Configuration of few parameters."""

    USER_CONFIG_DIR = USER_CONFIG_DIR
    USER_CONFIG_FILENAME = "config_reader.json"

    init_pyramid: bool = Field(
        True, title="Initialize pyramid", description="Initialize pyramid on data load.", in_app=True
    )
    auto_pyramid: bool = Field(True, title="Auto pyramid", description="Automatically create pyramid.", in_app=True)
    split_czi: bool = Field(
        True,
        title="Split CZI image by scene",
        description="When loading CZI image(s), split them by the scene",
        in_app=True,
    )
    split_rgb: bool = Field(
        True,
        title="Split RGB image by channel",
        description="When loading RGB image(s), split them by the channel",
        in_app=True,
    )
    split_roi: bool = Field(
        True,
        title="Split Bruker .d images by region of interest.",
        description="When loading Bruker .d image(s), split them by the region of interest.",
        in_app=True,
    )
    only_last_pyramid: bool = Field(False, title="Only last pyramid", description="Only use last pyramid.", in_app=True)
    view_type: ViewType = Field(ViewType.RANDOM, title="View type", description="IMS view type.", in_app=False)
    show_transformed: bool = Field(
        True, title="Show transformed", description="If checked, transformed moving image will be shown.", in_app=False
    )
    shape_display: ty.Literal["polygon", "path", "polygon or path", "points"] = Field(
        "path", title="Shape display", description="Shape display type.", in_app=False
    )
    subsample: bool = Field(True, title="Subsample", description="Subsample shapes for display.", in_app=False)
    subsample_ratio: float = Field(0.01, title="Subsample ratio", description="Shapes subsample ratio.", in_app=False)
    subsample_random_seed: float = Field(
        -1, title="Subsample random seed", description="Shapes subsample ratio.", in_app=False, save=False
    )

    # extra options
    multicore: bool = Field(True, title="Multicore", description="Use multicore processing.", in_app=False)

    @validator("view_type", pre=True, allow_reuse=True)
    def _validate_view_type(value: ty.Union[str, ViewType]) -> ViewType:  # type: ignore[misc]
        """Validate view_type."""
        return ViewType(value)

    @contextmanager
    def temporary_override(self, **kwargs: ty.Any) -> ty.ContextManager:
        """Temporary override configuration."""
        old_values = {key: getattr(self, key) for key in kwargs}
        for key, value in kwargs.items():
            setattr(self, key, value)
        yield
        for key, value in old_values.items():
            setattr(self, key, value)


CONFIG = Config()  # type: ignore[call-arg]


def get_reader_config() -> Config:
    """Get reader configuration."""
    global CONFIG

    if CONFIG is None:
        CONFIG = Config()
        CONFIG.load()
    return CONFIG
