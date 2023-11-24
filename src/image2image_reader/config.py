"""Configuration to override few parameters."""
import typing as ty

from koyo.config import BaseConfig
from pydantic import Field, validator

from image2image_reader.enums import ViewType
from image2image_reader.utils._appdirs import USER_CONFIG_DIR


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

    view_type: ViewType = Field(ViewType.RANDOM, title="View type", description="IMS view type.", in_app=False)
    show_transformed: bool = Field(
        True, title="Show transformed", description="If checked, transformed moving image will be shown.", in_app=False
    )

    @validator("view_type", pre=True, allow_reuse=True)
    def _validate_view_type(value: ty.Union[str, ViewType]) -> ViewType:  # type: ignore[misc]
        """Validate view_type."""
        return ViewType(value)


CONFIG = Config()  # type: ignore[call-arg]
