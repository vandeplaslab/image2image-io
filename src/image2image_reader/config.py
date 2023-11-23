"""Configuration to override few parameters."""
from koyo.config import BaseConfig
from pydantic import Field

from image2image_reader.utils._appdirs import USER_CONFIG_DIR


class Config(BaseConfig):
    """Configuration of few parameters."""

    USER_CONFIG_DIR = USER_CONFIG_DIR
    USER_CONFIG_FILENAME = "config_reader.json"

    auto_pyramid: bool = Field(True, title="Auto pyramid", description="Automatically create pyramid.", in_app=True)


CONFIG = Config()  # type: ignore[call-arg]
