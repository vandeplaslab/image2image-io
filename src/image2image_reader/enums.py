"""Enums."""
from enum import Enum, auto

DEFAULT_TRANSFORM_NAME: str = "Identity matrix"

TIME_FORMAT = "%d/%m/%Y-%H:%M:%S:%f"


class ViewType(str, Enum):
    """View type."""

    RANDOM = "random"
    OVERLAY = "overlay"
