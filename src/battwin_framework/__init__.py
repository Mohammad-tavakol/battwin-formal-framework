"""Modular BATTWIN coupled-model library with the original scientific behaviour."""

from .framework import Framework
from .data import load_dataset
from .scenarios import DisturbanceConfig
from .resources import verify_assets
from .core import NamedStateSpaceLine
from .indexed import StateSpaceLine

__version__ = "0.2.0"
__all__ = [
    "Framework",
    "load_dataset",
    "DisturbanceConfig",
    "NamedStateSpaceLine",
    "StateSpaceLine",
    "verify_assets",
]
