"""Model adapters and instance-local replacement; all inference is in physical units."""

from .trees import RFStageModel
from .registry import CallableStageModel, ModelRegistry, StageModel

__all__ = [
    "RFStageModel",
    "DNNStageModel",
    "MissingTargetTransformError",
    "CallableStageModel",
    "ModelRegistry",
    "StageModel",
]


def __getattr__(name):
    if name in {"DNNStageModel", "MissingTargetTransformError"}:
        from . import neural

        return getattr(neural, name)
    raise AttributeError(name)
