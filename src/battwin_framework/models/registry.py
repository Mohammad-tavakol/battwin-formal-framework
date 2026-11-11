"""Lazy model loading, stage contracts, and instance-local model replacement."""

from pathlib import Path
from typing import Protocol, runtime_checkable
import pandas as pd
from ..schema import STAGE_OUTPUTS, ALIASES, OUTPUT_COLUMNS
from ..simulation import _assemble, chain_description
from .catalogue import RF_FILES, DNN_FILES, DNN_TARGET_FILES
from .preprocessing import load_saved
from .trees import RFStageModel


@runtime_checkable
class StageModel(Protocol):
    """A stage consumes canonical inputs and returns ordered physical-unit outputs."""

    name: str
    family: str
    columns: list[str]
    outputs: list[str]

    def predict_frame(self, features: pd.DataFrame) -> pd.DataFrame: ...


class CallableStageModel:
    """Adapt a custom predictor to the stage contract, validating every result.

    ``predict`` receives only declared input columns. It must return a DataFrame
    with the original index and exactly the stage's physical-unit output columns.
    """

    def __init__(self, name, family, columns, predict):
        self.name, self.family = name, family
        self.columns = list(columns)
        self.outputs = list(STAGE_OUTPUTS[name])
        self.paths = []
        self._predict = predict

    def predict_frame(self, features):
        missing = set(self.columns) - set(features)
        if missing:
            raise KeyError(f"Missing model inputs: {sorted(missing)}")
        result = self._predict(features.loc[:, self.columns].copy())
        if not isinstance(result, pd.DataFrame):
            raise TypeError("Custom stages must return a DataFrame in physical units.")
        if not result.index.equals(features.index) or list(result) != self.outputs:
            raise ValueError(
                "Custom stage changed row order or its declared output schema."
            )
        import numpy as np

        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("Custom stage produced non-finite outputs.")
        return result


class ModelRegistry:
    """Own an independent set of stage models; bundled files are never overwritten."""

    def __init__(
        self, directory, *, cache_directory=None, allow_extra_trees=True, n_jobs=1
    ):
        self.directory = Path(directory).expanduser().resolve()
        self.cache_directory = cache_directory
        self.allow_extra_trees, self.n_jobs = allow_extra_trees, n_jobs
        self._models, self._chains = {}, {}

    def _load(self, family):
        if family not in {"RF", "DNN"}:
            raise ValueError("Model family must be RF or DNN.")
        if family in self._models:
            return
        directory = self.directory / "best models" / family
        if family == "RF":
            models = {
                stage: RFStageModel(
                    stage,
                    [directory / f for f in files],
                    allow_extra_trees=self.allow_extra_trees,
                    n_jobs=self.n_jobs,
                )
                for stage, files in RF_FILES.items()
            }
        else:
            from .neural import DNNStageModel

            models = {
                stage: DNNStageModel(
                    stage,
                    directory / files[0],
                    directory / files[1],
                    load_saved(directory / DNN_TARGET_FILES[stage]),
                    cache_directory=self.cache_directory,
                )
                for stage, files in DNN_FILES.items()
            }
        self._models[family] = models

    def chain(self, family):
        """Return the compiled, current chain for a model family."""
        self._load(family)
        if family not in self._chains:
            self._chains[family] = _assemble(dict(self._models[family]), family)
        return self._chains[family]

    def stage(self, family, stage):
        self._load(family)
        return self._models[family][stage]

    def describe(self, family):
        """Show each stage's exact reads, writes, base inputs and artifact names."""
        return pd.DataFrame(chain_description(self.chain(family)))

    def replace(self, family, stage, model):
        """Replace one stage in this registry after validating the compiled contract.

        Returns the previous model so the caller can restore it. No training,
        serialization or modification of source artifacts is performed.
        """
        self._load(family)
        if stage not in STAGE_OUTPUTS:
            raise ValueError(f"Unknown stage: {stage}")
        if model.name != stage or model.family != family:
            raise ValueError("Replacement stage name and model family must match.")
        if list(model.outputs) != STAGE_OUTPUTS[stage]:
            raise ValueError(
                "Replacement must preserve output names, order and physical units."
            )
        if not callable(getattr(model, "predict_frame", None)):
            raise TypeError("Replacement must implement predict_frame(DataFrame).")
        if len(set(model.columns)) != len(model.columns):
            raise ValueError("Replacement contains duplicate input columns.")
        proposed = dict(self._models[family])
        proposed[stage] = model
        line = _assemble(proposed, family)
        if set(line.get_required_input_columns()) & set(OUTPUT_COLUMNS):
            raise ValueError(
                "Replacement would expose a stage output as an original input."
            )
        old = self._models[family][stage]
        self._models[family], self._chains[family] = proposed, line
        return old

    def schema(self, family, stage):
        """Return canonical required stage inputs, plus physical output columns."""
        model = self.stage(family, stage)
        return {
            "inputs": [ALIASES.get(c, c) for c in model.columns],
            "outputs": list(model.outputs),
        }
