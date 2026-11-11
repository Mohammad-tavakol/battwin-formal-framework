"""Saved tree-stage pipelines, including the supplied ExtraTrees models."""

from pathlib import Path
import numpy as np
import pandas as pd
from ..schema import STAGE_OUTPUTS
from .preprocessing import load_saved, model_frame


class RFStageModel:
    family = "RF"

    def __init__(self, name, paths, allow_extra_trees=False, n_jobs=1):
        self.name = name
        self.outputs = STAGE_OUTPUTS[name]
        self.paths = [Path(p).expanduser().resolve() for p in paths]
        self.models = [load_saved(p) for p in self.paths]
        self.columns = list(self.models[0].feature_names_in_)
        self.regressor_types = []
        for model in self.models:
            if list(model.feature_names_in_) != self.columns:
                raise ValueError(f"Inconsistent schemas within stage {name}")
            regressor = model.named_steps["regressor"].regressor_
            kind = type(regressor).__name__
            allowed = {"RandomForestRegressor"}
            if allow_extra_trees:
                allowed.add("ExtraTreesRegressor")
            if kind not in allowed:
                raise ValueError(
                    f"{name}: {kind} is not an authorised RF-stage estimator. "
                    "The supplied calendering files contain ExtraTrees. "
                    "Set allow_extra_trees=True to use and label them explicitly."
                )
            self.regressor_types.append(kind)
            regressor.n_jobs = n_jobs
        self.preprocessor = self.models[0].named_steps["preprocessor"]

    def predict_frame(self, features):
        frame = model_frame(features, self.columns, self.preprocessor)
        # Pipeline.predict returns physical units, including log1p/standard inverses.
        values = np.column_stack([model.predict(frame) for model in self.models])
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite outputs from {self.name}")
        return pd.DataFrame(values, index=frame.index, columns=self.outputs)
