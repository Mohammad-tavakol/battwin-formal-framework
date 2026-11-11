"""Keras inference with saved preprocessing and physical target inverses."""

from pathlib import Path
from copy import copy
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from ..resources import default_cache_directory
from ..schema import STAGE_OUTPUTS
from .preprocessing import load_saved, model_frame


class MissingTargetTransformError(ValueError):
    pass


def load_keras_for_inference(path, *, cache_directory=None):
    """Handle null initializer fields added by Keras 3.15 on older Keras.

    Only serialization metadata with no inference effect is removed, in a local
    cache copy. Original model files and the entire saved weights payload stay
    byte-identical. Non-null axis metadata is never discarded.
    """
    from keras.models import load_model

    try:
        return load_model(path, compile=False)
    except TypeError as exc:
        if "unexpected keyword argument 'input_axes'" not in str(exc):
            raise
        with zipfile.ZipFile(path) as source:
            configuration = json.loads(source.read("config.json"))
            changed = 0

            def clean(value):
                nonlocal changed
                if isinstance(value, dict):
                    if value.get("class_name") == "GlorotUniform":
                        config = value.get("config", {})
                        for key in ["input_axes", "output_axes"]:
                            if key in config:
                                if config[key] is not None:
                                    raise TypeError(
                                        "Cannot discard non-null initializer axes"
                                    )
                                del config[key]
                                changed += 1
                    for child in value.values():
                        clean(child)
                elif isinstance(value, list):
                    for child in value:
                        clean(child)

            clean(configuration)
            if not changed:
                raise
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            cache = (
                (
                    Path(cache_directory)
                    if cache_directory is not None
                    else default_cache_directory()
                )
                / "keras_compat"
                / (digest + ".keras")
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(cache, "w") as target:
                for info in source.infolist():
                    contents = (
                        json.dumps(configuration).encode("utf-8")
                        if info.filename == "config.json"
                        else source.read(info.filename)
                    )
                    target.writestr(copy(info), contents)
            with zipfile.ZipFile(cache) as target:
                assert target.read("model.weights.h5") == source.read(
                    "model.weights.h5"
                )
        return load_model(cache, compile=False)


def inverse_target(raw, specification):
    """Inverse of a supplied target transform, never a fitted approximation."""
    kind = specification.get("kind")
    values = np.asarray(raw, dtype=float).reshape(-1, 1)
    scaler = specification.get("scaler")
    if scaler is not None:
        values = scaler.inverse_transform(values)
    elif "mean" in specification and "scale" in specification:
        values = values * float(specification["scale"]) + float(specification["mean"])
    elif kind == "standard":
        raise MissingTargetTransformError(
            "A standard target needs its saved scaler or mean/scale."
        )
    if kind == "log1p":
        values = np.expm1(values)
    elif kind not in {"identity", "standard"}:
        raise MissingTargetTransformError(f"Unknown target transform: {kind!r}")
    return values[:, 0]


class DNNStageModel:
    family = "DNN"

    def __init__(
        self, name, model_path, preprocessor_path, targets, *, cache_directory=None
    ):
        self.name = name
        self.outputs = STAGE_OUTPUTS[name]
        self.paths = [
            Path(model_path).expanduser().resolve(),
            Path(preprocessor_path).expanduser().resolve(),
        ]
        self.targets = targets
        if len(targets) != len(self.outputs):
            raise MissingTargetTransformError(f"{name}: wrong target metadata length")
        self.preprocessor = load_saved(self.paths[1])
        self.columns = list(self.preprocessor.feature_names_in_)
        try:
            from keras.models import load_model
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "DNN execution requires a working Keras backend. "
                "Install the pinned package dependencies in Python 3.11."
            ) from exc
        self.model = load_keras_for_inference(
            self.paths[0], cache_directory=cache_directory
        )
        if len(self.model.outputs) != len(self.outputs):
            raise ValueError(
                f"{name}: saved DNN head count does not match target metadata"
            )

    def predict_frame(self, features):
        frame = model_frame(features, self.columns, self.preprocessor)
        if len(frame) == 0:
            return pd.DataFrame(index=frame.index, columns=self.outputs, dtype=float)
        encoded = np.asarray(self.preprocessor.transform(frame), dtype=np.float32)
        if encoded.shape[1] != self.model.input_shape[-1]:
            raise ValueError(f"{self.name}: preprocessor width does not match the DNN")
        raw = self.model(encoded, training=False)
        heads = list(raw) if isinstance(raw, (tuple, list)) else [raw]
        values = np.column_stack(
            [
                inverse_target(np.asarray(head), spec)
                for head, spec in zip(heads, self.targets)
            ]
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite outputs from {self.name}")
        return pd.DataFrame(values, index=frame.index, columns=self.outputs)
