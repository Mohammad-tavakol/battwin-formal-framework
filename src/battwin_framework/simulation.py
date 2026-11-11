"""Stage-local actuation and independent or sequential chain execution."""

from copy import deepcopy
import pandas as pd
from .core import NamedStateSpaceLine
from .schema import ALIASES, FEEDBACK_INPUTS, chain_inputs


class StageAdapter:
    """Read r/u/theta from the named store and apply the existing gap actuator."""

    def __init__(self, model):
        self.model = model

    def predict_frame(self, features, zeta=None):
        local = features.copy()
        adjustments = FEEDBACK_INPUTS.get(self.model.name, {})
        zeta = zeta or {}
        unexpected = set(zeta) - set(adjustments)
        if unexpected:
            raise ValueError(
                f"Unsupported feedback keys for {self.model.name}: {unexpected}"
            )
        for key, col in adjustments.items():
            delta = float(zeta.get(key, 0.0))
            local[col] = local[col].astype(float) + delta
        return self.model.predict_frame(local)

    def __call__(self, features, zeta, control=None, theta=None):
        # Same stage-local actuator semantics as the Implementation notebook.
        frame = pd.DataFrame([deepcopy(features)])
        return self.predict_frame(frame, zeta).iloc[0].to_dict()


def _assemble(models, family):
    line = NamedStateSpaceLine()
    chain = []
    produced = set()
    for name, model in models.items():
        canonical_inputs = [ALIASES.get(c, c) for c in model.columns]
        adapter = StageAdapter(model)
        chain.append(
            {
                "name": name,
                "func": adapter,
                "control": {},
                "theta": {},
                "input_columns": canonical_inputs,
                "output_columns": model.outputs,
                "required_input_columns": [
                    c for c in canonical_inputs if c not in produced
                ],
            }
        )
        produced.update(model.outputs)
    line.load_sequence(chain)
    line.compile()
    line.family = family
    line.models = models
    types = {t for m in models.values() for t in getattr(m, "regressor_types", [])}
    line.label = (
        "RF + ExtraTrees calendering" if "ExtraTreesRegressor" in types else family
    )
    return line


def chain_description(line):
    return [
        {
            "chain": line.label,
            "stage": stage["name"],
            "reads": stage["input_columns"],
            "writes": stage["output_columns"],
            "base_inputs": stage["required_input_columns"],
            "models": [
                str(p.name) for p in getattr(line.models[stage["name"]], "paths", [])
            ],
        }
        for stage in line.chain_def
    ]


def simulate_without_feedback(line, frame):
    """Independent items in a batch, with no feedback or shared mutable state.

    Equivalent to NamedStateSpaceLine.simulate_items(use_feedback=False),
    with each model evaluating all independent rows in one call. Feedback and
    per-item traces use the original core's sequential methods instead.
    """
    if not line.compiled:
        raise RuntimeError("Framework must be compiled before simulation.")
    features = chain_inputs(line, frame)
    if len(features) == 0:
        for stage in line.chain_def:
            for col in stage["output_columns"]:
                features[col] = pd.Series(index=features.index, dtype=float)
        return features
    for stage in line.chain_def:
        output = stage["func"].predict_frame(features, zeta={})
        if not output.index.equals(features.index):
            raise ValueError(f"Stage {stage['name']} changed item order")
        for column in output:
            features[column] = output[column]
    return features


def simulate_with_trace(line, frame, **kwargs):
    inputs = chain_inputs(line, frame)
    history = line.simulate_items(inputs, **kwargs)
    results = pd.DataFrame(
        [record["features_final"] for record in history], index=inputs.index
    )
    return results, deepcopy(history)
