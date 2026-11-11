"""Permutation score drops for the complete RF-labelled tree chain."""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from ..schema import OUTPUT_COLUMNS, chain_inputs
from ..simulation import simulate_without_feedback


def rf_chain_pfi(
    line, frame, truth, features, outputs, repeats=12, seed=42, metric="r2"
):
    if line.family != "RF":
        raise ValueError("PFI belongs to the RF-labelled tree chain.")
    if metric not in {"r2", "rmse"} or repeats < 1:
        raise ValueError("Choose r2/rmse and at least one repeat.")
    inputs = chain_inputs(line, frame)
    if len(inputs) != len(truth) or not inputs.index.equals(truth.index):
        raise ValueError(
            "PFI inputs and measured targets must have matching row indices."
        )
    if set(features) - set(inputs.columns) or set(outputs) - set(OUTPUT_COLUMNS):
        raise ValueError("Select base input features and actual chain outputs.")
    true = {}
    masks = {}
    for output in outputs:
        if output not in truth:
            raise KeyError(f"No measured target for PFI: {output}")
        true[output] = pd.to_numeric(truth[output], errors="coerce").to_numpy(float)
        masks[output] = np.isfinite(true[output])
        if masks[output].sum() < 2:
            raise ValueError(f"PFI requires at least two measured values: {output}")

    def scores(prediction):
        values = {}
        for output in outputs:
            mask = masks[output]
            y, pred = true[output][mask], prediction[output].to_numpy(float)[mask]
            values[output] = (
                r2_score(y, pred)
                if metric == "r2"
                else np.sqrt(mean_squared_error(y, pred))
            )
        return values

    baseline = scores(simulate_without_feedback(line, inputs))
    rng = np.random.default_rng(seed)
    records = []
    for feature in features:
        deltas = {output: [] for output in outputs}
        for _ in range(repeats):
            permuted = inputs.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            permuted_score = scores(simulate_without_feedback(line, permuted))
            for output in outputs:
                delta = (
                    baseline[output] - permuted_score[output]
                    if metric == "r2"
                    else permuted_score[output] - baseline[output]
                )
                deltas[output].append(delta)
        for output, values in deltas.items():
            records.append(
                {
                    "chain": "RF",
                    "output": output,
                    "feature": feature,
                    f"baseline_{metric}": baseline[output],
                    "mean_importance": np.mean(values),
                    "std_importance": np.std(values, ddof=1) if repeats > 1 else 0,
                    "min_importance": min(values),
                    "max_importance": max(values),
                    "all_importances": values,
                    "n_repeats": repeats,
                    "random_state": seed,
                }
            )
    return pd.DataFrame(records).sort_values(
        ["output", "mean_importance"], ascending=[True, False]
    )
