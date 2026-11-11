"""Forward differences through the complete DNN chain in physical units."""

import numpy as np
import pandas as pd
from ..schema import OUTPUT_COLUMNS, chain_inputs
from ..simulation import simulate_without_feedback


def finite_difference_sensitivity(
    line, frame, feature_cols, *, output_cols=None, epsilon=1e-4
):
    if getattr(line, "family", None) != "DNN":
        raise ValueError(
            "Finite-difference importance is assigned to the DNN chain only"
        )
    inputs = chain_inputs(line, frame)
    output_cols = output_cols or OUTPUT_COLUMNS
    unknown = set(output_cols) - set(OUTPUT_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown sensitivity outputs: {unknown}")
    baseline = simulate_without_feedback(line, inputs)
    records = []
    for feature in feature_cols:
        if feature not in inputs:
            raise KeyError(
                f"Sensitivity feature must be an original chain input: {feature}"
            )
        if not pd.api.types.is_numeric_dtype(inputs[feature]):
            raise TypeError(f"Finite differences need numeric inputs: {feature}")
        step = float(epsilon[feature] if isinstance(epsilon, dict) else epsilon)
        if not np.isfinite(step) or step <= 0:
            raise ValueError(f"Positive finite epsilon required for {feature}")
        perturbed = inputs.copy()
        perturbed[feature] = perturbed[feature].astype(float) + step
        if not np.isfinite(perturbed[feature]).all():
            raise ValueError(f"Non-finite sensitivity input: {feature}")
        prediction = simulate_without_feedback(line, perturbed)
        for output in output_cols:
            derivatives = (
                prediction[output].to_numpy() - baseline[output].to_numpy()
            ) / step
            for item, derivative in enumerate(derivatives):
                records.append(
                    {
                        "chain": line.label,
                        "item_index": item + 1,
                        "feature": feature,
                        "output": output,
                        "epsilon": step,
                        "input_value": float(inputs.iloc[item][feature]),
                        "base_output": float(baseline.iloc[item][output]),
                        "derivative": float(derivative),
                    }
                )
    return pd.DataFrame(records)


def summarise_sensitivity(detail, reference_inputs=None):
    """Signed/absolute slopes. These have units and are not RF score drops."""
    if detail.empty:
        return pd.DataFrame()
    grouped = detail.groupby(["chain", "output", "feature", "epsilon"], sort=False)
    summary = (
        grouped["derivative"]
        .agg(n="count", mean_signed="mean", std="std", minimum="min", maximum="max")
        .reset_index()
    )
    summary["mean_absolute"] = (
        grouped["derivative"].apply(lambda d: d.abs().mean()).to_numpy()
    )
    if reference_inputs is not None:
        scales = reference_inputs.select_dtypes("number").std(ddof=0)
        summary["reference_input_sd"] = summary["feature"].map(scales)
        summary["mean_absolute_one_sd"] = (
            summary["mean_absolute"] * summary["reference_input_sd"]
        )
    ranking = (
        "mean_absolute_one_sd" if reference_inputs is not None else "mean_absolute"
    )
    return summary.sort_values(
        ["output", ranking], ascending=[True, False]
    ).reset_index(drop=True)


def sensitivity_with_feedback(line, frame, feature_cols, **kwargs):
    """Original sequential sensitivities with zeta fixed during each item's probes."""
    if getattr(line, "family", None) != "DNN":
        raise ValueError(
            "Finite-difference sensitivity is assigned to the DNN chain only"
        )
    return line.sensitivity_items(
        chain_inputs(line, frame), feature_cols, use_feedback=True, **kwargs
    )
