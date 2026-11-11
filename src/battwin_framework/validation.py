"""Measured-target metrics and standalone electrochemical comparisons."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .schema import OUTPUT_COLUMNS, GRAVIMETRIC, VOLUMETRIC, ALIASES


def validation_metrics(predictions, truth, chain_label):
    if len(predictions) != len(truth) or not predictions.index.equals(truth.index):
        raise ValueError("Predictions and truth must have matching item indices")
    records = []
    for col in OUTPUT_COLUMNS:
        if col not in truth:
            records.append(
                {
                    "chain": chain_label,
                    "output": col,
                    "n": 0,
                    "status": "No measured target in the supplied dataset",
                }
            )
            continue
        true = pd.to_numeric(truth[col], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(predictions[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(pred).all():
            raise ValueError(f"Non-finite prediction for {col}")
        mask = np.isfinite(true)
        n = int(mask.sum())
        if not n:
            records.append(
                {
                    "chain": chain_label,
                    "output": col,
                    "n": 0,
                    "status": "No finite measured values",
                }
            )
            continue
        error = pred[mask] - true[mask]
        nonzero = mask & (true != 0)
        records.append(
            {
                "chain": chain_label,
                "output": col,
                "n": n,
                "status": "Evaluated",
                "r2": float(r2_score(true[mask], pred[mask])) if n >= 2 else np.nan,
                "rmse": float(np.sqrt(mean_squared_error(true[mask], pred[mask]))),
                "mae": float(mean_absolute_error(true[mask], pred[mask])),
                "mean_error": float(np.mean(error)),
                "mape_percent": (
                    float(
                        np.mean(np.abs((pred[nonzero] - true[nonzero]) / true[nonzero]))
                        * 100
                    )
                    if nonzero.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def comparison_table(predictions_by_chain, truth):
    return pd.concat(
        [
            validation_metrics(pred, truth, name)
            for name, pred in predictions_by_chain.items()
        ],
        ignore_index=True,
    )


def line_metrics(measured, predicted):
    from sklearn.metrics import r2_score

    mask = np.isfinite(measured) & np.isfinite(predicted)
    y, p = measured[mask], predicted[mask]
    if len(y) < 2:
        return f"Measured pairs: n = {len(y)}"
    error = p - y
    text = rf"$R^2={r2_score(y, p):.3f}$"
    if (y > -1).all() and (p > -1).all():
        text += rf",  $R^2_{{\log(1+y)}}={r2_score(np.log1p(y), np.log1p(p)):.3f}$"
    text += (
        "\n"
        + rf"$\mathrm{{RMSE}}={np.sqrt(np.mean(error**2)):.2f}$,  $\mathrm{{MAE}}={np.mean(np.abs(error)):.2f}$"
    )
    text += f"\nMeasured pairs: n = {len(y)}"
    return text


def individual_electrochemical_predictions(chains, truth):
    """Same saved electrochemical models, supplied measured upstream inputs."""
    individual = {}
    capacity_targets = {GRAVIMETRIC, VOLUMETRIC}
    for family, line in chains.items():
        model = line.models["Electrochemical"]
        required = list(dict.fromkeys(ALIASES.get(c, c) for c in model.columns))
        if capacity_targets.intersection(required):
            raise ValueError(
                "A capacity target must never be an electrochemical input."
            )
        missing = set(required) - set(truth.columns)
        if missing:
            raise ValueError(
                f"Individual {family} model lacks measured inputs: {sorted(missing)}"
            )
        individual[family] = model.predict_frame(truth.loc[:, required].copy())
        if not individual[family].index.equals(truth.index):
            raise ValueError(
                "Individual predictions changed the measurement row order."
            )
    return individual


def electrochemical_metric_values(measured, predicted):
    from sklearn.metrics import r2_score

    y, p = np.asarray(measured, float), np.asarray(predicted, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if not len(y):
        raise ValueError("No measured pairs for electrochemical comparison.")
    return {
        "n": len(y),
        "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "mae": float(np.mean(np.abs(p - y))),
    }


def electrochemical_chain_effects(predictions, individual, truth):
    """Positive error deltas indicate larger errors after chaining."""
    records = []
    for family, frame in predictions.items():
        if not frame.index.equals(truth.index):
            raise ValueError("Chain row order mismatch.")
        for output in [GRAVIMETRIC, VOLUMETRIC]:
            chain = electrochemical_metric_values(truth[output], frame[output])
            single = electrochemical_metric_values(
                truth[output], individual[family][output]
            )
            records.append(
                {
                    "model": family,
                    "output": output,
                    "n": chain["n"],
                    **{
                        f"chain_{key}": value
                        for key, value in chain.items()
                        if key != "n"
                    },
                    **{
                        f"individual_{key}": value
                        for key, value in single.items()
                        if key != "n"
                    },
                    "chain_minus_individual_r2": chain["r2"] - single["r2"],
                    "chain_minus_individual_mae": chain["mae"] - single["mae"],
                }
            )
    return pd.DataFrame(records)
