"""Cartesian sweeps, original recipe helper, correlations and tolerance filters."""

from itertools import product
import numpy as np
import pandas as pd
from .schema import MASS, DRY_THICKNESS, CAL_THICKNESS, DRY_POROSITY, CAL_POROSITY
from .simulation import simulate_without_feedback


def parametric_inputs(base_row, ranges, *, derived=None, max_items=100_000):
    if not ranges or any(len(values) == 0 for values in ranges.values()):
        raise ValueError("Supply at least one nonempty parameter range")
    count = int(np.prod([len(v) for v in ranges.values()], dtype=object))
    if count > max_items:
        raise ValueError(f"Sweep has {count} items; max_items={max_items}")
    records = []
    for values in product(*ranges.values()):
        row = dict(base_row)
        row.update(zip(ranges, values))
        if derived is not None:
            row = derived(row)
        records.append(row)
    return pd.DataFrame(records)


def original_anode_recipe(row):
    """Original parametric notebook's shear, composition and density formulas.

    Explicitly opt in only for the GRA/SBR/CMC/water recipe used in that notebook.
    Does not run inside orchestration or importance calculations.
    """
    row = dict(row)
    if (
        row["Electrode type"],
        row["Chemistry"],
        row["Binder1 type"],
        row["Binder2 (Thickener) type"],
        row["Solvent"],
    ) != ("Anode", "GRA", "SBR", "CMC", "water"):
        raise ValueError("Original parametric recipe is for GRA/SBR/CMC/water anodes")
    sc = float(row["Solid Content (%)"])
    gap = float(row["Coating Gap (micron)"])
    if gap <= 0:
        raise ValueError("Coating gap must be positive")
    row["Shear rate"] = (float(row["Coating Speed (m/min)"]) / 60) / (gap / 1e6)
    row["Am/total weight%"] = 0.94 * sc
    row["Conductive additive/total weight %"] = 0.01 * sc
    row["Binder1/total weight %"] = 0.03 * sc
    row["Binder2 (thickener)/total weight %"] = 0.02 * sc
    row["Slurry density g/cm3"] = (
        row["Am/total weight%"] * 2.26 / 100
        + row["Conductive additive/total weight %"] * 1.6 / 100
        + row["Binder1/total weight %"] * 1.3 / 100
        + row["Binder2 (thickener)/total weight %"] * 1.6 / 100
        + (100 - sc * 1) / 100
    )
    return row


def run_parametric(line, inputs, batch_size=2048):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parts = [
        simulate_without_feedback(line, inputs.iloc[i : i + batch_size])
        for i in range(0, len(inputs), batch_size)
    ]
    return pd.concat(parts) if parts else simulate_without_feedback(line, inputs)


def physical_checks(results):
    """Flag observed physical violations; never delete rows or alter model outputs.

    Wet-coating thickness no longer exists. Dried porosity's units are not encoded
    in its saved target name, so avoid a guessed percentage/fraction upper bound.
    """
    checks = pd.DataFrame(index=results.index)
    for name in [MASS, DRY_THICKNESS, CAL_THICKNESS]:
        checks[f"{name}: positive"] = results[name].gt(0) & np.isfinite(results[name])
    checks["calendered_thinner_than_dry"] = (
        results[CAL_THICKNESS] < results[DRY_THICKNESS]
    )
    checks["roll_gap_less_than_dry_thickness"] = (
        results["Calendaring Gap (um)"] < results[DRY_THICKNESS]
    )
    checks["calendered_porosity_in_percent_range"] = results[CAL_POROSITY].between(
        0, 100
    )
    checks["dry_porosity_nonnegative_finite"] = results[DRY_POROSITY].ge(
        0
    ) & np.isfinite(results[DRY_POROSITY])
    checks["all_checks"] = checks.all(axis=1)
    return checks


def numeric_correlations(results, method="pearson", drop_columns=()):
    if method not in {"pearson", "spearman"}:
        raise ValueError("Correlation method must be pearson or spearman")
    data = results.drop(columns=list(drop_columns), errors="ignore").select_dtypes(
        "number"
    )
    data = data.loc[:, data.nunique(dropna=False) > 1].dropna()
    return data.corr(method=method)


def filter_tolerances(results, bounds):
    mask = pd.Series(True, index=results.index)
    for column, (lower, upper) in bounds.items():
        if lower > upper:
            raise ValueError(f"Invalid tolerance for {column}")
        mask &= results[column].between(lower, upper)
    return results.loc[mask].copy()
