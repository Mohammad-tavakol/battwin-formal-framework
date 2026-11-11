"""Independent recipe-centred random proposals and strict physical screening.

This is the notebook's finite random search, not an adaptive optimiser.
"""

import numpy as np
import pandas as pd
from .schema import (
    MASS,
    DRY_THICKNESS,
    CAL_THICKNESS,
    DRY_POROSITY,
    CAL_POROSITY,
    GRAVIMETRIC,
    VOLUMETRIC,
    chain_inputs,
)
from .porosity import CALCULATED_POROSITY, geometry_porosity
from .simulation import simulate_without_feedback

SEARCH_PROCESS_COLUMNS = [
    "Coating Gap (micron)",
    "Coating Speed (m/min)",
    "Coating ratio (%)",
    "Drying temp-Zone1",
    "Drying temp-Zone2",
    "Calendering Roll speed(m/min)",
    "Calendaring Roll temp",
    "Calendaring Gap (um)",
    "Discharge rate (*C)",
]


def sample_inverse_candidates(
    training,
    *,
    rf_line,
    dnn_line,
    n_sweeps=1000,
    random_seed=42,
    local_scale_fraction=0.1,
):
    """Independent, balanced recipe-centred proposals; targets do not guide draws."""
    from scipy.stats import truncnorm

    if n_sweeps < 2 or n_sweeps % 2:
        raise ValueError("Use an even candidate budget to balance cathodes and anodes.")
    if not np.isfinite(local_scale_fraction) or local_scale_fraction <= 0:
        raise ValueError("The local sampling scale must be positive and finite.")
    base = chain_inputs(rf_line, training).reset_index(drop=True)
    dnn_line.validate_input_dataframe(base)
    rng = np.random.Generator(np.random.PCG64(random_seed))
    blocks, bounds = ([], [])
    for electrode in ["Cathode", "Anode"]:
        indices = np.flatnonzero(base["Electrode type"].eq(electrode))
        if not len(indices):
            raise ValueError(f"No training recipes for {electrode}.")
        selected = rng.choice(indices, size=n_sweeps // 2, replace=True)
        sampled = base.iloc[selected].copy().reset_index(drop=True)
        for column in SEARCH_PROCESS_COLUMNS:
            values = pd.to_numeric(base.iloc[indices][column], errors="raise")
            low, high = (float(values.min()), float(values.max()))
            sigma = local_scale_fraction * (high - low)
            if sigma:
                centre = sampled[column].to_numpy(float)
                sampled[column] = truncnorm.rvs(
                    (low - centre) / sigma,
                    (high - centre) / sigma,
                    loc=centre,
                    scale=sigma,
                    random_state=rng,
                )
            else:
                sampled[column] = low
            bounds.append(
                {
                    "electrode_type": electrode,
                    "feature": column,
                    "lower": low,
                    "upper": high,
                    "sigma": sigma,
                    "fixed_within_polarity": not bool(sigma),
                }
            )
        sampled["Training anchor row"] = selected + 1
        blocks.append(sampled)
    candidates = (
        pd.concat(blocks, ignore_index=True)
        .iloc[rng.permutation(n_sweeps)]
        .reset_index(drop=True)
    )
    candidates.insert(0, "Candidate ID", [f"S{i + 1:04d}" for i in range(n_sweeps)])
    if candidates.duplicated(list(base.columns)).any():
        raise RuntimeError("Random proposals contain duplicate input settings.")
    existing = set(base.itertuples(index=False, name=None))
    if any(
        (
            row in existing
            for row in candidates[list(base.columns)].itertuples(index=False, name=None)
        )
    ):
        raise RuntimeError("A proposal repeats an original training setting exactly.")
    fixed = [c for c in base if c not in SEARCH_PROCESS_COLUMNS]
    original = base.iloc[
        candidates["Training anchor row"].to_numpy(int) - 1
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(candidates[fixed], original[fixed], check_dtype=False)
    description = {
        "algorithm": "independent local random search",
        "new_candidate_budget": n_sweeps,
        "random_seed": random_seed,
        "bit_generator": "PCG64",
        "training_rows": len(base),
        "per_electrode_type": n_sweeps // 2,
        "anchor_sampling": "uniform with replacement within electrode type",
        "process_sampling": "independent normal perturbations about each anchor, truncated to polarity-specific observed training ranges",
        "local_scale_fraction": local_scale_fraction,
        "sampling_bounds": bounds,
        "varied_process_columns": SEARCH_PROCESS_COLUMNS,
        "copied_recipe_columns": fixed,
        "candidate_draws_independent_of_targets": True,
        "new_unique_settings_verified": True,
        "feedback_used": False,
        "models_refitted": False,
        "discharge_rate_fixed": False,
        "additional_user_input_constraints": {},
        "training_data_role": "sampling anchors only; not counted as new candidate evaluations",
    }
    return (candidates, description)


def inverse_search_physical_checks(predictions):
    """Exactly the requested physical screens, plus positive applied coating gap."""
    checks = pd.DataFrame(index=predictions.index)
    mass, dry_t, cal_t = MASS, DRY_THICKNESS, CAL_THICKNESS
    dry_p, cal_p = (DRY_POROSITY, CAL_POROSITY)
    for column in [mass, dry_t, cal_t]:
        checks[f"{column}: positive finite"] = np.isfinite(
            predictions[column]
        ) & predictions[column].gt(0)
    checks["calendered_thickness_below_dried"] = predictions[cal_t].lt(
        predictions[dry_t]
    )
    checks["roll_gap_below_dried_thickness"] = np.isfinite(
        predictions["Calendaring Gap (um)"]
    ) & predictions["Calendaring Gap (um)"].lt(predictions[dry_t])
    checks["predicted_calendered_porosity_in_0_100"] = np.isfinite(
        predictions[cal_p]
    ) & predictions[cal_p].between(0, 100)
    calculated = predictions[CALCULATED_POROSITY]
    checks["calculated_calendered_porosity_in_0_100"] = np.isfinite(
        calculated
    ) & calculated.between(0, 100)
    checks["dried_porosity_nonnegative_finite"] = np.isfinite(
        predictions[dry_p]
    ) & predictions[dry_p].ge(0)
    checks["predicted_calendered_porosity_below_dried"] = predictions[cal_p].lt(
        predictions[dry_p]
    )
    checks["calculated_calendered_porosity_below_dried"] = calculated.lt(
        predictions[dry_p]
    )
    checks["positive_applied_coating_gap"] = np.isfinite(
        predictions["Applied coating gap (µm)"]
    ) & predictions["Applied coating gap (µm)"].gt(0)
    checks["Physical pass"] = checks.all(axis=1)
    return checks


def evaluate_inverse_candidates(
    candidates, chains, *, targets, tolerance_fraction=0.02
):
    if set(targets) != {GRAVIMETRIC, VOLUMETRIC}:
        raise ValueError("Specify both electrochemical capacity targets.")
    if (
        not np.isfinite(tolerance_fraction)
        or tolerance_fraction <= 0
        or (not all((np.isfinite(v) and v > 0 for v in targets.values())))
    ):
        raise ValueError("Capacity targets and tolerance must be positive and finite.")
    tolerance = {key: abs(value) * tolerance_fraction for key, value in targets.items()}
    results, summaries = ({}, [])
    for family, line in chains.items():
        data = simulate_without_feedback(line, candidates).reset_index(drop=True)
        data.insert(0, "Candidate ID", candidates["Candidate ID"].to_numpy())
        data["Training anchor row"] = candidates["Training anchor row"].to_numpy()
        data["Source"] = "random search"
        data["Source row"] = np.arange(1, len(data) + 1)
        data["Path ID"] = [f"{family}-{x}" for x in data["Candidate ID"]]
        data["Feedback correction (µm)"] = 0.0
        data["Applied coating gap (µm)"] = data["Coating Gap (micron)"]
        data[CALCULATED_POROSITY] = geometry_porosity(data)[CALCULATED_POROSITY]
        checks = inverse_search_physical_checks(data)
        scaled = pd.DataFrame(
            {c: (data[c] - v).abs() / tolerance[c] for c, v in targets.items()}
        )
        capacity_pass = scaled.le(1).all(axis=1) & np.isfinite(data[list(targets)]).all(
            axis=1
        )
        data["Capacity targets pass"] = capacity_pass
        data["Physical pass"] = checks["Physical pass"]
        data["Target match"] = capacity_pass & data["Physical pass"]
        data["Maximum tolerance-normalized target distance"] = scaled.max(axis=1)
        data["Failed physical checks"] = [
            "; ".join(checks.columns[:-1][~row[:-1].to_numpy(bool)])
            for _, row in checks.iterrows()
        ]
        matches = data.loc[data["Target match"]].copy()
        nearest = (
            data.loc[data["Physical pass"] & ~data["Target match"]]
            .sort_values("Maximum tolerance-normalized target distance", kind="stable")
            .groupby("Electrode type", sort=False)
            .head(5)
            .copy()
        )
        results[family] = {
            "all": data,
            "matches": matches,
            "nearest": nearest,
            "checks": checks,
        }
        for electrode, g in data.groupby("Electrode type", sort=False):
            summaries.append(
                {
                    "model": family,
                    "electrode_type": electrode,
                    "new_candidates": len(g),
                    "physical_pass": int(g["Physical pass"].sum()),
                    "physical_fail": int((~g["Physical pass"]).sum()),
                    "capacity_targets_pass_before_physical": int(
                        g["Capacity targets pass"].sum()
                    ),
                    "target_matches_rejected_by_physical": int(
                        (g["Capacity targets pass"] & ~g["Physical pass"]).sum()
                    ),
                    "accepted_target_matches": int(g["Target match"].sum()),
                    "best_physical_target_distance": float(
                        g.loc[
                            g["Physical pass"],
                            "Maximum tolerance-normalized target distance",
                        ].min()
                    ),
                }
            )
    return (results, pd.DataFrame(summaries), tolerance)


def verify_inverse_physical_screens():
    """Counterexamples ensure the new ordering tests reject otherwise valid paths."""
    row = {
        MASS: 10.0,
        DRY_THICKNESS: 100.0,
        CAL_THICKNESS: 70.0,
        "Calendaring Gap (um)": 60.0,
        DRY_POROSITY: 50.0,
        CAL_POROSITY: 30.0,
        CALCULATED_POROSITY: 35.0,
        "Applied coating gap (µm)": 120.0,
    }
    good = pd.DataFrame([row])
    assert inverse_search_physical_checks(good)["Physical pass"].iloc[0]
    bad_cases = [
        (MASS, 0.0),
        (DRY_THICKNESS, np.nan),
        (CAL_THICKNESS, 100.0),
        ("Calendaring Gap (um)", 100.0),
        (CAL_POROSITY, 101.0),
        (DRY_POROSITY, -1.0),
        (CAL_POROSITY, 50.0),
        (CALCULATED_POROSITY, 50.0),
        (CALCULATED_POROSITY, np.inf),
        ("Applied coating gap (µm)", 0.0),
    ]
    for column, value in bad_cases:
        bad = good.copy()
        bad[column] = value
        assert not inverse_search_physical_checks(bad)["Physical pass"].iloc[0], (
            column,
            value,
        )
    return {"valid_case_passed": True, "invalid_cases_rejected": len(bad_cases)}
