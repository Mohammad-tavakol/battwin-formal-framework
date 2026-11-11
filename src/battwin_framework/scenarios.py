"""Matched disturbance scenario from the notebook, with explicit configuration."""

from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import (
    CAL_THICKNESS,
    CAL_POROSITY,
    DRY_THICKNESS,
    MASS,
    GRAVIMETRIC,
    VOLUMETRIC,
    OUTPUT_COLUMNS,
    chain_inputs,
)
from .feedback import simulate_fixed_feedback_memory


@dataclass(frozen=True)
class DisturbanceConfig:
    row: int = 0
    kpi: str = CAL_THICKNESS
    target: float = 62.0
    tolerance: float = 0.5
    gain: float = -1.0
    initial_memory: float = 0.0
    items: int = 80
    disturbance_start: int = 26
    disturbance_column: str = "Calendaring Gap (um)"
    disturbance_size: float = 2.0


@dataclass
class DisturbanceResult:
    inputs: pd.DataFrame
    controlled: dict
    held: dict
    traces: dict
    histories: dict
    summary: pd.DataFrame
    config: DisturbanceConfig

    def save(self, directory, *, plots=False):
        path = Path(directory).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        self.inputs.to_csv(path / "identical_input_sequence.csv", index=False)
        self.summary.to_csv(path / "summary.csv", index=False)
        from .results import _json

        _json(path / "config.json", asdict(self.config))
        for family in self.controlled:
            destination = path / family
            destination.mkdir(exist_ok=True)
            self.controlled[family].to_csv(
                destination / "controlled_predictions.csv", index=False
            )
            self.held[family].to_csv(destination / "held_predictions.csv", index=False)
            self.traces[family].to_csv(destination / "trace.csv", index=False)
        if plots:
            self.plot(path)
        return path

    def plot(self, directory):
        from .plotting.feedback import disturbance_feedback_plot, feedback_capacity_plot
        from .plotting.output import output_directory

        with output_directory(directory) as path:
            for family in self.controlled:
                disturbance_feedback_plot(
                    self.traces[family],
                    family,
                    self.config.kpi,
                    self.config.disturbance_start,
                    self.config.tolerance,
                )
                feedback_capacity_plot(
                    self.controlled[family],
                    self.held[family],
                    family,
                    self.config.disturbance_start,
                )
        return path


def run_disturbance_scenario(
    framework, frame, reference, *, config, families=("RF", "DNN")
):
    """Include all post-disturbance items in means, including transients/invalid paths.

    Reference bounds are diagnostic only. Tolerance assesses settling; it never
    changes the law. The held branch freezes the correction already scheduled
    for the first disturbed item.
    """
    if not 1 < config.disturbance_start <= config.items:
        raise ValueError("Include a nominal phase before the disturbance.")
    base = chain_inputs(framework.chain(families[0]), frame).iloc[config.row].to_dict()
    inputs = pd.DataFrame([base] * config.items)
    step = config.disturbance_start - 1
    inputs.loc[step:, config.disturbance_column] += config.disturbance_size
    same = (reference["Electrode type"] == base["Electrode type"]) & (
        reference["Chemistry"] == base["Chemistry"]
    )
    gaps = reference.loc[same, "Coating Gap (micron)"]
    bounds = float(gaps.min()), float(gaps.max())
    controlled, held, traces, histories, summaries = {}, {}, {}, {}, []
    for family in families:
        run = framework.run_feedback(
            inputs,
            family=family,
            kpi_key=config.kpi,
            kpi_target=config.target,
            gain=config.gain,
            initial_memory=config.initial_memory,
        )
        values, trace = run.predictions, run.trace
        frozen = float(trace.memory_used.iloc[step])
        counterfactual = values.copy()
        post_values = simulate_fixed_feedback_memory(
            framework.chain(family),
            inputs.iloc[step:],
            {"CoatingDrying": {"delta_coating_gap": frozen}},
        )
        counterfactual.loc[post_values.index, OUTPUT_COLUMNS] = post_values[
            OUTPUT_COLUMNS
        ]
        trace["held_kpi"] = counterfactual[config.kpi].to_numpy()
        trace["held_coating_gap"] = trace.adjusted_coating_gap.copy()
        trace.loc[step:, "held_coating_gap"] = base["Coating Gap (micron)"] + frozen
        trace["calendering_gap"] = inputs[config.disturbance_column].to_numpy()
        trace["within_tolerance"] = trace.error.abs() <= config.tolerance
        trace["gap_within_reference_range"] = trace.adjusted_coating_gap.between(
            *bounds
        )
        trace["calendered_thinner_than_dry"] = (
            values[CAL_THICKNESS] < values[DRY_THICKNESS]
        )
        trace["roll_gap_less_than_dry_thickness"] = (
            inputs[config.disturbance_column] < values[DRY_THICKNESS]
        )
        trace["calendered_porosity_in_range"] = values[CAL_POROSITY].between(0, 100)
        trace["positive_mass_loading"] = values[MASS] > 0
        np.testing.assert_allclose(
            values.loc[step, OUTPUT_COLUMNS].to_numpy(float),
            counterfactual.loc[step, OUTPUT_COLUMNS].to_numpy(float),
            rtol=2e-5,
            atol=2e-4,
        )
        post = trace.iloc[step:]
        held_error = post.held_kpi - config.target
        settled = next(
            (
                int(post.item_index.iloc[j])
                for j in range(len(post))
                if post.within_tolerance.iloc[j:].all()
            ),
            None,
        )
        summaries.append(
            {
                "chain": family,
                "target_um": config.target,
                "tolerance_um": config.tolerance,
                "gain": config.gain,
                "disturbance_item": config.disturbance_start,
                "post_disturbance_mae_active_um": post.error.abs().mean(),
                "post_disturbance_mae_held_um": held_error.abs().mean(),
                "final_error_um": trace.error.iloc[-1],
                "settled_from_item": settled,
                "items_to_settle": (
                    settled - config.disturbance_start
                    if settled is not None
                    else np.nan
                ),
                "minimum_coating_gap_um": trace.adjusted_coating_gap.min(),
                "maximum_coating_gap_um": trace.adjusted_coating_gap.max(),
                "all_gaps_within_reference_range": bool(
                    trace.gap_within_reference_range.all()
                ),
                "final_gravimetric_active": values[GRAVIMETRIC].iloc[-1],
                "final_gravimetric_held": counterfactual[GRAVIMETRIC].iloc[-1],
                "final_volumetric_active": values[VOLUMETRIC].iloc[-1],
                "final_volumetric_held": counterfactual[VOLUMETRIC].iloc[-1],
            }
        )
        controlled[family], held[family] = values, counterfactual
        traces[family], histories[family] = trace, run.history
    return DisturbanceResult(
        inputs, controlled, held, traces, histories, pd.DataFrame(summaries), config
    )
