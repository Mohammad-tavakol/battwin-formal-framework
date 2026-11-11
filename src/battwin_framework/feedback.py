"""Original additive controller: error = prediction - target, next = used + gain*error.

There is no clipping, saturation, deadband or anti-windup. Updates affect the
next item. Only the CoatingDrying stage receives the adjusted coating gap.
"""

import numpy as np
import pandas as pd
from .schema import FEEDBACK_INPUTS, chain_inputs
from .simulation import simulate_with_trace


def run_feedback(
    line,
    frame,
    *,
    kpi_key,
    kpi_target,
    gain,
    stage="CoatingDrying",
    memory_key="delta_coating_gap",
    initial_memory=0.0,
):
    if stage not in FEEDBACK_INPUTS or memory_key not in FEEDBACK_INPUTS[stage]:
        raise ValueError(f"No feedback actuator {stage}.{memory_key}")
    if not np.isfinite([kpi_target, gain, initial_memory]).all():
        raise ValueError("Feedback target, gain and initial memory must be finite")
    if kpi_key not in [c for s in line.chain_def for c in s["output_columns"]]:
        raise ValueError(f"Unknown chain-output KPI: {kpi_key}")
    results, history = simulate_with_trace(
        line,
        frame,
        use_feedback=True,
        zeta_init={stage: {memory_key: initial_memory}},
        kpi_key=kpi_key,
        kpi_target=kpi_target,
        feedback_gains={stage: {memory_key: gain}},
        feedback_stage_name=stage,
    )
    trace = pd.DataFrame(
        [
            {
                "item_index": r["item_index"],
                "kpi_value": r["kpi_value"],
                "kpi_target": r["kpi_target"],
                "error": r["error"],
                "memory_used": r["zeta_used"][stage][memory_key],
                "memory_next": r["zeta_next"][stage][memory_key],
                "adjusted_coating_gap": float(frame.iloc[i]["Coating Gap (micron)"])
                + r["zeta_used"][stage][memory_key],
            }
            for i, r in enumerate(history)
        ],
        index=frame.index,
    )
    return results, trace, history


def simulate_fixed_feedback_memory(line, frame, memory):
    """Same stage maps and feature writes, with memory held fixed for all items."""
    features = chain_inputs(line, frame)
    for stage in line.chain_def:
        output = stage["func"].predict_frame(features, memory.get(stage["name"], {}))
        for column in output:
            features[column] = output[column]
    return features
