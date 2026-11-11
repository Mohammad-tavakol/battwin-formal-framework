"""Canonical physical-unit column names and stage contracts."""

import pandas as pd

VISCOSITY = "Viscosity (Pa.s)"

MASS = "Mass loading dry (mg/cm2)"

DRY_THICKNESS = "Dry Coating Thickness (\u00b5m)"

DRY_POROSITY = "Dried Porosity"

CAL_THICKNESS = "Calendered thickness (um)"

CAL_POROSITY = "Calendered porosity (%)"

GRAVIMETRIC = "Gravimetric capacity (mA h/g)"

VOLUMETRIC = "Volumetric capacity (mA h/cm3)"

STAGE_OUTPUTS = {
    "Mixing": [VISCOSITY],
    "CoatingDrying": [MASS, DRY_THICKNESS, DRY_POROSITY],
    "Calendering": [CAL_THICKNESS, CAL_POROSITY],
    "Electrochemical": [GRAVIMETRIC, VOLUMETRIC],
}

OUTPUT_COLUMNS = [c for outputs in STAGE_OUTPUTS.values() for c in outputs]

MEASURED_OUTPUTS = [MASS, CAL_THICKNESS, CAL_POROSITY, GRAVIMETRIC, VOLUMETRIC]

CATEGORICAL_COLUMNS = [
    "Electrode type",
    "Chemistry",
    "Conductive additive type",
    "Binder1 type",
    "Binder2 (Thickener) type",
    "Solvent",
    "Dispersent type",
]

ALIASES = {"slurry density g/cm3": "Slurry density g/cm3"}

FEEDBACK_INPUTS = {"CoatingDrying": {"delta_coating_gap": "Coating Gap (micron)"}}


def chain_inputs(line, frame):
    """Exclude measured intermediates/KPIs so they cannot bypass a stage."""
    required = line.get_required_input_columns()
    leaked = set(required).intersection(OUTPUT_COLUMNS)
    if leaked:
        raise ValueError(
            f"Stage outputs were incorrectly declared as base inputs: {leaked}"
        )
    line.validate_input_dataframe(frame)
    return frame.loc[:, required].copy()
