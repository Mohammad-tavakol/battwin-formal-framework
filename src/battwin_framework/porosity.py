"""Geometry diagnostic, separate from the learned porosity prediction."""

import numpy as np
import pandas as pd
from .schema import MASS, CAL_THICKNESS

ELECTRODE_DENSITIES = {"Cathode": 4.54, "Anode": 2.26}

FOIL_THICKNESS_UM = 0.0

CALCULATED_POROSITY = "Calculated calendered porosity (%)"


def geometry_porosity(frame):
    """Use total dry coating mass, coating thickness, and the electrode-type reference."""
    names = frame["Electrode type"].astype(str).str.strip().str.lower()
    mapping = {k.lower(): v for k, v in ELECTRODE_DENSITIES.items()}
    solid_density = names.map(mapping)
    if solid_density.isna().any():
        raise ValueError(
            f"Missing electrode density for {names[solid_density.isna()].unique().tolist()}"
        )
    if not np.isfinite(solid_density).all() or solid_density.le(0).any():
        raise ValueError("Electrode density references must be finite and positive.")
    mass = pd.to_numeric(frame[MASS], errors="coerce")
    thickness = pd.to_numeric(frame[CAL_THICKNESS], errors="coerce") - FOIL_THICKNESS_UM
    valid = np.isfinite(mass) & np.isfinite(thickness) & mass.gt(0) & thickness.gt(0)
    bulk = (10.0 * mass / thickness).where(valid)
    value = 100.0 * (1.0 - bulk / solid_density)
    return pd.DataFrame(
        {
            "Solid density reference (g/cm3)": solid_density,
            "Bulk coating density (g/cm3)": bulk,
            CALCULATED_POROSITY: value,
            "Geometry inputs valid": valid,
            "Calculated porosity in range": valid & value.between(0, 100),
        },
        index=frame.index,
    )
