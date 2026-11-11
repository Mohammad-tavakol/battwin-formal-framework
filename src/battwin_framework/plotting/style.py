"""Original chart style and explicit output routing (no notebook globals)."""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from .output import result_path

CHART_COLORS = {"Measured": "#1f77b4", "RF": "#2ca02c", "DNN": "#d62728"}


def slug(text):
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.labelsize": 19,
            "axes.titlesize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "axes.linewidth": 1.35,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def figure_label(column):
    labels = {
        "Gravimetric capacity (mA h/g)": r"Gravimetric capacity (mA h g$^{-1}$)",
        "Volumetric capacity (mA h/cm3)": r"Volumetric capacity (mA h cm$^{-3}$)",
        "Viscosity (Pa.s)": r"Viscosity (Pa s)",
        "Mass loading dry (mg/cm2)": r"Dry mass loading (mg cm$^{-2}$)",
        "Dry Coating Thickness (µm)": r"Dry coating thickness ($\mu$m)",
        "Calendered thickness (um)": r"Calendered thickness ($\mu$m)",
        "Calendered porosity (%)": "Calendered porosity (%)",
        "Coating Gap (micron)": r"Coating gap ($\mu$m)",
        "Calendaring Gap (um)": r"Calendering gap ($\mu$m)",
        "Drying temp-Zone2": "Zone 2 drying temperature",
        "Discharge rate (*C)": "Discharge rate (C)",
    }
    return labels.get(column, column)


def style_axes(ax, grid="both"):
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator

    ax.set_axisbelow(True)
    ax.grid(True, axis=grid, which="major", color="0.83", linestyle=":", linewidth=0.9)
    ax.tick_params(which="both", direction="out", top=False, right=False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    for spine in ax.spines.values():
        spine.set_color("0.22")


def save_figure(fig, relative_path):
    """300-dpi PNG for the notebook, vector PDF for manuscripts."""
    path = result_path(relative_path)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    return path
