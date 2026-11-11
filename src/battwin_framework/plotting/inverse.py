"""Original static and interactive target-path layouts with explicit targets."""

import numpy as np
import pandas as pd
from ..schema import (
    VISCOSITY,
    MASS,
    DRY_THICKNESS,
    CAL_THICKNESS,
    DRY_POROSITY,
    CAL_POROSITY,
    GRAVIMETRIC,
    VOLUMETRIC,
)
from ..porosity import CALCULATED_POROSITY
from .style import CHART_COLORS, _pyplot, save_figure
from .output import result_path


def path_axis_specs(tables, columns, *, targets, tolerances):
    """Common physical ranges across both families, including the whole grey background."""
    combined = pd.concat(list(tables.values()), ignore_index=True)
    specs = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(combined[col]):
            finite = combined[col][np.isfinite(combined[col])]
            if finite.empty:
                raise ValueError(f"No finite values to plot: {col}")
            lo, hi = (float(finite.min()), float(finite.max()))
            if lo == hi:
                pad = max(abs(lo) * 0.02, 0.5)
                lo, hi = (lo - pad, hi + pad)
            if col in targets:
                lo = min(lo, targets[col] - tolerances[col])
                hi = max(hi, targets[col] + tolerances[col])
            ticks = np.linspace(lo, hi, 3)
            specs[col] = (
                lo,
                hi,
                None,
                ticks,
                [f"{t:.3g}" if abs(t) < 10 else f"{t:.1f}" for t in ticks],
            )
        else:
            names = sorted(combined[col].astype(str).unique())
            mapping = {name: i for i, name in enumerate(names)}
            specs[col] = (
                -0.25,
                max(len(names) - 0.75, 0.75),
                mapping,
                list(mapping.values()),
                names,
            )
    return specs


def plot_target_paths(data, selected, family, columns, specs, *, targets, tolerances):
    import plotly.graph_objects as go
    import textwrap

    order = np.r_[
        np.flatnonzero(~selected.to_numpy()), np.flatnonzero(selected.to_numpy())
    ]
    ordered, chosen = (data.iloc[order], selected.iloc[order])
    dimensions, normalized = ([], [])
    labels = {
        "Chemistry": "Chemistry",
        "Coating Gap (micron)": "Coating gap (µm)",
        "Am/total weight%": "Active material (%)",
        "Coating Speed (m/min)": "Coater speed (m/min)",
        "Calendaring Gap (um)": "Roll gap (µm)",
        "Discharge rate (*C)": "Discharge rate (C)",
        VISCOSITY: "Viscosity (Pa s)",
        MASS: "Dry mass loading (mg/cm²)",
        DRY_THICKNESS: "Dry thickness (µm)",
        CAL_THICKNESS: "Calendered thickness (µm)",
        CAL_POROSITY: "Calendered porosity (model) (%)",
        CALCULATED_POROSITY: "Calendered porosity (geometry) (%)",
        DRY_POROSITY: "Dried porosity (%)",
        GRAVIMETRIC: "Gravimetric capacity (mAh/g)",
        VOLUMETRIC: "Volumetric capacity (mAh/cm³)",
    }
    for col in columns:
        lo, hi, mapping, ticks, texts = specs[col]
        values = (
            ordered[col].astype(str).map(mapping).to_numpy(float)
            if mapping
            else ordered[col].to_numpy(float)
        )
        normalized.append((values - lo) / (hi - lo))
        dimension = dict(
            label=labels.get(col, col),
            values=values,
            range=[lo, hi],
            tickvals=list(ticks),
            ticktext=texts,
        )
        if col in targets:
            dimension["label"] += f" (target {targets[col]:g} ± {tolerances[col]:g})"
        dimensions.append(dimension)
    title = f"{family}: {int(selected.sum())} matching evaluations / {len(data)} tested paths"
    fig = go.Figure(
        go.Parcoords(
            dimensions=dimensions,
            line=dict(
                color=chosen.astype(int),
                cmin=0,
                cmax=1,
                showscale=False,
                colorscale=[
                    [0, "#c9c9c9"],
                    [0.499, "#c9c9c9"],
                    [0.5, CHART_COLORS[family]],
                    [1, CHART_COLORS[family]],
                ],
            ),
            labelfont=dict(family="Times New Roman", size=12),
            tickfont=dict(size=11),
        )
    )
    fig.update_layout(
        height=680,
        width=max(1300, len(columns) * 110),
        margin=dict(l=85, r=85, t=150, b=60),
        paper_bgcolor="white",
    )
    html = result_path(f"inverse/{family}/target_paths.html")
    fig.write_html(html, include_plotlyjs=True, auto_open=False)
    plt = _pyplot()
    matrix = np.array(normalized).T
    fig, ax = plt.subplots(figsize=(max(13, len(columns) * 1.2), 7))
    x = np.arange(len(columns))
    for row, hit in zip(matrix, chosen):
        ax.plot(
            x,
            row,
            color=CHART_COLORS[family] if hit else ".75",
            alpha=0.8 if hit else 0.18,
            lw=1.5 if hit else 0.65,
            zorder=3 if hit else 1,
        )
    for i, col in enumerate(columns):
        lo, hi, _, ticks, texts = specs[col]
        if col in targets:
            a = (targets[col] - tolerances[col] - lo) / (hi - lo)
            b = (targets[col] + tolerances[col] - lo) / (hi - lo)
            ax.fill_between(
                [i - 0.065, i + 0.065],
                a,
                b,
                color=CHART_COLORS["Measured"],
                alpha=0.2,
                zorder=2,
            )
            ax.text(
                i,
                1.035,
                f"Target {targets[col]:g} ± {tolerances[col]:g}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=".2",
            )
        ax.vlines(i, 0, 1, color=".4", lw=0.85, zorder=4)
        for tick, text in zip(ticks, texts):
            y = (tick - lo) / (hi - lo)
            ax.plot([i - 0.025, i + 0.025], [y, y], color=".4", lw=0.8, zorder=4)
            ax.text(
                i + 0.045,
                y,
                text,
                fontsize=9,
                ha="left",
                va="center",
                zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2),
            )
    ax.set_xticks(
        x,
        ["\n".join(textwrap.wrap(labels.get(c, c), 19)) for c in columns],
        fontsize=11,
    )
    ax.set_xlim(-0.15, len(columns) - 0.6)
    ax.set_ylim(-0.04, 1.07)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    caption = f"{family}: {int(selected.sum())} matches among {len(data)} evaluations. Shared physical axis ranges. Grey: all tests; colored: target matches; blue marks: target intervals."
    if not selected.any():
        caption += " No tested path satisfies every target condition."
    fig.text(0.5, 0.02, caption, ha="center", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    png = save_figure(fig, f"inverse/{family}/target_paths.png")
    plt.close(fig)
    return (html, png)
