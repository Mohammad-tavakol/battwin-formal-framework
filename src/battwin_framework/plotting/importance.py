"""Original sensitivity, correlation and parametric visualisations."""

import numpy as np
import pandas as pd
from .style import CHART_COLORS, slug, _pyplot, figure_label, style_axes, save_figure
from .output import export_table, result_path


def importance_plot(
    table, value, output, family, xlabel, error=None, method_label=None
):
    plt = _pyplot()
    data = table.loc[table["output"] == output].sort_values(value)
    fig, ax = plt.subplots(figsize=(10.5, max(6.0, len(data) * 0.35)))
    color = "#2ca02c" if family == "RF" else "#d62728"
    positions = np.arange(len(data))
    bars = ax.barh(
        positions,
        data[value],
        xerr=data[error] if error else None,
        color=color,
        alpha=0.88,
        edgecolor="white",
        height=0.70,
        capsize=2.5,
        error_kw={"ecolor": "0.25", "elinewidth": 0.9},
    )
    values = data[value].to_numpy(float)
    uncertainties = data[error].to_numpy(float) if error else np.zeros(len(values))
    span = max(np.max(np.abs(values) + uncertainties), 1e-9)
    for bar, val, uncertainty in zip(bars, values, uncertainties):
        if val < 0:
            bar.set_hatch("//")
        ax.text(
            val + np.sign(val if val else 1) * (uncertainty + span * 0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3g}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=11,
        )
    ax.set_xlim(
        min(0, np.min(values - uncertainties)) - 0.13 * span,
        max(0, np.max(values + uncertainties)) + 0.21 * span,
    )
    ax.axvline(0, color="0.2", lw=1.0)
    method = (
        "Permutation feature importance"
        if family == "RF"
        else "Forward finite-difference sensitivity"
    )
    method_label = method_label or ("PFI" if family == "RF" else "Finite difference")
    ax.set_xlabel(f"{method_label} {figure_label(output)}", fontsize=16)
    style_axes(ax, grid="x")
    # Override the shared numeric locator: every feature needs its own tick.
    ax.set_yticks(positions, data["feature"].astype(str).tolist())
    from matplotlib.ticker import NullLocator

    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.65, len(data) - 0.35)
    ax.tick_params(axis="y", labelsize=12)
    if error:
        repeats = int(data.n_repeats.iloc[0])
        ax.text(
            0.99,
            0.015,
            f"Mean ± SD over {repeats} permutations",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
        )
    fig.text(0.5, 0.008, xlabel, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    path = save_figure(fig, f"importance/{family}/{slug(output)}.png")
    plt.close(fig)
    return path


def correlation_plot(correlation, family, method):
    if correlation.empty:
        return None
    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap

    n = len(correlation)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.68), max(9, n * 0.62)))
    cmap = LinearSegmentedColormap.from_list(
        "balanced", ["#1f77b4", "#f7f7f7", "#d62728"]
    )
    im = ax.imshow(correlation, vmin=-1, vmax=1, cmap=cmap)
    labels = [figure_label(column) for column in correlation.columns]
    ax.set_xticks(range(n), labels, rotation=55, ha="right", fontsize=11)
    ax.set_yticks(range(n), labels, fontsize=11)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    if n <= 18:
        for i in range(n):
            for j in range(n):
                val = correlation.iloc[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if abs(val) > 0.65 else "0.15",
                    )
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation coefficient", fontsize=16)
    colorbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    path = save_figure(fig, f"parametric/{family}/{method}.png")
    plt.close(fig)
    return path


def parallel_coordinates(results, columns, color, family, name="parallel_coordinates"):
    import plotly.express as px

    fig = px.parallel_coordinates(
        results,
        dimensions=columns,
        color=color,
        labels={col: col.replace(" ", "<br>") for col in columns},
        color_continuous_scale=["#1f77b4", "#f7f7f7", "#d62728"],
    )
    fig.update_layout(
        font=dict(family="Times New Roman", size=13),
        height=620,
        paper_bgcolor="white",
        margin=dict(l=75, r=75, t=100, b=70),
    )
    path = result_path(f"parametric/{family}/{name}.html")
    fig.write_html(path, include_plotlyjs=True, auto_open=False)
    return path


def feedback_plot(trace, family):
    plt = _pyplot()
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True)
    color = "#2ca02c" if family == "RF" else "#d62728"
    axes[0].plot(
        trace.item_index,
        trace.kpi_value,
        color=color,
        lw=1.9,
        marker="o",
        ms=3,
        label=f"{family} chain",
    )
    axes[0].plot(
        trace.item_index, trace.kpi_target, "--", color="0.2", lw=1.5, label="Target"
    )
    axes[0].set_ylabel("Controlled KPI")
    axes[0].legend(loc="best", frameon=True)
    axes[1].plot(trace.item_index, trace.error, color=color, lw=1.9, marker="o", ms=3)
    axes[1].axhline(0, color="0.25", lw=1.0, ls="--")
    axes[1].set_ylabel("KPI − target")
    axes[2].plot(
        trace.item_index,
        trace.adjusted_coating_gap,
        color=color,
        lw=1.9,
        marker="o",
        ms=3,
    )
    axes[2].set(xlabel="sample No.", ylabel=r"Coating gap ($\mu$m)")
    for i, ax in enumerate(axes):
        style_axes(ax)
        ax.text(
            0.015, 0.95, f"({chr(97+i)})", transform=ax.transAxes, va="top", fontsize=16
        )
    fig.tight_layout()
    path = save_figure(fig, f"feedback/{family}/trajectory.png")
    plt.close(fig)
    return path
