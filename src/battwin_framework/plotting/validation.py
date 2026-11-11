"""Current measured/chain/individual charts, with R2 and MAE legends."""

import numpy as np
import pandas as pd
from ..schema import GRAVIMETRIC, VOLUMETRIC
from ..validation import line_metrics, electrochemical_metric_values
from .style import CHART_COLORS, slug, _pyplot, figure_label, style_axes, save_figure
from .output import export_table


def validation_plots(predictions, truth, outputs, family):
    return [
        execution_line_plot({family: predictions}, truth, output, name=family)
        for output in outputs
        if output in truth
    ]


def execution_line_plot(
    predictions, truth, output, name="comparison", *, individual_predictions=None
):
    """Identical stable measured-value ordering for every prediction series."""
    plt = _pyplot()
    y = truth[output].to_numpy(dtype=float)
    order = np.argsort(np.where(np.isfinite(y), y, np.inf), kind="stable")
    y = y[order]
    x = np.arange(1, len(y) + 1)
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    colors = {"RF": "#2ca02c", "DNN": "#d62728"}
    ax.plot(x, y, color="#1f77b4", lw=1.65, label="Measured", zorder=5)
    extrema = list(y[np.isfinite(y)])
    for family, frame in predictions.items():
        if not frame.index.equals(truth.index):
            raise ValueError("Chain row order mismatch.")
        p = frame[output].to_numpy(dtype=float)[order]
        m = electrochemical_metric_values(y, p)
        chain_label = f"{family} chain" + rf" ($R^2$={m['r2']:.3f}, MAE={m['mae']:.2f})"
        ax.plot(
            x,
            p,
            color=colors.get(family, "#1b7837"),
            lw=1.85,
            ls="--",
            label=chain_label,
            zorder=2,
        )
        extrema.extend(p[np.isfinite(p)])
        if individual_predictions is not None:
            data = individual_predictions[family]
            if not data.index.equals(truth.index):
                raise ValueError("Individual model row order mismatch.")
            single = data[output].to_numpy(dtype=float)[order]
            m = electrochemical_metric_values(y, single)
            individual_label = (
                f"{family} individual" + rf" ($R^2$={m['r2']:.3f}, MAE={m['mae']:.2f})"
            )
            ax.plot(
                x,
                single,
                color=colors.get(family, "#1b7837"),
                lw=1.85,
                ls="-",
                label=individual_label,
                zorder=3,
            )
            extrema.extend(single[np.isfinite(single)])
    lower, upper = min(extrema), max(extrema)
    span = max(upper - lower, abs(upper) * 0.05, 1)
    ax.set_ylim(lower - 0.025 * span, upper + 0.025 * span)
    ax.set_xlim(0.5, len(y) + 0.5)
    ax.set(xlabel="sample No.", ylabel=figure_label(output))
    style_axes(ax)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor=".75",
        fontsize=10.5,
    )
    fig.tight_layout()
    plot_data = pd.DataFrame(
        {
            "sample No.": x,
            "Original row": order + 1,
            "Original index": truth.index.to_numpy()[order],
            "Measured": y,
            **{
                family: frame[output].to_numpy(float)[order]
                for family, frame in predictions.items()
            },
        }
    )
    if individual_predictions is not None:
        for family, frame in individual_predictions.items():
            plot_data[f"{family} individual"] = frame[output].to_numpy(float)[order]
    export_table(plot_data, f"validation/{name}/{slug(output)}_plot_data.csv")
    path = save_figure(fig, f"validation/{name}/{slug(output)}.png")
    plt.close(fig)
    return path
