"""Original disturbance-rejection and downstream-capacity figures."""

import numpy as np
from ..schema import GRAVIMETRIC, VOLUMETRIC
from .style import _pyplot, figure_label, style_axes, save_figure


def disturbance_feedback_plot(trace, family, kpi, disturbance_start, tolerance):
    plt = _pyplot()
    color = "#2ca02c" if family == "RF" else "#d62728"
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    x = trace.item_index
    axes[0].plot(x, trace.kpi_value, color=color, lw=2, label="Feedback active")
    axes[0].plot(
        x,
        trace.held_kpi,
        color="0.4",
        lw=1.8,
        ls="--",
        label="Correction held after disturbance",
    )
    target = float(trace.kpi_target.iloc[0])
    axes[0].axhline(target, color="0.15", lw=1.2, ls=":", label="Shared target")
    axes[0].axhspan(
        target - tolerance,
        target + tolerance,
        color="#1b7837",
        alpha=0.10,
        label=f"Example tolerance: ±{tolerance:g}",
    )
    axes[0].set_ylabel(figure_label(kpi), fontsize=16)
    axes[0].legend(loc="lower right", fontsize=11, ncol=2)
    axes[1].plot(x, trace.error, color=color, lw=2)
    axes[1].plot(x, trace.held_kpi - trace.kpi_target, color="0.4", lw=1.8, ls="--")
    axes[1].axhspan(-tolerance, tolerance, color="#1b7837", alpha=0.10)
    axes[1].axhline(0, color="0.2", lw=1, ls=":")
    axes[1].set_ylabel(r"Thickness error ($\mu$m)", fontsize=16)
    axes[2].plot(x, trace.adjusted_coating_gap, color=color, lw=2)
    axes[2].plot(x, trace.held_coating_gap, color="0.4", lw=1.8, ls="--")
    axes[2].set_ylabel(r"Applied coating gap ($\mu$m)", fontsize=16)
    axes[3].step(x, trace.calendering_gap, where="post", color="0.2", lw=1.8)
    axes[3].set(xlabel="sample No.", ylabel=r"Calender-roll gap ($\mu$m)")
    axes[3].yaxis.label.set_size(16)
    for i, ax in enumerate(axes):
        style_axes(ax)
        ax.axvline(disturbance_start - 0.5, color="0.45", lw=1.1, ls="-.")
        ax.text(
            0.012, 0.97, f"({chr(97+i)})", transform=ax.transAxes, va="top", fontsize=16
        )
    offset = trace.calendering_gap.iloc[-1] - trace.calendering_gap.iloc[0]
    post = trace.loc[trace.item_index >= disturbance_start]
    active_mae = post.error.abs().mean()
    held_mae = (post.held_kpi - post.kpi_target).abs().mean()
    caption = (
        f"After offset: MAE active = {active_mae:.3f} µm; held = {held_mae:.3f} µm. "
        f"Final error = {trace.error.iloc[-1]:+.3f} µm."
    )
    fig.text(0.5, 0.005, caption, ha="center", fontsize=12)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    path = save_figure(fig, f"feedback/scenario/{family}/disturbance_response.png")
    plt.close(fig)
    return path


def feedback_capacity_plot(controlled, held, family, disturbance_start):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = np.arange(1, len(controlled) + 1)
    color = "#2ca02c" if family == "RF" else "#d62728"
    for i, (ax, output) in enumerate(zip(axes, [GRAVIMETRIC, VOLUMETRIC])):
        ax.plot(x, controlled[output], color=color, lw=2, label="Feedback active")
        ax.plot(x, held[output], color="0.4", lw=1.8, ls="--", label="Correction held")
        ax.axvline(disturbance_start - 0.5, color="0.45", lw=1.1, ls="-.")
        ax.set_ylabel(figure_label(output), fontsize=16)
        style_axes(ax)
        ax.text(0.015, 0.97, f"({chr(97+i)})", transform=ax.transAxes, va="top")
    axes[0].legend(loc="best", fontsize=12)
    axes[1].set_xlabel("sample No.")
    fig.tight_layout()
    path = save_figure(fig, f"feedback/scenario/{family}/capacity_consequences.png")
    plt.close(fig)
    return path
