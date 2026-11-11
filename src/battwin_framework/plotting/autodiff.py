"""Plots for the original AD versus finite-difference comparison."""

from pathlib import Path
import json
import numpy as np
import pandas as pd


def plot_autodiff_comparison(result, directory, *, capacities):
    """Readable comparisons using the same physical-unit importance definition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary, paired = (result["importance"], result["derivatives"])

    def save(fig, name):
        fig.tight_layout()
        for extension in ["png", "pdf"]:
            fig.savefig(directory / f"{name}.{extension}", dpi=220, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(21, 10))
    for ax, output in zip(axes, capacities):
        data = summary[summary.output.eq(output)].sort_values("ad_importance")
        y = np.arange(len(data))
        ax.barh(
            y - 0.19,
            data.fd_importance,
            0.36,
            color=".65",
            label="Forward finite difference",
        )
        ax.barh(
            y + 0.19,
            data.ad_importance,
            0.36,
            color="#d62728",
            label="Automatic differentiation",
        )
        ax.set_yticks(y, data.feature, fontsize=10)
        ax.set_title(output, fontsize=12)
        ax.set_xlabel(
            "Mean absolute derivative × reference input SD\n(output units)", fontsize=11
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    save(fig, "capacity_importance")
    outputs = result["diagnostics"]["outputs"]
    features = result["diagnostics"]["numeric_features"]
    matrices = [
        summary.pivot(index="feature", columns="output", values=c)
        .reindex(index=features, columns=outputs)
        .to_numpy()
        for c in ["ad_importance", "fd_importance"]
    ]
    common = np.maximum(matrices[0].max(axis=0), matrices[1].max(axis=0))
    common = np.where(common == 0, 1, common)
    fig, axes = plt.subplots(1, 2, figsize=(21, 11), sharey=True)
    for ax, matrix, title in zip(
        axes, matrices, ["Automatic differentiation", "Forward finite difference"]
    ):
        im = ax.imshow(matrix / common, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(
            range(len(outputs)),
            ["\n".join(textwrap.wrap(c, 17)) for c in outputs],
            rotation=45,
            ha="right",
            fontsize=9,
        )
        ax.set_yticks(range(len(features)), features, fontsize=10)
    fig.subplots_adjust(left=0.22, right=0.9, bottom=0.25, wspace=0.1)
    colorbar_axis = fig.add_axes([0.93, 0.25, 0.012, 0.63])
    fig.colorbar(
        im,
        cax=colorbar_axis,
        label="SD-scaled importance / shared maximum for each output",
    )
    for extension in ["png", "pdf"]:
        fig.savefig(
            directory / f"all_outputs_importance.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, output in zip(axes.flat, outputs):
        data = paired[paired.output.eq(output)]
        ax.scatter(
            data.scaled_ad,
            data.scaled_fd,
            s=9,
            alpha=0.25,
            color="#d62728",
            rasterized=True,
        )
        low = min(data.scaled_ad.min(), data.scaled_fd.min())
        high = max(data.scaled_ad.max(), data.scaled_fd.max())
        if high == low:
            low, high = (low - 1, high + 1)
        ax.plot([low, high], [low, high], "k--", lw=0.8)
        metrics = result["agreement"].set_index("output").loc[output]
        ax.set_title("\n".join(textwrap.wrap(output, 27)), fontsize=11)
        ax.text(
            0.04,
            0.96,
            f"r = {metrics.signed_derivative_pearson:.4f}\nRelative MAE = {metrics.relative_scaled_derivative_mae:.2%}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        ax.set_xlabel("AD derivative × input SD", fontsize=9)
        ax.set_ylabel("FD derivative × input SD", fontsize=9)
        ax.grid(alpha=0.15)
    save(fig, "derivative_agreement")
    fig, ax = plt.subplots(figsize=(11, 6))
    for output, data in result["step_size"].groupby("output", sort=False):
        ax.plot(
            data.step_multiplier,
            data.relative_scaled_derivative_mae,
            marker="o",
            label=output,
        )
    ax.set_xscale("log", base=2)
    ax.set(
        xlabel="Finite-difference step / original step",
        ylabel="Mean absolute FD–AD error / mean absolute AD sensitivity",
    )
    ax.set_xticks(
        result["diagnostics"]["step_multipliers"],
        result["diagnostics"]["step_multipliers"],
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "step_size")


def save_autodiff_comparison(result, directory, *, capacities):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ["derivatives", "importance", "agreement", "step_size"]:
        result[name].to_csv(directory / f"{name}.csv", index=False)
    (directory / "diagnostics.json").write_text(
        json.dumps(result["diagnostics"], indent=2), encoding="utf-8"
    )
    plot_autodiff_comparison(result, directory, capacities=capacities)
