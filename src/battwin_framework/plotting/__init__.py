"""Public plots; every operation takes an explicit output directory."""

from .output import output_directory


def plot_validation(result, directory, *, outputs=None):
    """Solid blue measurements, dashed chains, solid individual models; R2/MAE."""
    from .validation import execution_line_plot
    from ..schema import MEASURED_OUTPUTS

    outputs = (
        [c for c in MEASURED_OUTPUTS if c in result.truth]
        if outputs is None
        else outputs
    )
    paths = []
    with output_directory(directory):
        for output in outputs:
            single = (
                result.individual
                if result.individual
                and all(output in df for df in result.individual.values())
                else None
            )
            paths.append(
                execution_line_plot(
                    result.predictions,
                    result.truth,
                    output,
                    individual_predictions=single,
                )
            )
    return paths


def plot_sensitivity(result, directory, *, outputs=None):
    from .importance import importance_plot

    pfi = result.method == "pfi"
    family = "RF" if pfi else "DNN"
    value = "mean_importance" if pfi else "mean_absolute_one_sd"
    label = {
        "pfi": "PFI",
        "finite_difference": "Finite difference",
        "autodiff": "Automatic differentiation",
    }[result.method]
    explanation = (
        "Whole-chain permutation score drop"
        if pfi
        else "Mean absolute derivative × reference input SD (output units)"
    )
    paths = []
    outputs = result.summary.output.unique() if outputs is None else outputs
    with output_directory(directory):
        for output in outputs:
            paths.append(
                importance_plot(
                    result.summary,
                    value,
                    output,
                    family,
                    explanation,
                    error="std_importance" if pfi else None,
                    method_label=label,
                )
            )
    return paths


def plot_search(result, directory):
    """Notebook capacity panels and interactive/static parallel target paths."""
    from .inverse import path_axis_specs, plot_target_paths
    from .style import CHART_COLORS
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    targets, tol = result.manifest["targets"], result.manifest["absolute_tolerances"]
    with output_directory(directory) as path:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        for row, (family, model) in enumerate(result.models.items()):
            for col, electrode in enumerate(["Cathode", "Anode"]):
                ax = axes[row, col]
                data = model.evaluated.query("`Electrode type` == @electrode")
                valid, hit = data["Physical pass"], data["Target match"]
                ax.scatter(
                    data.loc[~valid, GRAVIMETRIC],
                    data.loc[~valid, VOLUMETRIC],
                    s=14,
                    c=".75",
                    marker="x",
                    label="Physical check failed",
                )
                ax.scatter(
                    data.loc[valid, GRAVIMETRIC],
                    data.loc[valid, VOLUMETRIC],
                    s=17,
                    c=CHART_COLORS[family],
                    alpha=0.40,
                    label="Physical checks passed",
                )
                ax.scatter(
                    data.loc[hit, GRAVIMETRIC],
                    data.loc[hit, VOLUMETRIC],
                    s=50,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.1,
                    label="Both targets met",
                )
                ax.add_patch(
                    Rectangle(
                        (
                            targets[GRAVIMETRIC] - tol[GRAVIMETRIC],
                            targets[VOLUMETRIC] - tol[VOLUMETRIC],
                        ),
                        2 * tol[GRAVIMETRIC],
                        2 * tol[VOLUMETRIC],
                        fill=False,
                        edgecolor="black",
                        lw=1.5,
                    )
                )
                ax.plot(targets[GRAVIMETRIC], targets[VOLUMETRIC], "k+", markersize=9)
                ax.set(
                    xlabel=GRAVIMETRIC,
                    ylabel=VOLUMETRIC,
                    title=f"{family} — {electrode}: {int(valid.sum())}/{len(data)} physical; {int(hit.sum())} matches",
                )
                ax.grid(alpha=0.15)
                ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(path / f"capacity_search.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)
        columns = [
            "Electrode type",
            "Chemistry",
            "Am/total weight%",
            VISCOSITY,
            "Coating Gap (micron)",
            "Coating Speed (m/min)",
            DRY_THICKNESS,
            MASS,
            "Calendaring Gap (um)",
            CAL_THICKNESS,
            DRY_POROSITY,
            CAL_POROSITY,
            CALCULATED_POROSITY,
            "Discharge rate (*C)",
            GRAVIMETRIC,
            VOLUMETRIC,
        ]
        tables = {k: v.evaluated for k, v in result.models.items()}
        specs = path_axis_specs(tables, columns, targets=targets, tolerances=tol)
        for family, data in tables.items():
            plot_target_paths(
                data,
                data["Target match"],
                family,
                columns,
                specs,
                targets=targets,
                tolerances=tol,
            )
    return path


__all__ = ["plot_validation", "plot_sensitivity", "plot_search", "output_directory"]
