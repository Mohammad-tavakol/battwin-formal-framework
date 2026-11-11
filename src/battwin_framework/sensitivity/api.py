"""High-level sensitivity operations with named, reusable result objects."""

import numpy as np
import pandas as pd
from ..schema import chain_inputs
from ..data import load_dataset
from ..results import SensitivityResult, AutodiffResult, AutodiffComparison
from .finite_difference import finite_difference_sensitivity, summarise_sensitivity
from .permutation import rf_chain_pfi


def reference_steps(line, reference):
    """Notebook forward steps: max(0.001, 0.001*reference SD), population SD."""
    data = chain_inputs(line, reference).select_dtypes("number")
    return {c: max(0.001, 0.001 * float(data[c].std(ddof=0))) for c in data}


class Sensitivity:
    """Sensitivity service bound to a Framework's current model registry.

    Reference scaling always uses supplied reference rows or bundled train_valid,
    never the external test set by default.
    """

    def __init__(self, framework):
        self.framework = framework

    def _reference(self, reference):
        return (
            load_dataset("train_valid", directory=self.framework.directory)
            if reference is None
            else reference
        )

    def finite_difference(
        self, frame, *, features=None, outputs=None, epsilon=None, reference=None
    ):
        """Whole-chain DNN forward differences, with reference-SD scaled rankings."""
        line = self.framework.chain("DNN")
        ref = chain_inputs(line, self._reference(reference))
        features = (
            list(chain_inputs(line, frame).select_dtypes("number"))
            if features is None
            else list(features)
        )
        steps = reference_steps(line, ref) if epsilon is None else epsilon
        detail = finite_difference_sensitivity(
            line, frame, features, output_cols=outputs, epsilon=steps
        )
        return SensitivityResult(
            detail, summarise_sensitivity(detail, ref), "finite_difference"
        )

    def pfi(
        self,
        frame,
        *,
        truth=None,
        features=None,
        outputs=None,
        repeats=12,
        seed=42,
        metric="r2",
    ):
        """RF chain score drops; categories and numeric inputs can be permuted."""
        from ..schema import MEASURED_OUTPUTS

        line = self.framework.chain("RF")
        features = (
            list(chain_inputs(line, frame)) if features is None else list(features)
        )
        truth = frame if truth is None else truth
        outputs = (
            [c for c in MEASURED_OUTPUTS if c in truth]
            if outputs is None
            else list(outputs)
        )
        summary = rf_chain_pfi(
            line,
            frame,
            truth,
            features,
            outputs,
            repeats=repeats,
            seed=seed,
            metric=metric,
        )
        return SensitivityResult(pd.DataFrame(), summary, "pfi")

    def autodiff(self, frame, *, reference=None):
        """Exact automatic derivatives of the supported DNN graph, at fixed categories."""
        from .autodiff import DifferentiableDNNChain
        from ..models.preprocessing import model_frame
        from ..schema import ALIASES

        line = self.framework.chain("DNN")
        graph = DifferentiableDNNChain(
            line,
            frame,
            chain_inputs=chain_inputs,
            model_frame=model_frame,
            aliases=ALIASES,
        )
        prediction, jacobian, diagnostics = graph.jacobian()
        scales = chain_inputs(line, self._reference(reference))[graph.features].std(
            ddof=0
        )
        records = []
        for j, feature in enumerate(graph.features):
            for k, output in enumerate(graph.outputs):
                for i in range(len(frame)):
                    records.append(
                        {
                            "chain": line.label,
                            "item_index": i + 1,
                            "feature": feature,
                            "output": output,
                            "input_value": float(graph.numeric[i, j]),
                            "base_output": float(prediction[i, k]),
                            "derivative": float(jacobian[i, k, j]),
                        }
                    )
        detail = pd.DataFrame(records)
        summary = (
            detail.groupby(["chain", "output", "feature"], sort=False)
            .agg(
                n=("derivative", "count"),
                mean_signed=("derivative", "mean"),
                std=("derivative", "std"),
                minimum=("derivative", "min"),
                maximum=("derivative", "max"),
                mean_absolute=("derivative", lambda d: d.abs().mean()),
            )
            .reset_index()
        )
        summary["reference_input_sd"] = summary.feature.map(scales)
        summary["mean_absolute_one_sd"] = (
            summary.mean_absolute * summary.reference_input_sd
        )
        summary = summary.sort_values(
            ["output", "mean_absolute_one_sd"], ascending=[True, False]
        ).reset_index(drop=True)
        return AutodiffResult(
            detail,
            summary,
            "autodiff",
            pd.DataFrame(prediction, index=frame.index, columns=graph.outputs),
            jacobian,
            tuple(graph.features),
            tuple(graph.outputs),
            diagnostics,
        )

    def compare(
        self,
        frame,
        *,
        reference=None,
        finite_difference_detail=None,
        step_multipliers=(0.25, 0.5, 1.0, 2.0, 4.0),
    ):
        """Original paired AD/FD comparison and the five-scale step-size study."""
        from .autodiff import autodiff_finite_difference_comparison

        result = autodiff_finite_difference_comparison(
            self.framework.chain("DNN"),
            frame,
            self._reference(reference),
            finite_difference_detail=finite_difference_detail,
            step_multipliers=step_multipliers,
        )
        return AutodiffComparison(**result)
