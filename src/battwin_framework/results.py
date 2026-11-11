"""Named results with explicit save/plot operations; computation never exports implicitly."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd


def _directory(directory):
    path = Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json(path, value):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        ),
        encoding="utf-8",
    )


@dataclass
class ValidationResult:
    predictions: dict[str, pd.DataFrame]
    metrics: pd.DataFrame
    truth: pd.DataFrame
    individual: dict[str, pd.DataFrame] = field(default_factory=dict)

    def save(self, directory, *, plots=False):
        path = _directory(directory)
        self.metrics.to_csv(path / "metrics.csv", index=False)
        for family, data in self.predictions.items():
            data.to_csv(path / f"{family}_chain.csv", index=False)
        for family, data in self.individual.items():
            data.to_csv(path / f"{family}_individual.csv", index=False)
        if plots:
            self.plot(path)
        return path

    def plot(self, directory, *, outputs=None):
        from .plotting import plot_validation

        return plot_validation(self, directory, outputs=outputs)


@dataclass
class SensitivityResult:
    detail: pd.DataFrame
    summary: pd.DataFrame
    method: str

    def save(self, directory, *, plots=False):
        path = _directory(directory)
        if not self.detail.empty:
            self.detail.to_csv(path / "derivatives.csv", index=False)
        self.summary.to_csv(path / "importance.csv", index=False)
        _json(path / "method.json", {"method": self.method})
        if plots:
            self.plot(path)
        return path

    def plot(self, directory, *, outputs=None):
        from .plotting import plot_sensitivity

        return plot_sensitivity(self, directory, outputs=outputs)


@dataclass
class AutodiffResult(SensitivityResult):
    predictions: pd.DataFrame
    jacobian: np.ndarray
    features: tuple[str, ...]
    outputs: tuple[str, ...]
    diagnostics: dict

    def save(self, directory, *, plots=False):
        path = super().save(directory, plots=plots)
        self.predictions.to_csv(path / "predictions.csv", index=False)
        _json(path / "diagnostics.json", self.diagnostics)
        # Detail already stores every Jacobian entry with its feature/output/item.
        return path


@dataclass
class AutodiffComparison:
    derivatives: pd.DataFrame
    importance: pd.DataFrame
    agreement: pd.DataFrame
    step_size: pd.DataFrame
    diagnostics: dict

    def save(self, directory, *, plots=False):
        path = _directory(directory)
        for name in ["derivatives", "importance", "agreement", "step_size"]:
            getattr(self, name).to_csv(path / f"{name}.csv", index=False)
        _json(path / "diagnostics.json", self.diagnostics)
        if plots:
            self.plot(path)
        return path

    def plot(self, directory):
        from .plotting.autodiff import plot_autodiff_comparison
        from .schema import GRAVIMETRIC, VOLUMETRIC

        path = _directory(directory)
        plot_autodiff_comparison(
            asdict(self), path, capacities=[GRAVIMETRIC, VOLUMETRIC]
        )
        return path


@dataclass
class FeedbackResult:
    predictions: pd.DataFrame
    trace: pd.DataFrame
    history: list
    family: str

    def save(self, directory, *, plots=False):
        path = _directory(directory)
        self.predictions.to_csv(path / "predictions.csv", index=False)
        self.trace.to_csv(path / "trace.csv", index=False)
        if plots:
            self.plot(path)
        return path

    def plot(self, directory):
        from .plotting.output import output_directory
        from .plotting.importance import feedback_plot

        with output_directory(directory):
            return feedback_plot(self.trace, self.family)


@dataclass
class SearchModelResult:
    evaluated: pd.DataFrame
    matches: pd.DataFrame
    nearest: pd.DataFrame
    checks: pd.DataFrame


@dataclass
class SearchResult:
    candidates: pd.DataFrame
    models: dict[str, SearchModelResult]
    summary: pd.DataFrame
    manifest: dict

    def save(self, directory, *, plots=False):
        path = _directory(directory)
        self.candidates.to_csv(path / "sampled_inputs.csv", index=False)
        self.summary.to_csv(path / "summary.csv", index=False)
        _json(path / "search_manifest.json", self.manifest)
        for family, value in self.models.items():
            destination = path / family
            destination.mkdir(exist_ok=True)
            for attr, filename in [
                ("evaluated", "all_evaluated_paths"),
                ("matches", "matching_paths"),
                ("nearest", "nearest_evaluated_paths"),
            ]:
                getattr(value, attr).to_csv(
                    destination / f"{filename}.csv", index=False
                )
            checks = value.checks.copy()
            checks.insert(0, "Candidate ID", self.candidates["Candidate ID"].to_numpy())
            checks.to_csv(destination / "physical_checks.csv", index=False)
        if plots:
            self.plot(path)
        return path

    def plot(self, directory):
        from .plotting import plot_search

        return plot_search(self, directory)
