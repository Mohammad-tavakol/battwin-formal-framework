"""Public facade over modular scientific operations; no notebook execution."""

from pathlib import Path
import hashlib
import importlib.metadata
import pandas as pd
from .resources import ASSETS
from .data import load_dataset
from .models import ModelRegistry
from .schema import OUTPUT_COLUMNS, STAGE_OUTPUTS
from .simulation import simulate_without_feedback, simulate_with_trace
from .results import ValidationResult, FeedbackResult, SearchResult, SearchModelResult


class Framework:
    """An independent set of saved models and their coupled capabilities.

    Loading is lazy. Importing or constructing a Framework does not run the
    notebook, fit models, copy datasets or write experimental results.
    """

    def __init__(
        self, directory=None, *, cache_directory=None, allow_extra_trees=True, n_jobs=1
    ):
        self.directory = (
            ASSETS if directory is None else Path(directory).expanduser().resolve()
        )
        self.models = ModelRegistry(
            self.directory,
            cache_directory=cache_directory,
            allow_extra_trees=allow_extra_trees,
            n_jobs=n_jobs,
        )
        from .sensitivity import Sensitivity

        self.sensitivity = Sensitivity(self)

    @classmethod
    def from_bundle(cls, **kwargs):
        """Use the unchanged model artifacts and workbooks shipped in this package."""
        return cls(**kwargs)

    def load_dataset(self, name="test"):
        return load_dataset(name, directory=self.directory)

    def chain(self, family):
        """Access the original NamedStateSpaceLine for advanced sequential operations."""
        return self.models.chain(family)

    def describe(self, family="DNN"):
        return self.models.describe(family)

    def predict(self, frame, *, families=("RF", "DNN")):
        """Return family -> DataFrame, including base inputs and eight physical outputs."""
        return {
            family: simulate_without_feedback(self.chain(family), frame)
            for family in families
        }

    def predict_stage(self, frame, *, family, stage):
        """Run a standalone stage using supplied upstream measurements, without chaining."""
        return self.models.stage(family, stage).predict_frame(frame)

    def simulate(self, frame, *, family="DNN", **kwargs):
        """Original sequential simulation with full history and explicit core options."""
        return simulate_with_trace(self.chain(family), frame, **kwargs)

    def replace_model(self, family, stage, model):
        """Replace one instance-local stage and return its previous model."""
        return self.models.replace(family, stage, model)

    def validate(
        self,
        frame,
        *,
        truth=None,
        families=("RF", "DNN"),
        individual_stages=("Electrochemical",),
    ):
        """Chain accuracy and optional standalone accuracy on identical measurement rows.

        Additional standalone stages require their measured upstream inputs.
        Missing inputs raise an error; predicted substitutes are never introduced.
        """
        from .validation import (
            comparison_table,
            validation_metrics,
            individual_electrochemical_predictions,
        )

        truth = frame if truth is None else truth
        predictions = self.predict(frame, families=families)
        metrics = comparison_table(predictions, truth).assign(mode="chain")
        individual = {}
        for stage in individual_stages:
            if stage == "Electrochemical":
                values = individual_electrochemical_predictions(
                    {f: self.chain(f) for f in families}, truth
                )
            else:
                values = {
                    f: self.predict_stage(truth, family=f, stage=stage)
                    for f in families
                }
            for family, data in values.items():
                if family not in individual:
                    individual[family] = pd.DataFrame(index=truth.index)
                for column in data:
                    individual[family][column] = data[column]
                table = validation_metrics(
                    data, truth.loc[:, [c for c in data if c in truth]], family
                )
                table = table.loc[table.output.isin(STAGE_OUTPUTS[stage])].assign(
                    mode="individual", stage=stage
                )
                metrics = pd.concat([metrics, table], ignore_index=True)
        return ValidationResult(predictions, metrics, truth.copy(), individual)

    def run_feedback(
        self,
        frame,
        *,
        family="DNN",
        kpi_key,
        kpi_target,
        gain,
        stage="CoatingDrying",
        memory_key="delta_coating_gap",
        initial_memory=0.0,
    ):
        """Unbounded additive correction, applied stage-locally on the next item."""
        from .feedback import run_feedback

        values, trace, history = run_feedback(
            self.chain(family),
            frame,
            kpi_key=kpi_key,
            kpi_target=kpi_target,
            gain=gain,
            stage=stage,
            memory_key=memory_key,
            initial_memory=initial_memory,
        )
        return FeedbackResult(values, trace, history, family)

    def disturbance_rejection(
        self, frame, *, reference=None, config=None, families=("RF", "DNN")
    ):
        """Reproduce the matched active/held 80-item disturbance scenario."""
        from .scenarios import run_disturbance_scenario, DisturbanceConfig

        return run_disturbance_scenario(
            self,
            frame,
            self.load_dataset("train_valid") if reference is None else reference,
            config=DisturbanceConfig() if config is None else config,
            families=families,
        )

    def parametric_sweep(
        self,
        base_row,
        ranges,
        *,
        family="DNN",
        derived=None,
        max_items=100000,
        batch_size=2048,
    ):
        """Evaluate the original Cartesian sweep; no physical filtering is implicit."""
        from .parametric import parametric_inputs, run_parametric

        inputs = parametric_inputs(
            base_row, ranges, derived=derived, max_items=max_items
        )
        return run_parametric(self.chain(family), inputs, batch_size=batch_size)

    def random_search(
        self,
        *,
        targets,
        training=None,
        n_sweeps=1000,
        random_seed=42,
        local_scale_fraction=0.1,
        tolerance_fraction=0.02,
    ):
        """Draw and screen the same independent local candidates with RF and DNN.

        An even budget is balanced equally between cathodes and anodes. Targets
        do not steer the draws. There is no refitting or feedback in this search.
        """
        from .search import (
            sample_inverse_candidates,
            evaluate_inverse_candidates,
            verify_inverse_physical_screens,
        )
        from .porosity import ELECTRODE_DENSITIES

        training = self.load_dataset("train_valid") if training is None else training
        chains = {family: self.chain(family) for family in ("RF", "DNN")}
        protected = [
            p for p in (self.directory / "best models").rglob("*") if p.is_file()
        ]
        protected += list(self.directory.glob("*.xlsx"))
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        candidates, manifest = sample_inverse_candidates(
            training,
            rf_line=chains["RF"],
            dnn_line=chains["DNN"],
            n_sweeps=n_sweeps,
            random_seed=random_seed,
            local_scale_fraction=local_scale_fraction,
        )
        repeat, _ = sample_inverse_candidates(
            training,
            rf_line=chains["RF"],
            dnn_line=chains["DNN"],
            n_sweeps=n_sweeps,
            random_seed=random_seed,
            local_scale_fraction=local_scale_fraction,
        )
        pd.testing.assert_frame_equal(candidates, repeat, check_exact=True)
        models, summary, tolerance = evaluate_inverse_candidates(
            candidates, chains, targets=targets, tolerance_fraction=tolerance_fraction
        )
        manifest.update(
            {
                "targets": targets,
                "tolerance_fraction": tolerance_fraction,
                "absolute_tolerances": tolerance,
                "seed_reproducibility_verified": True,
                "geometry_density_g_cm3": ELECTRODE_DENSITIES,
                "dried_porosity_units": "percent, as in coating/drying training workbook",
                "porosity_ordering": "Both predicted and geometry-calculated calendered porosity must be strictly below predicted dried porosity",
                "models_evaluated": list(models),
                "forward_evaluations_per_model": n_sweeps,
                "capacity_targets_checked_jointly": True,
                "global_optimum_claimed": False,
                "physical_screen_tests": verify_inverse_physical_screens(),
                "reference_dataframe_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(training, index=True)
                    .to_numpy()
                    .tobytes()
                ).hexdigest(),
                "source_model_sha256": {
                    str(p.relative_to(self.directory)): h
                    for p, h in hashes.items()
                    if "best models" in p.parts
                },
                "active_models": {
                    f: self.describe(f).to_dict("records") for f in chains
                },
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in [
                        "numpy",
                        "scipy",
                        "pandas",
                        "tensorflow",
                        "keras",
                        "scikit-learn",
                    ]
                },
            }
        )
        if not all(
            hashlib.sha256(p.read_bytes()).hexdigest() == digest
            for p, digest in hashes.items()
        ):
            raise RuntimeError("A source model or dataset changed during the search.")
        manifest["original_model_and_dataset_hashes_verified"] = len(hashes)
        return SearchResult(
            candidates,
            {
                f: SearchModelResult(v["all"], v["matches"], v["nearest"], v["checks"])
                for f, v in models.items()
            },
            summary,
            manifest,
        )
