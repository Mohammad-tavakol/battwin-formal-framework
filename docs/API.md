# API map

`Framework(directory=None, *, cache_directory=None, allow_extra_trees=True, n_jobs=1)`
uses bundled resources by default. Imports and construction do not load Keras or
execute experiments. `help(...)` shows live signatures.

| Public method | Return |
|---|---|
| `load_dataset(name)` | Fresh DataFrame |
| `describe(family)` | Stage/schema/artifact DataFrame |
| `predict(frame, families=...)` | Family → chain DataFrame |
| `predict_stage(frame, family=..., stage=...)` | Standalone stage DataFrame |
| `validate(frame, truth=..., individual_stages=...)` | `ValidationResult` |
| `simulate(frame, family=..., **core_options)` | `(predictions, history)` |
| `replace_model(family, stage, model)` | Previous model |
| `run_feedback(frame, kpi_key=..., kpi_target=..., gain=...)` | `FeedbackResult` |
| `disturbance_rejection(frame, config=...)` | `DisturbanceResult` |
| `parametric_sweep(base_row, ranges, family=...)` | Chain DataFrame |
| `random_search(targets=..., n_sweeps=..., random_seed=...)` | `SearchResult` |
| `sensitivity.pfi(frame, ...)` | `SensitivityResult` |
| `sensitivity.finite_difference(frame, ...)` | `SensitivityResult` |
| `sensitivity.autodiff(frame, ...)` | `AutodiffResult` |
| `sensitivity.compare(frame, ...)` | `AutodiffComparison` |

## Result fields

Each result has `.save(directory, plots=False)` and `.plot(directory)`.
Validation/sensitivity plots accept an optional `outputs=[...]` subset.

| Type | Fields |
|---|---|
| `ValidationResult` | `predictions`, `metrics`, `truth`, `individual` |
| `SensitivityResult` | `detail`, `summary`, `method` |
| `AutodiffResult` | Above plus `predictions`, `jacobian`, `features`, `outputs`, `diagnostics` |
| `AutodiffComparison` | `derivatives`, `importance`, `agreement`, `step_size`, `diagnostics` |
| `FeedbackResult` | `predictions`, `trace`, `history`, `family` |
| `DisturbanceResult` | `inputs`, `controlled`, `held`, `traces`, `histories`, `summary`, `config` |
| `SearchResult` | `candidates`, `models`, `summary`, `manifest` |
| `SearchModelResult` | `evaluated`, `matches`, `nearest`, `checks` |

Search models are indexed by family. Sensitivity rows use one-based `item_index`.
AD tensor axes are named by `outputs` and `features`. PFI has an empty derivative
detail because it computes score drops rather than derivatives.

## Modules

| Module | Responsibility |
|---|---|
| `data`, `resources` | Read-only bundle/data access and writable inference cache |
| `schema` | Physical-unit columns, stage outputs, aliases and input leakage guard |
| `models.preprocessing` | Exact saved-column/category alignment |
| `models.trees`, `models.neural` | Saved inference, preprocessing and target inverses |
| `models.registry` | Stage protocol, callable adapter and model replacement |
| `simulation` | Batch/sequential execution and stage-local actuation |
| `validation` | Accuracy metrics and individual electrochemical comparisons |
| `feedback`, `scenarios` | Additive controller and matched disturbance experiment |
| `parametric` | Cartesian sweeps, original checks, correlations and tolerance filters |
| `porosity` | Original geometry calculation and density references |
| `search` | Random proposals, strict physical checks and target screening |
| `sensitivity.finite_difference` | Forward slopes, summaries and feedback probes |
| `sensitivity.permutation` | Whole-chain RF PFI |
| `sensitivity.autodiff` | Differentiable DNN chain and AD/FD comparison |
| `plotting` | Directory-explicit validation, sensitivity and search plots |
| `plotting.autodiff`, `.feedback`, `.importance` | Original specialised chart layouts |
| `core`, `indexed` | Unchanged named and indexed mathematics |

## Functional composition

The facade is optional. Functions accept explicit compiled lines and DataFrames:

```python
from battwin_framework import Framework
from battwin_framework.search import sample_inverse_candidates, evaluate_inverse_candidates
from battwin_framework.schema import GRAVIMETRIC, VOLUMETRIC

f = Framework.from_bundle()
lines = {family: f.chain(family) for family in ("RF", "DNN")}
inputs, sampling = sample_inverse_candidates(
    f.load_dataset("train_valid"), rf_line=lines["RF"], dnn_line=lines["DNN"],
    n_sweeps=1000, random_seed=42,
)
models, summary, tolerances = evaluate_inverse_candidates(
    inputs, lines, targets={GRAVIMETRIC: 165., VOLUMETRIC: 480.}, tolerance_fraction=.02,
)
```

Low-level plotting helpers use a per-call output context, restored on exit:

```python
from battwin_framework.plotting import output_directory
from battwin_framework.plotting.importance import correlation_plot
from battwin_framework.parametric import numeric_correlations

with output_directory("results/correlation"):
    correlation_plot(numeric_correlations(predictions["DNN"]), "DNN", "pearson")
```
