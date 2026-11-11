# Usage guide

## Load models and data

```python
from battwin_framework import Framework, load_dataset

framework = Framework.from_bundle()
test = load_dataset("test")
reference = load_dataset("train_valid")
predictions = framework.predict(test)
rf = framework.predict(test, families=("RF",))["RF"]
```

Aliases are `test` (38 aggregated external-test rows), `train_valid` (291 aggregated
development rows), `test_raw` and `train_valid_raw`. A CSV/XLSX path is also accepted.
Loading does not split, aggregate, impute or fit data. Global `load_dataset` uses
bundled aliases; `framework.load_dataset` resolves files against that instance's
project root.

```python
framework = Framework(r"C:\path\to\Framework-level", cache_directory=r"C:\temp\battwin-cache")
```

The project root must contain `best models/RF`, `best models/DNN` and the required
workbooks. Without a project path, models and data are read directly from bundled
assets. Compatibility copies use the system temporary cache or `cache_directory`.
Source artifacts are never overwritten. RF retains the supplied ExtraTrees
calendering models; `allow_extra_trees=False` rejects those models explicitly.
`n_jobs=1` is the original loaded tree-regressor default.

```python
print(framework.describe("DNN"))
print(framework.models.schema("RF", "Calendering"))
base_columns = framework.chain("DNN").get_required_input_columns()
```

Stages are `Mixing`, `CoatingDrying`, `Calendering`, `Electrochemical`. Prediction
frames retain the input index and include the 25 base inputs plus eight physical
outputs. Saved preprocessing, column ordering, category alignment and target
inverses are used directly. Measured intermediate/target columns are excluded from
chain base inputs, so they cannot bypass a stage.

## Compare chain and standalone accuracy

```python
evaluation = framework.validate(test)
print(evaluation.metrics[["chain", "mode", "output", "n", "r2", "rmse", "mae", "mape_percent"]])
evaluation.save("results/validation", plots=True)
single = framework.predict_stage(test, family="DNN", stage="Electrochemical")
```

The default standalone comparison runs the same electrochemical models using
measured upstream mass loading, calendered thickness and porosity. Capacity targets
are never model inputs. Additional standalone stages require all their measured
inputs:

```python
# measured must contain the inputs shown by models.schema for each stage.
evaluation = framework.validate(
    measured, individual_stages=("CoatingDrying", "Calendering", "Electrochemical")
)
```

Standalone coating/drying needs measured viscosity; standalone calendering needs
measured dry thickness and mass loading. Those source measurements exist in stage
workbooks but are not all in the compact framework exports. Source-row matching
and aggregation must be validated before joining them. Missing inputs raise errors;
the library does not silently merge recipes or replace measurements with predictions.
Use `truth=...` for a separate measurement table with an identical index.

Metrics include R², RMSE, MAE, signed mean error and MAPE (%). Non-finite measured
values are excluded per output; MAPE excludes measured zeros. R² needs two finite
measurements. Missing targets have `n=0` and an explicit status.

Plots retain solid blue measurements, green RF and red DNN curves, dashed for
chains and solid for individual models. All curves share a stable measured-value
ordering. Legends show R² and MAE; y limits cover the data with the existing small
padding. Exports include PNG, PDF and plot-data CSVs.

## Call the inverse-design optimiser

```python
from battwin_framework.schema import GRAVIMETRIC, VOLUMETRIC

search = framework.random_search(
    targets={GRAVIMETRIC: 165., VOLUMETRIC: 480.},
    training=reference,
    n_sweeps=1000, random_seed=42,
    local_scale_fraction=.1, tolerance_fraction=.02,
)
print(search.summary)
dnn_matches = search.models["DNN"].matches
all_paths = search.models["DNN"].evaluated
checks = search.models["DNN"].checks
nearest = search.models["DNN"].nearest
search.save("results/inverse", plots=True)
```

This preserves the current finite random-search algorithm:

1. Balance an even candidate budget equally between cathodes and anodes.
2. Draw training anchors uniformly with replacement within each electrode type,
   using NumPy PCG64 and the supplied seed.
3. Independently perturb nine process variables around each anchor with a normal
   distribution whose SD is `local_scale_fraction × within-type observed range`,
   truncated to that type's observed minimum/maximum. Zero-range variables stay fixed.
4. Copy all other formulation, density, shear, mixing and material inputs jointly
   from the chosen recipe, shuffle proposals, and evaluate identical candidates
   through both chains.
5. Accept only candidates meeting both target tolerances and all physical screens.

`battwin_framework.search.SEARCH_PROCESS_COLUMNS` lists the nine variables,
including discharge rate. Targets do not steer draws. Training anchors are not
counted as evaluated new settings. Duplicate proposals or exact training settings
raise errors as in the original implementation. Default training is `train_valid`.
The canonical bundled reference is eight RF and one DNN accepted cathode settings,
and no anode settings; replacement models or different environments can change it.

Both targets must be positive and finite. The ±2% tolerances here equal ±3.3 mAh/g
and ±9.6 mAh/cm³, with endpoints included. `nearest` contains up to five physically
passing nonmatches per electrode type, ranked by maximum tolerance-normalised
target distance. The saved manifest records distributions, bounds, copied inputs,
seed, tolerances, reference hash, source artifact hashes and active stage schemas.

Physical screens require:

- Positive finite dry mass loading, dry thickness and calendered thickness.
- Calendered thickness and finite roll gap strictly below dry thickness.
- Finite predicted and calculated calendered porosities in [0, 100] %.
- Finite non-negative dried porosity.
- Both predicted and calculated calendered porosity strictly below dried porosity.
- Positive finite applied coating gap.

No adaptive optimiser, additional process constraints, refitting, feedback, global
optimum claim or experimental manufacturing validation is introduced.

## Replace a stage model

Replacement affects only this Framework and subsequent calls. It does not train
or overwrite artifacts. The replacement chain is validated before registry mutation.
The previous model is returned so you can restore it.

Tree stages require fitted saved pipelines in output order, each retaining its
`preprocessor` and `regressor` steps and saved target inverse:

```python
from battwin_framework.models import RFStageModel

replacement = RFStageModel(
    "Calendering", ["my_models/thickness.joblib", "my_models/porosity.joblib"],
    allow_extra_trees=True, n_jobs=1,
)
previous = framework.replace_model("RF", "Calendering", replacement)
updated = framework.predict(test, families=("RF",))
framework.replace_model("RF", "Calendering", previous)
```

Neural stages require a saved network, fitted preprocessor and exact ordered target
metadata. Each target specification has a `kind` (`identity`, `standard`, `log1p`)
and its saved scaler or mean/scale when applicable. Scaling is inverted before log1p.

```python
from joblib import load
from battwin_framework.models import DNNStageModel

replacement = DNNStageModel(
    "Calendering", "my_models/network.keras", "my_models/preprocessor.joblib",
    targets=load("my_models/target_transform.joblib"),
)
previous = framework.replace_model("DNN", "Calendering", replacement)
```

A custom predictor may implement the `StageModel` protocol or use the checked adapter:

```python
from battwin_framework.models import CallableStageModel

# my_predictor returns a physical-unit DataFrame with the input index.
replacement = CallableStageModel(
    "Electrochemical", "RF",
    columns=framework.models.schema("RF", "Electrochemical")["inputs"],
    predict=my_predictor,
)
previous = framework.replace_model("RF", "Electrochemical", replacement)
```

Preserve stage/family names, output names/order and physical units. Custom frames
must preserve row indices and contain finite outputs. Changed input schemas require
matching data; stage outputs cannot become leaked original inputs. Caller-supplied
units and input meanings cannot be inferred from a Python function. Arbitrary
callables are not serialized; retain them separately for reproducibility.

## RF permutation importance

```python
pfi = framework.sensitivity.pfi(
    test, outputs=[GRAVIMETRIC, VOLUMETRIC], repeats=12, seed=42, metric="r2"
)
print(pfi.summary)
pfi.save("results/pfi", plots=True)
```

Defaults are all base features and measured framework targets present in the truth
table. `features=[...]` selects a subset; `truth=...` supplies separate aligned
measurements. One input column is permuted across rows per repeat, and the entire
RF-labelled chain is re-evaluated. R² importance is baseline minus permuted R²;
`metric="rmse"` uses permuted minus baseline RMSE. Negative importances remain.
The summary retains every repeat, mean, sample SD, extrema, repeat count and seed.
At least two finite target measurements are required. PFI is not a derivative.

## DNN finite differences

```python
fd = framework.sensitivity.finite_difference(test, reference=reference)
print(fd.detail)
print(fd.summary)
fd.save("results/fd", plots=True)

subset = framework.sensitivity.finite_difference(
    test, features=["Coating Gap (micron)"], outputs=[GRAVIMETRIC],
    epsilon={"Coating Gap (micron)": .01}, reference=reference,
)
```

Slopes use `(f(x+h)-f(x))/h` through the complete DNN chain. Defaults match the
notebook: all numeric base inputs and all eight outputs, with each physical-unit
step `h=max(0.001, 0.001 × reference SD)`. The reference is development data by
default, never test data, and uses population SD (`ddof=0`). `epsilon` may be a
positive scalar or a dictionary of positive steps. Categories and feedback stay
fixed. Summaries contain signed and absolute slopes, and mean absolute slope ×
reference input SD in output units. Constant reference features have zero scaled
importance even when their unscaled slopes are nonzero.

## DNN automatic differentiation and comparison

```python
ad = framework.sensitivity.autodiff(test, reference=reference)
print(ad.features, ad.outputs)
print(ad.jacobian.shape)  # (38, 8, 18) for bundled test data
print(ad.diagnostics)
ad.save("results/ad", plots=True)

comparison = framework.sensitivity.compare(test, reference=reference)
print(comparison.agreement)
print(comparison.importance)
print(comparison.step_size)
comparison.save("results/ad_vs_fd", plots=True)
```

The TensorFlow graph includes numeric preprocessing, all four DNNs and saved
physical target inverses, so upstream connections obey the full chain rule.
Categories remain fixed for each row; categorical derivatives are not defined.
Saved networks run at float32; preprocessing and target inverses use float64.
Forward output and encoded-feature parity are checked against ordinary inference.

Supported preprocessing is the current numeric SimpleImputer/StandardScaler
pipeline with fixed categorical encodings; supported layers are InputLayer, Dense
and inference-mode Dropout. Unsupported operations, indicator imputers, weighted
transformers or row-coupled layers raise errors. A replacement with different
structure may support prediction/FD but needs a verified differentiable adapter
before AD is valid.

The paired comparison retains identical cases, units and reference scaling,
using step multipliers `(0.25, 0.5, 1, 2, 4)`. It returns paired derivatives,
importance rankings, signed Pearson agreement, importance Spearman agreement,
top-five overlap and scaled errors. Four existing plot layouts are retained.
Pass `finite_difference_detail=fd.detail` to reuse a complete matching FD grid,
or `step_multipliers=...` for an explicitly changed step study.

These are local sensitivities, not causal effects. The independent indexed
`StateSpaceLine.get_jacobians` still uses its original finite differences; it is
separate from the TensorFlow AD operation above.

## Additive feedback and disturbance rejection

```python
import pandas as pd
from battwin_framework import DisturbanceConfig
from battwin_framework.schema import CAL_THICKNESS

sequence = pd.concat([test.iloc[[0]]] * 10, ignore_index=True)
controlled = framework.run_feedback(
    sequence, family="DNN", kpi_key=CAL_THICKNESS,
    kpi_target=62., gain=-1., initial_memory=0.,
)
print(controlled.trace)
controlled.save("results/controller", plots=True)

scenario = framework.disturbance_rejection(
    test, reference=reference,
    config=DisturbanceConfig(row=0, target=62., tolerance=.5, gain=-1.,
                            items=80, disturbance_start=26, disturbance_size=2.),
)
print(scenario.summary)
scenario.save("results/disturbance", plots=True)
```

The sign convention is `error = predicted KPI - target`, followed by
`memory_next = memory_used + gain × error`. Correction is fixed during an item's
chain execution and updated for the next item. It is added to the coating gap
only at `CoatingDrying.delta_coating_gap`; the nominal gap is retained elsewhere.
Initial memory defaults to zero. There is no clipping, saturation, deadband,
rate limit or anti-windup. Tolerance only assesses settling.

The default matched scenario uses identical recipe/rate inputs for both families,
with a sustained +2 µm roll-gap offset starting at item 26. The held branch shares
the nominal history, then freezes the correction already scheduled for item 26.
Post-disturbance averages include all 55 items 26–80, including transients and any
physical violations. Reference bounds and physical flags are diagnostics only.
Adding limits would change the original scientific behavior.

`framework.simulate(frame, family=..., **kwargs)` exposes the original sequential
core and returns `(predictions, history)`. Advanced local probes with feedback are
available through `sensitivity.sensitivity_with_feedback`: memory is fixed during
each item's perturbations and updated along the nominal sequence between items.

## Parametric sweeps, porosity and filters

```python
from battwin_framework.schema import chain_inputs
from battwin_framework.parametric import physical_checks, numeric_correlations, filter_tolerances
from battwin_framework.porosity import geometry_porosity

row = chain_inputs(framework.chain("DNN"), test).iloc[0].to_dict()
sweep = framework.parametric_sweep(
    row, {"Coating Gap (micron)": [100, 120, 140, 160],
          "Calendaring Gap (um)": [45, 55, 65]}, family="DNN",
)
flags = physical_checks(sweep)
correlation = numeric_correlations(sweep, method="spearman")
selected = filter_tolerances(sweep, {GRAVIMETRIC: (161.7, 168.3)})
geometry = geometry_porosity(sweep)
```

The sweep uses the Cartesian product in supplied insertion order, with the
original `max_items=100000` and `batch_size=2048`. `derived=...` is an explicit
optional callback. No recipe/density recalculation is enabled by default. The
historical `original_anode_recipe` helper remains available only as an opt-in.

`parametric.physical_checks` retains the earlier smaller set of sweep diagnostics;
`search.inverse_search_physical_checks` applies the stronger inverse-design
screens above. Neither changes predictions. The two flag sets are not equivalent.

Geometry porosity is `100 × (1 - 10 × dry mass loading / (solid density × coating
thickness))`, using mg/cm², g/cm³ and µm. Fixed density references remain 4.54 for
cathodes and 2.26 for anodes, with zero foil subtraction for coating-only thickness.
Invalid non-positive mass/thickness is flagged; values are not clipped. Calculated
porosity is diagnostic and does not replace learned porosity downstream.

## Saving and reproducibility

Computation writes no result files. `.save(directory)` exports tables;
`.save(directory, plots=True)` also creates figures. Reusing an output directory
overwrites corresponding exports. Choose separate method directories for AD and
FD, whose figure names share a layout. Model inference compatibility caches are
separate from result exports.

`verify_assets()` or `battwin-framework verify` checks original artifact hashes.
The tests cover reference prediction metrics, seeded search, physical
counterexamples, analytical gradients, feedback timing and replacement.
Arbitrary custom models remain the caller's reproducibility responsibility.

Use `Framework` or the explicit functional modules to compose your experiments.
No upstream models are trained by these APIs.
