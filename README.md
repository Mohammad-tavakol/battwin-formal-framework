# BATTWIN formal framework

A modular library for the coupled Mixing → Coating/Drying → Calendering →
Electrochemical framework, with explicit APIs for simulation, validation,
inverse design, sensitivity and feedback. The library uses saved model weights,
preprocessing and physical-unit target transforms.

paper preprint: Tavakol Sadrabadi, Mohammad and Su, Zhizhuo and Faraji Niri, Mona and Ceglarek, Darek, A Formal Digital Twinning Framework for Integrated Multi-Stage Modelling and Analysis of Battery Manufacturing (November 10, 2026). Available at SSRN: https://ssrn.com/abstract=7727858 or http://dx.doi.org/10.2139/ssrn.7727858

## Install

Use Python 3.11. On Windows, use a short checkout path such as `C:\src\battwin-framework`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps .
battwin-framework verify
```

The package contains the 20 required model artifacts and four workbooks. No model
or dataset download is needed for inference. Dependencies remain pinned.

## Start here

```python
from battwin_framework import Framework

framework = Framework.from_bundle()
test = framework.load_dataset("test")

predictions = framework.predict(test)  # independent RF and DNN DataFrames
validation = framework.validate(test)  # chain versus standalone electrochemical
print(validation.metrics)             # R², RMSE, MAE, mean error, MAPE, n, status
validation.save("results/validation", plots=True)

print(framework.describe("DNN"))
print(framework.models.schema("DNN", "Calendering"))
```

Computation stays in memory until `.save(directory)` or `.plot(directory)` is
called. Models load lazily; each Framework has its own replaceable model registry.

## Capabilities

| Operation | Entry point |
|---|---|
| Coupled predictions | `framework.predict(data)` |
| Standalone stage | `framework.predict_stage(data, family="DNN", stage="Calendering")` |
| Accuracy and comparison figures | `framework.validate(data)` |
| Inverse design / existing optimiser | `framework.random_search(targets=...)` |
| Replace one model | `framework.replace_model(family, stage, model)` |
| RF permutation importance | `framework.sensitivity.pfi(data)` |
| DNN forward finite differences | `framework.sensitivity.finite_difference(data)` |
| DNN automatic differentiation | `framework.sensitivity.autodiff(data)` |
| AD/FD comparison and step study | `framework.sensitivity.compare(data)` |
| Additive feedback | `framework.run_feedback(data, ...)` |
| Matched disturbance scenario | `framework.disturbance_rejection(data)` |
| Cartesian sweep | `framework.parametric_sweep(base_row, ranges)` |
| Porosity, physical screens, correlations | `porosity`, `search`, `parametric` modules |
| Sequential history / indexed maths | `framework.simulate(...)`, `NamedStateSpaceLine`, `StateSpaceLine` |

The inverse method remains the existing **independent local random search** around
training recipes, followed by target and physical screening.

```python
from battwin_framework.schema import GRAVIMETRIC, VOLUMETRIC

search = framework.random_search(
    targets={GRAVIMETRIC: 165., VOLUMETRIC: 480.},
    n_sweeps=1000, random_seed=42, tolerance_fraction=.02,
)
print(search.summary)
print(search.models["DNN"].matches)
search.save("results/inverse", plots=True)

ad = framework.sensitivity.autodiff(test)
print(ad.summary)
comparison = framework.sensitivity.compare(test)
comparison.save("results/ad_vs_fd", plots=True)
```

See the [usage guide](docs/USAGE.md) for model replacement, all sensitivity methods,
search sampling, feedback, units and outputs. The [API map](docs/API.md) explains
module boundaries and result fields. [Examples](examples/) are executable scripts.

## Command line

```powershell
battwin-framework describe --family DNN
battwin-framework predict --dataset test --family DNN --output results/prediction
battwin-framework validate --dataset test --output results/validation --plots
battwin-framework sensitivity --method pfi --dataset test --output results/pfi
battwin-framework sensitivity --method compare --dataset test --output results/ad_vs_fd --plots
battwin-framework search --cg 165 --cv 480 --sweeps 1000 --seed 42 --tolerance .02 --output results/inverse --plots
battwin-framework feedback --dataset test --output results/feedback --plots
```

Use `--project PATH` for a directory containing your `best models/` and datasets.
Each command has `--help`. Custom controller sequences and model replacement use
the Python API.

## Reproducibility

The library imports Python modules without executing notebook cells. Named and
indexed state-space mathematics are directly importable. All experiments use
the supplied frozen models; no upstream model training is performed by this package.

```powershell
python -m unittest discover -s tests -v
```

Tests check artifact integrity, reference prediction metrics, seeded search,
physical rejection, analytical gradients, feedback timing, model replacement,
and exported plot data. Dependencies, datasets, trained models and random-search
settings are supplied for reproduction. Numerical rounding can depend on the
TensorFlow/CPU environment; the forward graph includes explicit parity checks.
