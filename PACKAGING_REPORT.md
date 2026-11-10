# Packaging validation

Validated on 10 November 2026 with Python 3.11.9 on Windows.

The current `Framework-level/Framework.ipynb` is packaged without changing its
scientific implementation or substituting any trained model. The source folder
is unchanged. The package adds a Python API, importable mathematical classes,
a command-line runner, dependency pins and reproducibility checks.

| Check | Result |
|---|---|
| Original code cells | All 29 preserve their Python syntax trees |
| Importable named and indexed mathematics | Same syntax trees as the notebook implementations |
| Original notebook, 24 model files, four workbooks | All 29 source-file checksums unchanged |
| Bundled inputs, extracted source and numerical reference | 32 artifact hashes verified |
| Preservation unit tests | 4 passed, including tests against the installed wheel |
| Unseen aggregate | All 29 cells completed, 38 rows, exports and charts generated |
| Familiar aggregate | All 29 cells completed, 291 rows, exports and charts generated |
| Original notebook execution checks | Passed in both complete runs |
| Independent original-notebook comparison | All 56 selected tables exactly equal |
| Model fitting or scientific amendments | None |

The independent comparison executes the original notebook cells directly,
without the package's Experiment runner, in the same numerical environment.
It checks RF/DNN predictions, validation metrics, geometry diagnostics, feedback
predictions/traces, parameter sweeps, PFI, finite-difference detail/summary,
step stability, feedback derivatives, and inverse paths/selections. Export is
disabled for this direct baseline; scientific settings and calculations remain
unchanged. Exact comparison is made after the same CSV round-trip in both branches.

The original matching-path counts are preserved: unseen RF/DNN = 1/0 and
familiar RF/DNN = 4/9. These are evaluated-path selections, not an optimization.

## Historical-reference comparison

The strict comparison with saved historical exports passes 113 of 118 tables
at rtol=0.00002 and atol=0.0002. Four DNN finite-difference detail/summary tables
are flagged. In the unseen simulation, the largest DNN forward-output difference
from those historical exports is approximately 0.0000722 in physical units;
finite differences amplify small forward differences through division by the
unchanged perturbation step. The largest unseen raw-derivative difference is
approximately 0.188. The original notebook executed in the current environment
produces the package's derivative tables exactly. No step sizes, models,
formulas or tolerances were changed to conceal the historical differences.

The fifth flagged table is the familiar RF nearest-evaluated-path diagnostic.
It contains the same five candidates, with the order of two equally ranked
candidates swapped. Direct notebook execution has exactly the package's order.
The successful inverse matches and acceptance criteria remain unchanged.

`tests/check_reproduction.py` deliberately retains this strict historical test
and its nonzero exit status when differences occur. The separate
`tests/check_notebook_equivalence.py` verifies the packaging acceptance criterion:
the current package and current original notebook yield identical selected
scientific results. Machine-readable evidence is in `validation.json`.

## Reproduction boundary

The notebook uses frozen supplied models and performs no training. The package
reproduces its experiments; upstream training procedures are not introduced.
It uses the notebook's existing model selections, category handling, target
inverses, density references, feedback law, PFI and plotting code. The original
inference-compatibility cache logic is retained unchanged. Only the initial
project directory, input-dataset selection and export switch are configurable
at the execution boundary.

The wheel builds from a short Windows path, installs without editing the source
project, and contains the original model binaries and datasets. Building and
validation use system temporary folders; generated runs and build caches are
excluded from the repository. Pinned dependencies describe the verified Python
3.11 Windows environment. Figures use the original plotting implementation;
rendering can depend on installed fonts.
