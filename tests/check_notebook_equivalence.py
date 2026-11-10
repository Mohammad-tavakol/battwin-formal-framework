"""Compare package outputs to direct execution of original notebook cells.

Only the notebook project location, dataset and SAVE_RESULTS switch are changed.
This check does not use Experiment or its execution machinery.
"""
import argparse
import contextlib
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd
from battwin_framework.experiment import ASSETS


def compare(workspace):
    notebook = json.loads((ASSETS / "Framework.ipynb").read_text(encoding="utf-8"))
    records = []
    with tempfile.TemporaryDirectory(prefix="bf-notebook-") as temp:
        root = Path(temp)
        for file in ASSETS.glob("*.xlsx"):
            shutil.copy2(file, root / file.name)
        shutil.copytree(ASSETS / "best models", root / "best models")
        for dataset in ["test_aggregated_deterministic.xlsx", "train_valid_aggregated_deterministic.xlsx"]:
            scope = {"__name__": "original_notebook_baseline"}
            with contextlib.redirect_stdout(StringIO()):
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(notebook["cells"][index]["source"])
                    if index == 2:
                        source = source.replace("PROJECT_DIR = None", f"PROJECT_DIR = {str(root)!r}", 1)
                        source = source.replace("SAVE_RESULTS = True", "SAVE_RESULTS = False", 1)
                    elif index == 4:
                        source = source.replace('DATA_FILE = "test_aggregated_deterministic.xlsx"',
                                                f"DATA_FILE = {dataset!r}", 1)
                    exec(compile(source, f"Original_Framework.ipynb:cell-{index}", "exec"), scope)
                    if index == 2:
                        scope["display"] = lambda *args, **kwargs: None
                tables = {f"simulation/{family}/predictions.csv": frame
                          for family, frame in scope["predictions"].items()}
                tables.update({"importance/DNN/finite_difference_detail.csv": scope["fd_detail"],
                               "importance/DNN/finite_difference_summary.csv": scope["fd_summary"],
                               "importance/DNN/step_stability.csv": scope["step_stability"],
                               "importance/DNN/feedback_finite_differences.csv": scope["feedback_fd_table"],
                               "importance/RF/pfi.csv": scope["rf_pfi"],
                               "validation/comparison.csv": scope["comparison"],
                               "porosity/comparison.csv": scope["porosity_comparison"],
                               "porosity/metrics.csv": scope["porosity_metrics"],
                               "feedback/scenario/summary.csv": scope["scenario_summary"],
                               "inverse/summary.csv": scope["inverse_summary"]})
                for family in scope["chains"]:
                    for variable, suffix in [("feedback_results", "controlled_predictions"),
                                             ("held_results", "held_predictions"),
                                             ("feedback_traces", "trace")]:
                        tables[f"feedback/scenario/{family}/{suffix}.csv"] = scope[variable][family]
                    tables[f"parametric/{family}/predictions.csv"] = scope["sweep_results"][family]
                    tables[f"parametric/{family}/physical_checks.csv"] = scope["sweep_flags"][family]
                    for variable, suffix in [("inverse_paths", "all_evaluated_paths"),
                                             ("inverse_matches", "matching_paths"),
                                             ("inverse_nearest", "nearest_evaluated_paths")]:
                        tables[f"inverse/{family}/{suffix}.csv"] = scope[variable][family]
            for relative, frame in tables.items():
                path = Path(workspace) / "codex_work/results" / Path(dataset).stem / relative
                original = pd.read_csv(StringIO(frame.to_csv(index=False)))
                packaged = pd.read_csv(path)
                pd.testing.assert_frame_equal(original, packaged, check_dtype=False, check_exact=True)
                records.append({"dataset": dataset, "table": relative,
                                "exactly_equal": True, "rows": len(frame)})
            print(f"Original notebook versus package: exact equality for {dataset}", flush=True)
    return {"tables_compared": len(records), "all_exactly_equal": True, "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace")
    parser.add_argument("--report")
    args = parser.parse_args()
    result = compare(args.workspace)
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
