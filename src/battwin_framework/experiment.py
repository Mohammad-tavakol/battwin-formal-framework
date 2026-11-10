"""Execution boundary around the notebook's unmodified Python source.

Each experiment owns its namespace. Cell order and scientific statements are
preserved; only the initial project/data location and export switch can be
overridden. Installed package assets are never used as an output directory.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil

ASSETS = Path(__file__).resolve().parent / "assets"
MANIFEST = ASSETS / "manifest.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_assets():
    """Verify every packaged model, dataset, notebook and source block."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = ASSETS / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Packaged artifact changed or missing: {relative}")
    lines = (ASSETS / "implementation.py").read_text(encoding="utf-8").splitlines(keepends=True)
    for entry in manifest["cells"]:
        source = "".join(lines[entry["start"]:entry["stop"]])
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != entry["sha256"]:
            raise ValueError(f"Scientific source changed: notebook cell {entry['index']}")
    return manifest


def initialize_workspace(directory):
    """Copy bundled inputs into a self-contained writable experiment folder.

    Existing files must be identical. A modified workspace is never overwritten.
    """
    manifest = verify_assets()
    root = Path(directory).resolve()
    if root == ASSETS or ASSETS in root.parents:
        raise ValueError("Choose a workspace outside the installed package assets.")
    inputs = [p for p in manifest["files"]
              if p.startswith("best models/") or p.endswith(".xlsx") or p == "Framework.ipynb"]
    for relative in inputs:
        target = root / relative
        if target.exists() and (not target.is_file() or _sha256(target) != manifest["files"][relative]):
            raise FileExistsError(f"Refusing to replace an edited workspace file: {target}")
    for relative in inputs:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(ASSETS / relative, target)
    return root


class Experiment:
    """Run the original experiments or use their loaded chains from Python.

    ``prepare()`` loads both independent chains. ``run()`` executes every
    original code cell, including PFI, finite differences, feedback, inverse
    selection and the indexed appendix. Results remain in ``namespace``.
    """

    def __init__(self, directory, dataset="test_aggregated_deterministic.xlsx", *,
                 save_results=True, quiet=False):
        self.root = initialize_workspace(directory)
        dataset_path = (self.root / dataset).resolve()
        if not dataset_path.is_relative_to(self.root):
            raise ValueError("Dataset must be inside the experiment workspace.")
        self.dataset = str(dataset_path.relative_to(self.root))
        self.save_results = bool(save_results)
        self.quiet = bool(quiet)
        self.namespace = {"__name__": "battwin_framework.experiment_session"}
        self.executed_cells = []
        self._manifest = verify_assets()
        self._lines = (ASSETS / "implementation.py").read_text(encoding="utf-8").splitlines(keepends=True)

    def _execute(self, entry):
        index = entry["index"]
        source = "".join(self._lines[entry["start"]:entry["stop"]])
        tree = ast.parse(source)
        # These three notebook configuration assignments are the only overrides.
        overrides = ({"PROJECT_DIR": str(self.root), "SAVE_RESULTS": self.save_results}
                     if index == 2 else {"DATA_FILE": self.dataset} if index == 4 else {})
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in overrides:
                    node.value = ast.copy_location(ast.Constant(overrides[target.id]), node.value)
        ast.fix_missing_locations(tree)
        exec(compile(tree, f"Framework.ipynb:cell-{index}", "exec"), self.namespace)
        if index == 2 and self.quiet:
            self.namespace["display"] = lambda *args, **kwargs: None
        self.executed_cells.append(index)

    def run(self, *, through_cell=None):
        """Continue in original order; optionally stop after a notebook cell."""
        for entry in self._manifest["cells"]:
            if through_cell is not None and entry["index"] > through_cell:
                break
            if entry["index"] not in self.executed_cells:
                self._execute(entry)
        return self

    def prepare(self):
        """Load exact supplied RF/DNN chains and simulation functions."""
        return self.run(through_cell=18)

    def predict(self, frame=None):
        """Return both independent chains' eight physical-unit outputs."""
        self.prepare()
        data = self.namespace["df_input"] if frame is None else frame
        simulate = self.namespace["simulate_without_feedback"]
        return {family: simulate(line, data)
                for family, line in self.namespace["chains"].items()}
