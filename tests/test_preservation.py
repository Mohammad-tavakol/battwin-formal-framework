import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from battwin_framework.experiment import ASSETS, initialize_workspace, verify_assets


class PreservationTests(unittest.TestCase):
    def test_all_scientific_cells_match_original_notebook(self):
        manifest = verify_assets()
        notebook = json.loads((ASSETS / "Framework.ipynb").read_text(encoding="utf-8"))
        lines = (ASSETS / "implementation.py").read_text(encoding="utf-8").splitlines(keepends=True)
        self.assertEqual(len(manifest["cells"]), 29)
        for entry in manifest["cells"]:
            original = "".join(notebook["cells"][entry["index"]]["source"])
            extracted = "".join(lines[entry["start"]:entry["stop"]])
            self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), entry["original_sha256"])
            self.assertEqual(ast.dump(ast.parse(original)), ast.dump(ast.parse(extracted)))
        self.assertEqual(sum(p.startswith("best models/") for p in manifest["files"]), 24)
        self.assertEqual(sum(p.endswith(".xlsx") for p in manifest["files"]), 4)

    def test_importable_mathematics_matches_notebook(self):
        import inspect
        from battwin_framework import core, indexed
        notebook = json.loads((ASSETS / "Framework.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(ast.dump(ast.parse(inspect.getsource(core))),
                         ast.dump(ast.parse("".join(notebook["cells"][6]["source"]))))
        appendix = ast.parse("".join(notebook["cells"][57]["source"]))
        assignment = next(node for node in appendix.body if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "indexed_source" for t in node.targets))
        self.assertEqual(ast.dump(ast.parse(inspect.getsource(indexed))),
                         ast.dump(ast.parse(ast.literal_eval(assignment.value))))

    def test_initialization_does_not_overwrite_edited_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "test.xlsx"
            target.write_bytes(b"user-edited dataset")
            with self.assertRaises(FileExistsError):
                initialize_workspace(temp)
            self.assertEqual(target.read_bytes(), b"user-edited dataset")
            self.assertFalse((Path(temp) / "best models").exists())

    def test_original_feedback_law_and_one_item_delay(self):
        import numpy as np
        import pandas as pd
        notebook = json.loads((ASSETS / "Framework.ipynb").read_text(encoding="utf-8"))
        scope = {}
        exec("".join(notebook["cells"][6]["source"]), scope)
        line = scope["NamedStateSpaceLine"]()
        def stage(features, zeta, control, theta):
            return {"KPI": features["input"] + zeta.get("offset", 0.0)}
        line.load_sequence([{"name": "stage", "func": stage,
                             "required_input_columns": ["input"]}])
        line.compile()
        history = line.simulate_items(pd.DataFrame({"input": [10.0, 10.0, 10.0]}),
            use_feedback=True, zeta_init={"stage": {"offset": 0.0}},
            kpi_key="KPI", kpi_target=8.0, feedback_gains={"stage": {"offset": -0.5}},
            feedback_stage_name="stage")
        np.testing.assert_array_equal([x["kpi_value"] for x in history], [10.0, 9.0, 8.5])
        for before, after in zip(history, history[1:]):
            self.assertEqual(before["zeta_next"], after["zeta_used"])


if __name__ == "__main__":
    unittest.main()
