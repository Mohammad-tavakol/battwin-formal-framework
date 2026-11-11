"""Scientific contracts and reproducibility checks for the paper's library."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from battwin_framework import Framework, load_dataset, verify_assets
from battwin_framework.models import CallableStageModel
from battwin_framework.schema import (
    GRAVIMETRIC,
    VOLUMETRIC,
    CAL_THICKNESS,
    OUTPUT_COLUMNS,
)
from battwin_framework.sensitivity.autodiff import verify_known_autodiff_jacobian
from battwin_framework.search import verify_inverse_physical_screens


class FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.framework = Framework.from_bundle()
        cls.test = load_dataset("test")

    def test_artifacts_and_datasets(self):
        manifest = verify_assets()
        self.assertEqual(len(manifest["files"]), 24)
        self.assertEqual(len(self.test), 38)
        self.assertEqual(len(load_dataset("train_valid")), 291)

    def test_chain_metrics_and_target_leakage(self):
        result = self.framework.validate(self.test)
        anchors = {
            ("RF", "chain", GRAVIMETRIC): 0.911630525,
            ("DNN", "chain", GRAVIMETRIC): 0.953336194,
            ("RF", "individual", GRAVIMETRIC): 0.958167931,
            ("DNN", "individual", GRAVIMETRIC): 0.979165902,
            ("RF", "chain", VOLUMETRIC): 0.938636513,
            ("DNN", "chain", VOLUMETRIC): 0.951809753,
            ("RF", "individual", VOLUMETRIC): 0.949683448,
            ("DNN", "individual", VOLUMETRIC): 0.972399432,
        }
        indexed = result.metrics.set_index(["chain", "mode", "output"])
        for key, value in anchors.items():
            self.assertAlmostEqual(indexed.loc[key, "r2"], value, places=5)
        poisoned = self.test.copy()
        for output in OUTPUT_COLUMNS:
            poisoned[output] = -999999.0
        prediction = self.framework.predict(poisoned)
        for family in ["RF", "DNN"]:
            pd.testing.assert_frame_equal(
                prediction[family], result.predictions[family], check_exact=True
            )
        with self.assertRaises((KeyError, ValueError)):
            self.framework.predict_stage(self.test, family="RF", stage="CoatingDrying")

    def test_seeded_random_search_and_physical_rejection(self):
        result = self.framework.random_search(
            targets={GRAVIMETRIC: 165.0, VOLUMETRIC: 480.0}
        )
        self.assertEqual(len(result.candidates), 1000)
        self.assertEqual(
            result.candidates["Electrode type"].value_counts().to_dict(),
            {"Anode": 500, "Cathode": 500},
        )
        self.assertEqual(len(result.models["RF"].matches), 8)
        self.assertEqual(len(result.models["DNN"].matches), 1)
        for value in result.models.values():
            self.assertTrue(value.matches["Physical pass"].all())
            self.assertTrue(value.matches[GRAVIMETRIC].between(161.7, 168.3).all())
            self.assertTrue(value.matches[VOLUMETRIC].between(470.4, 489.6).all())
        self.assertEqual(
            verify_inverse_physical_screens()["invalid_cases_rejected"], 10
        )

    def test_analytic_gradient_and_finite_difference(self):
        self.assertTrue(
            verify_known_autodiff_jacobian()["analytic_two_stage_jacobian_passed"]
        )
        result = self.framework.sensitivity.compare(self.test.iloc[:4])
        self.assertTrue(result.diagnostics["forward_parity_passed"])
        self.assertEqual(len(result.derivatives), 4 * 8 * 18)
        self.assertEqual(len(result.step_size), 8 * 5)

    def test_seeded_pfi(self):
        options = {
            "features": ["Coating Gap (micron)"],
            "outputs": [GRAVIMETRIC],
            "repeats": 3,
            "seed": 42,
        }
        first = self.framework.sensitivity.pfi(self.test, **options)
        second = self.framework.sensitivity.pfi(self.test, **options)
        pd.testing.assert_frame_equal(first.summary, second.summary, check_exact=True)
        self.assertEqual(first.summary.n_repeats.iloc[0], 3)

    def test_feedback_sign_and_next_item_timing(self):
        sequence = pd.concat([self.test.iloc[[0]]] * 3, ignore_index=True)
        result = self.framework.run_feedback(
            sequence,
            family="DNN",
            kpi_key=CAL_THICKNESS,
            kpi_target=62.0,
            gain=-1.0,
            initial_memory=0.0,
        )
        trace = result.trace
        np.testing.assert_array_equal(trace.error, trace.kpi_value - 62.0)
        np.testing.assert_array_equal(
            trace.memory_next, trace.memory_used - trace.error
        )
        np.testing.assert_array_equal(
            trace.memory_used.iloc[1:], trace.memory_next.iloc[:-1]
        )
        self.assertEqual(trace.memory_used.iloc[0], 0.0)

    def test_replacement_and_restore(self):
        old = self.framework.models.stage("RF", "Electrochemical")
        columns = self.framework.models.schema("RF", "Electrochemical")["inputs"]
        model = CallableStageModel(
            "Electrochemical",
            "RF",
            columns,
            lambda data: pd.DataFrame(
                {GRAVIMETRIC: 100.0, VOLUMETRIC: 200.0}, index=data.index
            ),
        )
        previous = self.framework.replace_model("RF", "Electrochemical", model)
        self.assertIs(previous, old)
        try:
            values = self.framework.predict(self.test.iloc[:2], families=("RF",))["RF"]
            np.testing.assert_array_equal(values[GRAVIMETRIC], [100.0, 100.0])
        finally:
            self.framework.replace_model("RF", "Electrochemical", previous)
        bad = CallableStageModel("Mixing", "RF", [GRAVIMETRIC], lambda data: data)
        with self.assertRaises(ValueError):
            self.framework.replace_model("RF", "Mixing", bad)

    def test_result_export_and_plot_data(self):
        result = self.framework.validate(self.test)
        with tempfile.TemporaryDirectory(prefix="bf-test-") as temp:
            path = result.save(temp)
            plots = result.plot(temp, outputs=[GRAVIMETRIC])
            self.assertTrue((path / "metrics.csv").is_file())
            self.assertTrue(plots[0].with_suffix(".pdf").is_file())
            from battwin_framework.plotting.style import slug

            values = pd.read_csv(
                path / "validation/comparison" / f"{slug(GRAVIMETRIC)}_plot_data.csv"
            )
            self.assertIn("RF individual", values)
            self.assertIn("DNN individual", values)


if __name__ == "__main__":
    unittest.main()
