import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import soil_gsd_pipeline as pipeline  # noqa: E402


class SoilGsdPipelineTests(unittest.TestCase):
    def test_target_columns_keep_decimal_boundaries_distinct(self):
        resolved = pipeline.resolve_target_columns(["sample_id", "2.0", "20", "6.3", "63"])
        self.assertEqual(resolved["2.0"], "2")
        self.assertEqual(resolved["20"], "20")
        self.assertEqual(resolved["6.3"], "6.3")
        self.assertEqual(resolved["63"], "63")

    def test_repair_cdf_enforces_submission_constraints(self):
        raw = np.array([[30.0, 10.0, 105.0, -4.0, 20.0] + [0.0] * 6])
        repaired = pipeline.repair_cdf(raw)
        self.assertTrue(np.all(repaired >= 0.0))
        self.assertTrue(np.all(repaired <= 100.0))
        self.assertTrue(np.all(np.diff(repaired, axis=1) >= 0.0))
        self.assertEqual(repaired[0, -1], 100.0)

    def test_metric_is_zero_for_identical_curves(self):
        curve = np.array([[0.0, 1.0, 4.0, 8.0, 18.0, 35.0, 52.0, 68.0, 82.0, 94.0, 100.0]])
        self.assertAlmostEqual(pipeline.log_weighted_emd(curve, curve), 0.0, places=8)

    def test_metric_has_competition_scale(self):
        truth = np.zeros((1, 11), dtype=float)
        prediction = np.ones((1, 11), dtype=float) * 100.0
        prediction[:, -1] = 100.0
        expected = 100.0 * np.log10(200.0 / 0.002)
        self.assertAlmostEqual(pipeline.log_weighted_emd(truth, prediction), expected, places=8)

    def test_photo_aggregation_dimension_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            one = Path(directory) / "one.png"
            two = Path(directory) / "two.png"
            Image.fromarray(np.full((24, 24, 3), 100, dtype=np.uint8)).save(one)
            Image.fromarray(np.full((24, 24, 3), 150, dtype=np.uint8)).save(two)
            self.assertEqual(
                pipeline.aggregate_image_features([one]).shape,
                pipeline.aggregate_image_features([one, two]).shape,
            )

    def test_image_matching_handles_transliterated_sample_id(self):
        paths = [Path("iPhone14_HPC_M\N{REPLACEMENT CHARACTER}nster_BS6_9,0-10m (1).JPG")]
        matched = pipeline.match_images("HPC_Muenster_BS6_9_0-10m", paths)
        self.assertEqual(matched, paths)

    def test_target_column_resolution_accepts_numeric_headers(self):
        resolved = pipeline.resolve_target_columns(["0.0020", "0.006300", "200.0", "not_a_target"])
        self.assertEqual(resolved["0.0020"], "0.002")
        self.assertEqual(resolved["0.006300"], "0.0063")
        self.assertEqual(resolved["200.0"], "200")
        self.assertNotIn("not_a_target", resolved)

    def test_ppm_features_add_fixed_five_value_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            Image.fromarray(np.full((24, 24, 3), 100, dtype=np.uint8)).save(first)
            Image.fromarray(np.full((24, 24, 3), 150, dtype=np.uint8)).save(second)
            visual = pipeline.aggregate_image_features([first, second])
            calibrated = pipeline.aggregate_image_features([first, second], [10.0, 20.0])
            self.assertEqual(calibrated.shape[0], visual.shape[0] + 5)
            self.assertTrue(np.all(np.isfinite(calibrated)))

    def test_metric_repairs_nonmonotone_predictions_before_scoring(self):
        truth = np.array([[0.0, 20.0, 40.0, 60.0, 80.0, 90.0, 94.0, 97.0, 99.0, 100.0, 100.0]])
        raw = truth.copy()
        raw[0, 1] = 5.0
        raw[0, 3] = 120.0
        self.assertGreaterEqual(pipeline.log_weighted_emd(truth, raw), 0.0)
        self.assertTrue(np.all(np.diff(pipeline.repair_cdf(raw), axis=1) >= 0.0))

    def test_find_csv_helpers_choose_labels_and_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "Training_labels_updated.csv"
            submission = root / "sample_submission.csv"
            columns = ["sample_id", *pipeline.TARGET_COLUMNS]
            frame = np.zeros((1, len(columns)))
            frame[:, 0] = np.nan
            import pandas as pd
            pd.DataFrame([["A", *([1.0] * 11)]], columns=columns).to_csv(labels, index=False)
            pd.DataFrame([["B", *([0.0] * 11)]], columns=columns).to_csv(submission, index=False)
            self.assertEqual(pipeline.find_label_csv(root), labels)
            self.assertEqual(pipeline.find_submission_csv(root), submission)

    def test_repair_cdf_preserves_valid_curve_and_forces_endpoint(self):
        curve = np.array([[1.0, 3.0, 7.0, 12.0, 22.0, 35.0, 50.0, 68.0, 82.0, 95.0, 98.0]])
        repaired = pipeline.repair_cdf(curve)
        np.testing.assert_allclose(repaired[0, :-1], curve[0, :-1])
        self.assertEqual(repaired[0, -1], 100.0)


if __name__ == "__main__":
    unittest.main()
