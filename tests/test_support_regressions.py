"""Integration follow-ups: hidden masked values must not supply observations."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from src.depth_wizard.elevation_engine import ElevationEngine, robust_affine_calibration_irls
from src.depth_wizard.srtm_provider import SRTMElevationProvider
from src.depth_wizard.benchmark import DepthWizardBenchmark


class TestMaskedSupport(unittest.TestCase):
    def setUp(self):
        self.engine = ElevationEngine.__new__(ElevationEngine)
        self.surface = np.ma.array(np.zeros((40, 40)), mask=False)
        self.surface.mask[10, 10] = True

    def test_all_geometric_screens_reject_hidden_masked_pixels(self):
        results = [
            self.engine.detect_landing_zones(self.surface, self.surface, np.zeros((40, 40)),
                                              ground_res_m=1, pad_radius_m=4),
            self.engine.screen_landslide_risk(self.surface),
            self.engine.simulate_flood(self.surface, 1),
        ]
        for result in results:
            with self.subTest(scope=result["assessment_scope"]):
                self.assertEqual(result["status"], "NOT_ASSESSED")
                self.assertEqual(result["diagnostics"]["invalid_pixel_count"], 1)
                json.dumps(result, allow_nan=False)

    def test_export_cannot_turn_mask_into_finite_height(self):
        with self.assertRaisesRegex(ValueError, "NODATA"):
            self.engine.export_dsm_geotiff(self.surface)
        with self.assertRaisesRegex(ValueError, "NODATA"):
            self.engine.export_dsm_obj(self.surface)

    def test_calibration_preserves_masks_as_missing_support(self):
        for target in ("relative", "terrain", "provider"):
            with self.subTest(target=target):
                relative = self.surface if target == "relative" else np.zeros((40, 40))
                kwargs = {"terrain_grid": self.surface} if target == "terrain" else {}
                if target == "provider":
                    kwargs["geo_bounds"] = [0, 0, 1, 1]
                with patch.object(SRTMElevationProvider, "get_elevation_grid", return_value=self.surface):
                    result = self.engine.calibrate_to_absolute_dsm(relative, 50, **kwargs)
                self.assertFalse(result["stats"]["is_metric"])
                self.assertEqual(result["stats"]["status"], "NOT_ASSESSED")
                self.assertEqual(result["stats"]["valid_pixel_count"], 1599)

    def test_masked_controls_do_not_affect_solver(self):
        reference = np.ma.array([2., 5., 8., 1000.], mask=[False, False, False, True])
        fit = robust_affine_calibration_irls([0, 1, 2, 3], reference)
        self.assertTrue(fit["converged"])
        self.assertEqual(fit["valid_sample_count"], 3)
        self.assertEqual(fit["discarded_sample_count"], 1)
        self.assertAlmostEqual(fit["scale"], 3)
        with self.assertRaises(ValueError):
            SRTMElevationProvider._valid_elevations(reference, 4)

    def test_missing_metric_status_cannot_enter_the_accuracy_aggregate(self):
        class MissingStatusEngine:
            def load_gamus_scene(self, scene_id, **kwargs):
                return {"name": "Synthetic controlled fixture", "is_synthetic": True,
                        "terrain_type": "Fixture", "rgb_image": np.zeros((8, 8, 3), dtype=np.uint8),
                        "base_elevation_m": 10, "max_structural_height_m": 25,
                        "ground_truth_dsm": np.ones((8, 8)) * 10}

            def extract_relative_depth(self, image):
                return np.zeros((8, 8))

            def calibrate_to_absolute_dsm(self, **kwargs):
                return {"dsm": np.ones((8, 8)) * 10}  # No supported metric-status declaration.

        report = DepthWizardBenchmark.run_suite(MissingStatusEngine(), terrain_specs=[
            {"id": "fixture", "cat": "Fixture", "label": "Controlled fixture"}])
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["failed_scenes"][0]["reason"], "METRIC_SCALE_NOT_ESTABLISHED")
        self.assertEqual(report["benchmark_summary"]["evaluated_scenes_count"], 0)


if __name__ == "__main__":
    unittest.main()
