"""Offline regressions for honest terrain/control and geometric-screen semantics."""

import io
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from src.depth_wizard import elevation_engine as module
from src.depth_wizard.elevation_engine import ElevationEngine, robust_affine_calibration_irls
from src.depth_wizard.srtm_provider import ElevationSourceError, SRTMElevationProvider


class TestIndependentControlIRLS(unittest.TestCase):
    def test_affine_fit_and_finite_sample_filter(self):
        x = np.array([0.0, 0.2, 0.5, 0.8, 1.0, np.nan, 2.0])
        y = 12 * x + 4
        y[-1] = np.inf
        fit = robust_affine_calibration_irls(x, y)
        self.assertEqual(fit["status"], "FITTED")
        self.assertTrue(fit["converged"])
        self.assertAlmostEqual(fit["scale"], 12)
        self.assertAlmostEqual(fit["offset"], 4)
        self.assertLess(fit["rmse_m"], 1e-10)
        self.assertEqual(fit["valid_sample_count"], 5)
        self.assertEqual(fit["discarded_sample_count"], 2)

    def test_outlier_robustness(self):
        x = np.linspace(0, 1, 101)
        y = 20 * x + 3
        y[50] += 100
        fit = robust_affine_calibration_irls(x, y)
        self.assertTrue(fit["converged"])
        self.assertAlmostEqual(fit["scale"], 20, places=3)
        self.assertLess(abs(fit["offset"] - 3), 0.05)
        self.assertEqual(fit["inlier_count"], 100)
        self.assertGreater(fit["rmse_m"], 1)  # Actual residuals remain disclosed.

    def test_rank_weights_condition_and_invalid_lengths(self):
        for x, y, weights, status in [
            ([1, 1, 1], [2, 3, 4], None, "RANK_DEFICIENT"),
            ([0, 1], [2, 3], [0, 0], "INSUFFICIENT_SUPPORT"),
            ([0, 1], [2, 3], [1, -1], "INVALID_WEIGHTS"),
            ([0, 1], [2, 3], [1, np.nan], "INVALID_WEIGHTS"),
            ([0, 1], [2, 3], [1, np.inf], "INVALID_WEIGHTS"),
            ([0, 1], [2], None, "INVALID_INPUT"),
            ([0, 1], [2, 3], [1], "INVALID_INPUT"),
            ([0, 1, 2], [2, 3, 4], [1, 1e-30, 1e-30], "ILL_CONDITIONED"),
        ]:
            with self.subTest(status=status, x=x, weights=weights):
                fit = robust_affine_calibration_irls(x, y, weights=weights)
                self.assertFalse(fit["converged"])
                self.assertEqual(fit["status"], status)
                self.assertIsNone(fit["scale"])
                json.dumps(fit, allow_nan=False)

    def test_zero_weight_and_positive_scale_requirement(self):
        fit = robust_affine_calibration_irls([0, 1, 2], [4, 7, 1000], [1, 1, 0])
        self.assertTrue(fit["converged"])
        self.assertAlmostEqual(fit["scale"], 3)
        negative = robust_affine_calibration_irls([0, 1, 2], [5, 3, 1])
        self.assertFalse(negative["converged"])
        self.assertEqual(negative["status"], "NONPOSITIVE_SCALE")
        signed = robust_affine_calibration_irls([0, 1, 2], [5, 3, 1], require_positive_scale=False)
        self.assertTrue(signed["converged"])
        self.assertAlmostEqual(signed["scale"], -2)
        constant = robust_affine_calibration_irls([0, 1, 2], [3, 3, 3])
        self.assertFalse(constant["converged"])
        self.assertEqual(constant["status"], "NONPOSITIVE_SCALE")

    def test_invalid_solver_parameters_and_nonconvergence(self):
        for kwargs in [{"max_iter": 0}, {"huber_delta": 0}, {"tol": np.nan}]:
            fit = robust_affine_calibration_irls([0, 1, 2], [1, 3, 5], **kwargs)
            self.assertEqual(fit["status"], "INVALID_INPUT")
        x = np.linspace(0, 1, 20)
        y = 8 * x + 2
        y[-1] += 100
        fit = robust_affine_calibration_irls(x, y, max_iter=1, tol=1e-12)
        self.assertFalse(fit["converged"])
        self.assertEqual(fit["status"], "NOT_CONVERGED")


class TestTerrainConsistencyAndProvenance(unittest.TestCase):
    def setUp(self):
        # Numerical regressions need no neural model or HF downloads.
        self.engine = ElevationEngine.__new__(ElevationEngine)
        self.bounds = [72, 22, 73, 23]

    @staticmethod
    def verified_terrain():
        return {"source": "INDEPENDENT_FIELD_TERRAIN_SURVEY", "status": "VALIDATED",
                "source_surface_type": "DTM", "bare_earth_certified": True,
                "independent_of_image": True, "datum_verified": True, "vertical_datum": "SURVEY_LOCAL_DATUM",
                "horizontal_crs": "EPSG:4326", "grid_aligned": True, "support": {"valid_sample_count": 64}}

    @staticmethod
    def controls():
        return {"source": "independent_field_survey", "independent": True, "measurement_type": "SURVEYED_AGL",
                "pixel_x": [0, 3, 7], "pixel_y": [0, 4, 7],
                "heights_agl_m": [2, 2 + 20 * 3 / 7, 22]}

    def test_independent_relief_is_never_replaced_by_appearance_fit(self):
        y, x = np.mgrid[:64, :64]
        terrain = (150 + 50 * np.sin(x / 6) * np.cos(y / 8)).astype(np.float32)
        relative = (x / 63).astype(np.float32)
        with patch.object(SRTMElevationProvider, "get_elevation_grid", return_value=terrain):
            result = self.engine.calibrate_to_absolute_dsm(relative, 150, geo_bounds=self.bounds)
        np.testing.assert_array_equal(result["dtm"], terrain)
        np.testing.assert_allclose(result["dsm"], result["dtm"] + result["structural_heights"], atol=0)
        np.testing.assert_allclose(result["dsm"] - result["dtm"], result["structural_heights"], atol=2e-5)
        self.assertEqual(result["stats"]["irls_huber_fit"]["status"], "NOT_RUN")
        self.assertFalse(result["stats"]["is_metric"])
        self.assertEqual(result["stats"]["height_scale_status"], "ASSUMED")

    def test_bbox_and_assumed_plane_do_not_establish_metric_status(self):
        relative = np.ones((8, 8)) * 0.5
        for bounds in [None, self.bounds]:
            with patch.object(SRTMElevationProvider, "get_elevation_grid", side_effect=RuntimeError("offline")):
                result = self.engine.calibrate_to_absolute_dsm(relative, 50, geo_bounds=bounds)
            self.assertFalse(result["stats"]["is_metric"])
            self.assertEqual(result["stats"]["calibration_status"], "UNCALIBRATED")
            self.assertEqual(result["stats"]["vertical_datum"], "ASSUMED_LOCAL_BASE")
            self.assertEqual(result["stats"]["dtm_source"], "USER_SPECIFIED")
            if bounds:
                self.assertEqual(result["stats"]["terrain_status"], "UNAVAILABLE_ASSUMED_PLANE")
                self.assertIn("requested_source_failure", result["stats"]["terrain_provenance"])

    def test_invalid_and_nodata_terrain_anchors_are_not_metric(self):
        relative = np.broadcast_to(np.linspace(0, 1, 8), (8, 8)).copy()
        for invalid in [np.nan, np.inf, -9999]:
            terrain = np.full((8, 8), invalid)
            result = self.engine.calibrate_to_absolute_dsm(
                relative, 50, terrain_grid=terrain, terrain_provenance=self.verified_terrain(),
                independent_height_controls=self.controls(), geo_bounds=self.bounds)
            self.assertFalse(result["stats"]["is_metric"])
            self.assertEqual(result["stats"]["status"], "NOT_ASSESSED")
            self.assertEqual(result["stats"]["terrain_status"], "INVALID_SOURCE_SAMPLES")
            self.assertIsNone(result["stats"]["min_elevation_m"])
            json.dumps(result["stats"], allow_nan=False)

    def test_supported_independent_agl_transform_preserves_valid_terrain(self):
        relative = np.broadcast_to(np.linspace(0, 1, 8), (8, 8)).copy()
        terrain = (50 + np.arange(64).reshape(8, 8) / 5).astype(np.float32)
        result = self.engine.calibrate_to_absolute_dsm(
            relative, 50, terrain_grid=terrain, terrain_provenance=self.verified_terrain(),
            independent_height_controls=self.controls(), geo_bounds=self.bounds)
        self.assertTrue(result["stats"]["is_metric"])
        self.assertEqual(result["stats"]["height_scale_status"], "INDEPENDENT_AGL_CALIBRATED")
        np.testing.assert_array_equal(result["dtm"], terrain)
        np.testing.assert_allclose(result["structural_heights"], relative * 20 + 2, atol=2e-6)
        np.testing.assert_array_equal(result["dsm"], result["dtm"] + result["structural_heights"])

    def test_nodata_evaluation_and_zero_weight_controls_cannot_calibrate(self):
        relative = np.broadcast_to(np.linspace(0, 1, 8), (8, 8)).copy()
        for overrides in [{"heights_agl_m": [2, np.nan, 22]}, {"heights_agl_m": [2, -9999, 22]},
                          {"heights_agl_m": [10, 10, 10]},
                          {"nodata": 22}, {"measurement_type": []},
                          {"evaluation_only": True}, {"source": "benchmark_ground_truth"},
                          {"weights": [0, 1, 1]}, {"weights": [1, -1, 1]}]:
            with self.subTest(overrides=overrides):
                result = self.engine.calibrate_to_absolute_dsm(
                    relative, 50, terrain_grid=np.full((8, 8), 50), terrain_provenance=self.verified_terrain(),
                    independent_height_controls={**self.controls(), **overrides}, geo_bounds=self.bounds)
                self.assertFalse(result["stats"]["is_metric"])
                self.assertEqual(result["stats"]["height_scale_status"], "ASSUMED")

    def test_copernicus_remains_surface_reference_not_bare_earth(self):
        provenance = SRTMElevationProvider._provenance("OPEN_METEO_COPERNICUS_GLO30", 25)
        provenance["support"]["bounds_wgs84"] = self.bounds
        record = {"grid": np.full((8, 8), 50), "provenance": provenance}
        with patch.object(SRTMElevationProvider, "get_elevation_grid", return_value=record):
            result = self.engine.calibrate_to_absolute_dsm(np.ones((8, 8)) * 0.5, 50, geo_bounds=self.bounds)
        self.assertEqual(result["stats"]["terrain_reference_kind"], "DSM")
        self.assertFalse(result["stats"]["terrain_provenance"]["bare_earth_certified"])
        self.assertFalse(result["stats"]["is_metric"])
        self.assertEqual(result["stats"]["vertical_datum"], "UNKNOWN")
        self.assertEqual(result["stats"]["horizontal_reference_status"], "SOURCE_COORDINATES_AVAILABLE_ALIGNMENT_UNVERIFIED")


class TestPublicElevationSource(unittest.TestCase):
    def setUp(self):
        SRTMElevationProvider.clear_cache()

    def test_offline_source_raises_without_analytical_or_centre_fallback(self):
        with patch("src.depth_wizard.srtm_provider.HAS_URLLIB", False):
            for operation in [lambda: SRTMElevationProvider.get_elevation_at_point(23, 72),
                              lambda: SRTMElevationProvider.get_elevation_grid([72, 22, 73, 23], 8, 8)]:
                with self.assertRaises(ElevationSourceError) as caught:
                    operation()
                self.assertEqual(caught.exception.provenance["status"], "UNAVAILABLE")
                self.assertEqual(caught.exception.provenance["support"]["valid_sample_count"], 0)
        self.assertFalse(SRTMElevationProvider._cache)

    def test_invalid_api_samples_fail_closed(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({"elevation": [-9999] * 25}).encode()
        with patch("src.depth_wizard.srtm_provider.urllib.request.urlopen", side_effect=[response, OSError("offline")]):
            with self.assertRaises(ElevationSourceError):
                SRTMElevationProvider.get_elevation_grid([72, 22, 73, 23], 8, 8)

    def test_grid_north_to_south_and_coarse_support_provenance(self):
        def samples(lats, lons, timeout_s):
            return np.asarray(lats) * 10, SRTMElevationProvider._provenance("OPEN_METEO_COPERNICUS_GLO30", len(lats))

        with patch.object(SRTMElevationProvider, "_fetch_samples", side_effect=samples):
            record = SRTMElevationProvider.get_elevation_grid([72, 22, 73, 23], 8, 12, return_metadata=True)
            legacy = SRTMElevationProvider.get_elevation_grid([72, 22, 73, 23], 8, 12)
        self.assertEqual(record["grid"].shape, (8, 12))
        np.testing.assert_allclose(record["grid"][0], 230)
        np.testing.assert_allclose(record["grid"][-1], 220)
        np.testing.assert_array_equal(legacy, record["grid"])
        self.assertEqual(record["support"]["valid_sample_count"], 25)
        self.assertFalse(record["support"]["fine_resolution_observation"])
        self.assertEqual(record["source_surface_type"], "DSM")
        self.assertFalse(record["bare_earth_certified"])

    def test_provider_validates_coordinates_and_output_sizes_before_network(self):
        with patch.object(SRTMElevationProvider, "_fetch_samples") as fetch:
            for bounds, rows, cols in [([72, 23, 73, 22], 8, 8), ([1000, 22, 1001, 23], 8, 8),
                                      ([72, 22, 73, 23], 0, 8), ([72, 22, 73, 23], 2049, 8)]:
                with self.assertRaises(ValueError):
                    SRTMElevationProvider.get_elevation_grid(bounds, rows, cols)
            fetch.assert_not_called()


class TestGeometricSafetyScreens(unittest.TestCase):
    def setUp(self):
        self.engine = ElevationEngine.__new__(ElevationEngine)
        self.terrain = np.full((64, 64), 50.0)
        self.heights = np.zeros((64, 64))

    def assert_json_safe(self, value):
        json.dumps(value, allow_nan=False)

    def test_nan_surfaces_never_produce_landing_or_hazard_results(self):
        for field in ["dsm", "dtm", "structural_heights"]:
            inputs = {"dsm": self.terrain.copy(), "dtm": self.terrain.copy(), "structural_heights": self.heights.copy()}
            inputs[field][:] = np.nan
            result = self.engine.detect_landing_zones(**inputs)
            self.assertEqual(result["status"], "NOT_ASSESSED")
            self.assertEqual(result["detected_zones_count"], 0)
            self.assertEqual(result["candidate_zones"], [])
            self.assert_json_safe(result)
        risk = self.engine.screen_landslide_risk(np.full((64, 64), np.nan))
        self.assertEqual(risk["status"], "NOT_ASSESSED")
        self.assertIsNone(risk["mean_slope_deg"])
        self.assertEqual(risk["slope_band_counts"], {})
        self.assert_json_safe(risk)
        flood = self.engine.simulate_flood(np.full((64, 64), np.nan), 55)
        self.assertEqual(flood["status"], "NOT_ASSESSED")
        self.assert_json_safe(flood)

    def test_invalid_shape_sampling_pad_and_decomposition_are_not_assessed(self):
        for grid in [np.array([]), np.ones((1, 4)), np.ones(8), np.full((64, 64), np.inf)]:
            result = self.engine.screen_landslide_risk(grid)
            self.assertEqual(result["status"], "NOT_ASSESSED")
            self.assert_json_safe(result)
        for resolution in [0, -1, np.nan, np.inf, 1e-300]:
            result = self.engine.detect_landing_zones(self.terrain, self.terrain, self.heights, ground_res_m=resolution)
            self.assertEqual(result["status"], "NOT_ASSESSED")
            self.assert_json_safe(result)
        for kwargs in [{"pad_radius_m": 0}, {"pad_radius_m": -1}, {"pad_radius_m": np.nan},
                       {"pad_radius_m": 501}, {"max_slope_deg": 0}, {"max_slope_deg": np.inf}]:
            result = self.engine.detect_landing_zones(self.terrain, self.terrain, self.heights, **kwargs)
            self.assertEqual(result["status"], "NOT_ASSESSED")
            self.assert_json_safe(result)
        for dsm, terrain, height in [(self.terrain, self.terrain[:10], self.heights),
                                     (self.terrain + 10, self.terrain, self.heights)]:
            result = self.engine.detect_landing_zones(dsm, terrain, height)
            self.assertEqual(result["status"], "NOT_ASSESSED")
            self.assert_json_safe(result)

    def test_ring_obstacle_centres_and_complete_local_disks_remain_clear(self):
        size = 101
        yy, xx = np.mgrid[:size, :size]
        radial = np.hypot(xx - 50, yy - 50)
        heights = np.where((radial >= 20) & (radial <= 24), 8.0, 0.0)
        terrain = np.full((size, size), 50.0)
        result = self.engine.detect_landing_zones(terrain + heights, terrain, heights, ground_res_m=1, pad_radius_m=8)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertGreater(result["detected_zones_count"], 0)
        self.assertFalse(result["flight_safety_assessed"])
        for zone in result["candidate_zones"]:
            ix, iy = zone["pixel_x"], zone["pixel_y"]
            self.assertEqual(heights[iy, ix], 0)
            disk = np.hypot(xx - ix, yy - iy) <= 8 + 1 / np.sqrt(2)
            self.assertTrue(np.all(heights[disk] <= 0.5))
            self.assertGreaterEqual(ix - 8, 0)
            self.assertLess(ix + 8, size)
            self.assertNotIn("reconnaissance_verified", zone)
            self.assertEqual(zone["suitability"], "GEOMETRIC_CANDIDATE_ONLY")
        self.assert_json_safe(result)

    def test_border_and_undersized_scene_cannot_supply_a_complete_pad(self):
        terrain = np.full((10, 10), 50.0)
        result = self.engine.detect_landing_zones(terrain, terrain, np.zeros_like(terrain), ground_res_m=1, pad_radius_m=8)
        self.assertEqual(result["detected_zones_count"], 0)
        self.assertEqual(result["candidate_zones"], [])

    def test_screen_is_geometric_only_without_stability_or_standard_projection(self):
        terrain = self.terrain + np.arange(64)[None, :] * 0.25
        result = self.engine.screen_landslide_risk(terrain, ground_res_m=1)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["assessment_scope"], "GEOMETRIC_SCREEN_ONLY")
        self.assertFalse(result["landslide_risk_assessed"])
        self.assertFalse(result["standards_compliance_assessed"])
        self.assertTrue(result["missing_geotechnical_factors"])
        self.assertEqual(result["low_slope_pct"], 100)
        self.assertEqual(result["relative_relief_m"], 15.75)
        self.assertNotIn("stable_pct", result)
        self.assertFalse(any(key.startswith("bis_is14496") for key in result))
        self.assert_json_safe(result)


class TestSurfaceLoadingDisclosure(unittest.TestCase):
    def setUp(self):
        self.engine = ElevationEngine.__new__(ElevationEngine)

    def test_synthetic_fixtures_have_no_invented_geography_and_consistent_references(self):
        for scene_id in ["SAC_AHMEDABAD", "HILLY_RIDGE"]:
            scene = self.engine.load_gamus_scene(scene_id, 64)
            self.assertTrue(scene["is_synthetic"])
            self.assertIsNone(scene["geo_metadata"]["bounds"])
            self.assertEqual(scene["geo_metadata"]["crs"], "UNREFERENCED")
            self.assertIsNone(scene["reference_provenance"]["geography"])
            np.testing.assert_allclose(scene["ground_truth_dsm"], scene["ground_truth_dtm"] + scene["ground_truth_agl"])

    def test_local_agl_is_raw_with_fixed_base_disclosure(self):
        import h5py

        scene = self.engine.load_gamus_scene("DC_04_23", 64)
        step = scene["geo_metadata"]["pixel_stride"]
        with h5py.File("data/gamus_sample/DC_04_23_AGL.h5", "r") as sample:
            raw = np.asarray(sample["image"][::step, ::step])
        np.testing.assert_array_equal(scene["ground_truth_agl"], raw)
        np.testing.assert_array_equal(scene["ground_truth_dsm"], raw + 15)
        self.assertFalse(scene["is_synthetic"])
        self.assertFalse(scene["reference_provenance"]["absolute_dsm_available"])
        self.assertEqual(scene["max_structural_height_m"], 25)
        self.assertIsNone(scene["geo_metadata"]["bounds"])
        self.assertIn("RAW_AGL", scene["reference_surface_semantics"])

    def test_model_unavailability_and_claimed_upload_georeference_are_disclosed(self):
        rgb = np.full((64, 64, 3), 120, dtype=np.uint8)
        rgb[20:40, 20:40] = 230
        with patch.object(module, "HAS_TORCH_TRANSFORMERS", False):
            result = self.engine.extract_relative_depth(rgb)
            self.assertTrue(np.all(np.isfinite(result)))
            self.assertEqual(self.engine.last_inference_provenance["mode"], "APPEARANCE_HEURISTIC")
            self.assertFalse(self.engine.last_inference_provenance["neural_model_used"])
            self.assertFalse(self.engine.last_inference_provenance["metric_accuracy_validated"])
            payload = io.BytesIO()
            Image.fromarray(rgb).save(payload, format="PNG")
            scene = self.engine.process_image_file(payload.getvalue(), "optical.png", is_georeferenced=True)
        self.assertFalse(scene["is_georeferenced"])
        self.assertFalse(scene["is_metric"])
        self.assertIn("RDSM", scene["model_mode"])
        self.assertEqual(scene["geo_metadata"]["crs"], "UNREFERENCED")

    def test_loading_size_and_unknown_identifier_validation(self):
        for size in [0, -1, 2049, 1.5, True]:
            with self.assertRaises(ValueError):
                self.engine.load_gamus_scene("HILLY_RIDGE", size)
        with self.assertRaises(ValueError):
            self.engine.load_gamus_scene("made_up_04_23_scene")
        with self.assertRaises(ValueError):
            self.engine.process_image_file(b"image", "optical.png", target_resample_size=0)

    @unittest.skipUnless(module.HAS_RASTERIO, "rasterio is not installed")
    def test_projected_upload_keeps_crs_and_resampled_gsd_without_metric_height_claim(self):
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin

        with MemoryFile() as memory:
            with memory.open(driver="GTiff", width=64, height=64, count=3, dtype="uint8",
                             crs="EPSG:32643", transform=from_origin(500000, 2300000, 1, 1)) as dst:
                dst.write(np.full((3, 64, 64), 120, dtype=np.uint8))
            payload = memory.read()
        with patch.object(module, "HAS_TORCH_TRANSFORMERS", False), patch.object(
                SRTMElevationProvider, "get_elevation_grid", side_effect=RuntimeError("offline")) as source:
            scene = self.engine.process_image_file(payload, "projected.tif", target_resample_size=32)
        meta = scene["geo_metadata"]
        self.assertTrue(scene["is_georeferenced"])
        self.assertFalse(scene["is_metric"])
        self.assertEqual(meta["crs"], "EPSG:32643")
        self.assertAlmostEqual(meta["gsd_m"], 2)
        self.assertEqual(meta["gsd_status"], "PROJECTED_CRS_LINEAR_UNITS")
        self.assertEqual(meta["transform"], [2, 0, 500000, 0, -2, 2300000])
        bounds = meta["bounds_wgs84"]
        self.assertTrue(-180 <= bounds[0] < bounds[2] <= 180)
        self.assertTrue(-90 <= bounds[1] < bounds[3] <= 90)
        self.assertEqual(source.call_args.kwargs["bounds"], bounds)

    @unittest.skipUnless(module.HAS_RASTERIO, "rasterio is not installed")
    def test_float_export_preserves_height_values_transform_and_uncalibrated_tags(self):
        import rasterio
        from rasterio.io import MemoryFile

        surface = np.arange(64, dtype=np.float32).reshape(8, 8) / 3 + 52
        transform = [2, 0.5, 500000, 0.25, -2, 2300000]
        metadata = {"crs": "EPSG:32643", "bounds": [500000, 2299984, 500020, 2300002],
                    "transform": transform, "vertical_datum": "ASSUMED_LOCAL_BASE",
                    "height_calibration_status": "UNCALIBRATED"}
        payload = self.engine.export_dsm_geotiff(surface, metadata)
        with MemoryFile(payload) as memory, memory.open() as raster:
            np.testing.assert_array_equal(raster.read(1), surface)
            self.assertEqual(list(raster.transform)[:6], transform)
            self.assertEqual(raster.tags()["HEIGHT_CALIBRATION_STATUS"], "UNCALIBRATED")
            self.assertEqual(raster.tags()["VERTICAL_DATUM"], "ASSUMED_LOCAL_BASE")
        with patch.object(module, "HAS_RASTERIO", False):
            raw = self.engine.export_dsm_geotiff(surface)
            with Image.open(io.BytesIO(raw)) as image:
                np.testing.assert_array_equal(np.asarray(image), surface)
            with self.assertRaises(RuntimeError):
                self.engine.export_dsm_geotiff(surface, metadata)

    def test_model_loader_uses_only_cached_hf_and_records_unavailability(self):
        fake_torch = MagicMock()
        fake_torch.backends.mps.is_available.return_value = False
        fake_torch.cuda.is_available.return_value = False
        fake_processor = MagicMock()
        fake_processor.from_pretrained.side_effect = OSError("missing cache")
        with patch.object(module, "HAS_TORCH_TRANSFORMERS", True), patch.object(module, "torch", fake_torch, create=True), \
                patch.object(module, "AutoImageProcessor", fake_processor, create=True):
            backbone = module.DepthAnythingV2Backbone()
        self.assertEqual(backbone.load_status, "UNAVAILABLE")
        self.assertIsNone(backbone._model)
        self.assertTrue(fake_processor.from_pretrained.call_args.kwargs["local_files_only"])
        self.assertIn("cached", backbone.load_error)


if __name__ == "__main__":
    unittest.main()
