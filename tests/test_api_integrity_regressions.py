"""Offline API/benchmark regressions using independent, deterministic fake engines."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
import inspect
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.depth_wizard.benchmark import DepthWizardBenchmark as Benchmark
from src.depth_wizard.case_store import CaseStore
from src.depth_wizard.elevation_engine import ElevationEngine

# Importing the API must not load/download a neural model for these tests.
with patch.object(ElevationEngine, "__init__", lambda self: None):
    from src.depth_wizard import server
from src.depth_wizard import cli


class ControlledEngine:
    def __init__(self, error=0.0, failed=(), partial=()):
        self.error = error
        self.failed = set(failed)
        self.partial = set(partial)
        self.last_dsm = None
        self.safety_result = None

    def load_gamus_scene(self, scene_id, resample_size=512):
        if scene_id in self.failed or scene_id == "missing":
            raise FileNotFoundError("Fixture unavailable")
        level = 100.0 if scene_id == "hundred" else 10.0
        if scene_id in self.partial or scene_id == "bad_nan":
            level = 33.0
        return {
            "scene_id": scene_id, "name": scene_id, "terrain_type": "Controlled test surface",
            "is_synthetic": False, "rgb_image": np.full((4, 4, 3), level, dtype=np.uint8),
            "ground_truth_dsm": np.full((4, 4), level), "base_elevation_m": level,
            "max_structural_height_m": 25.0, "geo_metadata": {"gsd_m": 1.0, "crs": "UNREFERENCED"},
        }

    def extract_relative_depth(self, image):
        return image[:, :, 0].astype(float)

    def calibrate_to_absolute_dsm(self, rel_depth, **kwargs):
        # Evaluation labels are never supplied as calibration arguments.
        assert "ground_truth_dsm" not in kwargs and "ground_truth_agl" not in kwargs
        self.last_dsm = rel_depth + self.error
        if rel_depth[0, 0] == 33:
            self.last_dsm[0, 0] = np.nan
        return {"dsm": self.last_dsm, "dtm": self.last_dsm.copy(), "structural_heights": np.zeros((4, 4)),
                "stats": {"is_metric": True, "surface_type": "DSM", "gsd_m": 1.0,
                          "ground_sample_dist_m": 1.0, "dtm_source": "INDEPENDENT_TEST_CONTROL",
                          "control_status": "VALIDATED_TEST_CONTROL", "grid_dimensions": [4, 4]}}

    def process_image_file(self, file_bytes, **kwargs):
        level = float(file_bytes.decode())
        calibration = self.calibrate_to_absolute_dsm(np.full((4, 4), level))
        return {**calibration, "scene_id": "upload", "name": "upload", "terrain_type": "upload",
                "model_mode": "TEST_CONTROL", "is_georeferenced": True,
                "geo_metadata": {"crs": "UNREFERENCED", "gsd_m": 1.0},
                "rgb_image": np.zeros((4, 4, 3), dtype=np.uint8)}

    def verify_indian_territory(self, metadata):
        return {"status": "UNREFERENCED", "badge": "UNREFERENCED", "is_in_india": False}

    def measure_distance_between_points(self, dsm, p1, p2, **kwargs):
        return {"point_a_elevation_m": float(dsm[p1[1], p1[0]]), "point_b_elevation_m": float(dsm[p2[1], p2[0]])}

    def export_dsm_geotiff(self, dsm, geo_meta=None):
        stream = io.BytesIO()
        Image.fromarray(dsm.astype(np.float32)).save(stream, format="TIFF")
        return stream.getvalue()

    def export_dsm_obj(self, dsm, step=4):
        return f"v 0 0 {dsm[0, 0]}\n"

    def simulate_flood(self, dsm, water, **kwargs):
        return self.safety_result or {"status": "SUCCESS", "water_level_m": water}

    def detect_landing_zones(self, **kwargs):
        return self.safety_result or {"status": "SUCCESS", "candidate_zones": [], "detected_zones_count": 0}

    def screen_landslide_risk(self, dtm, **kwargs):
        return self.safety_result or {"status": "SUCCESS", "mean_elevation_m": float(np.mean(dtm))}


class SmallMesh:
    def __init__(self):
        self.dtm_seen = []
        self.reference_seen = []

    def generate_mesh_payload(self, dsm, rgb, stats, ground_truth_dsm=None, dtm=None):
        self.dtm_seen.append(None if dtm is None else np.asarray(dtm).copy())
        self.reference_seen.append(ground_truth_dsm is not None)
        return {"grid_size": 4, "metric_heights": np.asarray(dsm).ravel().tolist(), "stats": stats}


class TestApiCaseIntegrity(unittest.TestCase):
    def setUp(self):
        self.engine = ControlledEngine()
        self.mesh = SmallMesh()
        self.patches = [patch.object(server, "engine", self.engine), patch.object(server, "mesh_gen", self.mesh),
                        patch.object(server, "CASE_STORE", CaseStore())]
        for mocked in self.patches:
            mocked.start()
            self.addCleanup(mocked.stop)
        self.a = TestClient(server.app)
        self.b = TestClient(server.app)
        self.addCleanup(self.a.close)
        self.addCleanup(self.b.close)

    def select(self, client, scene):
        response = client.post("/api/scene/select", json={"scene_id": scene})
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def elevation(self, client, case_id=None):
        response = client.post("/api/measure", params={"case_id": case_id} if case_id else {},
                               json={"p1_x": 0, "p1_y": 0, "p2_x": 1, "p2_y": 1})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["measurement"]["point_a_elevation_m"]

    def exported_elevation(self, client, case_id=None):
        response = client.get("/api/export/dsm", params={"case_id": case_id} if case_id else {})
        self.assertEqual(response.status_code, 200, response.text[:100])
        return float(np.asarray(Image.open(io.BytesIO(response.content)))[0, 0])

    def test_two_clients_keep_10m_and_100m_surfaces_and_exports(self):
        first = self.select(self.a, "ten")
        self.select(self.b, "hundred")
        self.assertEqual(self.elevation(self.a), 10.0)
        self.assertEqual(self.elevation(self.b), 100.0)
        self.assertEqual(self.exported_elevation(self.a), 10.0)
        self.assertEqual(self.exported_elevation(self.b), 100.0)
        self.assertIn("HttpOnly", first.headers["set-cookie"])
        self.assertIn("SameSite=lax", first.headers["set-cookie"])
        stolen = self.b.get("/api/export/dsm", params={"case_id": first.json()["case_id"]})
        self.assertEqual(stolen.status_code, 404)

    def test_uploads_are_also_isolated_and_old_owned_case_is_retained(self):
        ten = self.a.post("/api/upload", files={"file": ("ten.png", b"10", "image/png")})
        hundred = self.b.post("/api/upload", files={"file": ("hundred.png", b"100", "image/png")})
        self.assertEqual(ten.status_code, 200, ten.text)
        self.assertEqual(hundred.status_code, 200, hundred.text)
        self.select(self.a, "hundred")
        self.assertEqual(self.elevation(self.a), 100.0)
        self.assertEqual(self.elevation(self.a, ten.json()["case_id"]), 10.0)
        self.assertEqual(self.exported_elevation(self.a, ten.json()["case_id"]), 10.0)
        self.assertEqual(self.exported_elevation(self.b), 100.0)

    def test_fresh_client_has_no_implicit_or_inherited_case(self):
        self.select(self.a, "ten")
        for endpoint in ("/api/export/dsm", "/api/export/obj", "/api/export/metadata"):
            self.assertEqual(self.b.get(endpoint).status_code, 409)
        for endpoint in ("/api/disaster/flood", "/api/disaster/landing-zones", "/api/disaster/landslide"):
            response = self.b.post(endpoint, json={})
            self.assertEqual(response.status_code, 409, response.text)
            self.assertEqual(response.json()["status"], "NOT_ASSESSED")
        self.assertEqual(self.b.post("/api/scene/select", json={"scene_id": "missing"}).status_code, 404)

    def test_health_discloses_service_availability_without_model_or_operational_readiness(self):
        # This deterministic engine deliberately has no cached neural backbone.
        health = self.a.get("/api/health")
        self.assertEqual(health.status_code, 200)
        data = health.json()
        self.assertEqual(data["status"], "AVAILABLE")
        self.assertEqual(data["health_scope"], "SERVICE_AVAILABILITY_ONLY")
        self.assertEqual(data["backbone_status"], "UNAVAILABLE")
        self.assertEqual(data["operational_readiness"], "NOT_ASSESSED")
        self.assertEqual(data["metric_accuracy_status"], "NOT_ASSESSED")

    def test_published_case_copies_arrays_and_is_immutable(self):
        response = self.select(self.a, "ten")
        self.engine.last_dsm[:] = 9999
        self.assertEqual(self.elevation(self.a), 10.0)
        case = server.CASE_STORE.get(self.a.cookies.get(server.SESSION_COOKIE), response.json()["case_id"])
        with self.assertRaises(ValueError):
            case.data["dsm"][0, 0] = 20
        with self.assertRaises(ValueError):
            case.data["dsm"].setflags(write=True)
        with self.assertRaises(TypeError):
            case.data["scene_id"] = "another"

    def test_bad_terrain_and_nan_assessments_are_explicit_json_4xx(self):
        self.select(self.a, "ten")
        failed = self.a.post("/api/scene/select", json={"scene_id": "bad_nan"})
        self.assertEqual(failed.status_code, 422)
        self.assertEqual(failed.json()["status"], "NOT_ASSESSED")
        self.assertEqual(self.elevation(self.a), 10.0)
        for status in ("NOT_ASSESSED", "SUCCESS"):
            self.engine.safety_result = {"status": status, "submergence_pct": np.nan, "reason": "Invalid support"}
            for endpoint in ("/api/disaster/flood", "/api/disaster/landing_zones", "/api/disaster/landslide_risk"):
                response = self.a.post(endpoint, json={})
                self.assertEqual(response.status_code, 422, response.text)
                self.assertIsNone(response.json()["submergence_pct"])
                json.dumps(response.json(), allow_nan=False)
        invalid_request = self.a.post("/api/disaster/flood", content='{"water_level_m": NaN}', headers={"Content-Type": "application/json"})
        self.assertEqual(invalid_request.status_code, 422)
        json.dumps(invalid_request.json(), allow_nan=False)

    def test_relative_surface_withholds_safety_and_metric_measurements(self):
        response = self.select(self.a, "ten")
        owner = self.a.cookies.get(server.SESSION_COOKIE)
        data = dict(server.CASE_STORE.get(owner, response.json()["case_id"]).data)
        data["stats"] = {**data["stats"], "is_metric": False, "refusal_reason": "No independent scale"}
        server.CASE_STORE.put(owner, data)
        response = self.a.post("/api/disaster/flood", json={})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error_code"], "METRIC_SCALE_NOT_ESTABLISHED")
        measurement = self.a.post("/api/measure", json={"p1_x": 0, "p1_y": 0, "p2_x": 1, "p2_y": 1})
        self.assertEqual(measurement.status_code, 422)

    def test_reference_view_exports_the_viewed_surface_not_prediction(self):
        self.engine.error = 2.0
        response = self.a.post("/api/scene/select", json={"scene_id": "ten", "surface_source": "lidar_gt"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["mesh_payload"]["metric_heights"][0], 10.0)
        self.assertEqual(response.json()["benchmark"]["rmse_meters"], 2.0)
        self.assertEqual(self.exported_elevation(self.a), 10.0)
        self.assertEqual(self.a.get("/api/export/metadata").json()["surface_source"], "lidar_gt")
        self.assertEqual(self.a.post("/api/disaster/flood", json={}).status_code, 422)

    def test_mesh_receives_calibrated_terrain_and_reference_difference_is_not_self_comparison(self):
        response = self.select(self.a, "ten")
        np.testing.assert_array_equal(self.mesh.dtm_seen[-1], np.full((4, 4), 10.0))
        self.assertTrue(self.mesh.reference_seen[-1])
        self.assertTrue(response.json()["mesh_payload"]["reference_comparison_available"])

        reference = self.a.post("/api/scene/select", json={"scene_id": "ten", "surface_source": "lidar_gt"})
        self.assertEqual(reference.status_code, 200, reference.text)
        self.assertFalse(self.mesh.reference_seen[-1])
        self.assertFalse(reference.json()["mesh_payload"]["reference_comparison_available"])

    def test_mesh_failure_has_a_typed_api_error(self):
        class BrokenMesh:
            def generate_mesh_payload(self, *args, **kwargs):
                raise ValueError("invalid render support")

        with patch.object(server, "mesh_gen", BrokenMesh()):
            response = self.a.post("/api/scene/select", json={"scene_id": "ten"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error_code"], "INVALID_MESH")

    def test_finite_not_assessed_terrain_does_not_publish_a_case(self):
        calibration = self.engine.calibrate_to_absolute_dsm(np.full((4, 4), 10.0))
        calibration["stats"]["status"] = "NOT_ASSESSED"
        calibration["stats"]["reason"] = "Control unavailable"
        with patch.object(self.engine, "calibrate_to_absolute_dsm", return_value=calibration):
            response = self.a.post("/api/scene/select", json={"scene_id": "ten"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.a.get("/api/export/dsm").status_code, 409)

    def test_procedural_catalog_api_and_exports_retain_provenance(self):
        catalog = self.a.get("/api/scenes").json()["scenes"]
        for scene in catalog[:2]:
            self.assertTrue(scene["is_synthetic"])
            self.assertFalse(scene["provenance"]["counts_as_real_performance"])
            self.assertIn("Synthetic", scene["name"])
        response = self.select(self.a, "isro_sac_ahmedabad")
        self.assertTrue(response.json()["provenance"]["is_synthetic"])
        export = self.a.get("/api/export/dsm")
        self.assertEqual(export.headers["x-depthwizard-provenance"], "synthetic")
        from rasterio.io import MemoryFile
        with MemoryFile(export.content) as memory, memory.open() as dataset:
            metadata = json.loads(dataset.tags()["DEPTHWIZARD_METADATA"])
            self.assertTrue(metadata["provenance"]["is_synthetic"])
            self.assertEqual(metadata["certification"], "NOT CERTIFIED")
        obj = self.a.get("/api/export/obj").text
        self.assertIn('"is_synthetic":true', obj)
        self.assertIn("NOT CERTIFIED", obj)


class TestBenchmarkIntegrity(unittest.TestCase):
    def test_constant_and_empty_support_metrics_are_json_null(self):
        metrics = Benchmark.evaluate(np.full((4, 4), 12.0), np.full((4, 4), 10.0))
        self.assertEqual(metrics["rmse_meters"], 2.0)
        self.assertIsNone(metrics["pearson_correlation_r"])
        self.assertIsNone(metrics["r_squared"])
        moderate = "moderate_slopes_5_to_15deg_rmse_m"
        self.assertIsNone(metrics["slope_stratification"][moderate])
        self.assertEqual(metrics["slope_stratum_status"][moderate], "NOT_ASSESSED")
        self.assertEqual(metrics["slope_stratum_sample_counts"][moderate], 0)
        empty = Benchmark.evaluate(np.full((3, 3), np.nan), np.ones((3, 3)))
        self.assertEqual(empty["status"], "NOT_ASSESSED")
        self.assertEqual(empty["valid_sample_count"], 0)
        self.assertIsNone(empty["rmse_meters"])
        json.dumps(metrics, allow_nan=False)
        json.dumps(empty, allow_nan=False)

    def test_invalid_support_is_excluded_and_reported_as_partial(self):
        gt = np.zeros((3, 3))
        pred = np.full((3, 3), 2.0)
        gt[0, 0], pred[0, 1], pred[1, 0] = np.nan, np.inf, -9999
        pred = np.ma.array(pred, mask=False)
        pred.mask[2, 2] = True
        metrics = Benchmark.evaluate(pred, gt)
        self.assertEqual(metrics["status"], "PARTIAL")
        self.assertEqual(metrics["valid_sample_count"], 5)
        self.assertEqual(metrics["invalid_sample_count"], 4)
        self.assertEqual(metrics["rmse_meters"], 2.0)
        json.dumps(metrics, allow_nan=False)
        self.assertEqual(Benchmark.evaluate(np.zeros((2, 2)), np.zeros((3, 3)))["status"], "NOT_ASSESSED")

    def test_raw_nodata_is_masked_before_assumed_base_and_negative_labels_are_preserved(self):
        agl = np.array([[-9999.0, -2.0], [2.0, 4.0]])
        scene = {"ground_truth_agl": agl, "ground_truth_dsm": 15 + agl, "geo_metadata": {"gsd_m": 1.0}}
        metrics = Benchmark.evaluate_scene(np.full((2, 2), 15.0), scene)
        self.assertEqual(metrics["status"], "PARTIAL")
        self.assertEqual(metrics["valid_sample_count"], 3)
        self.assertEqual(metrics["raw_reference_invalid_sample_count"], 1)
        self.assertEqual(metrics["rmse_meters"], 2.83)
        self.assertAlmostEqual(metrics["bias_meters"], -1.33)
        json.dumps(metrics, allow_nan=False)

    def test_unreferenced_statistics_do_not_use_masked_or_nodata_backing_values(self):
        dsm = np.ma.array(np.full((2, 2), 10.0), mask=[[True, False], [False, False]])
        dtm = np.full((2, 2), 8.0)
        heights = np.full((2, 2), 2.0)
        heights[1, 1] = -32768
        result = Benchmark.evaluate_unreferenced_scene(dsm, dtm, heights)
        self.assertEqual(result["valid_sample_count"], 2)
        self.assertEqual(result["invalid_sample_count"], 2)
        self.assertEqual(result["reconstruction_statistics"]["max_structural_value"], 2.0)
        json.dumps(result, allow_nan=False)

    def test_undefined_spacing_and_overflow_stay_json_safe(self):
        grid = np.ones((2, 2))
        metrics = Benchmark.evaluate(grid, grid, gsd_m=None)
        self.assertEqual(metrics["slope_assessment_status"], "NOT_ASSESSED")
        huge = Benchmark.evaluate(np.full((2, 2), 1e308), grid)
        self.assertEqual(huge["status"], "NOT_ASSESSED")
        self.assertIsNone(huge["rmse_meters"])
        json.dumps(metrics, allow_nan=False)
        json.dumps(huge, allow_nan=False)

    def test_all_failures_and_partial_failures_are_recorded_not_zero_success(self):
        ids = [spec["id"] for spec in Benchmark.TERRAIN_SPECS]
        failed = Benchmark.run_suite(ControlledEngine(failed=ids))
        self.assertEqual(failed["status"], "FAILED")
        self.assertEqual(len(failed["failed_scenes"]), 4)
        self.assertEqual(len(failed["performance_matrix"]), 4)
        self.assertIsNone(failed["benchmark_results"]["rmse_meters"])
        self.assertEqual(failed["benchmark_summary"]["valid_sample_count"], 0)
        partial = Benchmark.run_suite(ControlledEngine(failed=[ids[0]]))
        self.assertEqual(partial["status"], "PARTIAL")
        self.assertEqual(partial["benchmark_summary"]["evaluated_scenes_count"], 3)
        self.assertEqual(partial["failed_scenes"][0]["scene_id"], ids[0])
        support_partial = Benchmark.run_suite(ControlledEngine(partial=[ids[1]]))
        self.assertEqual(support_partial["status"], "PARTIAL")
        self.assertEqual(support_partial["benchmark_summary"]["partially_evaluated_scenes_count"], 1)
        json.dumps(failed, allow_nan=False)
        json.dumps(partial, allow_nan=False)
        json.dumps(support_partial, allow_nan=False)

    def test_catastrophic_metrics_never_receive_an_approval_tier(self):
        report = Benchmark.run_suite(ControlledEngine(error=1000))
        self.assertEqual(report["benchmark_results"]["rmse_meters"], 1000.0)
        self.assertEqual(report["certification"], "NOT CERTIFIED")
        serialized = json.dumps(report, allow_nan=False).upper()
        for unsupported in ("APPROVED", "ISRO_GRADE", "OVERALL_ISRO_COMPLIANCE", "TIER-", "OPERATIONAL GRADE"):
            self.assertNotIn(unsupported, serialized)
        with patch.object(cli, "ElevationEngine", lambda: ControlledEngine(error=1000)), contextlib.redirect_stdout(io.StringIO()) as output:
            result = cli.run_benchmark()
        self.assertEqual(result["benchmark_results"]["rmse_meters"], 1000.0)
        self.assertIn("1000.00", output.getvalue())
        self.assertIn("NOT CERTIFIED", output.getvalue())
        self.assertNotIn("APPROVED", output.getvalue())

    def test_real_and_synthetic_groups_do_not_share_aggregation(self):
        report = Benchmark.run_suite(ControlledEngine())
        groups = report["provenance_groups"]
        self.assertEqual(groups["real"]["evaluated_scene_count"], 2)
        self.assertEqual(groups["synthetic"]["evaluated_scene_count"], 2)
        self.assertEqual(report["benchmark_summary"]["valid_sample_count"], 32)
        self.assertEqual(report["benchmark_summary"]["all_source_valid_sample_count"], 64)
        synthetic = [row for row in report["scene_evaluations"] if row["provenance"]["is_synthetic"]]
        for row in synthetic:
            self.assertFalse(row["provenance"]["counts_as_real_performance"])
        unknown = Benchmark.scene_provenance({}, "unverifiable")
        self.assertEqual(unknown["data_kind"], "unknown")
        self.assertFalse(unknown["counts_as_real_performance"])

    def test_macro_and_pooled_metrics_have_different_weighting(self):
        evaluations = []
        for scene_id, size, error in (("small", 2, 0.0), ("large", 4, 10.0)):
            gt = np.zeros((size, size))
            evaluations.append({"scene_id": scene_id, "landscape_category": "Urban", "landscape_label": scene_id,
                                "metrics": Benchmark.evaluate(gt + error, gt),
                                "provenance": {"data_kind": "real", "is_synthetic": False}})
        report = Benchmark.summarize(evaluations, [], 2)
        summary = report["benchmark_summary"]
        self.assertEqual(summary["macro_metrics"]["rmse_meters"], 5.0)
        self.assertEqual(summary["pooled_metrics"]["rmse_meters"], 8.94)
        self.assertEqual(summary["pooled_metrics"]["mae_meters"], 8.0)
        self.assertIsNone(summary["pooled_metrics"]["le90_meters"])
        self.assertIsNone(summary["macro_metrics"]["pearson_correlation_r"])

    def test_callable_benchmark_has_no_request_and_http_reports_failure(self):
        self.assertEqual(len(inspect.signature(server.run_full_benchmark).parameters), 0)
        ids = [spec["id"] for spec in Benchmark.TERRAIN_SPECS]
        with patch.object(server, "engine", ControlledEngine(failed=ids)):
            direct = asyncio.run(server.run_full_benchmark())
            self.assertIsInstance(direct, dict)
            with TestClient(server.app) as client:
                response = client.get("/api/benchmark/run")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["status"], "FAILED")
        with patch.object(server, "engine", ControlledEngine(failed=ids[:1])):
            with TestClient(server.app) as client:
                self.assertEqual(client.get("/api/benchmark").status_code, 207)


class TestCaseStoreBounds(unittest.TestCase):
    def test_expiry_and_capacity_cannot_resurrect_or_inherit_cases(self):
        now = [0.0]
        store = CaseStore(ttl_seconds=10, max_cases=2, max_sessions=3, max_cases_per_session=2, clock=lambda: now[0])
        a, b = store.session(), store.session()
        old = store.put(a, {"dsm": np.full((2, 2), 10.0)})
        newest = store.put(a, {"dsm": np.full((2, 2), 20.0)})
        store.put(b, {"dsm": np.full((2, 2), 100.0)})
        with self.assertRaises(KeyError):
            store.get(a, old.case_id)
        self.assertEqual(store.get(a).case_id, newest.case_id)
        with self.assertRaises(KeyError):
            store.get(b, newest.case_id)
        now[0] = 11.0
        with self.assertRaises(KeyError):
            store.get(a)
        new_owner = store.session(a)
        self.assertNotEqual(a, new_owner)
        with self.assertRaises(KeyError):
            store.get(new_owner)

    def test_concurrent_publication_keeps_complete_owned_snapshots(self):
        store = CaseStore(max_cases=64, max_sessions=64)

        def publish(level):
            owner = store.session()
            case = store.put(owner, {"dsm": np.full((2, 2), level), "scene_id": str(level)})
            snapshot = store.get(owner, case.case_id)
            return snapshot.data["scene_id"], snapshot.data["dsm"].copy()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(publish, range(32)))
        for level, (scene, dsm) in enumerate(results):
            self.assertEqual(scene, str(level))
            np.testing.assert_array_equal(dsm, np.full((2, 2), level))


if __name__ == "__main__":
    unittest.main()
