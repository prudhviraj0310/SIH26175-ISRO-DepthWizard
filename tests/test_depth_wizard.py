"""
Unit tests for ISRO DepthWizard (SIH26175)
Using standard library unittest.
"""

import unittest
import numpy as np
from fastapi.testclient import TestClient

from src.depth_wizard.elevation_engine import ElevationEngine
from src.depth_wizard.mesh_generator import MeshGenerator
from src.depth_wizard.benchmark import DepthWizardBenchmark
from src.depth_wizard.server import app

class TestDepthWizard(unittest.TestCase):
    def setUp(self):
        self.engine = ElevationEngine()
        self.mesh_gen = MeshGenerator(target_grid_size=64)

    def test_elevation_engine_relative_depth(self):
        test_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        test_rgb[30:70, 30:70] = [255, 255, 255]  # bright square
        
        rel_depth = self.engine.extract_relative_depth(test_rgb)
        self.assertEqual(rel_depth.shape, (128, 128))
        self.assertGreaterEqual(float(np.min(rel_depth)), 0.0)
        self.assertLessEqual(float(np.max(rel_depth)), 1.0)
        self.assertGreater(float(np.mean(rel_depth[35:65, 35:65])), float(np.mean(rel_depth[0:20, 0:20])))

    def test_scale_calibration_and_dtm(self):
        test_rel = np.zeros((64, 64), dtype=np.float32)
        test_rel[20:40, 20:40] = 0.9  # Structure
        
        calib = self.engine.calibrate_to_absolute_dsm(
            rel_depth=test_rel,
            base_srtm_elevation_m=50.0,
            max_structural_height_m=30.0
        )
        
        dsm = calib["dsm"]
        dtm = calib["dtm"]
        structural_heights = calib["structural_heights"]
        stats = calib["stats"]
        
        self.assertEqual(dsm.shape, (64, 64))
        self.assertEqual(dtm.shape, (64, 64))
        self.assertTrue(np.all(dtm <= dsm + 1e-5))
        self.assertGreater(float(np.max(structural_heights)), 10.0)
        self.assertEqual(stats["base_srtm_m"], 50.0)

    def test_laser_caliper_measurement(self):
        dsm = np.full((100, 100), 50.0, dtype=np.float32)
        dsm[50, 50] = 80.0  # 30m tower at (50, 50)
        
        meas = self.engine.measure_distance_between_points(
            dsm=dsm,
            p1=(10, 10),
            p2=(50, 50),
            ground_resolution_m=1.0
        )
        self.assertEqual(meas["elevation_delta_m"], 30.0)
        self.assertGreater(meas["horizontal_distance_m"], 0)
        self.assertGreater(meas["true_3d_distance_m"], meas["horizontal_distance_m"])

    def test_mesh_generator(self):
        dsm = np.full((64, 64), 100.0, dtype=np.float32)
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        stats = {"min_elevation_m": 90.0, "max_elevation_m": 120.0}
        
        payload = self.mesh_gen.generate_mesh_payload(dsm, rgb, stats)
        self.assertIn("grid_size", payload)
        self.assertIn("min_elevation_m", payload)
        self.assertIn("max_elevation_m", payload)
        self.assertIn("metric_heights", payload)
        self.assertIn("normalized_z", payload)
        self.assertIn("texture_data_url", payload)
        self.assertEqual(len(payload["normalized_z"]), 64 * 64)

    def test_real_gamus_dataset_loading(self):
        # Verify that the downloaded real ISRO GAMUS dataset loads cleanly
        scene = self.engine.load_gamus_scene("DC_02_26", resample_size=64)
        self.assertIn("rgb_image", scene)
        self.assertIn("ground_truth_dsm", scene)
        self.assertEqual(scene["rgb_image"].shape, (64, 64, 3))
        self.assertEqual(scene["ground_truth_dsm"].shape, (64, 64))
        self.assertGreater(scene["max_structural_height_m"], 0.0)

    def test_disaster_management_battery(self):
        dsm = np.linspace(40, 80, 64*64, dtype=np.float32).reshape(64, 64)
        dtm = dsm.copy()
        struct_h = np.zeros((64, 64), dtype=np.float32)
        struct_h[20:30, 20:30] = 12.0  # Buildings

        # 1. Flood Inundation
        flood = self.engine.simulate_flood(dsm, water_level_m=60.0, structural_heights=struct_h)
        self.assertEqual(flood["status"], "SUCCESS")
        self.assertGreater(flood["submergence_pct"], 0.0)
        self.assertGreater(flood["max_depth_m"], 0.0)

        # 2. HLZ Detection
        hlz = self.engine.detect_landing_zones(dsm, dtm, struct_h, pad_radius_m=6.0, max_slope_deg=5.0)
        self.assertEqual(hlz["status"], "SUCCESS")
        self.assertIn("candidate_zones", hlz)

        # 3. Landslide Risk
        landslide = self.engine.screen_landslide_risk(dtm)
        self.assertEqual(landslide["status"], "SUCCESS")
        self.assertIn("critical_hazard_pct", landslide)

    def test_benchmark_metrics(self):
        gt = np.full((50, 50), 100.0, dtype=np.float32)
        # Create non-zero gradient to test slope partitioning
        for y in range(50):
            gt[y, :] += y * 0.5
        pred = gt + 2.0  # Constant 2m error with occasional noise
        pred[10, 10] += 5.0
        
        metrics = DepthWizardBenchmark.evaluate(pred, gt, "Urban Test")
        self.assertIn("rmse_meters", metrics)
        self.assertIn("mae_meters", metrics)
        self.assertIn("nmad_meters", metrics)
        self.assertIn("bias_meters", metrics)
        self.assertIn("r_squared", metrics)
        self.assertIn("slope_stratification", metrics)
        self.assertIn("flat_terrain_below_5deg_rmse_m", metrics["slope_stratification"])

    def test_server_api_endpoints(self):
        client = TestClient(app)
        
        # 1. Health check
        res = client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "OPERATIONAL")
        self.assertEqual(res.json()["ps_number"], "SIH26175")
        
        # 2. Scenes list
        res = client.get("/api/scenes")
        self.assertEqual(res.status_code, 200)
        self.assertGreaterEqual(len(res.json()["scenes"]), 3)
        
        # 3. Scene selection & processing
        res = client.post("/api/scene/select", json={"scene_id": "isro_sac_ahmedabad"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("mesh_payload", data)
        self.assertIn("benchmark", data)
        
        # 4. Measure
        res = client.post("/api/measure", json={
            "p1_x": 50, "p1_y": 50,
            "p2_x": 100, "p2_y": 100,
            "ground_res_m": 0.5
        })
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "SUCCESS")
        
        # 5. Full benchmark across all scenes
        res = client.get("/api/benchmark")
        self.assertEqual(res.status_code, 200)
        bench_data = res.json()
        self.assertIn("benchmark_summary", bench_data)
        self.assertEqual(bench_data["benchmark_summary"]["evaluated_scenes_count"], 3)

        # 6. Upload satellite image test (PNG/JPG)
        import io
        from PIL import Image
        test_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        upload_res = client.post(
            "/api/upload",
            files={"file": ("test_optical_scene.png", img_bytes.getvalue(), "image/png")},
            data={"is_georeferenced": "false", "base_srtm_elevation_m": "50.0"}
        )
        self.assertEqual(upload_res.status_code, 200)
        upload_data = upload_res.json()
        self.assertEqual(upload_data["status"], "SUCCESS")
        self.assertIn("RDSM", upload_data["model_mode"].upper())
        self.assertIn("mesh_payload", upload_data)

        # 7. Export DSM as 16-bit GeoTIFF / TIFF
        export_res = client.get("/api/export/dsm")
        self.assertEqual(export_res.status_code, 200)
        self.assertEqual(export_res.headers["content-type"], "image/tiff")
        self.assertGreater(len(export_res.content), 100)

        # 8. Export 3D Mesh as Wavefront OBJ
        export_obj_res = client.get("/api/export/obj")
        self.assertEqual(export_obj_res.status_code, 200)
        self.assertIn("model/obj", export_obj_res.headers["content-type"])
        self.assertTrue(export_obj_res.content.startswith(b"# DepthWizard"))
        self.assertIn(b"v ", export_obj_res.content)
        self.assertIn(b"f ", export_obj_res.content)

        # 9. Disaster Management API endpoints
        flood_res = client.post("/api/disaster/flood", json={"offset_m": 3.0})
        self.assertEqual(flood_res.status_code, 200)
        self.assertEqual(flood_res.json()["status"], "SUCCESS")
        self.assertIn("inundated_hectares", flood_res.json())

        hlz_res = client.post("/api/disaster/landing-zones", json={"pad_radius_m": 8.0})
        self.assertEqual(hlz_res.status_code, 200)
        self.assertEqual(hlz_res.json()["status"], "SUCCESS")
        self.assertIn("candidate_zones", hlz_res.json())

        ls_res = client.post("/api/disaster/landslide")
        self.assertEqual(ls_res.status_code, 200)
        self.assertEqual(ls_res.json()["status"], "SUCCESS")
        self.assertIn("critical_hazard_pct", ls_res.json())


if __name__ == "__main__":
    unittest.main()
