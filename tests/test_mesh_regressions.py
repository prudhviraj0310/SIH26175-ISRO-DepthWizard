"""Mesh and LoD-1 payload contracts for local, provenance-aware rendering."""

import unittest

import numpy as np

from src.depth_wizard.mesh_generator import MeshGenerator


class TestMeshPayloadContracts(unittest.TestCase):
    def setUp(self):
        self.mesh = MeshGenerator(target_grid_size=32)
        self.dtm = np.full((40, 40), 90.0, dtype=np.float32)
        self.dsm = self.dtm.copy()
        self.dsm[12:28, 12:28] += 10.0
        self.rgb = np.full((40, 40, 3), 120, dtype=np.uint8)

    def test_supplied_terrain_reference_drives_lod1_and_is_not_called_authoritative(self):
        stats = {
            "gsd_m": 1.0,
            "ground_sample_dist_m": 1.0,
            "dtm_source": "PUBLIC_SURFACE_REFERENCE",
            "terrain_reference_kind": "DSM",
            "vertical_reference_status": "UNVERIFIED_OR_ASSUMED",
            "terrain_provenance": {"bare_earth_certified": False, "datum_verified": False},
        }
        payload = self.mesh.generate_mesh_payload(self.dsm, self.rgb, stats, dtm=self.dtm)

        self.assertEqual(payload["lod1_dtm_source"], "PUBLIC_SURFACE_REFERENCE")
        self.assertEqual(payload["lod1_dtm_status"], "SUPPLIED_UNVERIFIED_REFERENCE")
        self.assertFalse(payload["lod1_dtm_authoritative"])
        self.assertEqual(payload["lod1_building_count"], 1)
        self.assertAlmostEqual(payload["lod1_buildings"][0]["base_elevation_m"], 90.0, delta=0.5)

    def test_verified_dtm_provenance_is_explicit_and_fallback_is_distinct(self):
        verified = {
            "gsd_m": 1.0,
            "terrain_reference_kind": "DTM",
            "vertical_reference_status": "VERIFIED",
            "terrain_provenance": {"bare_earth_certified": True, "datum_verified": True},
        }
        supplied = self.mesh.generate_mesh_payload(self.dsm, self.rgb, verified, dtm=self.dtm)
        fallback = self.mesh.generate_mesh_payload(self.dsm, self.rgb, verified)

        self.assertEqual(supplied["lod1_dtm_status"], "AUTHORITATIVE_VERIFIED_DTM")
        self.assertTrue(supplied["lod1_dtm_authoritative"])
        self.assertEqual(fallback["lod1_dtm_status"], "UNVERIFIED_MORPHOLOGICAL_FALLBACK")
        self.assertFalse(fallback["lod1_dtm_authoritative"])

    def test_empty_and_invalid_mesh_support_are_distinct(self):
        flat = np.full((40, 40), 90.0, dtype=np.float32)
        empty = self.mesh.generate_mesh_payload(flat, self.rgb, {"gsd_m": 1.0}, dtm=flat)
        self.assertEqual(empty["lod1_status"], "EMPTY")
        self.assertEqual(empty["lod1_building_count"], 0)

        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.mesh.generate_mesh_payload(
                np.where(np.indices(flat.shape).sum(axis=0) == 0, np.nan, flat),
                self.rgb, {"gsd_m": 1.0}, dtm=flat,
            )
        with self.assertRaisesRegex(ValueError, "matching spatial"):
            self.mesh.generate_mesh_payload(flat, self.rgb[:20], {"gsd_m": 1.0}, dtm=flat)
        with self.assertRaises(ValueError):
            self.mesh.generate_mesh_payload(flat, np.full(self.rgb.shape, "bad", dtype=object), {"gsd_m": 1.0}, dtm=flat)

    def test_boolean_gsd_is_not_a_physical_sampling_distance(self):
        for key in ("gsd_m", "ground_sample_dist_m"):
            for value in (True, np.bool_(True), False):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "boolean"):
                    self.mesh.generate_mesh_payload(self.dsm, self.rgb, {key: value}, dtm=self.dtm)

    def test_analytic_ramps_keep_physical_slope_across_resolutions_and_axes(self):
        for rows, cols, gsd in ((64, 64, 1.0), (128, 256, 0.5), (256, 128, 2.0), (512, 512, 1.0)):
            for slope_x, slope_y in ((0.2, 0.0), (0.0, 0.3), (0.2, 0.3)):
                with self.subTest(shape=(rows, cols), gsd=gsd, slope=(slope_x, slope_y)):
                    yy, xx = np.mgrid[:rows, :cols]
                    ramp = (50 + slope_x * xx * gsd + slope_y * yy * gsd).astype(np.float32)
                    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
                    payload = self.mesh.generate_mesh_payload(ramp, rgb, {"gsd_m": gsd}, dtm=ramp)
                    expected = np.degrees(np.arctan(np.hypot(slope_x, slope_y)))
                    slopes = np.asarray(payload["slope_degrees"]).reshape(32, 32)
                    # A linear ramp is unchanged in the interior by bilateral
                    # smoothing; the raster resize clamps its outer half-pixels.
                    np.testing.assert_allclose(slopes[4:-4, 4:-4], expected, atol=0.06, rtol=0)
                    self.assertAlmostEqual(payload["mean_slope_deg"], expected, delta=0.5)
                    self.assertEqual(payload["slope_grid_spacing_m"], [gsd * rows / 512, gsd * cols / 512])


if __name__ == "__main__":
    unittest.main()
