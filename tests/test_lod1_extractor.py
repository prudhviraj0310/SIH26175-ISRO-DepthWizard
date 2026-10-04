"""
Unit tests for DepthWizard LoD-1 Architectural Building Extrusion Engine.
Validates:
  - RDP polygonal line simplification
  - Input type and domain validation (GSD, dimensions, NaN/Inf robustness)
  - Synthetic building extrusion geometry (height, area, WebGL coordinates)
  - Touch-instance separation via marker-controlled watershed
"""

import math
import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.depth_wizard.lod1_extractor import extract_lod1_buildings, rdp_simplify
from src.depth_wizard import lod1_extractor as module


class TestLoD1Extractor(unittest.TestCase):

    def test_rdp_simplification(self):
        """Collinear points should simplify to end points."""
        line = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
        simplified = rdp_simplify(line, epsilon=0.1)
        self.assertEqual(len(simplified), 2)
        self.assertEqual(simplified[0], (0.0, 0.0))
        self.assertEqual(simplified[-1], (4.0, 0.0))

        # L-shape: should preserve corner
        corner = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
        simplified_corner = rdp_simplify(corner, epsilon=0.1)
        self.assertEqual(len(simplified_corner), 3)

    def test_input_validation(self):
        """Invalid inputs must be rejected with informative exceptions."""
        valid_grid = np.zeros((20, 20), dtype=np.float32)

        # Invalid GSD
        with self.assertRaises(ValueError):
            extract_lod1_buildings(valid_grid, valid_grid, gsd_m=0.0)
        with self.assertRaises(ValueError):
            extract_lod1_buildings(valid_grid, valid_grid, gsd_m=-1.0)
        with self.assertRaises(ValueError):
            extract_lod1_buildings(valid_grid, valid_grid, gsd_m=float("nan"))

        # Invalid types / dimensions
        with self.assertRaises(TypeError):
            extract_lod1_buildings("not_array", valid_grid)
        with self.assertRaises(ValueError):
            extract_lod1_buildings(valid_grid, np.zeros((10, 10)))

    def test_nan_inf_robustness(self):
        """Grids with NaN and Inf must not crash the extractor or leak into results."""
        dsm = np.full((30, 30), 10.0, dtype=np.float32)
        dtm = np.full((30, 30), 10.0, dtype=np.float32)

        # Add NaN and Inf in non-building and building zones
        dsm[0:5, 0:5] = np.nan
        dtm[5:10, 5:10] = np.inf

        # Add a valid 8x8 building with 8m height above DTM
        dsm[15:23, 15:23] = 18.0

        buildings = extract_lod1_buildings(dsm, dtm, gsd_m=1.0, min_height_m=2.5, min_area_m2=20.0)
        self.assertGreaterEqual(len(buildings), 1)

        b = buildings[0]
        self.assertTrue(math.isfinite(b["height_agl_m"]))
        self.assertTrue(math.isfinite(b["base_elevation_m"]))
        self.assertTrue(math.isfinite(b["roof_elevation_m"]))
        self.assertAlmostEqual(b["height_agl_m"], 8.0, delta=1.0)

    def test_masked_building_support_is_not_recovered_from_hidden_values(self):
        dsm = np.full((30, 30), 10.0, dtype=np.float32)
        dtm = np.full((30, 30), 10.0, dtype=np.float32)
        dsm[12:20, 12:20] = 18.0
        mask = np.zeros_like(dsm, dtype=bool)
        mask[12:20, 12:20] = True
        masked_dsm = np.ma.array(dsm, mask=mask)

        buildings = extract_lod1_buildings(masked_dsm, dtm, gsd_m=1.0, min_area_m2=20.0)
        self.assertEqual(buildings, [])

    def test_invalid_thresholds_and_rdp_inputs_fail_closed(self):
        grid = np.zeros((20, 20), dtype=np.float32)
        for kwargs in (
            {"min_height_m": np.nan}, {"min_area_m2": -1},
            {"simplify_epsilon_m": np.inf}, {"max_buildings": 0},
            {"gsd_m": 1e-12},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    extract_lod1_buildings(grid, grid, **kwargs)
        with self.assertRaises(ValueError):
            rdp_simplify([(0, 0), (1, 1)], epsilon=np.nan)
        with self.assertRaises(ValueError):
            rdp_simplify([(0, 0), (np.inf, 1)], epsilon=1.0)

    def test_synthetic_building_extrusion(self):
        """A synthetic 10m high block should yield accurate LoD-1 attributes."""
        # 50x50 terrain with base elevation 50m
        dtm = np.full((50, 50), 50.0, dtype=np.float32)
        dsm = dtm.copy()

        # Place 10x10 block of height +12m (from pixel 20 to 30)
        dsm[20:30, 20:30] += 12.0

        buildings = extract_lod1_buildings(
            dsm, dtm, gsd_m=1.0, min_height_m=3.0, min_area_m2=25.0
        )
        self.assertEqual(len(buildings), 1)
        b = buildings[0]

        self.assertEqual(b["geometry_type"], "LoD-1 Prism")
        self.assertAlmostEqual(b["base_elevation_m"], 50.0, delta=0.5)
        self.assertAlmostEqual(b["height_agl_m"], 12.0, delta=0.5)
        self.assertAlmostEqual(b["roof_elevation_m"], 62.0, delta=0.5)
        self.assertGreaterEqual(b["footprint_area_m2"], 50.0)
        self.assertLessEqual(b["footprint_area_m2"], 110.0)

        # Check WebGL coordinates are bounded in [-1.0, 1.0]
        for wx, wy in b["webgl_footprint_coords"]:
            self.assertTrue(-1.0 <= wx <= 1.0)
            self.assertTrue(-1.0 <= wy <= 1.0)

    def test_scipy_fallback_keeps_the_only_and_last_components(self):
        dtm = np.full((50, 60), 50.0, dtype=np.float32)
        dsm = dtm.copy()
        dsm[10:25, 8:23] += 10
        with patch.object(module, "HAS_CV2", False):
            single = extract_lod1_buildings(dsm, dtm)
            self.assertEqual(len(single), 1)
            self.assertEqual(single[0]["footprint_area_m2"], 225.0)
            self.assertEqual(single[0]["height_agl_m"], 10.0)
            dsm[30:40, 40:50] += 12
            both = extract_lod1_buildings(dsm, dtm)
            self.assertEqual(len(both), 2)
            self.assertEqual([b["height_agl_m"] for b in both], [12.0, 10.0])
            self.assertEqual(sum(b["footprint_area_m2"] for b in both), 325.0)

    @unittest.skipUnless(module.HAS_CV2, "OpenCV watershed is unavailable")
    def test_touching_blocks_have_unknown_shells_and_two_instances(self):
        dtm = np.full((50, 60), 50.0, dtype=np.float32)
        dsm = dtm.copy()
        dsm[15:30, 8:23] += 10
        dsm[15:30, 31:46] += 15
        dsm[21:24, 23:31] += 10  # A narrow touching bridge, not two disconnected masks.
        original_watershed = module.cv2.watershed

        def inspect_markers(image, markers):
            self.assertGreater(np.count_nonzero(markers == 0), 0)
            return original_watershed(image, markers)

        with patch.object(module.cv2, "watershed", side_effect=inspect_markers) as watershed:
            buildings = extract_lod1_buildings(dsm, dtm)
        watershed.assert_called_once()
        self.assertEqual(len(buildings), 2)
        self.assertEqual([b["height_agl_m"] for b in buildings], [15.0, 10.0])
        for building in buildings:
            self.assertGreaterEqual(building["footprint_area_m2"], 100.0)
            self.assertLessEqual(building["footprint_area_m2"], 249.0)
        self.assertLessEqual(sum(b["footprint_area_m2"] for b in buildings), 474.0)

    @unittest.skipUnless(module.HAS_CV2, "OpenCV watershed is unavailable")
    def test_morphology_and_watershed_cannot_count_or_sample_invalid_holes(self):
        dtm = np.full((40, 40), 50.0, dtype=np.float32)
        base = dtm.copy()
        base[10:25, 10:25] += 10
        missing = np.zeros(base.shape, dtype=bool)
        missing[16, 16] = True  # Closing would otherwise fill this unavailable observation.
        for kind in ("mask", "nan", "inf", "terrain_mask"):
            with self.subTest(kind=kind):
                dsm, terrain = base.copy(), dtm.copy()
                if kind == "mask":
                    dsm[missing] = 5000
                    dsm = np.ma.array(dsm, mask=missing)
                elif kind == "terrain_mask":
                    terrain[missing] = -5000
                    terrain = np.ma.array(terrain, mask=missing)
                else:
                    dsm[missing] = np.nan if kind == "nan" else np.inf
                # Even watershed output that grows beyond its seed mask must
                # remain clipped to observed support for attributes and area.
                with patch.object(module.cv2, "watershed", return_value=np.full(base.shape, 2, dtype=np.int32)):
                    buildings = extract_lod1_buildings(dsm, terrain)
                self.assertEqual(len(buildings), 1)
                self.assertEqual(buildings[0]["footprint_area_m2"], 224.0)
                self.assertEqual(buildings[0]["height_agl_m"], 10.0)
                self.assertEqual(buildings[0]["base_elevation_m"], 50.0)
                json.dumps(buildings, allow_nan=False)
                real = extract_lod1_buildings(dsm, terrain)
                self.assertEqual(len(real), 1)
                self.assertLessEqual(real[0]["footprint_area_m2"], 224.0)
                self.assertEqual(real[0]["height_agl_m"], 10.0)


if __name__ == "__main__":
    unittest.main()
