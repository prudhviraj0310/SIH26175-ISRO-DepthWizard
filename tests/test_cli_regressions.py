"""CLI regression checks that exercise the real exporter contract."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.depth_wizard import cli


class FakeEngine:
    def process_image_file(self, **kwargs):
        surface = np.full((4, 4), 10.0, dtype=np.float32)
        return {
            "dsm": surface,
            "stats": {
                "status": "SUCCESS", "surface_type": "rDSM", "is_metric": False,
                "min_elevation_m": 10.0, "max_elevation_m": 10.0,
                "height_scale_status": "ASSUMED",
            },
            "geo_metadata": {"crs": "UNREFERENCED", "gsd_m": 1.0},
        }

    def export_dsm_geotiff(self, dsm, geo_meta=None):
        return b"TIFF"

    def export_dsm_obj(self, dsm, step=4):
        return b"# mesh\nv 0 0 0\n"


class TestCliMeshExport(unittest.TestCase):
    def test_obj_flag_uses_engine_exporter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "scene.png"
            output_path = root / "scene.tif"
            input_path.write_bytes(b"not decoded by fake engine")
            with patch.object(cli, "ElevationEngine", FakeEngine):
                cli.process_image(str(input_path), str(output_path), export_obj=True)

            obj = (root / "scene.obj").read_bytes()
            self.assertTrue(obj.startswith(b"# DepthWizard metadata: "))
            self.assertIn(b"v 0 0 0", obj)


if __name__ == "__main__":
    unittest.main()
