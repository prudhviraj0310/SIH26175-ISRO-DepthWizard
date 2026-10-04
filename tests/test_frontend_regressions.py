"""Static browser contracts for the recorded-video-facing research UI."""

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src/depth_wizard/templates/index.html"
APP = ROOT / "src/depth_wizard/static/js/app.js"
FLYTHROUGH = ROOT / "src/depth_wizard/static/js/flythrough3d.js"


class TestFrontendContracts(unittest.TestCase):
    def test_visible_labels_do_not_claim_accuracy_or_geodetic_calibration(self):
        html = TEMPLATE.read_text()
        self.assertIn("Reference Difference (|ΔH|)", html)
        self.assertIn("Static Water-Level Geometry Screen", html)
        self.assertIn("3D Surface Transect", html)
        self.assertNotIn("Accuracy Error (ΔH)", html)
        self.assertNotIn("3D Geodetic Caliper", html)

    def test_network_or_assessment_errors_clear_stale_video_controls(self):
        app = APP.read_text()
        self.assertIn("setMetric('flood-submerged-val', null, '%')", app)
        self.assertIn("flythrough.renderLandingZones([])", app)
        self.assertIn("Benchmark unavailable — NOT CERTIFIED", app)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for JavaScript syntax smoke checks")
    def test_async_case_and_water_semantics_in_the_real_controllers(self):
        result = subprocess.run(
            ["node", "--test", "--test-reporter=tap", str(ROOT / "tests/frontend_runtime.cjs")],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("# pass 6", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for JavaScript syntax smoke checks")
    def test_browser_controllers_parse_with_node(self):
        for path in (APP, FLYTHROUGH):
            result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
