"""Compatibility discovery entry point for the canonical root test suite.

Run from the repository root with either ``unittest discover tests`` or
``unittest discover src/depth_wizard/tests``. Numeric legacy thresholds and
the formerly unreachable IRLS/refusal regression live in tests/test_depth_wizard.py.
"""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    # Unique names prevent the legacy entry point's test_depth_wizard module
    # from shadowing the root file during unittest discovery.
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        name = "_depthwizard_canonical_" + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        suite.addTests(loader.loadTestsFromModule(module))
    return suite


if __name__ == "__main__":
    unittest.main()
