import sys
import unittest
from pathlib import Path

import numpy as np


REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "updategs_mask_reference"
sys.path.insert(0, str(REFERENCE_ROOT))
from src.actualization_v1.ssim_evidence import _fit_affine


class FitAffineTests(unittest.TestCase):
    def _fit(self, x, y, fit_mask=None):
        valid = np.ones(np.asarray(x).shape, dtype=bool)
        return _fit_affine(np.asarray(x), np.asarray(y), valid, fit_mask)

    def test_recovers_known_affine(self):
        x = np.linspace(0.05, 0.95, 2000, dtype=np.float64)
        a, b, used, clipped, report = self._fit(x, 1.7 * x + 0.08)
        self.assertTrue(used)
        self.assertFalse(clipped)
        self.assertAlmostEqual(a, 1.7, places=7)
        self.assertAlmostEqual(b, 0.08, places=7)
        self.assertEqual(report["fit_reason"], "ok")

    def test_robust_weighting_limits_outlier_effect(self):
        x = np.linspace(0.02, 0.98, 3000, dtype=np.float64)
        y = 1.4 * x + 0.12
        y[::11] += 0.7
        a, b, used, clipped, report = self._fit(x, y)
        self.assertTrue(used)
        self.assertFalse(clipped)
        self.assertAlmostEqual(a, 1.4, delta=0.015)
        self.assertAlmostEqual(b, 0.12, delta=0.01)
        self.assertGreater(report["inlier_fraction"], 0.8)

    def test_empty_and_low_texture_fallbacks_are_preserved(self):
        valid = np.ones(5, dtype=bool)
        result = _fit_affine(np.arange(5.), np.arange(5.), np.zeros(5, dtype=bool))
        self.assertEqual(result[:4], (1.0, 0.0, False, False))
        self.assertEqual(result[4]["fit_reason"], "empty_fit_mask")
        result = _fit_affine(np.full(5, 0.25), np.linspace(0.1, 0.9, 5), valid)
        self.assertEqual(result[:4], (1.0, 0.0, False, False))
        self.assertEqual(result[4]["fit_reason"], "low_texture")

    def test_gain_clipping_recomputes_intercept_by_median(self):
        x = np.linspace(0.02, 0.98, 2000, dtype=np.float64)
        y = 7.0 * x + 0.1
        a, b, used, clipped, report = self._fit(x, y)
        self.assertTrue(used)
        self.assertTrue(clipped)
        self.assertEqual(a, 4.0)
        self.assertAlmostEqual(b, float(np.median(y - 4.0 * x)), places=10)
        self.assertEqual(report["fit_reason"], "gain_clipped")

    def test_degenerate_weighted_variance_has_explicit_fallback(self):
        # Pass the historical peak-to-peak gate, but remain below the
        # float64-resolvable weighted variance floor.
        x = np.array([0.0, 2e-8, 0.0, 2e-8], dtype=np.float64)
        y = np.array([0.0, 0.5, 0.1, 0.6], dtype=np.float64)
        result = self._fit(x, y)
        self.assertEqual(result[:4], (1.0, 0.0, False, False))
        self.assertEqual(result[4]["fit_reason"], "degenerate_weighted_variance")


if __name__ == "__main__":
    unittest.main()
