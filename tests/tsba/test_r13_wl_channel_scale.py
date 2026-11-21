"""Numerical, prefixed label-read and decision checks for the WL-radius supplement."""
from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import wl_channel_scale as scale, wl_channel_scoring


class RadiusScaleTests(unittest.TestCase):
    def test_three_scale_reorders_and_preserves_target_reference_mean(self):
        u = np.array([[2., -3., 0.], [-2., 3., 0.]])
        d = np.array([[2., -1., 5.], [-2., 1., 5.]])
        static = .1 + u.sum(axis=1)
        main, detail = scale.align(u, u, d, static)
        np.testing.assert_allclose(main, [1.1, -.9])
        self.assertAlmostEqual(main.mean(), static.mean())
        np.testing.assert_array_equal(detail['ratio'], [1, 1/3, 1])
        out, _ = scale.scale_predictions(np.array([1, 2]), static, d, u, u, u, .1)
        np.testing.assert_array_equal(out['Target'], [True, False])
        np.testing.assert_array_equal(out['Count'], [False, True])
        np.testing.assert_array_equal(out['Target'], out['Source'])

    def test_floor_includes_equality_and_both_reference_and_donor(self):
        x = np.array([[0., 0., 0.], [2e-12, 8., 4.]])
        d = np.array([[0., 0., 0.], [2., 2e-12, 8.]])
        f = x.sum(axis=1)
        _, scales = scale.align(x, x, d, f)
        np.testing.assert_array_equal(scales['ratio'], [1., 1., 2.])

    def test_gate_uses_protein_curve_and_exact_required_contrasts(self):
        setting = 'proteins_edge'
        overall = dict.fromkeys(scale.methods(setting), Fraction(1, 2))
        overall['Target'] = Fraction(101, 200)
        gain = {i: Fraction(1, 100) for i in range(4)}
        self.assertTrue(wl_channel_scoring.gate(setting, overall, gain, gain)['advance'])
        overall['CurvePool'] += Fraction(1, 1000)
        self.assertFalse(wl_channel_scoring.gate(setting, overall, gain, gain)['advance'])
        overall['CurvePool'] -= Fraction(1, 1000)
        overall['Source'] = overall['Target']
        self.assertFalse(wl_channel_scoring.gate(setting, overall, gain, gain)['advance'])

    def test_broken_blind_archive_is_rejected_before_any_e_grant(self):
        with (patch.object(scale, 'observed', return_value=({}, {})),
              patch.object(scale, 'verify_blind', side_effect=PermissionError),
              patch.object(wl_channel_scoring.data, 'ScopedLabels') as read):
            with self.assertRaises(PermissionError):
                wl_channel_scoring.evaluate(None, None, SimpleNamespace(cfg={}, roles={}), 'x', None, {})
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
