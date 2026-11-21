"""Target-only replay controls and exact supplement scope."""
from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mutagenicity.r19_分子路径闭合与分类迁移_20261121.src import target_change as delta, target_change_scoring


class TargetChangeTests(unittest.TestCase):
    def test_opportunity_change_matches_replay_and_constant_uses_reference(self):
        counts = {
            0: (4, 3, 0, 0, 0, 0, 0, 0, 0, 0),
            1: (6, 6, 12, 0, 0, 0, 2, 0, 0, 0),
            2: (8, 8, 8, 0, 0, 0, 2, 0, 0, 0),
            3: (8, 8, 16, 0, 0, 0, 2, 0, 0, 0),
        }
        model = dict(references={'U': [2, 3]}, pS=['1/2', '0', '0', '0'], pU=['1/4', '0', '0', '0'],
                     pMix=['3/8', '0', '0', '0'], intercept=.4,
                     background=dict(rms=[1., 0., 0., 0.], coef=[2., 0., 0., 0.]),
                     raw=dict(rms=[0.]*8, coef=[0.]*8))
        x = delta.decisions(counts, np.array([0, 1]), np.array([-.4, -.3]), model)
        np.testing.assert_array_equal(x['delta'], [0., .5])
        np.testing.assert_allclose(x['delta'], x['margin_TargetBg'] - x['margin_SourceBg'], atol=1e-15)
        self.assertEqual(x['offset'], .375)
        np.testing.assert_array_equal(x['MixtureDelta'], [False, False])
        np.testing.assert_array_equal(x['Delta'], [False, True])
        self.assertEqual(x['Delta'].sum(), x['Count'].sum())

    def test_only_u_complement_receives_a_label_grant(self):
        roles = dict(D=[1], C=[2], U=[6, 5, 4, 3], E=[7])
        self.assertEqual(delta.supplement_ids(roles, [5], [4, 5]), [3, 6])
        with self.assertRaises(PermissionError):
            delta.supplement_ids(roles, [7], [4])
        plan = {'x': dict(setting='mutagenicity_edge', status='ok', roles=roles, supplement=[3, 6])}
        delta.validate_score_receipt(plan, {'x': {'E': [3, 6]}})
        with self.assertRaises(PermissionError):
            delta.validate_score_receipt(plan, {'x': {'E': [7]}})

    def test_blind_must_agree_before_label_port_exists(self):
        with (patch.object(delta, 'observed', return_value={}),
              patch.object(delta, 'verify_blind', side_effect=PermissionError),
              patch.object(target_change_scoring.data, 'ScopedLabels') as reader):
            with self.assertRaises(PermissionError):
                from types import SimpleNamespace
                target_change_scoring.evaluate(None, None, SimpleNamespace(cfg={}), 'sha', None, {})
        reader.assert_not_called()

    def test_exact_gate_threshold_and_leave_source(self):
        scores = {m: Fraction(1, 2) for m in delta.ARMS}
        scores['Delta'] += Fraction(1, 200)
        source = {0: Fraction(-1, 400), 1: Fraction(1, 400), 2: Fraction(1, 400), 3: Fraction(1, 400)}
        self.assertTrue(target_change_scoring.gate(scores, source, {k: Fraction(1, 400) for k in source})['advance'])
        self.assertFalse(target_change_scoring.gate(scores, source, {0: Fraction(0)})['advance'])
        scores['Reverse'] = scores['Delta']
        self.assertFalse(target_change_scoring.gate(scores, source, {0: Fraction(1, 400)})['advance'])


if __name__ == '__main__':
    unittest.main()
