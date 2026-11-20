"""Operating point isolation, numerical ties, and gated target label access."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import fixed_mass


class FixedMassTests(unittest.TestCase):
    def test_seed_intervals_preserve_paired_graph_draws(self):
        settings = dict(strata=['target_domain', 'class'], percentiles=[2.5, 97.5], level=.95,
                        reuse_draws_across_methods_seeds_and_source_directions=True,
                        conditional_on_fitted_models=True, minimum_graphs_per_class_for_interval=2)
        plan = dict(phases={'P': dict(setting='toy', directions=[[0, 1]])}, real_model_seeds=[1, 2],
                    evaluation=dict(cohort_bootstrap_replicates=100, cohort_bootstrap_seed=13,
                                    paired_interval=settings))
        labels = np.array([0, 0, 0, 1, 1, 1])
        old = np.zeros((6, 2))
        new = old.copy()
        new[[0, 3, 4], 1] = 1.
        arms = [{'Left': new, 'Right': old}, {'Left': old, 'Right': new}]
        blocks = [dict(setting='toy', source=0, target=1, seed=seed, ids=np.arange(6), y=labels, logits=z)
                  for seed, z in enumerate(arms, 1)]
        result = fixed_mass.pilot.classification_intervals(plan, blocks, [('Left', 'Right')], .005,
                                                          seed_intervals=True)
        self.assertFalse(result['metadata']['candidate_max_recomputed_each_replicate'])
        self.assertNotIn('directional_gap', result['metadata'])
        a, b = [r for r in result['direction_seed_intervals'] if r['metric'] == 'Accuracy']
        self.assertLess(a['lower'], a['upper'])
        np.testing.assert_allclose([a['lower'], a['upper']], [-b['upper'], -b['lower']])
        pooled = next(r for r in result['interval_rows'] if r['scope'] == 'direction' and r['metric'] == 'Accuracy')
        self.assertEqual((pooled['lower'], pooled['upper']), (0., 0.))

    def test_both_slope_signs_match_the_unlabeled_a_mean_and_bracket(self):
        base = np.array([-2., -.4, .3, 1.7, 4.])
        v = np.array([.2, .6, .8, 0., .4])
        for slope in (-5., 5.):
            result = fixed_mass.fit_intercept(base, v, slope, 1e-10)
            self.assertLessEqual(result['bracket'][0], result['intercept'])
            self.assertLessEqual(result['intercept'], result['bracket'][1])
            moved = fixed_mass.apply(base, v, result)
            self.assertAlmostEqual(expit(moved).mean(), expit(base).mean(), places=12)
            h_v = np.array([.0, 1.])
            h_base = np.array([-10., 4.])
            np.testing.assert_allclose(fixed_mass.apply(h_base, h_v, result),
                                       h_base+result['intercept']+slope*h_v, rtol=0, atol=1e-14)

    def test_zero_slope_and_singleton_bracket_return_original_values(self):
        base = np.array([-3., -0., .7, 2.])
        for slope, v in ((0., np.arange(4)*.2), (4., np.repeat(.25, 4))):
            result = fixed_mass.fit_intercept(base, v, slope, 1e-10)
            np.testing.assert_array_equal(fixed_mass.apply(base, v, result), base)
            self.assertEqual(result['residual'], 0.)

    def test_constant_shift_checks_rank_ties_and_rejects_changed_slope(self):
        z = np.array([0., .25, .25, -.5])
        info = fixed_mass.compare_shift(z, z+.125, .125, 1e-12)
        self.assertEqual(info['changed_average_ranks'], 0)
        self.assertEqual(info['tied_excess_rows_before'], 1)
        with self.assertRaises(ValueError):
            fixed_mass.compare_shift(z, z*2+.125, .125, 1e-12)

    def test_bad_obs_cannot_open_evaluation_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'OBS.json').write_text('{}')
            with patch.object(fixed_mass.r5_inputs, 'evaluation_labels') as labels:
                with self.assertRaises(PermissionError):
                    fixed_mass.score(root, 'incorrect-digest', {}, {}, root)
                labels.assert_not_called()
