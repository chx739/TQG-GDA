"""Regressions for rank loss and finite-logit feasibility in the R8 pilot."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.special import expit, logit

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import projection


PLAN = Path(__file__).resolve().parents[2] / (
    'experiments/mutagenicity/r8_图结构观察量与类别约束_20261121/configs/pilot-plan.json'
)
OPTIONS = json.loads(PLAN.read_text())['projection']


class ExactMomentTests(unittest.TestCase):
    def test_nearly_equal_means_identify_the_same_prior_after_replication(self):
        mu = [.5 - .2e-8, .5 + .8e-8]
        prior = (.5 - mu[0]) / (mu[1] - mu[0])
        for n_pairs in (1, 25, 200):
            with self.subTest(n_pairs=n_pairs):
                v = np.tile([0., 1.], n_pairs)
                base = np.full(len(v), logit(.8))
                bands = projection.Bands.build(v, mu, [0, 0], 0)
                interval = projection.prior_interval(bands, OPTIONS)
                self.assertEqual(interval['status'], 'feasible', interval)
                np.testing.assert_allclose([interval['lower'], interval['upper']], prior, atol=1e-9, rtol=0)
                joint = projection.joint_projection(base, bands, interval, OPTIONS, exact=True)
                self.assertTrue(joint['valid'], joint)
                # For binary v, E[qv]=mu1*prior separately determines q(0), q(1).
                expected = np.where(v, 2*mu[1]*prior, 2*(1-mu[1])*prior)
                q = expit(projection.projected_logits(base, v, joint))
                np.testing.assert_allclose(q, expected, atol=1e-8, rtol=0)

    def test_boundary_only_even_with_a_nearly_extreme_source(self):
        v = np.array([0., 1.])
        bands = projection.Bands.build(v, [0., 1.], [0., 0.], 0.)
        interval = projection.prior_interval(bands, OPTIONS)
        result = projection.joint_projection(np.array([-25., 25.]), bands, interval, OPTIONS, exact=True)
        self.assertFalse(result['valid'], result)
        self.assertEqual(result['status'], 'boundary_only_or_near_boundary')

    def test_prior_singleton_at_boundary_even_with_extreme_logits(self):
        for target, log_odds in ((0., -40.), (1., 40.)):
            result = projection.prior_projection(np.array([log_odds]),
                dict(status='feasible', lower=target, upper=target), OPTIONS)
            self.assertFalse(result['valid'], result)
            self.assertEqual(result['status'], 'boundary_only')

    def test_equal_class_means_can_leave_prior_free_or_make_moments_impossible(self):
        v = np.array([0., 1.])
        for mu, expected in (([.5, .5], 'feasible'), ([.51, .51], 'infeasible')):
            with self.subTest(mu=mu):
                bands = projection.Bands.build(v, mu, [0., 0.], 0.)
                interval = projection.prior_interval(bands, OPTIONS)
                self.assertEqual(interval['status'], expected, interval)
                if expected == 'feasible':
                    np.testing.assert_allclose([interval['lower'], interval['upper']], [0., 1.])
                    base = np.array([-2., 1.])
                    result = projection.joint_projection(base, bands, interval, OPTIONS, exact=True)
                    self.assertTrue(result['valid'], result)
                    np.testing.assert_allclose(projection.projected_logits(base, v, result), [-.5, -.5], atol=1e-8)

    def test_equal_means_with_exact_cancellation_at_different_scales(self):
        v = np.array([.5, .5-2**-40, 2**-40-3*2**-67, .5, 2**-67, 2**-67, .5, 2**-67])
        bands = projection.Bands.build(v, [.25, .25], [0., 0.], 0.)
        interval = projection.prior_interval(bands, OPTIONS)
        self.assertEqual(interval['status'], 'feasible', interval)
        np.testing.assert_allclose([interval['lower'], interval['upper']], [0., 1.], atol=1e-9)

    def test_decimal_endpoint_does_not_round_outside_the_feasible_interval(self):
        for mu in ([.2, .8], [.8, .2]):
            for label in (0, 1):
                v = np.full(2, mu[label])
                bands = projection.Bands.build(v, mu, [0., 0.], 0.)
                interval = projection.prior_interval(bands, OPTIONS)
                self.assertEqual(interval['status'], 'feasible', interval)
                np.testing.assert_allclose([interval['lower'], interval['upper']], label, atol=1e-12)
                result = projection.joint_projection(np.zeros(2), bands, interval, OPTIONS, exact=True)
                self.assertEqual(result['status'], 'boundary_only_or_near_boundary')
