"""Generated topology, analytic KL projection, and transfer counterexample."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.special import expit, logit
import torch

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, projection

ROOT = Path(__file__).resolve().parents[2] / 'experiments/mutagenicity/r8_图结构观察量与类别约束_20261121'
OPTIONS = json.loads((ROOT/'configs/pilot-plan.json').read_text())['projection']


class PilotTests(unittest.TestCase):
    def test_unordered_pairs_unknown_and_edgeless(self):
        def graph(nodes, edge_index, kinds):
            return dict(x=torch.nn.functional.one_hot(torch.tensor(nodes), 3),
                        edge_index=torch.tensor(edge_index, dtype=torch.int64).reshape(2, -1),
                        edge_attr=torch.nn.functional.one_hot(torch.tensor(kinds, dtype=torch.long), 2))
        original = graph([1, 0], [[0, 1], [1, 0]], [1, 1])
        fit = [composition.graph_counts(original, 3, 2)]
        vocabulary = composition.Vocabulary.fit(fit)
        query = [composition.graph_counts(graph([2, 2], [[0, 1], [1, 0]], [0, 0]), 3, 2),
                 composition.graph_counts(graph([0], [[], []], []), 3, 2)]
        np.testing.assert_array_equal(vocabulary.encode([*fit, *query]),
                                      [[.5, .5, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            composition.graph_counts(original | {'y': torch.tensor(0)}, 3, 2)
        with self.assertRaises(ValueError):
            composition.graph_counts(graph([1, 0], [[0, 1], [1, 0]], [1, 0]), 3, 2)

    def test_base_feasibility_returns_original_logits_exactly(self):
        v = np.array([0., 1.]*40)
        z = logit(np.tile([.22, .78], 40))
        bands = projection.Bands.build(v, np.array([.2, .8]), np.array([.1, .1]), .05)
        interval = projection.prior_interval(bands, OPTIONS)
        result = projection.joint_projection(z, bands, interval, OPTIONS)
        self.assertEqual(result['status'], 'base_feasible')
        np.testing.assert_array_equal(projection.projected_logits(z, v, result), z)
        self.assertEqual(projection.prior_projection(z, interval, OPTIONS)['status'], 'base_feasible')

    def test_two_score_exact_solution_and_prior_control(self):
        v = np.tile([0., 1.], 40)
        z = logit(np.tile([.62, .38], 40))
        bands = projection.Bands.build(v, np.array([.2, .8]), np.zeros(2), 0.)
        interval = projection.prior_interval(bands, OPTIONS)
        np.testing.assert_allclose([interval['lower'], interval['upper']], .5, atol=1e-10, rtol=0)
        result = projection.joint_projection(z, bands, interval, OPTIONS, exact=True)
        self.assertTrue(result['valid'], result)
        np.testing.assert_allclose(expit(projection.projected_logits(z, v, result)),
                                   np.tile([.2, .8], 40), atol=1e-8, rtol=0)
        self.assertLess(result['dual_kkt_inf'], 1e-8)
        self.assertEqual(projection.prior_projection(z, interval, OPTIONS)['status'], 'base_feasible')

    def test_banded_solution_kkt_and_pointwise_analytic_limits(self):
        v = np.array([0., 1.]*40)
        z = logit(np.full(len(v), .5))
        # Symmetry sets E[q]=.5. Both moments allow q(1) >= .8-2*eps_A.
        bands = projection.Bands.build(v, np.array([.2, .8]), np.zeros(2), .025)
        result = projection.joint_projection(z, bands, projection.prior_interval(bands, OPTIONS), OPTIONS)
        self.assertTrue(result['valid'], result)
        self.assertLess(result['dual_kkt_inf'], 1e-8)
        np.testing.assert_allclose(expit(projection.projected_logits(z, v, result)),
                                   np.tile([.25, .75], 40), rtol=0, atol=1e-8)

    def test_exact_empty_and_boundary_have_distinct_status(self):
        v = np.repeat(.1, 6)
        bands = projection.Bands.build(v, np.array([.2, .8]), np.zeros(2), 0.)
        self.assertEqual(projection.prior_interval(bands, OPTIONS)['status'], 'infeasible')
        v = np.tile([0., 1.], 5)
        bands = projection.Bands.build(v, np.array([0., 1.]), np.zeros(2), 0.)
        intervals = projection.prior_interval(bands, OPTIONS)
        self.assertEqual(projection.joint_projection(np.zeros(len(v)), bands, intervals, OPTIONS, exact=True)['status'],
                         'boundary_only_or_near_boundary')

    def test_reflection_matches_prior_and_marginal_but_projection_harms(self):
        v = np.tile([0., 1.], 50)
        stable, reflected = .2+.6*v, .8-.6*v
        self.assertAlmostEqual(stable.mean(), reflected.mean(), places=14)
        logits = logit(reflected)
        bands = projection.Bands.build(v, np.array([.2, .8]), np.zeros(2), 0.)
        result = projection.joint_projection(logits, bands, projection.prior_interval(bands, OPTIONS), OPTIONS, exact=True)
        after = projection.projected_logits(logits, v, result)
        expected_nll = lambda z: (np.logaddexp(0., z)-reflected*z).mean()
        self.assertGreater(expected_nll(after), expected_nll(logits)+.8)
        self.assertLess(bands.slack(reflected).min(), -.2)
