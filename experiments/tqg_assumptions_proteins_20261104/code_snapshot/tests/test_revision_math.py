"""D0 numerical contracts independent of real graphs and target labels."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tqg_gda.matching import (MatchingConfig, matching_weights,
                              normalize_embedding, permute_weights)
from tqg_gda.residual_solver import (project_residual, select_ridge_cv,
                                    simplex_projection, uniform_ridge,
                                    weighted_ridge)


class RevisionMathTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(730)
        self.phi = torch.randn(11, 3, 4, generator=self.generator, dtype=torch.float64)
        self.residual = torch.randn(11, 3, generator=self.generator, dtype=torch.float64)

    def test_uniform_weights_recover_global_ridge(self):
        weights = torch.full((11, 5), 1 / 11, dtype=torch.float64)
        expected = uniform_ridge(self.phi, self.residual, .13)
        local = weighted_ridge(self.phi, self.residual, weights, .13)
        torch.testing.assert_close(local, expected.expand(5, -1), atol=1e-13, rtol=1e-12)

    def test_weighted_solution_matches_independent_augmented_least_squares(self):
        weights = torch.randn(11, 5, generator=self.generator, dtype=torch.float64).softmax(0)
        actual = weighted_ridge(self.phi, self.residual, weights, .17)
        for j in range(5):
            observation_weights = weights[:, j].sqrt().repeat_interleave(3)
            design = self.phi.reshape(-1, 4) * observation_weights[:, None]
            rhs = self.residual.flatten() * observation_weights
            design = torch.cat((design, math.sqrt(.17) * torch.eye(4, dtype=torch.float64)))
            rhs = torch.cat((rhs, torch.zeros(4, dtype=torch.float64)))
            expected = torch.linalg.lstsq(design, rhs).solution
            torch.testing.assert_close(actual[j], expected, atol=1e-12, rtol=1e-12)

    def test_regression_is_invariant_to_joint_source_permutation(self):
        weights = torch.randn(11, 6, generator=self.generator, dtype=torch.float64).softmax(0)
        order = torch.randperm(11, generator=self.generator)
        actual = weighted_ridge(self.phi[order], self.residual[order], weights[order], .1)
        expected = weighted_ridge(self.phi, self.residual, weights, .1)
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)

    def test_four_fold_cv_matches_numpy_and_uses_actual_fit_count(self):
        order = torch.tensor([7, 2, 9, 0, 4, 10, 3, 6, 1, 8, 5])
        candidates = (.01, .1, 1.)
        selected, scores = select_ridge_cv(self.phi, self.residual, order, candidates)
        folds = np.array_split(order.numpy(), 4)
        phi, residual = self.phi.numpy(), self.residual.numpy()
        numpy_scores, incorrectly_normalized_scores = {}, {}
        for ridge in candidates:
            error, wrong_error = 0., 0.
            for held in folds:
                fit = np.setdiff1d(np.arange(11), held)
                design = phi[fit].reshape(-1, 4)
                rhs = residual[fit].reshape(-1)
                a = np.linalg.lstsq(np.vstack((design / math.sqrt(len(fit)), math.sqrt(ridge) * np.eye(4))),
                                    np.concatenate((rhs / math.sqrt(len(fit)), np.zeros(4))), rcond=None)[0]
                wrong_a = np.linalg.solve(design.T @ design / 11 + ridge * np.eye(4), design.T @ rhs / 11)
                error += np.square(phi[held] @ a - residual[held]).sum()
                wrong_error += np.square(phi[held] @ wrong_a - residual[held]).sum()
            numpy_scores[ridge], incorrectly_normalized_scores[ridge] = error / 11, wrong_error / 11
            self.assertAlmostEqual(scores[ridge], numpy_scores[ridge], places=12)
        self.assertGreater(max(abs(scores[r] - incorrectly_normalized_scores[r]) for r in candidates), 1e-3)
        self.assertEqual(selected, min(numpy_scores, key=numpy_scores.get))

    def test_cv_audit_graphs_never_enter_corresponding_training_fold(self):
        # Encode a unique graph ID into a feature coordinate to inspect the
        # actual solver calls rather than trusting the fold documentation.
        phi = self.phi.clone()
        phi[:, 0, 0] = torch.arange(11, dtype=phi.dtype)
        order = torch.tensor([7, 2, 9, 0, 4, 10, 3, 6, 1, 8, 5])
        folds = torch.tensor_split(order, 4)
        observed = []
        def recording_solver(features, residual, ridge):
            observed.append(features[:, 0, 0].long().tolist())
            return uniform_ridge(features, residual, ridge)
        with patch('tqg_gda.residual_solver.uniform_ridge', side_effect=recording_solver):
            select_ridge_cv(phi, self.residual, order, candidates=(.1,))
        for held, training in zip(folds, observed):
            self.assertFalse(set(held.tolist()) & set(training))
            self.assertEqual(set(held.tolist()) | set(training), set(range(11)))

    def test_cv_ties_choose_largest_ridge_and_zero_column_dictionary_is_supported(self):
        order = torch.randperm(11, generator=self.generator)
        zero_phi = self.phi[:, :, :0]
        self.assertEqual(uniform_ridge(zero_phi, self.residual, .1).shape, (0,))
        weights = torch.full((11, 3), 1 / 11, dtype=torch.float64)
        self.assertEqual(weighted_ridge(zero_phi, self.residual, weights, .1).shape, (3, 0))
        selected, scores = select_ridge_cv(zero_phi, self.residual, order)
        self.assertEqual(selected, 1.)
        self.assertEqual(len(set(scores.values())), 1)
        selected, _ = select_ridge_cv(self.phi, self.residual * 0, order, candidates=(1., .01, .1))
        self.assertEqual(selected, 1.)

    def test_projection_is_legal_and_never_increases_error_to_any_true_class(self):
        for classes in (2, 3, 7):
            p = torch.randn(61, classes, generator=self.generator, dtype=torch.float64).softmax(-1)
            d = torch.randn(61, classes, generator=self.generator, dtype=torch.float64) * 5
            projected = project_residual(p, d)
            q = p + projected
            self.assertTrue(bool((q >= -1e-13).all()))
            torch.testing.assert_close(q.sum(-1), torch.ones(61, dtype=q.dtype), atol=1e-12, rtol=0)
            # Exhaust all true classes, not just a favorable sampled label.
            for true_class in range(classes):
                truth = torch.zeros_like(p)
                truth[:, true_class] = 1
                error_after = (projected - (truth - p)).square().sum(-1)
                error_before = (d - (truth - p)).square().sum(-1)
                self.assertTrue(bool((error_after <= error_before + 1e-11).all()))

    def test_projection_has_kkt_and_translation_invariance(self):
        value = torch.tensor([[.2, .5, -.4], [5., -2., 3.], [-4., -4., -4.]], dtype=torch.float64)
        q = simplex_projection(value)
        expected = torch.tensor([[.35, .65, 0.], [1., 0., 0.], [1/3, 1/3, 1/3]], dtype=torch.float64)
        torch.testing.assert_close(q, expected, atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(simplex_projection(value + 1e6), q, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(simplex_projection(q), q, atol=1e-13, rtol=1e-12)

    def test_outputs_are_detached_float64(self):
        phi = self.phi.float().requires_grad_(True)
        residual = self.residual.float().requires_grad_(True)
        weights = torch.full((11, 2), 1/11, dtype=torch.float64, requires_grad=True)
        values = [uniform_ridge(phi, residual, .1), weighted_ridge(phi, residual, weights, .1),
                  simplex_projection(residual), normalize_embedding(residual)]
        p = self.residual.softmax(-1).requires_grad_(True)
        values.append(project_residual(p, residual))
        for value in values:
            self.assertFalse(value.requires_grad)
            self.assertEqual(value.dtype, torch.float64)
        self.assertIsNone(phi.grad)
        self.assertIsNone(residual.grad)

    def test_invalid_regression_cv_and_projection_are_errors(self):
        bad_phi = self.phi.clone()
        bad_phi[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            uniform_ridge(bad_phi, self.residual, .1)
        for ridge in (0, -.1, float('inf')):
            with self.assertRaises(ValueError):
                uniform_ridge(self.phi, self.residual, ridge)
        for order in (torch.zeros(11, dtype=torch.long), torch.arange(11).double(), torch.arange(10)):
            with self.assertRaises(ValueError):
                select_ridge_cv(self.phi, self.residual, order)
        with self.assertRaises(ValueError):
            weighted_ridge(self.phi, self.residual, torch.ones(11, 2), .1)
        with self.assertRaises(ValueError):
            project_residual(torch.tensor([[.3, .3]]), torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            simplex_projection(torch.tensor([[0., float('inf')]]))


class MatchingMathTests(unittest.TestCase):
    def test_scale_uses_mathematical_median_and_threshold_excludes_diagonal(self):
        source = torch.tensor([0., 1., 3., 7.], dtype=torch.float64).reshape(4, 1, 1)
        target = torch.tensor([.5, 30.], dtype=torch.float64).reshape(2, 1, 1)
        result = matching_weights(source, target)
        # Positive pair distances: 1,9,49,4,36,16 -> median (9+16)/2.
        self.assertAlmostEqual(float(result['distance_scale']), 12.5, places=13)
        # Source LOO distances: 1,1,4,16. Linear q95=14.2.
        self.assertAlmostEqual(float(result['coverage_threshold']), 14.2 / 12.5, places=13)
        self.assertEqual(result['gate'].tolist(), [True, False])
        expected_raw = (-result['distances'] / 12.5).softmax(0)
        torch.testing.assert_close(result['weights'], .25 / 4 + .75 * expected_raw, atol=1e-14, rtol=1e-13)
        torch.testing.assert_close(result['ess'], 1 / result['weights'].square().sum(0))

    def test_distance_is_coordinate_mean_not_class_sum(self):
        source = torch.tensor([[[0., 1.], [2., 3.]], [[1., 2.], [3., 4.]]])
        target = torch.tensor([[[1., 1.], [1., 1.]]])
        result = matching_weights(source, target)
        torch.testing.assert_close(result['distances'][:, 0], torch.tensor([1.5, 3.5], dtype=torch.float64))

    def test_matching_is_invariant_to_joint_feature_and_source_permutations(self):
        gen = torch.Generator().manual_seed(481)
        source = torch.randn(9, 3, 4, generator=gen, dtype=torch.float64)
        target = torch.randn(5, 3, 4, generator=gen, dtype=torch.float64)
        source_order = torch.randperm(9, generator=gen)
        feature_order = torch.randperm(12, generator=gen)
        original = matching_weights(source, target)
        changed = matching_weights(source.flatten(1)[source_order][:, feature_order], target.flatten(1)[:, feature_order])
        torch.testing.assert_close(changed['weights'], original['weights'][source_order])
        self.assertTrue(torch.equal(changed['gate'], original['gate']))

    def test_uniform_limit_and_uninformative_signatures(self):
        source = torch.arange(8, dtype=torch.float64).reshape(4, 2)
        target = torch.tensor([[2., 3.], [4., 5.]], dtype=torch.float64)
        result = matching_weights(source, target, MatchingConfig(gamma=0))
        torch.testing.assert_close(result['weights'], torch.full((4, 2), .25, dtype=torch.float64))
        for source in (torch.zeros(4, 3), torch.zeros(1, 3)):
            result = matching_weights(source, torch.ones(2, 3))
            self.assertTrue(result['statistics']['uninformative_signature'])
            self.assertIsNone(result['distance_scale'])
            self.assertIsNone(result['coverage_threshold'])
            self.assertFalse(bool(result['gate'].any()))
            torch.testing.assert_close(result['weights'], torch.full((len(source), 2), 1/len(source), dtype=torch.float64))

    def test_ess_gate_rejects_concentrated_match_even_when_covered(self):
        # 8 separated graphs; with almost-hard local weights ESS~1 < nS/4.
        source = torch.arange(8, dtype=torch.float64)[:, None]
        result = matching_weights(source, source[:1], MatchingConfig(gamma=1, tau=.0001))
        self.assertEqual(float(result['nearest_scaled_distance'][0]), 0.)
        self.assertAlmostEqual(float(result['ess'][0]), 1.)
        self.assertFalse(bool(result['gate'][0]))

    def test_permutation_preserves_column_multisets_and_does_not_consume_global_rng(self):
        weights = torch.tensor([[.05, .4], [.15, .3], [.3, .2], [.5, .1]], dtype=torch.float64)
        state = torch.random.get_rng_state().clone()
        actual = permute_weights(weights, torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        torch.testing.assert_close(actual.sort(dim=0).values, weights.sort(dim=0).values)
        torch.testing.assert_close(actual.square().sum(0).reciprocal(), weights.square().sum(0).reciprocal())
        self.assertFalse(torch.equal(actual, weights))
        self.assertTrue(torch.equal(actual, permute_weights(weights, torch.Generator().manual_seed(42))))

    def test_embedding_normalization_handles_zero_and_detaches(self):
        value = torch.tensor([[3., 4.], [0., 0.], [1e-10, 0.]], requires_grad=True)
        actual = normalize_embedding(value)
        expected = torch.tensor([[.6, .8], [0., 0.], [.01, 0.]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)
        self.assertEqual(actual.dtype, torch.float64)
        self.assertFalse(actual.requires_grad)

    def test_matching_outputs_detach_and_nonfinite_is_not_silent_rejection(self):
        source = torch.tensor([[0., 1.], [2., 3.]], requires_grad=True)
        target = torch.tensor([[1., 2.]], requires_grad=True)
        result = matching_weights(source, target)
        for value in result.values():
            if isinstance(value, torch.Tensor):
                self.assertFalse(value.requires_grad)
                if value.dtype != torch.bool:
                    self.assertEqual(value.dtype, torch.float64)
        for invalid in (torch.tensor([[float('nan'), 1.]]), torch.tensor([[float('inf'), 1.]])):
            with self.assertRaises(ValueError):
                matching_weights(source, invalid)
        with self.assertRaises(ValueError):
            matching_weights(torch.zeros(2, 3, 0), torch.zeros(1, 3, 0))
        with self.assertRaises(ValueError):
            matching_weights(torch.tensor([[1e308], [-1e308]]), torch.tensor([[0.]]))
        for args in ({'gamma': 1.1}, {'tau': 0}, {'eps_distance': float('nan')}, {'ess_fraction': -.1}):
            with self.assertRaises(ValueError):
                MatchingConfig(**args)


if __name__ == '__main__':
    unittest.main()
