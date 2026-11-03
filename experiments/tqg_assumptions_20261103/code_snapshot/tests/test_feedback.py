import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from tqg_gda.feedback import (ProbeConfig, Snapshot, build_feedback, feedback_loss,
                              fit_residual, shared_response_scale)
from tqg_gda.models import center_logits


class Tiny(nn.Module):
    def __init__(self, batchnorm=False):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4) if batchnorm else nn.Identity(),
                                     nn.Tanh(), nn.Dropout(.4) if batchnorm else nn.Identity(), nn.Linear(4, 3), nn.Tanh())
        self.head = nn.Linear(3, 2)

    def forward(self, x):
        return center_logits(self.head(self.encoder(x)))


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(31)
        self.model = Tiny().double().eval()
        self.source = torch.randn(4, 2, dtype=torch.float64)
        self.target = torch.randn(3, 2, dtype=torch.float64)
        self.labels = torch.tensor([0, 1, 0, 1])

    def query(self, config=ProbeConfig(count=2)):
        return build_feedback(self.model, self.source, self.labels, self.target, config,
                              torch.Generator().manual_seed(7))

    def test_centering_preserves_probabilities_and_is_per_graph(self):
        z = torch.tensor([[5., 3.], [1., 7.]], dtype=torch.float64)
        torch.testing.assert_close(center_logits(z), torch.tensor([[1., -1.], [-3., 3.]], dtype=z.dtype))
        torch.testing.assert_close(z.softmax(-1), center_logits(z).softmax(-1))

    def test_queries_are_stateless_and_all_feedback_is_stopped(self):
        before = copy.deepcopy(self.model.state_dict())
        feedback = self.query()
        self.assertEqual(feedback.statistics['valid_directions'], 2)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        for name in ('source_features', 'target_features', 'source_residual', 'coefficients', 'reference', 'correction', 'scales'):
            self.assertFalse(getattr(feedback, name).requires_grad, name)
        self.assertTrue(all(n.startswith('encoder.') for d in feedback.directions for n in d))
        torch.testing.assert_close(feedback.correction.sum(-1), torch.zeros(3, dtype=torch.float64), atol=1e-10, rtol=0)

    def test_jvp_target_normalization_and_finite_responses_match_definition(self):
        config = ProbeConfig(count=1, radius=.03)
        feedback = self.query(config)
        snapshot = Snapshot(self.model)
        direction = tuple(feedback.directions[0][n] for n in snapshot.names)
        tangent = snapshot.jvp(self.target, direction)
        self.assertAlmostEqual(float(tangent.square().sum(-1).mean().sqrt()), 1., places=5)
        with torch.no_grad():
            plus = tuple(p + config.radius * v for p, v in zip(snapshot.values, direction))
            minus = tuple(p - config.radius * v for p, v in zip(snapshot.values, direction))
            for batch, measured in ((self.source, feedback.source_features), (self.target, feedback.target_features)):
                zp, zm, z0 = snapshot.logits(batch, plus), snapshot.logits(batch, minus), snapshot.logits(batch)
                expected = torch.stack(((zp-zm)/(2*config.radius), (zp+zm-2*z0)/config.radius**2), -1)
                torch.testing.assert_close(measured * feedback.scales, expected, atol=1e-11, rtol=1e-10)
        self.assertGreater(float(feedback.source_features[..., 1].abs().max()), 1e-5)

    def test_shared_scaling_is_equal_domain_rms_without_sample_centering(self):
        source = torch.tensor([[[1., 3.], [-1., -3.]]], dtype=torch.float64)
        target = source.repeat(3, 1, 1) * 2
        s, t, d = shared_response_scale(source, target, 1e-8)
        expected = (.5*source.square().sum(1).mean(0) + .5*target.square().sum(1).mean(0) + 1e-16).sqrt()
        torch.testing.assert_close(d, expected)
        torch.testing.assert_close(s * d, source); torch.testing.assert_close(t * d, target)
        self.assertGreater(float(s[:, 0].mean()), 0.)

    def test_ridge_solution_minimizes_the_specified_class_sum_objective(self):
        features = torch.randn(5, 2, 4, dtype=torch.float64)
        residual = torch.randn(5, 2, dtype=torch.float64)
        ridge = .07
        a = fit_residual(features, residual, ridge).requires_grad_(True)
        objective = (features @ a - residual).square().sum(-1).mean() + ridge * a.square().sum()
        gradient, = torch.autograd.grad(objective, a)
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)
        zero = fit_residual(features, torch.zeros_like(residual), ridge)
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))

    def test_feedback_loss_gradient_equals_minus_target_jacobian_times_correction(self):
        feedback = self.query()
        parameters = tuple(self.model.parameters())
        expected = torch.autograd.grad(-(self.model(self.target) * feedback.correction).sum() / len(self.target), parameters)
        actual = torch.autograd.grad(feedback_loss(self.model, self.target, feedback), parameters)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
        self.assertGreater(float(actual[-2].norm()), 0.)  # Head is trained although not probed.

    def test_random_source_odd_and_linear_ablations(self):
        for direction in ('target', 'random', 'source'):
            for responses, multiplier in (('odd_even', 2), ('odd', 1), ('linear', 1)):
                with self.subTest(direction=direction, responses=responses):
                    feedback = self.query(ProbeConfig(count=2, direction=direction, responses=responses))
                    self.assertEqual(feedback.target_features.shape[-1], multiplier*2)
                    self.assertTrue(torch.isfinite(feedback.reference).all())

    def test_no_target_sensitivity_falls_back_without_retry(self):
        with torch.no_grad():
            self.model.head.weight.zero_()
        feedback = self.query(ProbeConfig(count=3))
        self.assertEqual(feedback.statistics['candidate_directions'], 3)
        self.assertEqual(feedback.statistics['valid_directions'], 0)
        self.assertEqual(len(feedback.statistics['discarded']), 3)
        self.assertEqual(feedback.target_features.shape[-1], 0)
        self.assertTrue(torch.equal(feedback.correction, torch.zeros_like(feedback.correction)))
        self.assertEqual(float(feedback_loss(self.model, self.target, feedback)), 0.)

    def test_batchnorm_buffers_frozen_and_dropout_deterministic(self):
        self.model = Tiny(batchnorm=True).double().train()
        before = copy.deepcopy(self.model.state_dict())
        first, second = self.query(), self.query()
        self.assertTrue(torch.equal(first.correction, second.correction))
        self.assertFalse(self.model.training)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_bad_probe_parameters_are_rejected(self):
        for kwargs in ({'radius': 0}, {'ridge': 0}, {'count': 0}, {'epsilon': float('nan')}):
            with self.assertRaises(ValueError):
                ProbeConfig(**kwargs)

    def test_nonfinite_target_sensitivity_discards_candidates_in_both_domains(self):
        with patch.object(Snapshot, 'jvp', return_value=torch.full((3, 2), float('inf'), dtype=torch.float64)):
            feedback = self.query()
        self.assertEqual(feedback.statistics['valid_directions'], 0)
        self.assertEqual(feedback.source_features.shape[-1], 0)
        self.assertEqual(feedback.target_features.shape[-1], 0)
        self.assertTrue(torch.isfinite(feedback.reference).all())

    def test_float32_probes_and_feedback_remain_finite(self):
        self.model = self.model.float()
        feedback = build_feedback(self.model, self.source.float(), self.labels, self.target.float(),
                                  ProbeConfig(count=2, radius=.1), torch.Generator().manual_seed(7))
        self.assertEqual(feedback.reference.dtype, torch.float32)
        self.assertTrue(torch.isfinite(feedback.reference).all())

    def test_multiclass_feedback_uses_all_class_coordinates(self):
        self.model.head = nn.Linear(3, 3).double()
        feedback = build_feedback(self.model, self.source, torch.tensor([0, 1, 2, 0]), self.target,
                                  ProbeConfig(count=2), torch.Generator().manual_seed(7))
        self.assertEqual(feedback.correction.shape, (3, 3))
        self.assertEqual(feedback.source_residual.shape, (4, 3))
        torch.testing.assert_close(feedback.correction.sum(-1), torch.zeros(3, dtype=torch.float64), atol=1e-10, rtol=0)


if __name__ == '__main__':
    unittest.main()
