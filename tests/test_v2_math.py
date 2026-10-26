import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from tqg_gda.calibration_v2 import CalibrationConfig, project_density, solve_calibration
from tqg_gda.feedback_v2 import (V2Config, ParameterSnapshot, build_gradient_feedback,
    contrast_basis, parameter_feedback_loss, query_matrices, signature_kernels, target_space)
from tqg_gda.models import center_logits


class Tiny(nn.Module):
    def __init__(self, classes=3, stochastic=False):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 4),
            nn.BatchNorm1d(4) if stochastic else nn.Identity(), nn.Tanh(),
            nn.Dropout(.5) if stochastic else nn.Identity())
        self.head = nn.Linear(4, classes)

    def forward(self, batch):
        return center_logits(self.head(self.encoder(batch)))


def flatten(values, parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for g, p in zip(values, parameters)])


class V2MathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.model = Tiny().double().eval()
        self.source = torch.randn(5, 2, dtype=torch.float64)
        self.target = torch.randn(3, 2, dtype=torch.float64)
        self.labels = torch.tensor([0, 1, 2, 1, 0])

    def test_contrast_coordinates_remove_only_the_common_logit_shift(self):
        for classes in (2, 3, 6):
            q = contrast_basis(classes, dtype=torch.float64, device='cpu')
            torch.testing.assert_close(q.mT @ q, torch.eye(classes-1, dtype=q.dtype))
            projector = torch.eye(classes, dtype=q.dtype) - torch.ones(classes, classes, dtype=q.dtype)/classes
            torch.testing.assert_close(q @ q.mT, projector)
            logits = torch.randn(4, classes, dtype=q.dtype)
            torch.testing.assert_close((logits @ q @ q.mT).softmax(-1), logits.softmax(-1))

    def test_full_space_reconstructs_actual_target_ce_gradient_including_head(self):
        snapshot = ParameterSnapshot(self.model)
        q = contrast_basis(3, dtype=torch.float64, device='cpu')
        jacobian = snapshot.jacobian(self.target, q)
        v, stats = target_space(jacobian)
        torch.testing.assert_close(jacobian @ v @ v.mT, jacobian, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(v.mT @ v, torch.eye(stats['rank'], dtype=q.dtype), atol=1e-12, rtol=1e-10)
        target_labels = torch.tensor([0, 2, 1])  # Oracle test only, never passed to the algorithm.
        residual = (F.one_hot(target_labels, 3).double()-self.model(self.target).detach().softmax(-1)) @ q
        a = (jacobian @ v).reshape(3, 2, -1)
        d = torch.einsum('bcr,bc->r', a, residual)/3
        params = tuple(self.model.parameters())
        actual = flatten(torch.autograd.grad(F.cross_entropy(self.model(self.target), target_labels), params), params)
        torch.testing.assert_close(actual, -v @ d, atol=1e-12, rtol=1e-10)
        self.assertTrue(any(name.startswith('head.') for name in snapshot.names))
        self.assertGreater(float(actual[-self.model.head.bias.numel():].norm()), 0)
        source_response = snapshot.response(self.source, q, v)
        expected = (snapshot.jacobian(self.source, q) @ v).reshape(5, 2, -1)
        torch.testing.assert_close(source_response, expected, atol=1e-12, rtol=1e-10)
        self.assertTrue(all(p.grad is None for p in params))

    def test_kernel_scales_use_only_source_pairs_with_shared_target_coordinates(self):
        source_h = torch.tensor([[0., 0.], [1., 0.], [3., 0.]], dtype=torch.float64)
        target_h = torch.tensor([[4., 0.], [5., 0.]], dtype=torch.float64)
        source_a = source_h[:, None, :]
        target_a = target_h[:, None, :]
        kss, kst, ktt, info = signature_kernels(source_h, target_h, source_a, target_a)
        self.assertEqual(info['scale_h_squared'], 4.)
        self.assertEqual(info['scale_a_squared'], 4.)
        h = torch.cat((source_h, target_h))
        expected = torch.exp(-(h[:, None, :]-h[None, :, :]).square().sum(-1)/4.)
        torch.testing.assert_close(kss, expected[:3, :3])
        torch.testing.assert_close(kst, expected[:3, 3:])
        torch.testing.assert_close(ktt, expected[3:, 3:])
        _, _, _, changed = signature_kernels(source_h, target_h*100, source_a, target_a*100)
        self.assertEqual(changed['scale_h_squared'], info['scale_h_squared'])
        self.assertEqual(changed['scale_a_squared'], info['scale_a_squared'])

    def test_query_matrices_match_explicit_graph_major_block_kernels(self):
        kst = torch.rand(4, 3, dtype=torch.float64)
        ktt = torch.rand(3, 3, dtype=torch.float64)
        a = torch.randn(3, 2, 5, dtype=torch.float64)
        hst, htt = query_matrices(kst, ktt, a)
        b = a.reshape(6, 5)/3
        identity = torch.eye(2, dtype=a.dtype)
        torch.testing.assert_close(hst, torch.kron(kst, identity) @ b)
        torch.testing.assert_close(htt, b.mT @ torch.kron(ktt, identity) @ b)

    def test_mean_only_is_exact_kernel_ridge_for_multiclass_blocks(self):
        x = torch.randn(4, 4, dtype=torch.float64)
        kss = x @ x.mT
        kst = torch.randn(4, 3, dtype=torch.float64)
        ktt = kst.mT @ torch.linalg.solve(kss, kst)+torch.eye(3, dtype=x.dtype)
        hst, htt = query_matrices(kst, ktt, torch.randn(3, 2, 3, dtype=x.dtype))
        cfg = CalibrationConfig(mode='mean_only', lambda_w=.13)
        w, info = solve_calibration(kss, hst, htt, cfg)
        expected = torch.linalg.solve(torch.kron(kss, torch.eye(2, dtype=x.dtype))
                                      +.13*torch.eye(8, dtype=x.dtype), hst).mT
        torch.testing.assert_close(w, expected, atol=1e-12, rtol=1e-10)
        self.assertTrue(info['converged'])
        self.assertEqual(info['termination'], 'closed_form_krr')

    def test_joint_calibration_matches_an_independent_diagonal_solution(self):
        kss = torch.eye(2, dtype=torch.float64)
        hst = torch.diag(torch.tensor([1., .3], dtype=kss.dtype))
        htt = torch.diag(torch.tensor([2., .2], dtype=kss.dtype))
        cfg = CalibrationConfig(lambda_w=.4, gap_atol=1e-12, gap_rtol=1e-10)
        w, info = solve_calibration(kss, hst, htt, cfg)
        # E_11 is strictly the worst direction. Omega*=diag(1,0), D*=diag(3,1).
        expected = torch.diag(torch.tensor([3/3.4, .3/1.4], dtype=kss.dtype))
        torch.testing.assert_close(w, expected, atol=1e-10, rtol=1e-10)
        e = htt-w@hst-hst.mT@w.mT+w@kss@w.mT
        objective = e.trace()/2+torch.linalg.eigvalsh(e)[-1]+.4*w.square().sum()/2
        self.assertAlmostEqual(info['primal'], float(objective), places=12)
        self.assertAlmostEqual(info['primal']-info['dual'], info['gap'], places=12)
        self.assertTrue(info['converged'])

    def test_rank_one_joint_has_effective_ridge_lambda_over_two(self):
        kss = torch.eye(3, dtype=torch.float64)
        hst = torch.tensor([[.2], [.1], [.3]], dtype=kss.dtype)
        w, info = solve_calibration(kss, hst, torch.ones(1, 1, dtype=kss.dtype), CalibrationConfig(lambda_w=.4))
        torch.testing.assert_close(w, hst.mT/1.2)
        self.assertEqual(info['gap'], 0.)

    def test_backtracking_converges_when_two_worst_directions_must_balance(self):
        kss = torch.eye(2, dtype=torch.float64)
        hst = torch.diag(torch.tensor([1., .9], dtype=kss.dtype))
        htt = torch.diag(torch.tensor([1.2, 1.05], dtype=kss.dtype))
        cfg = CalibrationConfig(lambda_w=1., initial_step=1e6)
        w, info = solve_calibration(kss, hst, htt, cfg)
        # Independent 1-D solution: the interior optimal density balances E_11/E_22.
        lower, upper = 0., 1.
        for _ in range(80):
            omega = (lower+upper)/2
            difference = .2+1/(2+2*omega)**2-(.24+.81/(4-2*omega)**2)
            if difference > 0:
                lower = omega
            else:
                upper = omega
        omega = (lower+upper)/2
        d1, d2 = 1+2*omega, 3-2*omega
        expected = torch.diag(torch.tensor([d1/(d1+1), .9*d2/(d2+1)], dtype=kss.dtype))
        torch.testing.assert_close(w, expected, atol=3e-6, rtol=0)
        self.assertTrue(info['converged'])
        self.assertGreater(info['backtracks'], 0)

    def test_iteration_cap_reports_a_matched_nonconverged_gap(self):
        kss = torch.eye(2, dtype=torch.float64)
        hst = torch.diag(torch.tensor([1., .3], dtype=kss.dtype))
        htt = torch.diag(torch.tensor([2., .2], dtype=kss.dtype))
        cfg = CalibrationConfig(lambda_w=.4, max_iterations=1)
        w, info = solve_calibration(kss, hst, htt, cfg)
        torch.testing.assert_close(w, hst.mT/1.2)  # Initial Omega=I/r => D=2I.
        e = htt-w@hst-hst.mT@w.mT+w@w.mT
        gap = torch.linalg.eigvalsh(e)[-1]-e.trace()/2
        self.assertAlmostEqual(info['gap'], float(gap), places=12)
        self.assertFalse(info['converged'])
        self.assertEqual(info['termination'], 'iteration_limit')

    def test_singular_source_kernel_is_regularized_and_no_support_gives_zero(self):
        kss = torch.ones(3, 3, dtype=torch.float64)
        hst = torch.zeros(6, 2, dtype=kss.dtype)
        w, info = solve_calibration(kss, hst, torch.eye(2, dtype=kss.dtype))
        self.assertTrue(torch.equal(w, torch.zeros_like(w)))
        self.assertTrue(info['converged'])
        with self.assertRaises(FloatingPointError):
            solve_calibration(-torch.eye(3, dtype=kss.dtype), hst, torch.eye(2, dtype=kss.dtype))

    def test_rotation_does_not_change_physical_gradient_or_kernel(self):
        feedback = build_gradient_feedback(self.model, self.source, self.labels, self.target)
        self.assertEqual(feedback.statistics['status'], 'gradient_feedback')
        a, b = feedback.source_response, feedback.target_response
        o, _ = torch.linalg.qr(torch.randn(a.shape[-1], a.shape[-1], dtype=a.dtype))
        sh, th = self.model.encoder(self.source), self.model.encoder(self.target)
        first = signature_kernels(sh, th, a, b)
        second = signature_kernels(sh, th, a@o, b@o)
        for left, right in zip(first[:3], second[:3]):
            torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-10)
        hst, htt = query_matrices(*second[1:3], b@o)
        w, info = solve_calibration(second[0], hst, htt)
        physical = -(feedback.basis@o) @ (w@feedback.source_residual)
        torch.testing.assert_close(physical, feedback.gradient, atol=3e-6, rtol=3e-5)

    def test_feedback_is_stopped_label_values_do_not_change_readout_and_loss_gradient_is_exact(self):
        model = Tiny(stochastic=True).double().train()
        before = copy.deepcopy(model.state_dict())
        first = build_gradient_feedback(model, self.source, self.labels, self.target)
        second = build_gradient_feedback(model, self.source, (self.labels+1)%3, self.target)
        self.assertEqual(first.statistics['status'], 'gradient_feedback')
        torch.testing.assert_close(first.readout, second.readout, atol=0, rtol=0)
        self.assertGreater(float((first.coefficients-second.coefficients).norm()), 0.)
        for name in ('snapshot', 'basis', 'coefficients', 'contrasts', 'source_response',
                     'target_response', 'source_residual', 'readout'):
            self.assertFalse(getattr(first, name).requires_grad, name)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        parameters = tuple(model.parameters())
        actual = flatten(torch.autograd.grad(parameter_feedback_loss(model, first), parameters), parameters)
        torch.testing.assert_close(actual, first.gradient, atol=1e-12, rtol=1e-10)
        self.assertAlmostEqual(float(parameter_feedback_loss(model, first)),
                               float(first.coefficients.square().sum()/2), places=12)

    def test_numerical_fallback_supplies_a_zero_parameter_gradient(self):
        with patch('tqg_gda.feedback_v2.solve_calibration', side_effect=FloatingPointError('test anomaly')):
            feedback = build_gradient_feedback(self.model, self.source, self.labels, self.target)
        self.assertEqual(feedback.statistics['status'], 'numerical_fallback_source_only')
        self.assertTrue(torch.equal(feedback.gradient, torch.zeros_like(feedback.gradient)))
        grads = torch.autograd.grad(parameter_feedback_loss(self.model, feedback), tuple(self.model.parameters()))
        self.assertTrue(all(torch.equal(g, torch.zeros_like(g)) for g in grads))

    def test_numerical_zero_rank_is_handled_without_truncated_space_claims(self):
        v, stats = target_space(torch.zeros(6, 20, dtype=torch.float64))
        self.assertEqual(v.shape, (20, 0))
        self.assertEqual(stats['rank'], 0)
        self.assertEqual(stats['rank_policy'], 'all_numerically_nonzero')

    def test_zero_source_residual_cannot_create_feedback(self):
        with torch.no_grad():
            self.model.head.bias.copy_(torch.tensor([1000., -1000., -1000.]))
        feedback = build_gradient_feedback(self.model, self.source, torch.zeros(5, dtype=torch.long), self.target)
        self.assertEqual(feedback.statistics['status'], 'gradient_feedback')
        self.assertTrue(torch.equal(feedback.source_residual, torch.zeros_like(feedback.source_residual)))
        self.assertTrue(torch.equal(feedback.gradient, torch.zeros_like(feedback.gradient)))

    def test_density_projection_is_psd_trace_one_even_with_large_common_shift(self):
        for matrix in (torch.randn(4, 4, dtype=torch.float64),
                       torch.eye(4, dtype=torch.float64)*1e14):
            omega = project_density(matrix)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(omega)[0]), -1e-12)
            self.assertAlmostEqual(float(omega.trace()), 1., places=12)


if __name__ == '__main__':
    unittest.main()
