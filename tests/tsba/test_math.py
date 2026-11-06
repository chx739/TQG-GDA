import dataclasses
import unittest
import torch
from torch import nn
from torch.func import grad
from tqg_gda.tsba.classifier import Coordinates, ResidualClassifier, ParameterMap
from tqg_gda.tsba.config import Config
from tqg_gda.tsba.boundary import build_boundaries, project
from tqg_gda.tsba.problem import TsbaProblem
from tqg_gda.tsba.subproblem import GGNSystem, solve
from tqg_gda.tsba.score import bandwidth, denoising_loss
from tqg_gda.tsba.utils import NumericalError

torch.set_num_threads(1)


class ConstantScore(nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x, self.value)


def fixture(**overrides):
    torch.manual_seed(7)
    model = ResidualClassifier(1, 2, 2, dtype=torch.float64)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0], [0.0]], dtype=torch.float64))
        model.linear.bias.zero_()
    source = torch.tensor([[-2.0], [2.0]], dtype=torch.float64)
    target = torch.tensor([[-0.2], [0.3]], dtype=torch.float64)
    labels = torch.tensor([1, 0])
    cfg = Config(cg_tolerance=1e-9, **overrides)
    anchors = build_boundaries(model, target, ConstantScore(), 1.0, cfg)
    return model, source, labels, target, cfg, anchors


class MathTests(unittest.TestCase):
    def test_coordinates_preserve_probabilities_and_center_logits(self):
        torch.manual_seed(4)
        h = torch.randn(13, 5, dtype=torch.float64) * 3 + 7
        coords = Coordinates.fit(h, 1e-8)
        source = nn.Linear(5, 3, dtype=torch.float64)
        classifier = ResidualClassifier.from_source(source, coords, 4, 19)
        torch.testing.assert_close(classifier(coords(h)).softmax(-1), source(h).softmax(-1), atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(classifier(coords(h)).sum(1), torch.zeros(13, dtype=h.dtype), atol=1e-13, rtol=0)
        self.assertFalse(coords.mean.requires_grad)

    def test_projection_sign_normalization_and_detachment(self):
        model, _, _, _, _, b = fixture()
        self.assertEqual(len(b), 2)
        torch.testing.assert_close(b.points, torch.zeros_like(b.points), atol=1e-9, rtol=0)
        torch.testing.assert_close(b.zeta, torch.full_like(b.zeta, .1))
        for tensor in (b.points, b.normals, b.normal_norm, b.old_margin, b.zeta):
            self.assertFalse(tensor.requires_grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_invalid_boundary_pair_radius_and_zero_gradient(self):
        cfg = Config()
        model = nn.Linear(1, 3, dtype=torch.float64)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.], [0.], [0.]]))
            model.bias.copy_(torch.tensor([0., 0., 1.]))
        value, reason = project(model, torch.tensor([.2], dtype=torch.float64), torch.tensor([0, 1]), 1., 1e-5, cfg)
        self.assertIsNone(value)
        self.assertEqual(reason, "inactive_class_pair")
        model, _, _, _, _, _ = fixture()
        value, _ = project(model, torch.tensor([3.], dtype=torch.float64), torch.tensor([0, 1]), .1, 1e-5, cfg)
        self.assertIsNone(value)
        with torch.no_grad():
            model.linear.weight.zero_()
        value, reason = project(model, torch.tensor([.2], dtype=torch.float64), torch.tensor([0, 1]), 1., 1e-5, cfg)
        self.assertEqual(reason, "small_normal")

    def test_fixed_noise_bandwidth_with_duplicates(self):
        x = torch.tensor([[0., 0.], [0., 0.], [3., 4.]], dtype=torch.float64)
        self.assertAlmostEqual(bandwidth(x, 1), 5 / 2 ** .5)
        self.assertIsNone(bandwidth(x[:2], 1))
        self.assertIsNone(bandwidth(x[:1], 1))

    def test_distance_overflow_is_not_reported_as_duplicate_target_points(self):
        x = torch.tensor([[0.], [1e20]], dtype=torch.float32)
        with self.assertRaises(NumericalError):
            bandwidth(x, 1.)

    def test_normal_norm_overflow_cannot_create_valid_anchors(self):
        model = nn.Linear(2, 2, dtype=torch.float32)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1e20, 1e20], [0., 0.]]))
            model.bias.zero_()
        with self.assertRaises(NumericalError):
            build_boundaries(model, torch.tensor([[0., 0.], [1., 1.]]), ConstantScore(), 1., Config())

    def test_gaussian_score_denoising_objective(self):
        # A point mass smoothed by Gaussian noise has s(x+sigma*e)=-e/sigma.
        class ExactScore(nn.Module):
            def forward(self, x):
                return -(x - 2) / .25
        x = torch.full((4, 3), 2., dtype=torch.float64)
        noise = torch.randn_like(x)
        self.assertLess(float(denoising_loss(ExactScore(), x, .5, noise)), 1e-28)

    def test_ggn_dense_matrix_free_and_objective_gradient(self):
        model, source, labels, _, cfg, b = fixture()
        problem = TsbaProblem(model, source, labels, b, cfg)
        system = GGNSystem(problem)
        matrix = system.dense()
        vector = torch.randn_like(problem.base)
        torch.testing.assert_close(matrix @ vector, system.matvec(vector), atol=1e-12, rtol=1e-11)
        torch.testing.assert_close(-system.rhs, grad(problem.objective)(problem.base), atol=1e-12, rtol=1e-11)
        self.assertGreater(float(torch.linalg.eigvalsh(matrix).min()), 0)
        direct, direct_info = solve(problem)
        problem.cfg = dataclasses.replace(cfg, solver="cg")
        cg, cg_info = solve(problem)
        self.assertTrue(direct_info["valid"])
        self.assertTrue(cg_info["valid"])
        torch.testing.assert_close(direct, cg, atol=1e-8, rtol=1e-8)
        epsilon = 1e-6
        finite_difference = (problem.objective(problem.base + epsilon * vector)
                             - problem.objective(problem.base - epsilon * vector)) / (2 * epsilon)
        torch.testing.assert_close(finite_difference, (-system.rhs) @ vector, atol=1e-8, rtol=1e-8)

    def test_multiclass_nonlinear_response_matches_finite_difference(self):
        torch.manual_seed(43)
        model = ResidualClassifier(2, 3, 3, dtype=torch.float64)
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[1., 0.], [-1., 0.], [0., 1.]]))
            model.linear.bias.copy_(torch.tensor([0., 0., -1.]))
            model.last.weight.normal_(0, .1)  # exercise nonzero hidden-layer parameter responses
        source = torch.tensor([[1., 0.], [-1., 0.], [0., 2.]], dtype=torch.float64)
        labels = torch.tensor([0, 1, 2])
        target = torch.tensor([[.2, .1], [-.2, -.1]], dtype=torch.float64)
        cfg = Config(cg_tolerance=1e-9)
        anchors = build_boundaries(model, target, ConstantScore(), 1., cfg)
        self.assertEqual(len(anchors), 2)
        problem = TsbaProblem(model, source, labels, anchors, cfg)
        system = GGNSystem(problem)
        direction = torch.randn_like(problem.base)
        torch.testing.assert_close(system.dense() @ direction, system.matvec(direction), atol=1e-11, rtol=1e-10)
        epsilon = 1e-6
        fd = (problem.objective(problem.base + epsilon * direction)
              - problem.objective(problem.base - epsilon * direction)) / (2 * epsilon)
        torch.testing.assert_close(fd, (-system.rhs) @ direction, atol=1e-8, rtol=1e-8)
        direct, info = solve(problem)
        problem.cfg = dataclasses.replace(cfg, solver="cg")
        cg, other = solve(problem)
        self.assertTrue(info["valid"] and other["valid"])
        torch.testing.assert_close(direct, cg, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
