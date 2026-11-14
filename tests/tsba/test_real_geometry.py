"""Synthetic checks of registered geometry/density observations (no real data)."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from tsba.boundary import project
from tsba.config import Config
from tsba.classifier import ResidualClassifier
from workflows.tsba.validation.real_data.density import (
    GaussianKDE, bootstrap_counts, derived_seed, finite_support, valley_observation,
)
from workflows.tsba.validation.real_data.geometry import (
    analytic_projection, cluster_anchors, coverage, fixed_query_ids, observed_project,
    reference_consensus, scan_reference,
    run_cell, run_source_coverage,
)


ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "experiments/mutagenicity/r1-20261111/stage1-plan.json") as handle:
    PLAN = json.load(handle)


class MarginModel(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        margin = self.function(x)
        return torch.stack((margin / 2, -margin / 2), -1)


class RealGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_production_projection_is_unchanged_and_analytic_failure_is_separate(self):
        cfg = Config()
        model = MarginModel(lambda x: 2 * x[:, 0] + x[:, 1] - .5)
        x = torch.tensor([.2, .1], dtype=torch.float64)
        expected, failure = project(model, x, (0, 1), 4., 1e-5, cfg)
        result, actual_failure, counts = observed_project(model, x, 1., cfg)
        self.assertEqual(actual_failure, failure)
        for actual, want in zip(result, expected):
            torch.testing.assert_close(actual, want, rtol=0, atol=0)
        self.assertGreaterEqual(counts["iterations"], 1)
        rows = coverage(model, torch.stack((x, x + 50)), [2, 3], 1., cfg, True)
        self.assertTrue(rows[0]["success"])
        self.assertLess(rows[0]["analytic"]["projection_error_sigma"], 1e-10)
        self.assertFalse(rows[1]["success"])
        self.assertIsNone(rows[1]["point"])
        self.assertEqual(rows[1]["analytic"]["status"], "outside_radius")
        edge = analytic_projection(torch.tensor([4., 0.], dtype=torch.float64),
                                   torch.tensor([1., 0.], dtype=torch.float64), 0., 1.)
        self.assertEqual(edge["status"], "radius_edge")

    def test_standardized_reference_chain_rule_and_ray_intersection(self):
        mean = torch.tensor([.3, -.4], dtype=torch.float64)
        scale = torch.tensor([2., .5], dtype=torch.float64)
        weight = torch.tensor([3., -2.], dtype=torch.float64)
        model = MarginModel(lambda x: ((x - mean) / scale) @ weight - .2)
        point = torch.zeros(2, dtype=torch.float64)
        normal = torch.tensor([1., 0.], dtype=torch.float64)
        observation = scan_reference(model, point, normal, 1., PLAN, True)
        expected = float((mean / scale) @ weight + .2) / float(weight[0] / scale[0])
        self.assertEqual(observation["status"], "detected_unique")
        self.assertAlmostEqual(observation["t"], expected, places=7)
        self.assertLess(observation["analytic_error_sigma"], 1e-7)

    def test_scan_preserves_no_root_multiple_touch_parallel_and_coincident(self):
        point, normal = torch.zeros(2, dtype=torch.float64), torch.tensor([1., 0.], dtype=torch.float64)
        cases = [
            (lambda x: x[:, 0] ** 2 + 1, False, "no_detected_root"),
            (lambda x: x[:, 0] ** 2 - 1, False, "multiple_detected_roots"),
            (lambda x: x[:, 0] ** 2, False, "degenerate_normal"),
            (lambda x: x[:, 1] + 1, True, "parallel"),
            (lambda x: x[:, 1], True, "line_near_coincident"),
            (lambda x: x[:, 0] - .01, True, "near_coincident"),
        ]
        for function, linear, expected in cases:
            with self.subTest(expected=expected):
                result = scan_reference(MarginModel(function), point, normal, 1., PLAN, linear)
                self.assertEqual(result["status"], expected)

    def test_fixed_queries_clusters_and_consensus_have_fixed_denominators(self):
        query = fixed_query_ids(list(range(100)), "protocol", 1)
        self.assertEqual(len(query), 64)
        self.assertEqual(query, fixed_query_ids(list(reversed(range(100))), "protocol", 1))
        anchors = [{"origin_id": i, "success": i != 4, "point": torch.tensor([v], dtype=torch.float64)}
                   for i, v in [(3, .05), (1, 0), (2, .5), (4, 0)]]
        result = cluster_anchors(anchors, 1.)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["members"][1], [1, 3])
        roots = [{"architecture": "linear" if i < 4 else "hidden32_silu_mlp",
                  "status": "detected_unique", "t": 1. + .01 * i} for i in range(8)]
        self.assertEqual(reference_consensus(roots, 1., PLAN)["status"], "consensus")
        for root in roots[4:]:
            root["t"] *= -1
        self.assertTrue(reference_consensus(roots, 1., PLAN)["architecture_conflict"])
        self.assertEqual(reference_consensus(roots[:4], 1., PLAN)["denominator"], 8)
        self.assertEqual(reference_consensus(roots[:4], 1., PLAN)["status"], "reference_uncertain")

    def test_phase_guards_reject_postseal_geometry_without_reading_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source/seed3405").mkdir(parents=True)
            (root / "inputs").mkdir()
            (root / "seals").mkdir()
            generator = torch.Generator().manual_seed(391)
            x = torch.randn((128, 16), dtype=torch.float64, generator=generator)
            source = ResidualClassifier(16, 2, 16, dtype=torch.float64)
            with torch.no_grad():
                source.linear.weight.zero_()
                source.linear.bias.zero_()
                source.linear.weight[0, 0] = 1.
            torch.save(dict(x=x, initial_head=source.state_dict()), root / "source/seed3405/frozen.pt")
            torch.save(torch.arange(256, dtype=torch.float64).reshape(128, 2), root / "inputs/size.pt")
            roles = {"targets": {"1": {"ids": list(range(128)), "A": list(range(64)),
                      "B": list(range(64, 128)), "final": []}}}
            (root / "roles.json").write_text(json.dumps(roles))
            with self.assertRaises(PermissionError):
                run_cell(root, 3405, 1, PLAN)
            run_source_coverage(root, 3405, 1, PLAN)
            coverage_file = root / "geometry/seed3405_domain1/source_coverage.pt"
            sealed_bytes = coverage_file.read_bytes()
            (root / "seals/S1.json").write_text("{}")
            with self.assertRaises(PermissionError):
                run_source_coverage(root, 3405, 1, PLAN)
            self.assertEqual(coverage_file.read_bytes(), sealed_bytes)
            (root / "seals/S2.json").write_text("{}")
            with self.assertRaises(PermissionError):
                run_cell(root, 3405, 1, PLAN)


class RealDensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_kde_derivative_matches_autograd_and_finite_difference(self):
        x = torch.tensor([[-1., .2], [.3, -.4], [2., 1.]], dtype=torch.float64)
        point = torch.tensor([[.2, .1]], dtype=torch.float64, requires_grad=True)
        normal = torch.tensor([[.6, .8]], dtype=torch.float64)
        kde = GaussianKDE(x, .7)
        result = kde.evaluate(point, normals=normal)
        gradient, = torch.autograd.grad(result["logp"].sum(), point)
        torch.testing.assert_close(result["g"][0], (gradient * normal).sum(-1), rtol=1e-12, atol=1e-12)
        epsilon = 1e-5
        lp = kde.evaluate(torch.cat((point + epsilon * normal, point - epsilon * normal)))["logp"][0]
        self.assertAlmostEqual(float((lp[0] - lp[1]) / (2 * epsilon)), float(result["g"][0, 0]), places=8)

    def test_bootstrap_stable_when_near_kernel_has_zero_count_and_loo_renormalizes(self):
        kde = GaussianKDE(torch.tensor([[0.], [1.], [2.]], dtype=torch.float64), 1.)
        log_kernel = torch.tensor([[-1e6, -1e6 - 1000, -1e6 - 1002]], dtype=torch.float64)
        counts = torch.tensor([[0., 1., 2.]], dtype=torch.float64)
        result = kde.from_log_kernel(log_kernel, counts)
        expected = torch.logsumexp(log_kernel[0, 1:] + counts[0, 1:].log(), 0) - counts.sum().log() - kde.log_normalizer
        self.assertAlmostEqual(float(result["logp"][0, 0]), float(expected), places=9)
        points = torch.tensor([[0.]], dtype=torch.float64)
        actual = kde.evaluate(points, drop_index=0)
        reference = GaussianKDE(kde.x[1:], 1.).evaluate(points)
        torch.testing.assert_close(actual["logp"], reference["logp"])
        torch.testing.assert_close(actual["ess"], reference["ess"])
        equal = GaussianKDE(torch.zeros((8, 1), dtype=torch.float64), 1.).evaluate(points)
        self.assertAlmostEqual(float(equal["ess"][0, 0]), 8.)

    def test_shared_counts_rng_and_explicit_bootstrap_recomputation(self):
        seed = derived_seed("protocol", "density_bootstrap", 3405, 1)
        counts = bootstrap_counts(12, 200, seed)
        torch.testing.assert_close(counts, bootstrap_counts(12, 200, seed), rtol=0, atol=0)
        self.assertTrue(bool((counts.sum(-1) == 12).all()))
        x = torch.linspace(-2, 2, 12, dtype=torch.float64)[:, None]
        point, normal = torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)
        observation = valley_observation(GaussianKDE(x, .8), point, normal, .8, 1., counts)
        resample = x[torch.arange(len(x)).repeat_interleave(counts[0].long())]
        expected = GaussianKDE(resample, .8).evaluate(torch.tensor([[0.], [.8], [-.8]], dtype=torch.float64))["logp"][0]
        self.assertAlmostEqual(float(observation["V_bootstrap"][0]), float(expected[1:].min() - expected[0]), places=10)

    def test_valley_uses_minimum_and_finite_tubes_exclude_far_points_and_origin(self):
        x = torch.tensor([[1., 0.], [-1., 0.], [1., 100.], [-1., 100.], [.5, 0.], [1.5, 0.], [-.5, 0.], [-1.5, 0.]], dtype=torch.float64)
        support = finite_support(x, list(range(8)), 0, torch.zeros(2, dtype=torch.float64),
                                 torch.tensor([1., 0.], dtype=torch.float64), 1., 1., [1, 2, 4, 6], 1, 1)
        self.assertEqual(support["positive_ids"], [4])
        self.assertEqual(support["negative_ids"], [1, 6])
        self.assertTrue(support["unlabeled_supported"])
        self.assertTrue(support["B_supported"])
        generator = torch.Generator().manual_seed(37)
        points = torch.cat((torch.randn((100, 1), generator=generator, dtype=torch.float64) * .1 + 1.,
                            torch.randn((100, 1), generator=generator, dtype=torch.float64) * .1 - 1.))
        observation = valley_observation(GaussianKDE(points, .2), torch.zeros(1, dtype=torch.float64),
                                        torch.ones(1, dtype=torch.float64), 1., 1., bootstrap_counts(200, 200, 37))
        self.assertEqual(observation["valley_state"], "valley_supported")
        self.assertAlmostEqual(observation["V"], float(observation["logp"][1:].min() - observation["logp"][0]))


if __name__ == "__main__":
    unittest.main()
