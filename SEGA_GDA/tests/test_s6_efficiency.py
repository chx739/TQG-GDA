"""S6: efficiency implementations must reproduce the S2/S4 definitions exactly.

Covers the matrix-vector Taylor quadratic form (vs torch.matrix_exp), the
explicit reentrant chunk recompute (vs plain unrolled execution) and the
optional per-step relation view (default = full fixed memory, bitwise equal).
"""
import copy
import json
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from s4_fixture import make_fixture
from sega_gda.metric import LocalMetricField, taylor_degree
from sega_gda.geodesic import GeodesicSolver, NormalizedGeodesicDistance
from test_s2 import field_fixture, endpoints, scale_references

DTYPE = torch.float64


class S6Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_taylor_degree_is_fixed_by_the_spectral_bound(self):
        self.assertEqual([taylor_degree(h) for h in (0., .25, .5, 1., 2.)], [0, 12, 15, 19, 26])
        with self.assertRaises(ValueError):
            taylor_degree(-1.)
        self.assertEqual(LocalMetricField(torch.zeros(1, 2, dtype=DTYPE), torch.ones(1, dtype=DTYPE), 1.).taylor_degree, 15)
        # The degenerate Euclidean field keeps one exactly-zero term so B stays in the graph.
        self.assertEqual(LocalMetricField(torch.zeros(1, 2, dtype=DTYPE), torch.ones(1, dtype=DTYPE), 0.).taylor_degree, 1)

    def test_quadratic_form_matches_matrix_exponential_values_and_gradients(self):
        for bound in (0., 1., 2.5):
            with self.subTest(bound=bound):
                field = LocalMetricField(torch.randn(3, 4, dtype=DTYPE), torch.rand(3, dtype=DTYPE) + .5, bound)
                field.project_(torch.randn(3, 4, 4, dtype=DTYPE) * (bound + 1.))
                points = torch.randn(6, 5, 3, 4, dtype=DTYPE, requires_grad=True)
                delta = torch.randn(6, 5, 1, 4, dtype=DTYPE, requires_grad=True)
                wide = delta.expand(6, 5, 3, 4)
                exact = torch.einsum('...i,...ij,...j->...', wide, field(points), wide)
                fast = field.quadratic_form(points, delta)
                torch.testing.assert_close(fast, exact, atol=1e-12, rtol=1e-12)
                if bound == 0.:
                    # Exactly the Euclidean form (the single Taylor term is an exact zero).
                    torch.testing.assert_close(fast, wide.square().sum(-1), atol=0, rtol=0)
                weights = torch.randn_like(fast)
                for a, b in zip(torch.autograd.grad((fast * weights).sum(), (points, delta, field.B)),
                                torch.autograd.grad((exact * weights).sum(), (points, delta, field.B))):
                    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)

    def test_path_energy_matches_full_matrix_definition(self):
        field = field_fixture()
        solver = GeodesicSolver(field).double()
        x, y = endpoints()
        _, info = solver(x, y, diagnostics=True)
        path = info['path']
        delta = path[:, 1:] - path[:, :-1]
        points = path[:, :-1, None, :] + solver.nodes[None, None, :, None] * delta[:, :, None, :]
        reference = (path.shape[-2] - 1) * (torch.einsum('mld,mlqde,mle->mlq', delta, field(points), delta)
                                           * solver.weights).sum(dim=(-1, -2))
        torch.testing.assert_close(solver.energy(path), reference, atol=1e-12, rtol=1e-12)

    def test_recomputed_chunks_match_plain_execution_in_both_phases(self):
        x0, y0 = endpoints()
        # Content phase: primitives frozen, endpoints differentiable.
        outputs, gradients = [], []
        for recompute in (False, True):
            field = field_fixture()
            field.B.requires_grad_(False)
            distance = NormalizedGeodesicDistance(GeodesicSolver(field, chunk_size=1, checkpoint_chunks=recompute).double(),
                                                  scale_references())
            x, y = x0.clone().requires_grad_(True), y0.clone().requires_grad_(True)
            values = distance.pairwise(x, y)
            outputs.append(values.detach())
            gradients.append(torch.autograd.grad((values * values.new_tensor([[1., .5], [2., .7]])).sum(), (x, y)))
        torch.testing.assert_close(outputs[1], outputs[0], atol=1e-12, rtol=1e-12)
        for a, b in zip(gradients[1], gradients[0]):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)
        # Space phase: frozen endpoints, differentiable primitives (and scale).
        results = []
        for recompute in (False, True):
            field = field_fixture()
            distance = NormalizedGeodesicDistance(GeodesicSolver(field, chunk_size=1, checkpoint_chunks=recompute).double(),
                                                  scale_references())
            values = distance.pairwise(x0, y0)
            results.append((values.detach(), torch.autograd.grad((values * values.new_tensor([[1., .5], [2., .7]])).sum(),
                                                                 field.B)[0]))
        torch.testing.assert_close(results[1][0], results[0][0], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(results[1][1], results[0][1], atol=1e-10, rtol=1e-8)
        self.assertGreater(float(results[0][1].norm()), 1e-6)

    def test_relation_view_option_default_is_full_memory_and_bitwise_unchanged(self):
        config = json.loads((ROOT / 'configs/s4_evolution_debug.json').read_text())
        plain = make_fixture(copy.deepcopy(config))
        explicit = copy.deepcopy(config)
        explicit['relation_triplets_per_step'] = None
        with_null = make_fixture(explicit)
        self.assertEqual(plain.run_round(), with_null.run_round())
        for k, v in plain.model.state_dict().items():
            self.assertTrue(torch.equal(v, with_null.model.state_dict()[k]), k)
        full = [len(plain.source_memory.triplets), len(plain.target_memory.triplets)]
        self.assertEqual(plain.run_round()['content'][0]['relation_triplets'], full)
        self.assertGreaterEqual(min(full), 1)
        budgeted = copy.deepcopy(config)
        budgeted['relation_triplets_per_step'] = 1
        engine = make_fixture(budgeted)
        row = engine.run_round()
        self.assertEqual(row['content'][0]['relation_triplets'], [min(1, n) for n in full])
        self.assertEqual(row['space']['relation_triplets'], [min(1, n) for n in full])
        self.assertEqual(engine.content_step_count, config['content_steps'])
        with self.assertRaises(ValueError):
            bad = copy.deepcopy(config)
            bad['relation_triplets_per_step'] = 0
            make_fixture(bad)


if __name__ == '__main__':
    unittest.main()
