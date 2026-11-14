"""Focused S2 mathematics tests. No target data or adaptation training."""
import copy
import math
import unittest
import torch
from torch.func import functional_call
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sega_gda.metric import LocalMetricField, project_log_metrics, fit_fixed_coverage
from sega_gda.geodesic import GeodesicSolver, NormalizedGeodesicDistance, symmetric_quadrature
from sega_gda.geometry_head import RiemannianClassifier
from sega_gda.models import SourceClassifier
from sega_gda.initialization import InitializedClassifier


DTYPE = torch.float64
RESULTS = {}


def field_fixture(seed=3):
    torch.manual_seed(seed)
    field = LocalMetricField(torch.tensor([[-.45, .1], [.7, -.2]], dtype=DTYPE),
                             torch.tensor([.65, .9], dtype=DTYPE), 1.)
    field.project_(torch.tensor([[[.45, .12], [.12, -.45]],
                                [[-.22, .28], [.28, .22]]], dtype=DTYPE))
    return field


def endpoints():
    return (torch.tensor([[-1., .25], [.1, -.4]], dtype=DTYPE),
            torch.tensor([[.9, .4], [.75, .85]], dtype=DTYPE))


def scale_references():
    return torch.tensor([[[-.8, -.25], [.3, .5]], [[-.1, -.55], [.7, .2]],
                         [[-.4, .45], [.8, -.3]]], dtype=DTYPE)


class S2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_metric_background_anisotropy_and_constraints(self):
        field = field_fixture()
        points = torch.cat((field.centers, torch.randn(12, 2, dtype=DTYPE),
                            torch.tensor([[100., 100.]], dtype=DTYPE)))
        field.validate()
        metric = field(points)
        values = torch.linalg.eigvalsh(metric)
        trace_error = field.B.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max().item()
        logdet_error = torch.linalg.slogdet(metric)[1].abs().max().item()
        self.assertLessEqual(logdet_error, 1e-9)
        self.assertLessEqual(trace_error, 1e-9)
        self.assertGreater(values.min().item(), 0.)
        self.assertGreaterEqual(values.min().item(), math.exp(-1)-1e-9)
        self.assertLessEqual(values.max().item(), math.exp(1)+1e-9)
        self.assertLess(float((metric-metric.transpose(-1, -2)).abs().max()), 1e-9)
        torch.testing.assert_close(field.weights(points).sum(-1), torch.ones(len(points), dtype=DTYPE),
                                   atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(metric[-1], torch.eye(2, dtype=DTYPE), atol=1e-12, rtol=0)
        self.assertGreater(float((metric[0]-metric[1]).norm()), .05)
        self.assertGreater(float((field.B[0]@field.B[1]-field.B[1]@field.B[0]).norm()), .01)
        self.assertFalse(field.centers.requires_grad)
        self.assertFalse(field.bandwidths.requires_grad)
        RESULTS['metric'] = dict(max_logdet_error=logdet_error, max_B_trace_error=trace_error,
                                  min_eigenvalue=values.min().item(), max_eigenvalue=values.max().item(),
                                  regional_difference=float((metric[0]-metric[1]).norm()))

    def test_projection_spectral_trace_and_idempotence(self):
        candidate = torch.randn(4, 3, 3, dtype=DTYPE)*20
        p = project_log_metrics(candidate, .6)
        self.assertLess(float((p-p.transpose(-1, -2)).abs().max()), 1e-9)
        self.assertLess(float(p.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max()), 1e-9)
        self.assertLessEqual(float(torch.linalg.eigvalsh(p).abs().max()), .6+1e-9)
        torch.testing.assert_close(project_log_metrics(p, .6), p, atol=1e-9, rtol=1e-9)
        exact = torch.diag(torch.tensor([1., -1.], dtype=DTYPE))
        torch.testing.assert_close(project_log_metrics(torch.diag(torch.tensor([3., -.1], dtype=DTYPE)), 1.),
                                   exact, atol=1e-9, rtol=1e-9)
        self.assertFalse(p.requires_grad)

    def test_fixed_equal_domain_coverage_and_bandwidth_floor(self):
        source = torch.tensor([[0., 0.], [2., 0.]], dtype=DTYPE, requires_grad=True)
        target = torch.tensor([[10., 0.]]*8, dtype=DTYPE, requires_grad=True)
        centers, h, info = fit_fixed_coverage(source, target, 1, .1, seed=4)
        torch.testing.assert_close(centers[0], torch.tensor([5.5, 0.], dtype=DTYPE))
        self.assertFalse(centers.requires_grad)
        self.assertFalse(h.requires_grad)
        self.assertEqual((info['source_weight'], info['target_weight']), (.5, .5))
        c, h, info = fit_fixed_coverage(source[:1].detach()*0, target[:2].detach()*0, 2, .2)
        torch.testing.assert_close(h, torch.full_like(h, .2))
        self.assertEqual(len(info['empty_clusters']), 1)

    def test_quadrature_and_identity_path_all_budgets(self):
        x, y = endpoints()
        identity_error = 0.
        for q in (1, 3, 5):
            nodes, weights = symmetric_quadrature(q)
            self.assertTrue(bool((weights > 0).all()))
            torch.testing.assert_close(nodes, 1-nodes.flip(0), atol=1e-14, rtol=0)
            torch.testing.assert_close(weights.sum(), torch.tensor(1., dtype=DTYPE))
            for segments in (1, 4):
                f = LocalMetricField(torch.zeros(1, 2, dtype=DTYPE), torch.ones(1, dtype=DTYPE))
                solver = GeodesicSolver(f, segments=segments, quadrature=q).double()
                energy = solver(x, y)
                reference = (x-y).square().sum(-1)
                torch.testing.assert_close(energy, reference, atol=1e-10, rtol=1e-8)
                identity_error = max(identity_error, float((energy-reference).abs().max()))
        RESULTS['identity'] = dict(max_energy_error=identity_error)

    def test_path_relaxation_symmetry_same_point_and_no_grad(self):
        field = field_fixture()
        solver = GeodesicSolver(field, chunk_size=1).double()
        x, y = endpoints()
        energy, info = solver(x, y, diagnostics=True)
        self.assertTrue(bool(torch.isfinite(energy).all()))
        self.assertTrue(bool(torch.isfinite(info['stationarity']).all()))
        self.assertLessEqual(float((info['energy_history'][1:]-info['energy_history'][:-1]).max()), 1e-10)
        reverse, reversed_info = solver(y, x, diagnostics=True)
        torch.testing.assert_close(energy, reverse, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(info['path'], reversed_info['path'].flip(-2), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(solver(x, x), torch.zeros(len(x), dtype=DTYPE), atol=1e-12, rtol=0)
        with torch.no_grad():
            evaluation = solver(x, y)
        self.assertFalse(evaluation.requires_grad)
        torch.testing.assert_close(evaluation, energy, atol=1e-10, rtol=1e-8)
        RESULTS['relaxation'] = dict(initial_energy=info['energy_history'][0].tolist(),
                                      final_energy=energy.detach().tolist(),
                                      final_path_gradient_norm=info['stationarity'].tolist(),
                                      symmetry_max_error=float((energy-reverse).abs().max()))

    def test_actual_rbf_full_unroll_gradcheck(self):
        field = field_fixture()
        solver = GeodesicSolver(field).double()
        x, y = (v.clone().requires_grad_(True) for v in endpoints())
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: solver(a, b), (x, y),
                         eps=1e-6, atol=1e-5, rtol=1e-3))
        base = field.B.detach().clone()
        directions = torch.randn(2, *base.shape, dtype=DTYPE)
        directions = (directions+directions.transpose(-1, -2))/2
        directions -= torch.eye(2, dtype=DTYPE)*directions.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]/2
        directions /= directions.flatten(1).norm(dim=1)[:, None, None, None]
        def coefficients(t):
            basis = base+torch.einsum('a,akij->kij', t, directions)
            return functional_call(solver, {'metric.B': basis}, (x.detach(), y.detach()))
        self.assertTrue(torch.autograd.gradcheck(coefficients, (torch.zeros(2, dtype=DTYPE, requires_grad=True),),
                         eps=1e-6, atol=1e-5, rtol=1e-3))
        RESULTS['gradcheck'] = dict(actual_rbf_endpoints=True, actual_rbf_feasible_B_directions=True,
                                      L=4, Q=3, J=8, eps=1e-6, atol=1e-5, rtol=1e-3)

    def test_finite_difference_scale_and_normalized_denominator(self):
        field = field_fixture()
        solver = GeodesicSolver(field).double()
        fixed = scale_references().requires_grad_(True)
        normalized = NormalizedGeodesicDistance(solver, fixed)
        x, y = endpoints()
        scale = normalized.scale2()
        value = normalized(x, y).sum()
        gs, gv = (torch.autograd.grad(v, field.B)[0] for v in [scale, value])
        direction = torch.tensor([[[.3, .4], [.4, -.3]], [[-.2, .1], [.1, .2]]], dtype=DTYPE)
        direction /= direction.norm()
        base = field.B.detach().clone()
        expected = [float((g*direction).sum()) for g in [gs, gv]]
        differences = []
        for h in (1e-4, 1e-5, 1e-6):
            outputs = []
            for sign in (-1, 1):
                with torch.no_grad():
                    field.B.copy_(base+sign*h*direction)
                    outputs.append((float(normalized.scale2()), float(normalized(x, y).sum())))
            fd = [(outputs[1][j]-outputs[0][j])/(2*h) for j in range(2)]
            errors = [abs(a-b) for a, b in zip(expected, fd)]
            for a, b in zip(expected, fd):
                self.assertLessEqual(abs(a-b), 1e-5+1e-3*abs(b))
            differences.append(dict(h=h, finite_difference=fd, absolute_errors=errors))
        with torch.no_grad():
            field.B.copy_(base)
        denominator = normalized.scale2()
        numerator = solver(x, y)
        wrong = torch.autograd.grad((numerator/denominator.detach()).sum(), field.B)[0]
        gap = abs(float(((wrong-gv)*direction).sum()))
        self.assertGreater(gap, 1e-3)
        self.assertIsNone(fixed.grad)
        self.assertFalse(normalized.reference_pairs.requires_grad)
        before = normalized.reference_pairs.clone()
        with torch.no_grad():
            fixed.add_(100)
        torch.testing.assert_close(normalized.reference_pairs, before, atol=0, rtol=0)
        old_scale = float(normalized.scale2().detach())
        field.project_(base+.05*direction)
        self.assertGreater(abs(float(normalized.scale2().detach())-old_scale), 1e-6)
        RESULTS['scale_gradient'] = dict(autograd_directional=expected, checks=differences,
                                          detached_denominator_error=gap, reference_coordinates_fixed=True)

    def test_detaching_internal_path_is_detected(self):
        field = field_fixture()
        solver = GeodesicSolver(field).double()
        x, y = endpoints()
        exact, info = solver(x, y, diagnostics=True)
        correct = torch.autograd.grad(exact.sum(), field.B)[0]
        wrong = torch.autograd.grad(solver.energy(info['path']).sum(), field.B)[0]
        gap = float((correct-wrong).norm())
        self.assertGreater(gap, 1e-5)
        RESULTS['negative_internal_detach'] = dict(missing_gradient_norm=gap, detected=True)

    def test_position_derivative_when_B_is_frozen(self):
        field = field_fixture()
        field.B.requires_grad_(False)
        solver = GeodesicSolver(field).double()
        x, y = endpoints()
        x = x.requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(lambda a: solver(a, y), (x,),
                         eps=1e-6, atol=1e-5, rtol=1e-3))
        class DetachedPositionField(LocalMetricField):
            # Detach positions at the shared entry of forward() and quadratic_form().
            def log_metric(self, points, B=None):
                return super().log_metric(points.detach(), B)
        wrong_field = DetachedPositionField(field.centers, field.bandwidths)
        wrong_field.project_(field.B.detach())
        wrong_field.B.requires_grad_(False)
        wrong_solver = GeodesicSolver(wrong_field).double()
        direction = torch.tensor([[.5, -.2], [.4, .1]], dtype=DTYPE)
        automatic = float((torch.autograd.grad(wrong_solver(x, y).sum(), x)[0]*direction).sum())
        with torch.no_grad():
            fd = float((wrong_solver(x+1e-5*direction, y).sum()-wrong_solver(x-1e-5*direction, y).sum())/(2e-5))
        self.assertGreater(abs(automatic-fd), 1e-3)
        RESULTS['negative_position_detach'] = dict(directional_error=abs(automatic-fd), detected=True)

    def test_chunking_checkpoint_and_shared_scale_gradients(self):
        field = field_fixture()
        x0, y0 = endpoints()
        outputs, gradients = [], []
        for chunk, recompute in [(8, False), (1, False), (1, True)]:
            metric = copy.deepcopy(field)
            distance = NormalizedGeodesicDistance(GeodesicSolver(metric, chunk_size=chunk,
                                                         checkpoint_chunks=recompute).double(), scale_references())
            x, y = x0.clone().requires_grad_(True), y0.clone().requires_grad_(True)
            values = distance.pairwise(x, y)
            outputs.append(values.detach())
            gradients.append(torch.autograd.grad((values*values.new_tensor([[1., .5], [2., .7]])).sum(),
                                                 (x, y, metric.B)))
        for output, gradient in zip(outputs[1:], gradients[1:]):
            torch.testing.assert_close(output, outputs[0], atol=1e-10, rtol=1e-8)
            for a, b in zip(gradient, gradients[0]):
                torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-6)
        RESULTS['chunking'] = dict(plain_vs_chunk_max_error=float((outputs[0]-outputs[1]).abs().max()),
                                    checkpoint_max_gradient_error=max(float((a-b).abs().max())
                                                                     for a, b in zip(gradients[0], gradients[2])))

    def test_noncommuting_matrix_exponential_and_zero_initialization(self):
        a = torch.diag(torch.tensor([.4, -.4], dtype=DTYPE)).requires_grad_(True)
        h = torch.tensor([[0., .7], [.7, 0.]], dtype=DTYPE)
        _, tangent = torch.autograd.functional.jvp(torch.matrix_exp, a, h)
        block = torch.cat((torch.cat((a.detach(), h), 1),
                           torch.cat((torch.zeros_like(a), a.detach()), 1)), 0)
        reference = torch.matrix_exp(block)[:2, 2:]
        torch.testing.assert_close(tangent, reference, atol=1e-10, rtol=1e-8)
        self.assertGreater(float((tangent-torch.matrix_exp(a)@h).norm()), .01)
        self.assertTrue(torch.autograd.gradgradcheck(torch.matrix_exp, (a,), eps=1e-6, atol=1e-5, rtol=1e-3))
        metric = LocalMetricField(torch.zeros(1, 2, dtype=DTYPE), torch.ones(1, dtype=DTYPE))
        distance = NormalizedGeodesicDistance(GeodesicSolver(metric).double(),
                            torch.tensor([[[0., 0.], [1., 1.]]], dtype=DTYPE))
        u = torch.zeros(1, 2, dtype=DTYPE)
        q = torch.eye(2, dtype=DTYPE)
        loss = -(-distance.pairwise(u, q)).log_softmax(-1)[0, 0]
        g = torch.autograd.grad(loss, metric.B)[0]
        h = torch.diag(torch.tensor([1., -1.], dtype=DTYPE))[None]
        self.assertGreater(abs(float((g*h).sum())), .01)
        metric.project_(-.001*g)
        new_loss = -(-distance.pairwise(u, q)).log_softmax(-1)[0, 0]
        self.assertLess(float(new_loss), float(loss))

    def test_actual_axis_rbf_against_independent_continuous_distance(self):
        metric = LocalMetricField(torch.zeros(1, 2, dtype=DTYPE), torch.tensor([.8], dtype=DTYPE))
        metric.project_(torch.diag(torch.tensor([-.6, .6], dtype=DTYPE))[None])
        solver = GeodesicSolver(metric).double()
        x, y = torch.tensor([[-1., 0.]], dtype=DTYPE), torch.tensor([[1., 0.]], dtype=DTYPE)
        energy = float(solver(x, y).detach())
        references = []
        for count in (4096, 8192):
            t = torch.linspace(-1., 1., count+1, dtype=DTYPE)
            r = torch.exp(-t.square()/(2*.8**2))
            f = torch.exp(-.3*r/(1+r))
            integral = (2/count)/3*(f[0]+f[-1]+4*f[1:-1:2].sum()+2*f[2:-1:2].sum())
            references.append(float(integral.square()))
        self.assertLess(abs(references[1]-references[0]), 1e-10)
        relative = abs(energy-references[1])/references[1]
        self.assertLessEqual(relative, 1e-3)
        RESULTS['axis_rbf_continuous_reference'] = dict(finite_energy=energy,
                      independent_distance_squared=references[1], relative_error=relative,
                      reference_refinement_error=abs(references[1]-references[0]),
                      predeclared_relative_tolerance=1e-3)

    def test_nonfinite_inputs_and_invalid_scales_fail_explicitly(self):
        metric = field_fixture()
        solver = GeodesicSolver(metric).double()
        x, y = endpoints()
        x[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            solver(x, y)
        with self.assertRaises(ValueError):
            LocalMetricField(metric.centers, metric.bandwidths*0)
        with self.assertRaises(ValueError):
            NormalizedGeodesicDistance(solver, torch.zeros(1, 2, 2, dtype=DTYPE))
        with torch.no_grad():
            metric.B[0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            solver(*endpoints())

    def test_classifier_uses_trainable_path_interface_without_moving_coordinates(self):
        graphs = [Data(x=torch.randn(3, 2, dtype=DTYPE),
                       edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), y=torch.tensor([i]))
                  for i in range(2)]
        batch = next(iter(DataLoader(graphs, batch_size=2)))
        source = SourceClassifier(dict(input_dim=2, hidden_dim=4, num_layers=2,
                                       edge_dim=0, readout='mean'), 2).double()
        initial = InitializedClassifier(source, torch.tensor(2., dtype=DTYPE), 1.)
        references = torch.tensor([[[0., 0., 0., 0.], [1., 1., 0., 0.]]], dtype=DTYPE)
        metric = LocalMetricField(torch.zeros(2, 4, dtype=DTYPE), torch.ones(2, dtype=DTYPE))
        model = RiemannianClassifier.from_s1(initial, references, metric, chunk_size=1)
        before = initial.encoder(batch)
        torch.testing.assert_close(model.encoder(batch), before, atol=0, rtol=0)
        torch.testing.assert_close(model(batch).softmax(-1), source(batch).softmax(-1), atol=1e-10, rtol=1e-8)
        metric.project_(torch.randn_like(metric.B)*.1)
        loss = torch.nn.functional.cross_entropy(model(batch), batch.y)
        loss.backward()
        for parameter in [model.q, model.beta, metric.B, *model.encoder.parameters()]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertGreater(float(metric.B.grad.norm()), 0.)


if __name__ == '__main__':
    unittest.main()
