"""Focused S3 tests: correspondence, fixed-ID memory, and actual S2 gradients."""
import hashlib
import io
import json
import unittest
from dataclasses import replace
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.optimize import minimize

from sega_gda.transport import (class_balanced_masses, sample_source_references, uniform_target_masses,
    generalized_kl, ot_objective, solve_semi_relaxed, fixed_plan_objective, OTTolerances, OTConvergenceError)
from sega_gda.relations import (RelationRules, RelationMemory, build_source_memory, build_target_memory,
    encode_graph_ids, relation_loss)
from sega_gda.objectives import alignment_loss, learning_objective
from sega_gda.models import GraphEncoder
from sega_gda.geodesic import GeodesicSolver, NormalizedGeodesicDistance
from test_s2 import field_fixture, scale_references, endpoints


DTYPE = torch.float64
RESULTS = {}
RULES = RelationRules(2, 1, 3, 1000000, 2, 2, .2, .01)


def graph_fixture():
    torch.manual_seed(3)
    encoder = GraphEncoder(2, 2, 1).double()
    # Positive activations make the gradient fixture independent of dead ReLUs.
    with torch.no_grad():
        for name, p in encoder.named_parameters():
            if 'bias' in name:
                p.fill_(.15)
            elif p.ndim == 2:
                p.copy_(torch.tensor([[.6, .1], [.1, .4]], dtype=DTYPE))
    graphs, labels = {}, {}
    for domain, count in [('s', 6), ('t', 5)]:
        for i in range(count):
            gid = f'{domain}:{i}'
            x = torch.tensor([[.25+i*.12, .3+(i % 2)*.5], [.4+i*.08, .6],
                              [.5+(i % 3)*.2, .2+i*.07]], dtype=DTYPE)
            if domain == 't':
                x = x+torch.tensor([.15, -.05], dtype=DTYPE)
            graphs[gid] = Data(x=x, edge_index=torch.tensor([[0,1,1,2,2,0],[1,0,2,1,0,2]]))
            if domain == 's':
                labels[gid] = 0 if i < 4 else 1
    return graphs, labels, encoder


def encoder_hash(encoder):
    stream = io.BytesIO()
    # Fingerprint tensor values and names, independent of serialization container.
    for name, value in sorted(encoder.state_dict().items()):
        stream.write(name.encode())
        stream.write(value.detach().cpu().numpy().tobytes())
    return hashlib.sha256(stream.getvalue()).hexdigest()


def distance_fixture(identity=False, chunk_size=8):
    metric = field_fixture()
    if identity:
        with torch.no_grad():
            metric.B.zero_()
    return NormalizedGeodesicDistance(GeodesicSolver(metric, chunk_size=chunk_size), scale_references())


class S3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_reference_masses_and_sampling(self):
        labels = {'s:0':0, 's:1':0, 's:2':0, 's:3':1}
        ids, ys = sample_source_references(labels, 2, 23)
        self.assertEqual((ids, ys), sample_source_references(labels, 2, 23))
        self.assertEqual(len(set(ids)), 3)
        a = class_balanced_masses(ids, list(ys), [0,1])
        self.assertAlmostEqual(float(a[torch.tensor(ys)==0].sum()), .5)
        self.assertAlmostEqual(float(a[torch.tensor(ys)==1].sum()), .5)
        torch.testing.assert_close(uniform_target_masses(3), torch.full((3,), 1/3, dtype=DTYPE))
        with self.assertRaises(ValueError):
            class_balanced_masses(['s:0', 's:0'], [0,1], [0,1])
        with self.assertRaises(ValueError):
            class_balanced_masses(['s:0'], [0], [0,1])
        self.assertAlmostEqual(float(generalized_kl(torch.tensor([0.,1.]), torch.tensor([.5,.5]))), np.log(2), places=6)

    def test_rho_zero_closed_form_and_extreme_log_costs(self):
        a = torch.tensor([.25,.25,.5], dtype=DTYPE)
        b = uniform_target_masses(2)
        costs = [torch.tensor([[.2,1.1],[1.4,.4],[2.3,.1]], dtype=DTYPE),
                 torch.tensor([[0.,1e6],[1e6,0.],[2e6,2e6]], dtype=DTYPE)]
        records = []
        for c in costs:
            result = solve_semi_relaxed(c.requires_grad_(), a, b, .3, 0.)
            analytic = b * torch.softmax(a.log()[:,None]-c.detach()/.3, dim=0)
            torch.testing.assert_close(result.pi, analytic, atol=1e-10, rtol=1e-8)
            self.assertFalse(result.pi.requires_grad)
            self.assertIsNone(result.pi.grad_fn)
            torch.testing.assert_close(result.pi.sum(0), b, atol=1e-10, rtol=0)
            self.assertTrue(bool(torch.isfinite(result.pi).all()))
            self.assertGreater(float((result.pi.sum(1)-a).abs().sum()), .01)
            if result.diagnostics['underflow_count']:
                self.assertGreater(result.diagnostics['underflow_mass_upper_bound'], 0.)
                self.assertLess(result.diagnostics['log_underflow_mass'], -700)
            records.append(result.diagnostics)
        RESULTS['rho_zero'] = records

    def test_finite_rho_independent_constrained_reference(self):
        c = torch.tensor([[.1,1.2,.5],[1.1,.3,1.8],[2.,1.,.2]], dtype=DTYPE)
        a, b = torch.tensor([.25,.25,.5], dtype=DTYPE), uniform_target_masses(3)
        eps, rho = .4, 1.7
        result = solve_semi_relaxed(c, a, b, eps, rho)
        av, bv, cv = a.numpy(), b.numpy(), c.numpy()
        def objective(v):
            p = v.reshape(cv.shape)
            m = p.sum(1)
            value = (p*cv).sum()+eps*(p*np.log(p/(av[:,None]*bv))-p+av[:,None]*bv).sum()
            value += rho*(m*np.log(m/av)-m+av).sum()
            grad = cv+eps*np.log(p/(av[:,None]*bv))+rho*np.log(m/av)[:,None]
            return value, grad.ravel()
        constraints = dict(type='eq', fun=lambda v: v.reshape(cv.shape).sum(0)-bv,
                           jac=lambda v: np.tile(np.eye(3), (1,3)))
        ref = minimize(objective, (av[:,None]*bv).ravel(), method='SLSQP', jac=True,
                       bounds=[(1e-14,None)]*9, constraints=constraints,
                       options=dict(ftol=1e-13, maxiter=2000))
        self.assertTrue(ref.success, ref.message)
        actual = float(ot_objective(result.pi, c, a, b, eps, rho)['total'])
        self.assertLessEqual(abs(actual-ref.fun), 1e-8+1e-7*abs(ref.fun))
        torch.testing.assert_close(result.pi, torch.from_numpy(ref.x.reshape(3,3)), atol=2e-7, rtol=2e-6)
        self.assertGreater(result.diagnostics['source_row_l1'], .01)
        self.assertLess(result.diagnostics['column_abs'], 1e-10)
        self.assertLess(result.diagnostics['stationarity_abs'], 1e-7)
        RESULTS['finite_rho'] = dict(diagnostics=result.diagnostics, iterations=result.iterations,
                                    independent_method='SciPy SLSQP direct primal with equality columns',
                                    reference_objective=float(ref.fun), objective_error=abs(actual-ref.fun),
                                    plan_max_error=float(np.max(abs(result.pi.numpy()-ref.x.reshape(3,3)))),
                                    reference_min_mass=float(ref.x.min()),
                                    reference_column_error=float(abs(ref.x.reshape(3,3).sum(0)-bv).max()))

    def test_extreme_finite_rho_and_failure_is_not_consumed(self):
        a, b = uniform_target_masses(3), uniform_target_masses(2)
        c = torch.tensor([[0.,10000.],[10000.,0.],[20000.,20000.]], dtype=DTYPE)
        result = solve_semi_relaxed(c, a, b, .3, 1.)
        self.assertTrue(result.converged)
        self.assertGreater(result.diagnostics['underflow_count'], 0)
        failed = solve_semi_relaxed(c, a, b, .3, 1., replace(OTTolerances(), max_iter=1), raise_on_failure=False)
        self.assertFalse(failed.converged)
        self.assertEqual(failed.status, 'MAX_ITER')
        with self.assertRaises(OTConvergenceError):
            fixed_plan_objective(failed, c, a, b)
        with self.assertRaises(OTConvergenceError):
            solve_semi_relaxed(c, a, b, .3, 1., replace(OTTolerances(), max_iter=1))
        with self.assertRaises(ValueError):
            solve_semi_relaxed(c*float('nan'), a, b, .3, 1.)
        # Export checks must reject float32 plans when tiny-double tolerances cannot be met.
        c32 = torch.tensor([[.2,1.2],[1.3,.4],[.7,.9]], dtype=torch.float32)
        export = solve_semi_relaxed(c32, a.float(), b.float(), .3, 0., raise_on_failure=False)
        self.assertEqual(export.status, 'EXPORT_FAILED')
        zero = solve_semi_relaxed(torch.zeros_like(c), a, b, .3, 1.)
        torch.testing.assert_close(zero.pi, a[:,None]*b, atol=1e-10, rtol=0)
        RESULTS['extreme_finite_rho'] = result.diagnostics
        RESULTS['intentional_failure'] = dict(status=failed.status, iterations=failed.iterations,
                                              float32_export_status=export.status)

    def test_fixed_plan_full_objective_gradient_is_pi(self):
        c = torch.tensor([[.2,.9],[1.7,.4]], dtype=DTYPE, requires_grad=True)
        a, b = uniform_target_masses(2), uniform_target_masses(2)
        result = solve_semi_relaxed(c, a, b, .3, 1.)
        loss = fixed_plan_objective(result, c, a, b)
        gradient, = torch.autograd.grad(loss['total'], c)
        torch.testing.assert_close(gradient, result.pi, atol=0, rtol=0)
        self.assertGreater(float(loss['total']-loss['transport']), 0)
        self.assertFalse(loss['entropy_kl'].requires_grad)
        self.assertFalse(loss['source_kl'].requires_grad)

    def test_ot_actual_rbf_endpoint_and_metric_gradients(self):
        distance = distance_fixture()
        x, y = [p.requires_grad_() for p in endpoints()]
        a, b = uniform_target_masses(2), uniform_target_masses(2)
        values, result, cost = alignment_loss(x, y, distance, a, b, .3, 1.)
        gradients = torch.autograd.grad(values['total'], (x,y,distance.solver.metric.B))
        self.assertTrue(cost.requires_grad)
        for grad in gradients:
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(float(grad.norm()), 1e-6)
        # Fixed Pi FD, not differentiation of the OT program. Actual nonzero RBF.
        direction = torch.tensor([[.2,-.4],[.1,.3]], dtype=DTYPE)
        h = 1e-5
        with torch.no_grad():
            plus = fixed_plan_objective(result, distance.pairwise(x+h*direction,y), a,b)['total']
            minus = fixed_plan_objective(result, distance.pairwise(x-h*direction,y), a,b)['total']
        error = abs(float((plus-minus)/(2*h)-(gradients[0]*direction).sum()))
        self.assertLess(error, 1e-6)
        metric = distance.solver.metric
        original = metric.B.detach().clone()
        b_direction = torch.tensor([[[.2,.1],[.1,-.2]],[[-.1,.2],[.2,.1]]],dtype=DTYPE)
        with torch.no_grad():
            metric.B.copy_(original+h*b_direction)
            plus = fixed_plan_objective(result,distance.pairwise(x,y),a,b)['total']
            metric.B.copy_(original-h*b_direction)
            minus = fixed_plan_objective(result,distance.pairwise(x,y),a,b)['total']
            metric.B.copy_(original)
        b_error = abs(float((plus-minus)/(2*h)-(gradients[2]*b_direction).sum()))
        self.assertLess(b_error, 1e-6)
        RESULTS['ot_gradients'] = dict(norms=[float(g.norm()) for g in gradients], endpoint_fd_error=error,metric_fd_error=b_error)

    def test_memories_use_fixed_ids_initial_ordering_and_no_target_labels(self):
        graphs, labels, encoder = graph_fixture()
        source_ids, target_ids = tuple(labels), tuple(g for g in graphs if g.startswith('t:'))
        distance = distance_fixture(identity=True)
        provenance = dict(initial_encoder_sha256=encoder_hash(encoder), dataset='synthetic_development')
        with torch.no_grad():
            s = encode_graph_ids(source_ids, graphs.__getitem__, encoder)
            t = encode_graph_ids(target_ids, graphs.__getitem__, encoder)
        source = build_source_memory(source_ids,s,labels,distance,RULES,101,provenance)
        target = build_target_memory(target_ids,t,distance,RULES,102,provenance)
        self.assertGreater(len(source.triplets), 0)
        self.assertGreater(len(target.triplets), 0)
        for i,j,k in source.triplets:
            self.assertNotEqual(i,j)
            self.assertEqual(labels[i],labels[j])
            self.assertNotEqual(labels[i],labels[k])
        with torch.no_grad():
            rows = distance.pairwise(t,t)
        for i,j,k in target.triplets:
            ii,jj,kk = [target_ids.index(g) for g in (i,j,k)]
            rank = sorted((r for r in range(len(t)) if r != ii),key=lambda r:(float(rows[ii,r]),target_ids[r]))
            self.assertIn(jj, rank[:RULES.target_near_neighbors])
            self.assertIn(kk, rank[RULES.target_far_rank_low-1:RULES.target_far_rank_high])
            self.assertGreaterEqual(float(rows[ii,kk]), (1+RULES.zeta)*float(rows[ii,jj])+RULES.epsilon)
        reverse = build_target_memory(target_ids[::-1],t.flip(0),distance,RULES,102,provenance)
        self.assertEqual(target.to_dict(), reverse.to_dict())
        self.assertEqual(target.to_dict(), RelationMemory.from_dict(json.loads(json.dumps(target.to_dict()))).to_dict())
        self.assertFalse(any('coordinate' in k for k in target.to_dict()))
        # Poison label access rather than supplying target class information.
        class InputsOnly:
            def __init__(self,g): self.x,self.edge_index,self.edge_attr=g.x,g.edge_index,g.edge_attr
            @property
            def y(self): raise AssertionError('Target label accessed')
        wrapped = {g:InputsOnly(graphs[g]) for g in target_ids}
        t2 = encode_graph_ids(target_ids, wrapped.__getitem__, encoder)
        again = build_target_memory(target_ids,t2,distance,RULES,102,provenance)
        self.assertEqual(target.to_dict(), again.to_dict())
        empty = build_target_memory(target_ids[:2],t[:2],distance,RULES,102,provenance)
        self.assertEqual(empty.triplets, ())
        RESULTS['memories'] = dict(source=source.to_dict(), target=target.to_dict(), label_access_guard_passed=True)

    def test_relation_exact_reduction_empty_side_current_reencoding_and_gradients(self):
        graphs, labels, encoder = graph_fixture()
        distance = distance_fixture()
        source = RelationMemory('source', (('s:0','s:1','s:5'),('s:1','s:0','s:4')), {})
        target = RelationMemory('target', (('t:0','t:1','t:4'),), {})
        calls = []
        def lookup(g):
            calls.append(g)
            return graphs[g]
        margin = 3.
        values = relation_loss(source,target,lookup,encoder,distance,margin)
        manual = []
        for memory in (source,target):
            one = []
            for ids in memory.triplets:
                u = encode_graph_ids(ids,graphs.__getitem__,encoder)
                one.append(torch.relu(margin+distance(u[0],u[1])-distance(u[0],u[2])))
            manual.append(torch.stack(one).mean())
        torch.testing.assert_close(values['total'], (manual[0]+manual[1])/2, atol=1e-12, rtol=1e-12)
        empty_target = RelationMemory('target', (), {})
        single = relation_loss(source,empty_target,lookup,encoder,distance,margin)
        torch.testing.assert_close(single['total'], values['source']/2, atol=1e-12, rtol=1e-12)
        empty_source = RelationMemory('source', (), {})
        both = relation_loss(empty_source,empty_target,lookup,encoder,distance,margin)
        self.assertEqual(float(both['total']), 0.)
        gradients = torch.autograd.grad(values['total'], (*encoder.parameters(),distance.solver.metric.B))
        encnorm = sum(float(g.square().sum()) for g in gradients[:-1])**.5
        self.assertGreater(encnorm, 1e-6)
        self.assertGreater(float(gradients[-1].norm()), 1e-6)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        with torch.no_grad():
            for p in encoder.parameters(): p.mul_(1.1)
        updated = relation_loss(source,target,lookup,encoder,distance,margin)
        self.assertGreater(abs(float(updated['total'].detach()-values['total'].detach())), 1e-6)
        self.assertEqual(source.triplets[0], ('s:0','s:1','s:5'))
        # Hinge equality boundary, using S2 distances rather than a surrogate metric.
        u = encode_graph_ids(('s:0','s:1','s:5'),graphs.__getitem__,encoder)
        gap = float((distance(u[0],u[2])-distance(u[0],u[1])).detach())
        boundary = relation_loss(RelationMemory('source',(source.triplets[0],),{}),empty_target,
                                 lookup,encoder,distance,max(gap,0.))
        self.assertLess(abs(float(boundary['total'].detach())), 1e-10)
        self.assertGreater(len(calls), 0)
        RESULTS['relation_gradients'] = dict(encoder_norm=encnorm, metric_norm=float(gradients[-1].norm()),
            loss=float(values['total'].detach()), reencoded_loss=float(updated['total'].detach()),
            manual_error=float((values['total']-(manual[0]+manual[1])/2).abs().detach()),
            empty_side_half_weight=True, boundary_loss=float(boundary['total'].detach()))

    def test_shared_encoder_ot_graph_gradients_and_complete_objective(self):
        graphs, labels, encoder = graph_fixture()
        distance = distance_fixture()
        sids,tids = ('s:0','s:4'),('t:0','t:4')
        source = encode_graph_ids(sids,graphs.__getitem__,encoder)
        target = encode_graph_ids(tids,graphs.__getitem__,encoder)
        source.retain_grad(); target.retain_grad()
        a,b = uniform_target_masses(2),uniform_target_masses(2)
        values, result, _ = alignment_loss(source,target,distance,a,b,.3,1.)
        values['total'].backward()
        norms = [float(source.grad.norm()),float(target.grad.norm()),float(distance.solver.metric.B.grad.norm()),
                 sum(float(p.grad.square().sum()) for p in encoder.parameters())**.5]
        self.assertTrue(all(n > 1e-6 for n in norms))
        # Freeze B as in content updates: its position derivatives must still reach encoder.
        encoder.zero_grad(); distance.zero_grad(); distance.solver.metric.B.requires_grad_(False)
        s = encode_graph_ids(sids,graphs.__getitem__,encoder)
        t = encode_graph_ids(tids,graphs.__getitem__,encoder)
        fixed, _, _ = alignment_loss(s,t,distance,a,b,.3,1.)
        fixed['total'].backward()
        self.assertIsNone(distance.solver.metric.B.grad)
        frozen_norm = sum(float(p.grad.square().sum()) for p in encoder.parameters())**.5
        self.assertGreater(frozen_norm, 1e-6)
        ce, align, rel = (torch.tensor(v,dtype=DTYPE,requires_grad=True) for v in (1.,2.,3.))
        total = learning_objective(ce,align,rel,.7,.2)
        self.assertAlmostEqual(float(total), 3.)
        torch.testing.assert_close(torch.stack(torch.autograd.grad(total,(ce,align,rel))),torch.tensor([1.,.7,.2],dtype=DTYPE))
        RESULTS['graph_alignment_gradients'] = dict(source_endpoint=norms[0],target_endpoint=norms[1],
            metric=norms[2],encoder=norms[3],frozen_metric_encoder=frozen_norm, pi_detached=not result.pi.requires_grad)


if __name__ == '__main__':
    unittest.main()
