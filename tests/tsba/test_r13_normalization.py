"""Regression of BN mathematics, native order, label gating, and pilot criteria."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import (
    historical, normalization as norm, observation, scoring,
)
from tsba.encoder import GraphEncoder
from tsba.pretrain import SourceModel


def fixture(conv, sizes):
    torch.manual_seed(117)
    model = SourceModel(GraphEncoder(2, 1, hidden=5, num_layers=2, out_dim=3, dropout=.4, conv=conv), 2)
    model.eval().requires_grad_(False)
    model.encoder.freeze()
    graphs = []
    for i, n in enumerate(sizes):
        edges = torch.tensor([(a, b) for a in range(n) for b in (a-1, a+1) if 0 <= b < n], dtype=torch.long).T
        graphs.append(Data(x=torch.rand(n, 2) + i*.7, edge_index=edges,
                           edge_attr=torch.rand(edges.shape[1],1)))
    return model, graphs


def individual_graph_reference(original, data, weighting):
    """Collect full activations graph by graph and use explicit two-pass variance."""
    model = deepcopy(original).eval()
    states = dict(model.encoder.named_modules())
    results, order = {}, []
    for i in range(2):
        order.extend((f'convs.{i}.nn.1', f'norms.{i}'))
    with torch.no_grad():
        for name in order:
            layer = states[name]
            rows = []
            attach = layer.register_forward_pre_hook(lambda _m, args: rows.append(args[0].detach().double().numpy()))
            for graph in data:
                model.encoder(Batch.from_data_list([graph]))
            attach.remove()
            x = np.concatenate(rows)
            weights = np.concatenate([np.full(len(row), 1/len(row) if weighting=='graph' else 1.) for row in rows])
            mu = np.average(x, weights=weights, axis=0)
            variance = np.average((x-mu)**2, weights=weights, axis=0)
            layer.running_mean.copy_(torch.from_numpy(mu))
            layer.running_var.copy_(torch.from_numpy(variance))
            results[name] = (mu, variance)
    return results, model


class BatchNormTests(unittest.TestCase):
    def test_weighted_population_variance_includes_between_graph_means(self):
        activation = torch.tensor([[0.], [2.], [4.], [10.]], dtype=torch.float32)
        offsets = torch.tensor([0, 3, 4])
        expected = {'node': (4.,14.), 'graph': (6., 52/3)}
        for mode, (mean, var) in expected.items():
            accumulator = norm.PopulationMoments(mode)
            accumulator.update(activation, offsets)
            actual = accumulator.finish()
            self.assertEqual(float(actual[0]), mean)
            self.assertAlmostEqual(float(actual[1]), var)
            self.assertEqual((accumulator.graphs, accumulator.nodes), (2,4))

    def test_equal_size_graphs_give_identical_node_and_graph_predictions(self):
        original, data = fixture('gin', [3,3,3,3,3])
        node, stats_node, _ = norm.recalibrate(original, data, 'node', 2)
        graph, stats_graph, _ = norm.recalibrate(original, data, 'graph', 2)
        for k, value in stats_node.items():
            np.testing.assert_allclose(value['mean'], stats_graph[k]['mean'], atol=1e-14, rtol=1e-14)
            np.testing.assert_allclose(value['variance'], stats_graph[k]['variance'], atol=1e-14, rtol=1e-14)
        np.testing.assert_array_equal(norm.logits(node,data,2), norm.logits(graph,data,2))

    def test_gin_and_gine_interleave_and_hold_under_repartition(self):
        for conv in ('gin','gine'):
            with self.subTest(conv=conv):
                original, data = fixture(conv, [2,3,8,4])
                archive = {key: v.clone() for key, v in original.state_dict().items()}
                for method in ('node','graph'):
                    ref, expected = individual_graph_reference(original, data, method)
                    replay, measured, changed = norm.recalibrate(original, data, method, 2)
                    third, measurements, _ = norm.recalibrate(original, data, method, 3)
                    self.assertEqual(list(measured), ['convs.0.nn.1','norms.0','convs.1.nn.1','norms.1'])
                    self.assertTrue(changed)
                    self.assertTrue(all(not m.training for m in replay.modules()))
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in replay.parameters()))
                    for k in measured:
                        for i, field in enumerate(('mean','variance')):
                            np.testing.assert_allclose(measured[k][field], ref[k][i], atol=2e-6, rtol=2e-6)
                            np.testing.assert_allclose(measured[k][field], measurements[k][field], atol=2e-6, rtol=2e-6)
                    norm.compare_logits(norm.logits(replay,data,2), norm.logits(expected,data,1),
                                        dict(atol=2e-5, rtol=2e-5))
                    norm.compare_logits(norm.logits(replay,data,2), norm.logits(third,data,3),
                                        dict(atol=2e-5, rtol=2e-5))
                for key, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value,archive[key]), key)

    def test_affine_and_nonmoment_buffers_are_checked(self):
        original, data = fixture('gin', [2,3,4])
        fitted, stats, _ = norm.recalibrate(original,data,'graph',2)
        with torch.inference_mode():
            fitted.encoder.norms[0].weight[0] += .25
        with self.assertRaises(ValueError):
            norm.verify_frozen(original, fitted, stats)

    def test_topk_diagnostic_tie_is_smallest_numeric_id(self):
        ids = np.array([30, 1, 8, 20])
        z = np.column_stack((np.zeros(4), [0.,1.,1.,-2.]))
        np.testing.assert_array_equal(observation.equal_positive(ids,z,1), [0,1,0,0])
        np.testing.assert_array_equal(observation.equal_positive(ids,z,0), [0,0,0,0])
        np.testing.assert_array_equal(observation.equal_positive(ids,z,4), [1,1,1,1])

    def test_same_k_and_real_alien_roles_are_swapped_only_within_group(self):
        entries = [dict(target=t, ids={'source':[1,2,3,4,5], 'validation':[6],
                                     'adapt':list(range(10*t,10*t+a)), 'holdout':[10*t+8]})
                   for t,a in [(1,4),(2,2),(3,6)]]
        k, role = observation.roles_for_group('g', entries)
        self.assertEqual(k, 2)
        for key, pool in [('S',entries[0]['ids']['source'])] + [(f"A{x['target']}",x['ids']['adapt']) for x in entries]:
            self.assertEqual(role[key],historical.selected_ids('g',pool,2))
        self.assertEqual(observation.name_map(2,[1,3],('node',))['Other_node_2'],'A3_node')
        entries[-1]['ids']['holdout'] = [2]
        with self.assertRaises(PermissionError):
            observation.roles_for_group('g',entries)

    def test_no_verified_obs_no_h_reader(self):
        with TemporaryDirectory() as tmp, patch.object(scoring.r11_inputs,'GraphLabelSession') as labels:
            root = Path(tmp)
            plan = dict(settings=['g'],domains=[0,1],r7_final_digest='fixture')
            with self.assertRaises(FileNotFoundError):
                scoring.score(root,root,plan,[3405],('node','graph'),'missing',pilot=True)
            labels.assert_not_called()

    def test_exact_paired_interaction_and_other_mean(self):
        ys = np.array([0,1,0,1])
        decisions = dict(A_graph=np.array([0,1,0,1]), A_node=np.array([0,0,0,0]),
                         S_graph=np.array([0,0,0,0]), S_node=np.array([0,1,0,0]),
                         Other_graph_1=np.array([0,1,0,0]), Other_graph_2=np.array([1,0,0,1]))
        correct = {k:z==ys for k,z in decisions.items()}
        weights = np.array([[1.,0.,0.,0.],[0.,.5,.5,0.]])
        draw = {k:weights@v for k,v in correct.items()}
        diff, draws, extra = scoring.compare('A_graph','Other_graph',decisions,correct,draw)
        self.assertEqual(diff,Fraction(3,8))
        self.assertEqual(extra['corrected_exact'],'3/2')
        np.testing.assert_allclose(draws,draw['A_graph'] - (draw['Other_graph_1']+draw['Other_graph_2'])/2)
        interaction, _, _ = scoring.compare(*scoring.INTERACTION,decisions,correct,draw)
        self.assertEqual(interaction,Fraction(3,4))

    def test_gate_is_exact_and_prefers_graph_only_when_both_settings_pass(self):
        plan = dict(settings=['a','b','c'],domains=list(range(4)),phase1_seed=3405,
                    minimum_positive_directions=7,minimum_positive_settings=2)
        directions = [(s,t) for s in plan['domains'] for t in plan['domains'] if s!=t]
        values = {(name,s,t,3405,arm,baseline): Fraction(1,100)
                  for name in ('a','b') for s,t in directions
                  for arm, baselines in (('A_graph',scoring.GRAPH_GATES),('A_node',scoring.NODE_GATES))
                  for baseline in baselines}
        self.assertEqual(scoring.gate(plan,values)['branch'],'graph')
        for i,(s,t) in enumerate(directions):
            values['b',s,t,3405,'A_graph','SourceCal'] = Fraction((-1)**i,307)
        self.assertEqual(scoring.gate(plan,values)['branch'],'node')
        for s,t in directions[:6]:
            values['b',s,t,3405,'A_node','B0'] = Fraction(0)
        self.assertEqual(scoring.gate(plan,values)['branch'],'stop')
