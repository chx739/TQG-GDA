"""Regressions for the source-only R10 structural reference and its label gate."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.sparse import issparse
import torch
from torch_geometric.data import Data

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import fit_logistic, scores
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural_observation as obs
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural_scoring as scoring


ROOT = Path(__file__).resolve().parents[2] / 'experiments/mutagenicity/r10_原生图适配基线与增量_20261121'
PLAN = json.loads((ROOT / 'configs/structural-reference.json').read_bytes())


def graph(kinds, edges):
    return Data(x=torch.nn.functional.one_hot(torch.tensor(kinds), 3),
                edge_index=torch.tensor(edges, dtype=torch.int64).reshape(-1, 2).T.contiguous())


class StructuralReferenceTests(unittest.TestCase):
    def test_uniform_proteins_edge_matches_typed_wl_and_graph_size_counts_unordered_pairs(self):
        g = graph([0, 1, 1], [(0, 1), (1, 0), (1, 2), (2, 1)])
        row, sizes = structural.features(g, 'proteins_edge')
        reference = dict(x=g.x, edge_index=g.edge_index, edge_attr=torch.ones((4, 1)))
        counts, _ = wl.fingerprints(reference, {'node_types': 3, 'edge_types': 1})
        self.assertEqual(row, wl.normalize(counts))
        np.testing.assert_array_equal(sizes, np.log1p([3, 2]))
        repeated = graph([0, 1, 1], [(0, 1), (1, 0), (0, 1), (1, 2), (2, 1), (2, 2)])
        np.testing.assert_array_equal(structural.size(repeated), sizes)
        g.edge_attr = torch.ones((4, 1))
        with self.assertRaises(ValueError):
            structural.features(g, 'proteins_edge')

    def test_exact_validation_order_and_zero_margin(self):
        self.assertEqual(structural.choose([
            dict(C=0.1, correct=2, nll=.4), dict(C=1., correct=3, nll=.5),
            dict(C=10., correct=3, nll=np.nextafter(.5, 0.)), dict(C=100., correct=3, nll=.5),
        ]), 10.)
        self.assertEqual(structural.choose([
            dict(C=10., correct=3, nll=.2), dict(C=0.1, correct=3, nll=.2),
        ]), .1)
        np.testing.assert_array_equal(structural.decisions(np.array([-1., -0., 0., 1.])), [0, 0, 0, 1])

    def test_sparse_source_only_vocabulary_fixed_model_and_size_replay(self):
        source = [graph([0, 0], []), graph([0, 1], []),
                  graph([1, 1], [(0, 1), (1, 0)]), graph([0, 1], [(0, 1), (1, 0)])]
        validation = [graph([0, 2], []), graph([1, 2], [(0, 1), (1, 0)])]
        sf, vf = ([structural.features(g, 'proteins_edge') for g in pool] for pool in (source, validation))
        labels, val_labels = np.array([0, 0, 1, 1]), np.array([0, 1])
        with patch.object(structural, 'fit_logistic', wraps=fit_logistic) as logistic:
            fitted = structural.fit_models(sf, vf, labels, val_labels, PLAN)
        self.assertEqual(logistic.call_count, 8)
        self.assertTrue(all(issparse(call.args[0]) for call in logistic.call_args_list[:4]))
        self.assertEqual(sum(call.kwargs['penalty'] == 1. for call in logistic.call_args_list[:4]), 1)
        self.assertEqual(fitted['size']['scale'][0], 1.)
        rows = [row for row, _ in sf + vf]
        keys = wl.vocabulary(rows, range(4))
        self.assertEqual(fitted['wl']['vocabulary'], [[d, color.hex()] for d, color in keys])
        held = wl.sparse_rows(rows, [4, 5], keys)
        self.assertTrue(np.all(np.asarray(held.multiply(held).sum(1)) < 1))
        direct = fit_logistic(wl.sparse_rows(rows, range(4), keys), labels, np.zeros(4), PLAN['solver'])
        predicted = structural.predict(fitted, vf)
        np.testing.assert_array_equal(predicted['WL_fixed'], scores(held, direct))
        for method in predicted:
            np.testing.assert_allclose(structural.predict(fitted, vf[:1])[method], predicted[method][:1])
        fitted['wl']['selected_C'] = 1.
        reused = structural.predict(fitted, vf)
        np.testing.assert_array_equal(reused['WL_fixed'], predicted['WL_fixed'])
        np.testing.assert_array_equal(reused['WL_selected'], reused['WL_fixed'])

    def test_line_search_retries_same_fit_at_existing_stationarity_limit(self):
        x = np.array([[-1.], [1.]])
        y = np.array([0, 1])
        default = PLAN['solver']
        plan = dict(PLAN, C=[1.])
        failure = ValueError('Source logistic fit failed: ABNORMAL: , stationarity 1.82529e-09')
        answer = dict(weights=[1.], intercept=0., training_objective=.5, gradient_inf=2e-9)
        with patch.object(structural, 'fit_logistic', side_effect=[failure, answer]) as logistic:
            fitted = structural.fit_candidates(x, x, y, y, plan)
        initial, retry = logistic.call_args_list
        self.assertIs(initial.args[0], retry.args[0])
        self.assertIs(initial.args[1], retry.args[1])
        self.assertIs(initial.args[2], retry.args[2])
        self.assertEqual(initial.kwargs, retry.kwargs)
        self.assertEqual(initial.args[3], default)
        self.assertEqual(retry.args[3], dict(default, gradient_tolerance=default['stationarity_tolerance']))
        self.assertEqual(fitted['candidates']['1.0']['line_search_recovery']['original_error'], str(failure))
        with patch.object(structural, 'fit_logistic', side_effect=ValueError('Invalid binary source logistic inputs')) as logistic:
            with self.assertRaisesRegex(ValueError, 'Invalid binary'):
                structural.fit_candidates(x, x, y, y, plan)
            logistic.assert_called_once()

    def test_role_guard_rejects_a_h_and_reordered_ids_at_source_fit(self):
        pools = [dict(source=[2, 1], validation=[3], adapt=[4+t], holdout=[7+t]) for t in range(3)]
        structural.check_roles(pools)
        store = Mock()
        structural.take_role(store, [2, 1], pools[0], 'source', fitting=True)
        for ids, role in (([4], 'adapt'), ([7], 'holdout'), ([1, 2], 'source')):
            with self.assertRaises(PermissionError):
                structural.take_role(store, ids, pools[0], role, fitting=True)
        with self.assertRaises(PermissionError):
            structural.take_role(store, [4], pools[0], 'adapt', fitting=False)
        store.take.assert_called_once_with([2, 1])
        pools[1]['holdout'] = pools[0]['adapt']
        with self.assertRaises(PermissionError):
            structural.check_roles(pools)

    def test_reference_requires_role_hash_task_and_order(self):
        task = dict(setting='mutagenicity_edge', source=0, target=1, seed=3405)
        ids = [5, 3, 8]
        exact = dict(task, h_ids=ids, r7_manifest_sha256='r7 seal digest')
        check = dict(checksum_name='r7_manifest_sha256', ids_name='h_ids')
        obs.check_reference_role(exact, task, 'r7 seal digest', ids, **check)
        for bad in (dict(exact, r7_manifest_sha256='other'), dict(exact, seed=3406),
                    dict(exact, h_ids=sorted(ids))):
            with self.assertRaises(PermissionError):
                obs.check_reference_role(bad, task, 'r7 seal digest', ids, **check)

    def test_incomplete_fit_or_observation_never_constructs_holdout_reader(self):
        structural.validate_plan(PLAN)
        self.assertEqual(len(obs.tasks(PLAN)), 180)
        self.assertEqual(len(obs.fit_members(PLAN)) - 4, 60)
        with TemporaryDirectory() as tmp, patch.object(scoring.inputs, 'GraphLabelSession') as label:
            output = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                obs.freeze(output, ROOT, PLAN, 'incomplete')
            with self.assertRaises(FileNotFoundError):
                scoring.score(output, ROOT, PLAN, 'incomplete', 'incomplete')
            label.assert_not_called()
        with patch.object(obs, 'verify_fit', return_value=({}, {}, {})), \
                patch.object(scoring.inputs, 'GraphLabelSession') as label, TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                scoring.score(Path(tmp), ROOT, PLAN, 'x', 'incomplete')
            label.assert_not_called()


if __name__ == '__main__':
    unittest.main()
