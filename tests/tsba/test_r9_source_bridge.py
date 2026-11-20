"""Generated typed-WL, bridge exclusions, exact gate and label guard regressions."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import bridge, roles, screen, wl


ROOT = Path(__file__).resolve().parents[2] / 'experiments/mutagenicity/r9_目标近邻与源标签传递_20261121'
PLAN = json.loads((ROOT/'configs/source-bridge.json').read_text())


def graph(types, edges):
    pairs = [(u, v, e) for u, v, e in edges for u, v in ((u, v), (v, u))]
    return dict(x=torch.nn.functional.one_hot(torch.tensor(types), PLAN['node_types']),
                edge_index=torch.tensor([(u, v) for u, v, _ in pairs], dtype=torch.int64).reshape(-1, 2).T.contiguous(),
                edge_attr=torch.nn.functional.one_hot(torch.tensor([e for _, _, e in pairs], dtype=torch.int64),
                                                       PLAN['edge_types']))


class SourceBridgeTests(unittest.TestCase):
    def test_three_level_signature_hand_example_and_feature_denominators(self):
        counts, _, rows = wl.make_records({
            0: graph([0, 1], [(0, 1, 0)]),
            1: graph([0, 1], [(0, 1, 1)]),
            2: graph([0, 1, 1], [(0, 1, 0)]),
        }, PLAN)
        self.assertEqual([len(bag) for bag in counts[0]], [2, 2, 2])
        np.testing.assert_allclose(wl.kernel(rows, [0, 1], [0, 1]), [[1, 1/3], [1/3, 1]], atol=1e-14)
        vocab = wl.vocabulary(rows, [0])
        truncated = wl.sparse_rows(rows, [2], vocab)
        self.assertLess(float(truncated.multiply(truncated).sum()), 1.)
        self.assertEqual(truncated[0, vocab[(0, wl.hash_color(b'R9-WL0\0'+bytes([0, 0, 0, 1])))]],
                         2/np.sqrt(15))

    def test_reindex_and_disconnected_copies_share_primitive_fingerprint(self):
        one = graph([1, 0, 2], [(0, 1, 1), (1, 2, 2)])
        renumbered = graph([2, 1, 0], [(1, 2, 1), (2, 0, 2)])
        duplicated = graph([1, 0, 2, 1, 0, 2], [(0, 1, 1), (1, 2, 2), (3, 4, 1), (4, 5, 2)])
        counts, _, rows = wl.make_records(dict(enumerate((one, renumbered, duplicated))), PLAN)
        self.assertNotEqual(counts[0], counts[2])
        keys = {i: wl.count_key(row) for i, row in counts.items()}
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(keys[0], keys[2])
        self.assertEqual(rows[0], rows[2])
        np.testing.assert_array_equal(bridge.source_allowed([0], [0, 1, 2, 3], keys | {3: b'other'}),
                                      [[False, False, False, True]])

    def test_weights_use_id_tie_and_uniform_only_over_allowed(self):
        matrix = np.array([[.5, .5, .6], [0., 0., 0.]])
        mask = np.array([[True, True, False], [True, False, True]])
        weights, fallback = bridge.row_weights(matrix, [28, 10, 12], 1, allowed=mask)
        np.testing.assert_array_equal(weights, [[0, 1, 0], [.5, 0, .5]])
        self.assertEqual(fallback, 1)
        with self.assertRaises(bridge.NoCandidates):
            bridge.row_weights(matrix, [28, 10, 12], 1, allowed=np.zeros_like(mask))

    def test_reassignment_is_noop_when_weight_ranks_match(self):
        a = np.array([[.4, .6, 0.], [.2, .7, .1]])
        k = np.array([[.6, .8, .1], [.4, .9, .2]])
        np.testing.assert_array_equal(bridge.reassign(a, k, [5, 0, 3]), a)
        other = bridge.reassign(a, np.flip(k, axis=1), [5, 0, 3])
        np.testing.assert_array_equal(np.sort(a, axis=1), np.sort(other, axis=1))
        self.assertFalse(np.array_equal(a, other))

    def test_mixed_fragment_middle_makes_wrong_endpoint_win(self):
        rows = {0:{(0, b'x'): 1.}, 1:{(0, b'y'): 1.}, 2:{(0, b'x'): .6, (0, b'y'): .8},
                3:{(0, b'x'): 1.}}
        result, checks = bridge.predict(rows, {i: bytes([i]) for i in rows}, [0, 1], [3], [2], [0],
                                        np.array([0, 1]), np.array([-.1]), PLAN | {'k': 1})
        self.assertEqual(result['action_B'].item(), 0)
        self.assertEqual(result['action_Direct'].item(), 0)
        self.assertEqual(result['action_TargetBridge'].item(), 1)
        self.assertEqual(result['action_MatchedWeights'].item(), 0)
        np.testing.assert_array_equal(result['weights_PooledBridge'], result['a_to_s'].mean(0)[None])
        self.assertTrue(checks['exact_weight_multiset_match'])

    def test_roles_same_seed_replays_and_no_normalized_component_crosses_h_a(self):
        graphs = {i: graph([i%3, (i//3)%4, 5], [(0, 1, i%2)] + ([(1, 2, 2)] if i >= 12 else []))
                  for i in range(24)}
        counts, edges, _ = wl.make_records(graphs, PLAN)
        splits = roles.build(edges, counts, PLAN)
        self.assertEqual(splits, roles.build(dict(reversed(list(edges.items()))), counts, PLAN))
        for d in splits['directions']:
            held = {wl.count_key(counts[i]) for i in d['holdout_ids']}
            for p in d['perturbations'].values():
                self.assertTrue(held.isdisjoint(wl.count_key(counts[i]) for i in p['adapt']))
                self.assertEqual(len(p['adapt']), len(p['source_middle']))

    def test_exact_one_pp_bound_can_score_but_smaller_aborts_before_labels(self):
        plan = copy.deepcopy(PLAN)
        plan['adaptation_seeds'] = [3405]
        a = np.zeros(100, dtype=np.int64)
        b = a.copy()
        b[0] = 1
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'predictions').mkdir()
            (root/'directions').mkdir()
            directions = [dict(source=i, target=1-i, holdout_ids=list(range(100*i, 100*i+100)))
                          for i in range(2)]
            write_json(root/'roles.json', {'directions': directions})
            statuses = {}
            for key, d, _ in screen.tasks({'directions': directions}, plan):
                rel = f"directions/ps{d['source']}_to_{d['target']}"
                for suffix in ('checks', 'model'):
                    write_json(root/f'{rel}-{suffix}.json', {})
                statuses[key] = dict(status='available')
                # Pooled is irrelevant for the four gate comparisons.
                values = {'action_'+arm: b if arm in screen.CRITICAL else a for arm in screen.METHODS}
                screen.fit.write_npz(root/screen.prediction_path(key), h_ids=np.array(d['holdout_ids']), **values)
            write_json(root/'block-status.json', statuses)
            seal = root/'OBS.json'

            def seal_observation():
                files = {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()}
                write_json(seal, dict(stage='R9_OBS', r8_parent_sha256=plan['r8_observation_sha256'], files=files))
                return digest(seal)

            sha = seal_observation()
            frozen = screen.read_frozen(root, sha, plan)
            self.assertTrue(screen.action_gate(*frozen, plan)['scoring_permitted'])
            seal.unlink()
            observation = root/screen.prediction_path(next(iter(statuses)))
            observation.unlink()
            screen.fit.write_npz(observation, h_ids=np.array(directions[0]['holdout_ids']),
                                 **{'action_'+name: a for name in screen.METHODS})
            sha = seal_observation()
            with patch.object(screen.inputs, 'SourceEvidence') as evidence:
                result = screen.finish(root, sha, plan, ROOT)
                evidence.assert_not_called()
            self.assertFalse(result['passed'])
            self.assertEqual(result['scoring_label_pools'], [])
            self.assertEqual(set(result['blocked_comparators']), set(screen.CRITICAL))

    def test_strict_matched_direction_condition_checks_exact_correct_counts(self):
        plan = copy.deepcopy(PLAN)
        plan['adaptation_seeds'] = [3405, 3406]
        rows = []
        for s, t in plan['directions']:
            for seed in plan['adaptation_seeds']:
                for arm in screen.COMPARATORS:
                    net = 4 if arm != 'MatchedWeights' or s == 0 else 0
                    rows.append(dict(source=s, target=t, seed=seed, comparator=arm, n=100, net=net))
        gate = screen.paired_verdict(rows, plan)
        self.assertFalse(gate['passed'])
        self.assertFalse(gate['passed_each_critical']['MatchedWeights'])

    def test_all_one_group_target_is_archived_as_unavailable(self):
        a = graph([0, 1], [(0, 1, 0)])
        graphs = {0:a, 1:a, 2:graph([0, 0, 1], [(0, 1, 0), (1, 2, 1)]),
                  3:graph([0, 1, 1], [(0, 1, 0), (1, 2, 1)])}
        counts, sizes, rows = wl.make_records(graphs, PLAN)
        record = roles.build(sizes, counts, PLAN)
        unavailable = next(d for d in record['directions'] if d['target'] == 0)
        self.assertEqual(unavailable['unavailable_reason'], 'target_has_fewer_than_two_normalized_fingerprint_groups')
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'directions').mkdir()
            with patch.object(screen.inputs, 'SourceEvidence') as evidence:
                reader = screen.inputs.RoleLabels(evidence.return_value, record, 'fit')
                statuses, reports, _ = screen.fit_direction(
                    unavailable, reader, rows, {i: wl.count_key(c) for i, c in counts.items()}, root, PLAN)
                evidence.return_value.label_values.assert_not_called()
            self.assertTrue(all(v['status']=='unavailable' for v in statuses.values()))
            self.assertFalse(reports)
            details = json.loads((root/'directions/ps1_to_0-checks.json').read_text())
            self.assertEqual(details['source_label_ids'], [])
            self.assertEqual(len(details['status']), 5)

    def test_label_reads_are_direction_bound_and_scoring_follows_complete_obs(self):
        plan = PLAN | {'adaptation_seeds': [3405], 'cohort_bootstrap_replicates': 10}
        roles_record = {'directions': [
            dict(source=0, target=1, source_ids=[0, 1, 2, 3], holdout_ids=[4, 5]),
            dict(source=1, target=0, source_ids=[4, 5, 6, 7], holdout_ids=[0, 1])]}
        events = []
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output/'directions').mkdir()
            (output/'predictions').mkdir()
            write_json(output/'roles.json', roles_record)

            def label_values(requested, allowed):
                events.append((tuple(requested), tuple(allowed), (output/'OBS.json').exists()))
                return np.array(requested) % 2

            with patch.object(screen.inputs, 'SourceEvidence') as mocked:
                evidence = mocked.return_value
                evidence.label_values.side_effect = label_values
                reader = screen.inputs.RoleLabels(evidence, roles_record, 'fit')
                with self.assertRaises(PermissionError):
                    reader.load(0, 1, [4, 5])
                statuses = {}
                for key, d, seed in screen.tasks(roles_record, plan):
                    reader.load(d['source'], d['target'], d['source_ids'])
                    stem = f"directions/ps{d['source']}_to_{d['target']}"
                    write_json(output/f'{stem}-checks.json', {'source_label_ids': reader.actual[d['source'],d['target']]})
                    write_json(output/f'{stem}-model.json', {})
                    statuses[key] = {'status': 'available'}
                    actions = {arm: np.array([0,1]) if arm=='TargetBridge' else np.array([1,0])
                               for arm in screen.METHODS}
                    screen.fit.write_npz(output/screen.prediction_path(key), h_ids=np.array(d['holdout_ids']),
                        **{f'action_{arm}': value for arm,value in actions.items()},
                        **{f'logp_{arm}': np.eye(2)[value] for arm,value in actions.items()})
                write_json(output/'block-status.json', statuses)
                files = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}
                write_json(output/'OBS.json', dict(stage='R9_OBS', r8_parent_sha256=plan['r8_observation_sha256'], files=files))
                result = screen.finish(output, digest(output/'OBS.json'), plan, ROOT)
            self.assertTrue(result['passed'])
            self.assertEqual(events, [((0,1,2,3),(0,1,2,3),False), ((4,5,6,7),(4,5,6,7),False),
                                      ((4,5),(4,5),True), ((0,1),(0,1),True)])
            self.assertTrue((output/'FINAL.json').is_file())
