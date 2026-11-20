"""Generated regressions for the R15 origin-only screen."""
from collections import Counter, deque
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from experiments.common.io import write_json
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r15_新局部结构与图类别_20261121.src import scoring, screen, source_inputs, structure


PLAN = dict(minimum_votes=2, shrinkage_pseudocount=2, bootstrap_seed=20261121,
            bootstrap_replicates=2000, minimum_accuracy_gain_fraction=[1, 200],
            r9_observation_sha256='generated')


def graph(labels, pairs):
    arcs = [(u, v) for x, y in pairs for u, v in ((x, y), (y, x))]
    return dict(x=torch.nn.functional.one_hot(torch.tensor(labels), num_classes=4),
                edge_index=torch.tensor(arcs).T.contiguous(),
                edge_attr=torch.ones((len(arcs), 1), dtype=torch.long))


def ball(neighbors, node, depth):
    found = {node}
    queue = deque([(node, 0)])
    while queue:
        center, r = queue.popleft()
        if r < depth:
            for adjacent in neighbors[center]:
                if adjacent not in found:
                    found.add(adjacent)
                    queue.append((adjacent, r + 1))
    return found


class SourceScreenTest(TestCase):
    def test_center_alignment_and_multiple_centers_disjoint_neighborhoods(self):
        g = graph([0] * 9 + [1, 0, 2],
                  [(i, i+1) for i in range(8)] + [(9, 10), (10, 11)])
        plan = dict(node_types=4, edge_types=1)
        record = structure.encode(g, plan)
        self.assertEqual((record.counts, record.edges), wl.fingerprints(g, plan))
        centers = [0, 8]
        self.assertEqual(record.layers[2][0], record.layers[2][8])
        self.assertEqual(record.layers[1][0], record.layers[1][8])
        self.assertEqual(structure.distances(record.neighbors, centers), [0,1,2,3,4,3,2,1,0,-1,-1,-1])
        word_balls = set.union(*(ball(record.neighbors, center, 2) for center in centers))
        for r in range(3):
            for i, d in enumerate(structure.distances(record.neighbors, centers)):
                self.assertEqual(d > 2+r, d >= 0 and ball(record.neighbors, i, r).isdisjoint(word_balls))

    def test_context_normalization_keeps_unknown_and_excludes_unreachable(self):
        path = [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4], []]
        layers = [[b'x'] * 3 + [b'a', b'a', b'z', b'x'],
                  [b'z'] * 4 + [b'z', b'a', b'x'],
                  [b'z'] * 5 + [b'a', b'x']]
        coefficients = {(0, b'a'): 2., (1, b'a'): 3., (2, b'a'): 5.,
                        (0, b'x'): 100., (1, b'x'): 100., (2, b'x'): 100.}
        rec = structure.GraphRecord(layers, path, 5)
        expected = 4. / np.sqrt(15) + 3. / np.sqrt(6) + 5. / np.sqrt(3)
        self.assertAlmostEqual(structure.context_score(rec, [0], coefficients), expected)
        self.assertIsNone(structure.context_score(rec, [2, 3], coefficients))

    def test_independent_support_pair_order_swap_and_full_e_denominator(self):
        words = [bytes([c]) for c in range(1, 6)]
        if_votes = [(2, 7, -2.), (3, 6, 2.), (2, 8, 5.), (2, 7, 1.), (1, 6, -5.)]
        votes = {
            word: [dict(graph_id=i, edges=edges, score=score) for i in range(n)]
            for word, (n, edges, score) in zip(words, if_votes)
        }
        parents = {word: b'p' for word in words}
        with patch.object(structure, 'collect_votes', return_value=(parents, votes)):
            dictionary = structure.fit_dictionary({}, {}, PLAN)
        self.assertEqual([p['words'] for p in dictionary['pairs']], [['01', '04']])
        self.assertEqual(dictionary['words']['01']['weight'], -.75)
        self.assertEqual(dictionary['words']['04']['weight'], .75)
        self.assertEqual(dictionary['words']['02']['weight'], 0.)
        # Move a third vote into the same bin and check the different recipient factors.
        votes[words[1]][0]['edges'] = 9
        votes[words[2]][0]['edges'] = 6
        votes[words[3]] = votes[words[3]][:1]
        with patch.object(structure, 'collect_votes', return_value=(parents, votes)):
            dictionary = structure.fit_dictionary({}, {}, PLAN)
        self.assertEqual([p['words'] for p in dictionary['pairs']], [['01', '03']])
        self.assertEqual(dictionary['words']['03']['weight'], 1.75)
        votes[words[2]] = votes[words[2]][:1]
        with patch.object(structure, 'collect_votes', return_value=(parents, votes)):
            dictionary = structure.fit_dictionary({}, {}, PLAN)
        self.assertEqual([p['words'] for p in dictionary['pairs']], [['01', '02']])
        self.assertEqual(dictionary['words']['01']['weight'], -1.)
        self.assertEqual(dictionary['words']['02']['weight'], 1.2)
        self.assertEqual(dictionary['words']['02']['swapped_weight'], -1.2)
        row = wl.normalize([Counter({b'p': 6}), Counter({b'q': 6}), Counter({words[1]: 2, b'other': 4})])
        actual, swapped = structure.predict_delta([row], dictionary)[0]
        self.assertAlmostEqual(actual, 2 * 1.2 / np.sqrt(3 * (2**2 + 4**2)))
        self.assertEqual(actual, -swapped)

    def test_minimum_original_id_for_each_u_fingerprint(self):
        direction = dict(source_ids=[0, 1], holdout_ids=[9, 10],
                         groups=[dict(pool='U', graph_ids=[7, 4], group_sha256='a'),
                                 dict(pool='U', graph_ids=[5], group_sha256='b'),
                                 dict(pool='H', graph_ids=[9, 10], group_sha256='c')])
        self.assertEqual(source_inputs.direction_roles(direction),
                         dict(D=[0, 1], U=[4, 5, 7], U_representatives=[4, 5], E=[9, 10]))

    def test_actions_tie_orientation_and_nll_slope(self):
        ids = np.array([30, 15, 20, 10])
        margin = np.array([0.5, 0., 0., -0.5])
        delta = np.array([1., 2., 0., 0.])
        p = scoring.predictions(ids, margin, delta, -delta)
        np.testing.assert_array_equal(p['action_B'], [1, 0, 0, 0])
        np.testing.assert_array_equal(p['action_' + scoring.RANK], [1, 1, 0, 0])
        score = scoring.score_direction(p, np.array([1, 1, 0, 0]), 100, np.random.default_rng(1))
        self.assertGreater(score['nll_improvement_slope']['value'], 0)
        self.assertEqual(score['correspondence_comparisons']['B']['corrected'], 1)
        self.assertGreater(score['correspondence_comparisons']['B']['gains']['nll']['value'], 0)
        np.testing.assert_array_equal(scoring.match_positive_count(np.array([30, 20, 15]), np.zeros(3), 1),
                                      [0, 0, 1])

    def test_freeze_both_directions_before_labels_and_stop_for_either_direction(self):
        directions = [
            dict(source=s, target=t, source_ids=source, holdout_ids=[target[0]],
                 groups=[dict(pool='H', graph_ids=[target[0]], group_sha256='a'),
                         dict(pool='U', graph_ids=[target[1]], group_sha256='b')])
            for s, t, source, target in [(0, 1, [0, 1], [2, 3]), (1, 0, [2, 3], [0, 1])]
        ]
        records = {i: object() for i in range(4)}
        rows = {i: {(2, b'x'): 1.} for i in records}
        paired = dict(pairs=[{}],
                      words={'78': dict(pair=0, n=2, votes=[dict(graph_id=1), dict(graph_id=3)],
                                        weight=1., swapped_weight=-1.)})
        zero = dict(paired, words={'78': dict(paired['words']['78'], weight=0., swapped_weight=0.)})
        empty = dict(pairs=[], words={})
        for final_dictionary, expected in [(empty, 'stopped_without_labels'), (zero, 'stopped_without_labels'),
                                           (paired, 'scored')]:
            with self.subTest(expected=expected, pairs=bool(final_dictionary['pairs'])), TemporaryDirectory() as tmp:
                output = Path(tmp)
                requested = []

                def labels(ids, allowed):
                    for name in ('ps0_to_1', 'ps1_to_0'):
                        self.assertTrue((output / (name + '-predictions.npz')).is_file())
                        self.assertTrue((output / (name + '-dictionary.json')).is_file())
                    self.assertTrue((output / 'OBS.json').is_file())
                    self.assertEqual(ids, list(allowed))
                    requested.extend(ids)
                    return np.ones(len(ids), dtype=np.int64)

                original = SimpleNamespace(
                    roles=dict(directions=directions), records=lambda: (records, rows),
                    model=lambda *args: (np.array([-.2]), {}, {}), provenance={},
                    obs_sha256=PLAN['r9_observation_sha256'],
                    evidence=SimpleNamespace(label_values=labels))
                write_json(output / 'executed-config.json', {})
                with patch.object(structure, 'fit_dictionary', side_effect=[paired, final_dictionary]):
                    obs = screen.freeze(output, PLAN, original)
                self.assertEqual(requested, [])
                _, verdict = screen.finish(output, obs, PLAN, original)
                self.assertEqual(verdict['status'], expected)
                self.assertEqual(requested, [2, 0] if expected == 'scored' else [])
                (output / 'ps1_to_0-dictionary.json').write_text('{}')
                with self.assertRaises(PermissionError):
                    screen.verify_frozen(output, obs, original)
