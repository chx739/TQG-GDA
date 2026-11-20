from itertools import permutations
from unittest import TestCase

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import paired_action_support as p


def graph(nodes, arcs, *, typed=True):
    edges = torch.tensor([(s, t) for s, t, _ in arcs], dtype=torch.long).reshape(-1, 2).T.contiguous()
    kinds = torch.as_tensor([k for _, _, k in arcs], dtype=torch.long)
    return Data(x=torch.eye(3)[nodes], edge_index=edges, edge_attr=torch.eye(2)[kinds] if typed else None)


def count(qs, rs, dz, da):
    return p.count_queries(qs, rs, np.array(dz, dtype=np.float64), np.array(da, dtype=np.float64))


class PairedActionSupportTests(TestCase):
    def test_one_type_change_survives_all_relabelings(self):
        nodes = [0, 2, 0, 1]
        arcs = [(0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0), (2, 2, 1), (0, 2, 0), (2, 0, 0)]
        base = p.signature(graph(nodes, arcs), 3, 2)
        for old_to_new in permutations(range(4)):
            for i in range(4):
                new_nodes = [0] * 4
                for old, new in enumerate(old_to_new):
                    new_nodes[new] = (nodes[old] + (old == i)) % 3
                new_arcs = [(old_to_new[s], old_to_new[t], k) for s, t, k in reversed(arcs)]
                ref = p.signature(graph(new_nodes, new_arcs), 3, 2)
                self.assertEqual(base[0], ref[0])
                self.assertEqual(count([base], [ref], [1.], [-2.]),
                                 dict(structural=1, opposite=1, can_flip=1))

    def test_loop_parallel_type_and_isolates_affect_signature(self):
        arcs = [(0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0), (2, 2, 1)]
        base = p.signature(graph([0, 0, 0, 0], arcs), 3, 2)[0]
        for nodes, edges in (
                ([0, 0, 0], arcs), ([0, 0, 0, 0], arcs[2:]), ([0, 0, 0, 0], arcs[:-1]),
                ([0, 0, 0, 0], arcs[:-1] + [(2, 2, 0)]),
                ([0, 0, 0, 0], [(s, t, 1) for s, t, _ in arcs])):
            self.assertNotEqual(base, p.signature(graph(nodes, edges), 3, 2)[0])
        empty = p.signature(graph([0, 1], [], typed=False), 3, 0)
        self.assertEqual(empty[0][1], ((0, 0, 0), (0, 0, 0)))

    def test_union_of_possible_actions_and_strict_zero(self):
        q = p.signature(graph([0, 0], []), 3, 2)
        r = p.signature(graph([0, 1], []), 3, 2)
        ignored = p.signature(graph([1, 1], []), 3, 2)
        self.assertEqual(count([q]*3, [r, ignored, r], [-1., 1., 0.], [1., 100., -1.]),
                         dict(structural=3, opposite=3, can_flip=2))
        self.assertEqual(count([q], [r, r], [0.], [0., -0.]),
                         dict(structural=1, opposite=0, can_flip=0))
        tiny = np.nextafter(0., 1.)
        self.assertEqual(count([q], [r], [-tiny], [2*tiny])['can_flip'], 1)
        self.assertEqual(count([q], [], [0.], [])['can_flip'], 0)
        with self.assertRaises(ValueError):
            p.count_queries([q], [r], np.array([0.], dtype=np.float32), np.array([0.]))

    def test_nonisomorphic_graph_can_pass_the_loose_filter(self):
        a = [(i, (i+1) % 6, 0) for i in range(6)]
        b = [(i, offset+(i-offset+1) % 3, 0) for offset in (0, 3) for i in range(offset, offset+3)]
        q = p.signature(graph([0]*6, a), 3, 2)
        r = p.signature(graph([0]*5+[1], b), 3, 2)
        self.assertEqual(count([q], [r], [-2.], [3.])['can_flip'], 1)

    def test_equal_direction_gate_at_exact_threshold(self):
        cfg = dict(settings=list(p.WIDTHS), minimum_flip_fraction=[1, 200])
        rows = [dict(setting=s, source=u, target=v, q_graphs=n, r_graphs=1, structural=1,
                     opposite=1, can_flip=int(s != 'mutagenicity_node'))
                for s in cfg['settings'] for u, v in permutations(range(4), 2) for n in [200]]
        _, gate = p.summarize(rows, cfg)
        self.assertTrue(gate['proceed_to_exact_pairing'])
        self.assertFalse(gate['setting_passes']['mutagenicity_node'])
        rows[-1]['q_graphs'] = 201
        _, gate = p.summarize(rows, cfg)
        self.assertFalse(gate['proceed_to_exact_pairing'])
