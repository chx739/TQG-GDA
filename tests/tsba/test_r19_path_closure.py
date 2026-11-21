import json
import unittest
from fractions import Fraction

import numpy as np

from experiments.mutagenicity.r19_分子路径闭合与分类迁移_20261121.src import path_closure as c, source_screen


def row(n, o, closed):
    return (n, 9, *o, *closed)


class PathClosureTests(unittest.TestCase):
    def test_class_mix_explains_prism_and_bipartite_reference(self):
        prism = row(6, [0, 42, 0, 30], [0, 30, 0, 18])
        bipartite = row(6, [0, 36, 0, 36], [0, 0, 0, 36])
        d = [prism, bipartite]
        u = [prism]*3 + [bipartite]
        p_s = c.rate(*c.moments(d), (0,)*4)
        p_u = c.rate(*c.moments(u), p_s)
        self.assertEqual(p_s, (0, Fraction(5, 13), 0, Fraction(9, 11)))
        self.assertEqual(p_u, (0, Fraction(5, 9), 0, Fraction(5, 7)))
        self.assertEqual(c.mixture(d, [1, 0], p_s, p_u, c.moments(u)[1], p_s),
                         (p_u, Fraction(3, 4), 0))
        self.assertEqual(c.mixture(d, [0, 0], p_s, p_u, c.moments(u)[1], p_s), (p_s, None, None))
        self.assertEqual(c.mixture(d, [1, 0], p_s, p_s, (0,)*4, p_s)[1], 0)

    def test_frozen_zero_rms_exchange_and_no_path(self):
        empty = row(2, [0]*4, [0]*4)
        path = row(6, [3, 0, 2, 0], [1, 0, 0, 0])
        closed = row(6, [3, 0, 2, 0], [3, 0, 2, 0])
        rows = dict(enumerate([path, path, empty, path, empty, closed]))
        roles = dict(D=[0], C=[1, 2], U=[3], E=[4, 5])
        model = c.fit('toy', rows, roles, np.array([0]), np.array([1, 0]), np.array([.1, -.2]))
        self.assertEqual(model['background']['rms'], [0.]*4)
        self.assertEqual(model['pS'], model['pU'])
        self.assertEqual(model['source_rate_origin'], ['D', 'zero', 'D', 'zero'])
        self.assertEqual(model['target_rate_origin'], ['U', 'pS', 'U', 'pS'])
        static = np.array([-.2, .3])
        frozen = json.loads(json.dumps(model))
        arrays = c.actions(rows, np.array([4, 5]), static, frozen)
        np.testing.assert_array_equal(arrays['margin_SourceBg'], arrays['margin_TargetBg'])
        np.testing.assert_array_equal(arrays['margin_TargetBg'], static)
        for arm in ('SourceBg', 'TargetBg', 'Raw', 'Mixture'):
            self.assertEqual(arrays[f'margin_{arm}'][0], static[0])
        self.assertTrue(all(np.isfinite(x).all() for x in arrays.values()))

    def test_source_reference_falls_back_to_all_d(self):
        empty = row(2, [0]*4, [0]*4)
        path = row(6, [3, 0, 0, 0], [1, 0, 0, 0])
        roles = dict(D=[7, 2], C=[1], U=[5], E=[3])
        first, = c.reference_ids('toy', roles)['D']
        second, = set(roles['D']) - {first}
        rows = {1: path, first: empty, second: path, 5: empty, 3: empty}
        m = c.fit('toy', rows, roles, np.array([1, 0]), np.array([1]), np.array([.3]))
        self.assertEqual(m['source_rate_origin'], ['D_all', 'zero', 'zero', 'zero'])
        self.assertEqual(m['target_rate_origin'], ['pS']*4)
        self.assertEqual(m['pS'], ['1/3', '0', '0', '0'])
        self.assertEqual(m['pU'], m['pS'])

    def test_gate_exact_half_point_and_strict_comparisons(self):
        score = dict.fromkeys(c.METHODS, Fraction(1, 2))
        score['TargetBg'] += Fraction(1, 200)
        groups = dict.fromkeys(range(4), Fraction(1, 99))
        groups[3] = 0
        self.assertTrue(source_screen.gate(score, groups)['advance'])
        score['Count'] = score['TargetBg']
        self.assertFalse(source_screen.gate(score, groups)['advance'])


if __name__ == '__main__':
    unittest.main()
