from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np

from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import strong_rank_swaps as swaps


class StrongRankSwapTests(TestCase):
    def test_four_strata_reconcile_exactly(self):
        e,c,b = (np.array(q) for q in ([1,1,0,0,0],[0,0,1,1,0],[0,1,1,0,1]))
        masks,counts = swaps.membership(e,c,b)
        self.assertEqual(counts,(1,1,1,1))
        self.assertEqual([np.flatnonzero(q).tolist() for q in masks],[[0],[2],[1],[3]])

    def test_swap_expectation_agrees_with_exhaustive_equal_mass_draws(self):
        b,y = (np.array(q) for q in ([0,0,0,1,1,1,1],[1,0,1,0,1,0,1]))
        start = swaps.roc_score.confusion(b,y)
        samples = []
        for add in combinations(range(3),2):
            for drop in combinations(range(3,7),2):
                z = b.copy()
                z[list(add)],z[list(drop)] = 1,0
                self.assertEqual(int(z.sum()),int(b.sum()))
                decision = z==y
                conf = swaps.roc_score.confusion(z,y)
                samples.append((*conf, np.sum(decision & (b!=y)), np.sum(~decision & (b==y)),
                                swaps.roc_score.f1(conf)))
        expected = [swaps.mean(map(F,column)) for column in zip(*samples,strict=True)]
        conf,good,bad = swaps.swap_expectation(start,3,2,4,2,2)
        self.assertEqual((*conf,good,bad,swaps.roc_score.f1(conf)),tuple(expected))
        self.assertEqual(good-bad,F(2,3))
        self.assertEqual(swaps.swap_expectation(start,3,2,0,0,0),(tuple(map(F,start)),F(0),F(0)))

    def test_common_budget_comparison_and_direction_weighting(self):
        arms = {}
        for head,delta in [('Own','1/5'),('Other1','1/10'),('Other2','-1/5')]:
            arms[head] = {}
            for k,d in [('m',F(delta)),('L',F(delta)/2)]:
                arms[head][k] = dict(n=10,budget=1 if k=='L' else 2,delta_accuracy_exact=str(d),
                                   accuracy_exact=str(F(1,2)+d),delta_macro_f1_exact=str(d),
                                   macro_f1_exact=str(F(3,5)+d))
        got = swaps.effect_row({},arms)
        self.assertEqual([F(r['accuracy_exact']) for r in got],[F(1,5),F(1,4),F(1,8)])
        dirs = swaps.directions([dict(source=0,target=1,v=i) for i in (0,1)]+
                               [dict(source=1,target=0,v=0)],lambda r:F(r['v']))
        self.assertEqual(swaps.mean(dirs.values()),F(1,4))

    def test_score_refuses_unverified_obs_before_opening_h(self):
        with TemporaryDirectory() as tmp,mock.patch.object(swaps.prior_run,'verified_members',return_value={
                'blocks.csv':b'','graphs.csv':b''}),mock.patch.object(swaps,'authorized_h') as labels:
            out = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                swaps.score(out,out,None,None,({},{},{}),{},dict.fromkeys(swaps.PARENTS,'sha'),'sha',{})
            labels.assert_not_called()
