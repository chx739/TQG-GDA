from fractions import Fraction
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np

from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import (
    head_crossover_scoring as scoring, transfer_flips as transfer)


class TransferFlipsTests(TestCase):
    def test_all_binary_combinations_partition_between_transferred_and_already_corrected(self):
        b0,e,base,y = np.array(list(product((0,1), repeat=4))).T
        new = transfer.flip(b0, e, base)
        chosen = (e!=b0) & (b0==base)
        np.testing.assert_array_equal(new, np.where(chosen,e,base))
        parts = transfer.partition(b0,e,base,new,y)
        self.assertEqual([(r['flips'],r['corrected'],r['harmed']) for r in parts], [(2,1,1)]*4)

    def test_net_gain_to_b0_can_all_come_from_cases_base_already_corrects(self):
        b0,e,base,y = np.array([[0,0,0],[1,1,1],[1,1,0],[1,1,0]])
        moved = transfer.flip(b0,e,base)
        parts = transfer.partition(b0,e,base,moved,y)
        self.assertEqual([r['corrected']-r['harmed'] for r in parts], [-1,2,0,0])
        self.assertEqual(int((moved==y).sum())-int((base==y).sum()),-1)

    def test_two_alternative_donors_use_shared_draw_and_exact_zero(self):
        flags = {k:np.array(a, dtype=bool) for k,a in
                 {'o':[0,1,1], 'r1':[1,0,1], 'r2':[0,1,0]}.items()}
        counts = np.array([[0,0,3],[1,1,1],[2,1,0]])
        for alternatives,num in ((('r1',),[0,0,-1]), (('r1','r2'),[3,1,-1])):
            curve = transfer.exact_curve(('o',),alternatives,flags,counts,np.arange(3))
            np.testing.assert_array_equal(curve[0],num)
            self.assertEqual(curve[1],3*len(alternatives))
        zero_curve = transfer.exact_curve(('o',),('o',),flags,counts,np.arange(3))
        self.assertEqual(scoring.exact_accuracy_interval([zero_curve],[2.5,97.5]),[0,0])

    def test_gate_obeys_exact_cutoff_zero_lower_and_strict_matching(self):
        plan = dict(primary_bases={'mutagenicity_edge':'WL','mutagenicity_node':'WL','proteins_edge':'Size'},
                    minimum_accuracy_fraction=[1,200])
        summary = [dict(setting=s,scope='complete',kind='contrasts',name=f'{b}OwnMinus{arm}',source=None,
                        accuracy_exact='1/200',accuracy_lo_exact='1/400',accuracy_pp=.5,
                        accuracy_lo_pp=.25,accuracy_hi_pp=.75)
                   for s,b in plan['primary_bases'].items() for arm in ('Base','OtherMean')]
        self.assertTrue(transfer.gate(summary,plan)['continue_cross_setting_transfer'])
        summary[-2].update(accuracy_lo_exact='0',accuracy_lo_pp=1e-18)
        self.assertFalse(transfer.gate(summary,plan)['continue_cross_setting_transfer'])
        summary[-2].update(accuracy_lo_exact='1/400')
        summary[-1].update(accuracy_exact='0',accuracy_pp=1e-18)
        self.assertFalse(transfer.gate(summary,plan)['continue_cross_setting_transfer'])
        summary[-1].update(accuracy_exact='1/200')
        summary[-2].update(accuracy_exact=str(Fraction(1,200)-Fraction(1,10**30)))
        self.assertFalse(transfer.gate(summary,plan)['continue_cross_setting_transfer'])

    def test_h_reader_cannot_run_until_new_obs_is_verified(self):
        with TemporaryDirectory() as td, mock.patch.object(scoring,'observed',return_value=({}, {}, {})), \
                mock.patch.object(scoring,'read_h_cohorts') as read:
            out = Path(td)
            with self.assertRaises(FileNotFoundError):
                transfer.score(out, out, 'parent', out, None, {}, {}, 'missing')
            read.assert_not_called()
