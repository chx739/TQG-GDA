from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np

from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import (
    centered_target_update as centered, strong_anchor_scoring as scoring)


class CenteredTargetTests(TestCase):
    def test_subtract_weights_before_the_held_projection(self):
        old = {'WL_Full':np.array([1e16,1.]), 'WL_CalTeacher':np.array([1e16,0.])}
        h = np.array([[1.,1.]])
        self.assertEqual(float((h@old['WL_Full']-h@old['WL_CalTeacher'])[0]),0.)
        np.testing.assert_array_equal(h@centered.teacher_difference(old,'WL'),[1.])

    def test_local_foreign_reversed_and_equal_count(self):
        role = dict(target=0,reference_targets=[0,1,3],setting='toy',source=2,seed=3405)
        h = np.array([[1.,1.],[-2.,1.],[0.,1.]])
        x = {'ids':np.array([20,9,2],dtype=np.int64)}
        for a in ('WL','Size'):
            x[f'margin_{a}'] = np.array([-.5,-.5,.25])
            x[f'margin_{a}_Full_A0'] = np.array([2.,-1.,1.])
            x[f'margin_{a}_CalTeacher'] = np.array([4.,1.,1.])
        group = {i:{a:w for a in ('WL','Size')} for i,w in {
            0:np.array([1.,0.]),1:np.array([0.,1.]),3:np.zeros(2)}.items()}
        y = centered.predict(role,x,h,group)
        np.testing.assert_array_equal(y['WL_Centered_A0'],[1,0,1])
        np.testing.assert_array_equal(y['WL_Reversed'],[0,1,1])
        np.testing.assert_array_equal(y['WL_Centered_A1'],[1,1,1])
        np.testing.assert_array_equal(y['WL_Centered_A3'],y['WL'])
        np.testing.assert_array_equal(y['WL_EqualCount'],[0,1,1])
        _,_,_,paired = scoring.score_block(role,y,np.array([1,0,1]),active='Centered',
                         make_views=centered.aliases,comparisons=centered.COMPARISONS)
        self.assertEqual(paired['WL','OtherMean']['acc'],Fraction(1,3))
        self.assertEqual(paired['WL','EqualCount']['corrected'],2)

    def test_primary_and_equal_count_gates_use_exact_fraction(self):
        anchors = {'mutagenicity_edge':'WL','mutagenicity_node':'WL','proteins_edge':'Size'}
        cfg = dict(primary_anchors=anchors,minimum_accuracy_fraction=[1,200],minimum_positive_directions=7)
        rows = [dict(setting=s,anchor=a,right=m,accuracy_exact='1/200',positive=7,tied=2,negative=3)
                for s,a in anchors.items() for m in centered.COMPARISONS]
        self.assertTrue(centered.gate(rows,cfg)['continue_remaining_seeds'])
        p = lambda m: next(r for r in rows if r['setting']=='proteins_edge' and r['right']==m)
        p('EqualCount')['accuracy_exact']='0'
        g=centered.gate(rows,cfg)
        self.assertTrue(g['passes_cross_setting_accuracy'])
        self.assertFalse(g['continue_remaining_seeds'])
        p('Anchor')['accuracy_exact']=str(Fraction(1,200)-Fraction(1,10**30))
        self.assertFalse(centered.gate(rows,cfg)['passes_cross_setting_accuracy'])

    def test_no_label_reads_after_invalid_obs(self):
        cfg=dict(phase1_seed=3405,settings=['proteins_edge'],domains=[0,1,2,3])
        with TemporaryDirectory() as tmp, mock.patch.object(centered.holdout_score,'read_h_cohorts') as port:
            out=Path(tmp)
            with self.assertRaises(FileNotFoundError):
                centered.score(out,out,out,dict(anchor_obs_sha256='old'),cfg,None,'bad',execution={})
            port.assert_not_called()
            self.assertFalse((out/'score-execution.json').exists())
