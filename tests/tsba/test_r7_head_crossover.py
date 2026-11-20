from fractions import Fraction as F
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np

from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import (
    head_crossover as cross, head_crossover_scoring as score, paired)
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import roc_score


class HeadCrossoverTests(TestCase):
    def test_equal_count_breaks_boundary_ties_by_original_id(self):
        ids=np.array([45,9,3,21])
        z=np.array([1.,0.,0.,0.])
        a,tie=cross.equal_count(ids,z,2)
        np.testing.assert_array_equal(a,[1,0,1,0])
        self.assertEqual(tie,dict(tied=3,tied_selected=1))
        for k in (0,1,4):
            self.assertEqual(cross.equal_count(ids,z,k)[1],dict(tied=0,tied_selected=0))

    def test_cross_projects_donor_on_fixed_h_and_fails_back_before_count_matching(self):
        raw=np.array([[0.,-1.],[0.,0.],[0.,1.]])
        u=np.array([[1.,1.],[2.,1.],[3.,1.]])
        zero=np.zeros((2,2))
        donor=np.array([[0.,0.],[0.,1.25]])
        valid=lambda v:{m:dict(valid=v) for m in cross.HEADS}
        diag,_=cross.project(raw,u,zero,True)
        b0=diag.argmax(1)
        group={1:dict(z=raw,u=u,ids={'holdout':[22,10,31]},valid=valid(True),
                      archived={f'holdout_logits_{m}':diag for m in ('B0',*cross.HEADS)},
                      ws={f'w_{m}':zero for m in cross.HEADS}),
               2:dict(valid=valid(True),ws={f'w_{m}':donor for m in cross.HEADS}),
               3:dict(valid=valid(False),ws={f'w_{m}':donor for m in cross.HEADS})}
        actions,table=cross.group_actions(group,1,{'B0':b0})
        for m in cross.HEADS:
            np.testing.assert_array_equal(actions[cross.head_name(m,2)],[1,1,1])
            np.testing.assert_array_equal(actions[cross.head_name(m,3)],b0)
            np.testing.assert_array_equal(actions[cross.head_name(m,3,count=True)],b0)
            self.assertEqual(table[cross.head_name(m,1)]['diagonal_margin_error'],0.)
            self.assertTrue(table[cross.head_name(m,3)]['fallback'])

    def test_both_metrics_reuse_target_class_draws(self):
        y=np.array([1,0,1,0,0])
        ids=np.array([22,19,5,38,11])
        order=np.argsort(ids)
        c=paired._cohort('fixture',0,ids[order],y[order],100,123,1)
        actions={'E':np.array([1,0,0,0,1]),'J':np.array([0,0,0,0,0])}
        cells,point,draws=score.metric_tables(actions,y,c['weights'],order)
        self.assertEqual(cells['E'],(2,1,1,1))
        self.assertEqual(point['E'][0],F(3,5))
        for m,p in actions.items():
            np.testing.assert_allclose(draws[m][:,0],c['weights']@(p[order]==y[order]),atol=1e-14)
            np.testing.assert_allclose(draws[m][:,1],roc_score.f1_draws(p[order],y[order],c['weights']),atol=1e-14)

    def test_foreign_mean_gains_and_harms_use_same_two_pairs(self):
        y=np.array([0,1,1,0])
        actions={'own':np.array([0,1,0,0]),'r1':np.array([1,1,1,0]),'r2':np.array([1,0,1,0])}
        flags={k:a==y for k,a in actions.items()}
        self.assertEqual(score.paired_counts(('own',),('r1','r2'),actions,flags),(F(3,2),F(1),F(5,2)))

    def test_aggregate_balances_directions_with_unequal_seed_counts(self):
        setting='proteins_edge'
        roles={'a':dict(setting=setting,source=0,target=1,complete=True),
               **{str(i):dict(setting=setting,source=0,target=2,complete=True) for i in range(5)}}
        points={k:dict(absolute={},contrasts=dict(EMatch=((F(1),F(1)) if k=='a' else (F(0),F(0)),
                np.array([[1.,1.]]) if k=='a' else np.zeros((1,2)),(F(0),F(0),F(0),F(1)))),
                exact_E_match=(np.array([int(k=='a')]),1))
                for k in roles}
        lines=score.aggregate(dict(settings=[setting]),roles,points,[2.5,97.5])
        row=next(r for r in lines if r['scope']=='complete' and r['source'] is None)
        self.assertEqual((row['blocks'],row['directions'],row['accuracy_pp'],row['accuracy_lo_pp']),(6,2,50.,50.))

    def test_gate_requires_exact_match_strict_intervals_and_b0(self):
        opts=dict(settings=['mutagenicity_edge','mutagenicity_node','proteins_edge'],minimum_match_accuracy_fraction=[1,200])
        rows=[dict(setting=s,scope='complete',kind='contrasts',source=None,
                   name=m,accuracy_pp=.5,accuracy_exact='1/200',accuracy_lo_pp=.01,accuracy_hi_pp=1.,accuracy_lo_exact='1/10000')
              for s in opts['settings'] for m in ('EMatch','EOwnMinusB0')]
        self.assertTrue(score.gate(rows,opts)['continue_cross_setting_target_match'])
        rows[-2]['accuracy_lo_pp']=5.55e-17
        rows[-2]['accuracy_lo_exact']='0'
        self.assertFalse(score.gate(rows,opts)['continue_cross_setting_target_match'])

    def test_exact_accuracy_interval_keeps_zero_bootstrap_lower_endpoint(self):
        y=np.array([0]*10+[1]*11)
        ids=np.arange(21)
        own=np.array(list(map(int,'101011000011101001101')))==y
        foreign=[np.array(list(map(int,string)))==y for string in
                 ('101011011011000001000','010100000110001100000')]
        c=paired._cohort('toy',1,ids,y,2000,20261119,10)
        counts=score.integer_draws(c['weights'],len(y))
        curve=score.exact_match_curve(own,foreign,counts,ids)
        low,high=score.exact_accuracy_interval([curve],[2.5,97.5])
        self.assertEqual(low,F(0))
        self.assertGreater(high,F(0))
        np.testing.assert_allclose(curve[0]/curve[1],
            c['weights']@own.astype(float)-.5*(c['weights']@foreign[0]+c['weights']@foreign[1]),atol=1e-14)

    def test_no_label_reader_before_verified_complete_obs(self):
        cfg=dict(phase1_seed=3405,reserve_seeds=[],settings=['proteins_edge'],domains=[0,1],
                 r7_final_digest='old',r10_obs_sha256='r10',r11_obs_sha256='r11')
        with TemporaryDirectory() as directory,mock.patch.object(score.inputs,'GraphLabelSession') as labels:
            with self.assertRaises(FileNotFoundError):
                score.score(Path(directory),None,cfg,None,'missing')
            labels.assert_not_called()
