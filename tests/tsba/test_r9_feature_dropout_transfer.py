from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
from scipy.sparse import csr_matrix

from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import (
    feature_dropout_transfer as model, feature_dropout_transfer_scoring as scoring)


class FeatureDropoutReplayTests(TestCase):
    def test_weighted_ridge_matches_primal_and_no_intercept_penalty(self):
        x = np.array([[1.,0.,.3],[2.,2.,.1],[0.,1.,.5],[4.,3.,.1]])
        y = np.array([-1,1,-1,1])
        r = np.array([.5,0.,1.])
        pen = np.sum(x*x,axis=0)
        w,b = model.ridge(csr_matrix(x),y,r,pen,1.)
        z = np.c_[x[:,r>0],np.ones(len(x))]
        direct = np.linalg.solve(z.T@z+np.diag(np.r_[1.+pen[r>0]*(1/r[r>0]-1),0.]),z.T@y)
        np.testing.assert_allclose(w[r>0],direct[:-1],atol=1e-13)
        self.assertEqual(w[1],0.)
        self.assertAlmostEqual(b,direct[-1],places=13)

    def test_count_matching_preserves_original_id_tie_rule(self):
        ids = np.array([9,5,23,1,29])
        z = np.array([2.,2.,4.,2.,0.])
        np.testing.assert_array_equal(model.count_matched(ids,z,2),[0,0,1,1,0])
        np.testing.assert_array_equal(model.count_matched(ids,z,0),np.zeros(5))
        np.testing.assert_array_equal(model.count_matched(ids,z,5),np.ones(5))

    def test_fixed_feature_ratios_do_not_use_supplement_or_e_rows(self):
        word = (0,b'word')
        features = {0:{word:1.},1:{word:.5},2:{word:1.},3:{word:.5},4:{word:0.},5:{word:1.}}
        row = dict(roles=dict(D=[0,1],C=[2],U=[3,4],E=[5]),
                   target_reference=[3],source_reference=[2],supplement=[4],m=1)
        initial = model.fit(features,row,np.array([0,1]),np.array([1]))
        changed = dict(features)
        changed[4],changed[5] = {(0,b'unseen'):1.},{}
        replay = model.fit(changed,row,np.array([0,1]),np.array([1]))
        self.assertEqual(initial[0],replay[0])
        for method in model.ORIGINAL:
            np.testing.assert_array_equal(initial[1][method],replay[1][method])
        self.assertEqual(initial[2:],replay[2:])

    def test_supplement_is_only_unused_u_and_disjoint_from_all_initial_e(self):
        roles = {}
        for i in range(8):
            ids = [100*i+k for k in range(10)]
            roles[str(i)] = dict(setting='toy',roles=dict(D=ids[:2],C=ids[2:4],E=ids[4:6],U=ids[6:]))
        cfg = dict(supplement_counts={'toy':16})
        ref = SimpleNamespace(roles=roles)
        obtained = model.scope(ref,cfg)
        for key,row in obtained.items():
            self.assertEqual(set(row['supplement']),set(roles[key]['roles']['U'])-set(row['target_reference']))
            self.assertEqual(row['supplement'],sorted(row['supplement']))
        roles['1']['roles']['E'][0] = obtained['0']['supplement'][0]
        with self.assertRaises(PermissionError):
            model.scope(ref,cfg)

    def test_protein_gate_requires_both_size_baselines_and_omission_robustness(self):
        cfg = dict(minimum_accuracy_fraction=[1,200])
        arms = ('Static','Source','Perm','Count',*model.PROTEIN)
        rows = [dict(setting='proteins_edge',right=m,accuracy_exact='1/200') for m in arms]
        leave = [dict(setting='proteins_edge',accuracy_exact='1/100') for _ in range(4)]
        self.assertEqual(scoring.decide(cfg,rows,leave)['eligibility'],'consider_real_proteins_only')
        rows[-1]['accuracy_exact']='0'
        self.assertEqual(scoring.decide(cfg,rows,leave)['eligibility'],'local_effect_only')
        rows[0]['accuracy_exact']='0'
        self.assertEqual(scoring.decide(cfg,rows,leave)['eligibility'],'stop_fixed_dropout')

    def test_no_supplement_truth_until_replay(self):
        with TemporaryDirectory() as tmp,mock.patch.object(model,'observed',side_effect=PermissionError), \
                mock.patch.object(scoring.data,'ScopedLabels') as labels:
            p = Path(tmp)
            with self.assertRaises(PermissionError):
                scoring.score(p,p,None,{},'bad',{})
            labels.assert_not_called()
            self.assertEqual(list(p.iterdir()),[])
