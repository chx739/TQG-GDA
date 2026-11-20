"""Generated checks for KRR, scope and subtraction of matched controls."""
from fractions import Fraction
from itertools import permutations
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mutagenicity.r16_目标图结构与监督分数蒸馏_20261121.src import models, scoring


class R16Tests(unittest.TestCase):
    def test_linear_kernel_agrees_with_explicit_primal_and_zero_kernel_queries(self):
        x=np.array([[0.,1.],[1.,0.],[2.,4.],[3.,1.]])
        y=np.array([1.,2.,5.,3.])
        q=np.array([[3.,-2.],[0.,0.]])
        center=x-x.mean(0)
        w=np.linalg.solve(center.T@center+np.eye(2),center.T@(y-y.mean()))
        expected=y.mean()+(q-x.mean(0))@w
        fitted=models.fit_ridge(x@x.T,y,1e-10)
        np.testing.assert_allclose(models.apply_ridge(q@x.T,fitted),expected,atol=1e-13,rtol=0.)
        self.assertNotAlmostEqual(float(expected[1]),float(y.mean()))

    def test_centre_reproduces_unpenalized_intercept(self):
        k=np.full((3,3),2.)
        y=np.array([-3.,-1.,1.])
        model=models.fit_ridge(k,y,1e-10)
        np.testing.assert_allclose(models.apply_ridge(np.full((4,3),16.),model),np.full(4,y.mean()))

    def test_uniform_residual_permutation_expectation_is_zero(self):
        x=np.array([[.5,1.],[.7,0.],[-1.,3.]])
        y=np.array([-3.,-1.,4.])
        train=models.rbf(x,x)
        query=models.rbf(x+1.,x)
        rows=[models.apply_ridge(query,models.fit_ridge(train,np.asarray(perm),1e-10))
              for perm in permutations(y)]
        np.testing.assert_allclose(np.mean(rows,axis=0),0.,atol=1e-14)

    def test_full_wl_keeps_unknown_words_in_query_denominator(self):
        word=sha256(b'known').digest()
        unknown=sha256(b'unseen').digest()
        ref=[{(0,word):1.}]
        query=[{(0,word):.6,(0,unknown):.8}]
        lexicon=models.vocabulary(ref)
        self.assertNotIn((0,unknown),lexicon)
        actual=(models.sparse(query,lexicon)@models.sparse(ref,lexicon).T).toarray()
        np.testing.assert_array_equal(actual,[[.6]])

    def test_controls_share_own_low_and_zero_wg_is_identity(self):
        a=lambda u:np.array([float(u)])
        b,g=a(1),a(2)
        own=(a(4),a(10),a(20))
        v=(a(1000),a(30),a(300))
        other=[(a(2000),a(40),a(400)),(a(3000),a(50),a(500))]
        result=models.compose(b,g,2.,own,v,other,.5)
        target={'Low':3,'Own':13,'LowTwice':23,'VResidual':33,'OtherResidual1':43,
                'OtherResidual2':53,'VFull':1029}
        for name,value in result.items():
            self.assertEqual(float(value[0]),target[name])
        for value in models.compose(b,g,0.,own,v,other,.5).values():
            np.testing.assert_array_equal(value,b)

    def test_quadratic_and_constant_scale(self):
        x=np.array([[1.,2.,3.],[1.,4.,5.]])
        transformed=models.apply_standardizer(x,models.standardizer(x))
        np.testing.assert_array_equal(transformed[:,0],0.)
        np.testing.assert_array_equal(models.quadratic(x)[0],[1.,2.,3.,4.,6.,9.])

    def test_gate_requires_exact_positive_other_and_proteins(self):
        plan=dict(settings=['mutagenicity_edge','mutagenicity_node','proteins_edge'],domains=list(range(4)))
        cfg=dict(phase1_seed=3405,minimum_accuracy_fraction=[1,200],minimum_positive_directions=7)
        results={(s,a,b,3405,'Own',arm):Fraction(1,200) for s in plan['settings'] for a in plan['domains']
                 for b in plan['domains'] if a!=b for arm in scoring.GATES}
        self.assertTrue(scoring.gate(plan,cfg,results)['extend'])
        for k in results:
            if k[0]=='proteins_edge' and k[-1]=='OtherResidual':
                results[k]=Fraction(1 if k[1]<k[2] else -1,307)
        self.assertFalse(scoring.gate(plan,cfg,results)['extend'])

    def test_averaged_other_is_paired_inside_the_same_draw(self):
        actions=dict(Own=np.array([1,0,1]),OtherResidual1=np.array([0,0,1]),OtherResidual2=np.array([1,1,0]))
        flags={k:v==[1,0,1] for k,v in actions.items()}
        ordered=np.array([[.2,.7,.1],[.1,.1,.8]])
        samples={k:ordered@v for k,v in flags.items()}
        v,sample,row=scoring.contrast('Own','OtherResidual',actions,flags,samples)
        self.assertEqual(v,Fraction(1,2))
        self.assertEqual(row['changed_exact'],'3/2')
        np.testing.assert_allclose(sample,samples['Own']-np.mean(
            [samples['OtherResidual1'],samples['OtherResidual2']],axis=0))

    def test_missing_fit_never_constructs_label_session(self):
        with TemporaryDirectory() as tmp, patch.object(scoring.r11,'GraphLabelSession') as labels, patch.object(
            scoring.inputs,'Historical') as archive:
            archive.return_value.plan=dict(settings=['g'],domains=[0,1])
            with self.assertRaises(FileNotFoundError):
                scoring.score(Path(tmp),Path(tmp),{'structural_fit_sha256':'s','r14_fit_sha256':'v'},
                              [3405],'fit','obs',pilot=True)
            labels.assert_not_called()
