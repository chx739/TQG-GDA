from fractions import Fraction as F
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np

from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import (
    size_exposure as curve, size_exposure_observation as obs, size_exposure_scoring as scoring)


class SizeExposureTests(TestCase):
    def test_mle_score_and_constants(self):
        n = np.array([2,5,13,21])
        y = np.array([1,1,0,0])
        fit = curve.fit(n,y)
        self.assertTrue(fit['converged'])
        self.assertLess(abs(fit['derivative']),1e-12)
        p = curve.predict(n,fit)
        self.assertTrue(np.all(np.diff(p)<0))
        for label in (0,1):
            record = curve.fit(n,np.full(len(n),label))
            np.testing.assert_array_equal(curve.predict(n,record),np.full(len(n),label))
            self.assertEqual(curve.exposure_prior(n,record),F(label))

    def test_true_half_threshold_with_unequal_sizes_and_only_fit_nodes(self):
        n = np.array([5,5,10,10,10])
        y = np.array([1,1,0,0,0])
        fit = curve.fit(n,y)
        self.assertEqual(curve.threshold_sign(5,fit),0)
        self.assertEqual(curve.predict(np.array([5]),fit)[0],.5)
        self.assertEqual(curve.exposure_prior(np.array([5,5]),fit),F(1,2))
        self.assertEqual([curve.threshold_sign(q,fit) for q in (4,6)],[1,-1])
        fixed = curve.fit(np.array([33]*8),np.array([1]+[0]*7))
        self.assertEqual(curve.predict(np.array([11]),fixed)[0],.5)
        self.assertEqual(curve.exposure_prior(np.array([11,11]),fixed),F(1,2))

    def test_half_pool_is_order_invariant_and_bracketed_by_true_non_ties(self):
        m = curve.fit(np.array([1]*5),np.array([1,1,1,0,0]))
        a = np.array([1]*7+[2]*5)
        for v in (a,a[::-1],np.roll(a,4)):
            self.assertEqual(curve.exposure_prior(v,m),F(1,2))
        self.assertGreater(curve.exposure_prior(np.r_[a,1],m),F(1,2))
        self.assertLess(curve.exposure_prior(np.r_[a,2],m),F(1,2))
        mixed = curve.fit(np.array([21,1]+[2]*7),np.array([1]+[0]*8))
        a = np.array([1]*5+[3]*16)
        self.assertEqual(curve.mixed_pool_sign([1,3],[5,16],mixed),0)
        self.assertEqual(curve.exposure_prior(a,mixed),F(1,2))
        self.assertEqual(curve.mixed_pool_sign([1,3],[6,16],mixed),1)
        self.assertEqual(curve.mixed_pool_sign([1,3],[4,16],mixed),-1)

    def test_donor_changes_one_input_and_keeps_a_ties(self):
        a = {0:np.array([-2.,1.,1.,3.]),2:np.array([0.,4.,4.,8.]),3:np.array([-9.,-2.,0.,1.])}
        p = {0:F(1,2),2:F(1),3:F(0)}
        h = np.array([-1.,0.,1.,3.,5.])
        acts,notes = curve.policies(0,p,a,h,F(1,2),np.full(len(h),.5),dict(positive=2,n_source=4))
        self.assertFalse(acts['CurvePool'].any())
        self.assertFalse(acts['Curve'].any())
        self.assertEqual(notes['CurveDirectP']['a_positive'],1)
        self.assertEqual(notes['CurveDirectP']['cut'],1.)
        self.assertTrue(acts['PriorOther1'].all())
        self.assertFalse(acts['PriorOther2'].any())
        self.assertEqual(notes['PriorOther1']['reference'],0)
        self.assertEqual(notes['PriorOther1']['prior_donor'],2)
        self.assertEqual(notes['QuantileOther1']['reference'],2)
        self.assertEqual(notes['QuantileOther1']['p'],'1/2')

    def test_new_prior_preserves_existing_float64_directp_semantics(self):
        fit = curve.fit(np.ones(6,dtype=np.int64),np.array([1,1,1,1,1,0]))
        p = curve.exposure_prior(np.ones(3,dtype=np.int64),fit)
        self.assertEqual(p,F.from_float(5/6))
        self.assertEqual(curve.roc_rate.quantile(np.array([0.,1.,2.]),p),(-np.inf,3))
        self.assertEqual(curve.roc_rate.quantile(np.array([0.,1.,2.]),F(5,6)),(0.,2))

    def test_best_fixed_scope_and_exchange_gate_are_strict(self):
        cfg = dict(minimum_accuracy_fraction=[1,200],minimum_positive_directions=7)
        rows = [dict(scope='reserve',left='CurveDirectP',right=m,wins=7,accuracy_exact='1/200')
                for m in (*scoring.BEST,*scoring.SWAPS)]
        self.assertTrue(scoring.gate(cfg,rows)['advance'])
        rows[-1]['accuracy_exact'] = '0'
        self.assertFalse(scoring.gate(cfg,rows)['advance'])

    def test_invalid_observation_never_releases_h_labels(self):
        with TemporaryDirectory() as tmp,mock.patch.object(obs,'observed',side_effect=PermissionError), \
                mock.patch.object(scoring,'cohorts') as read:
            p = Path(tmp)
            with self.assertRaises(PermissionError):
                scoring.score(p,p,None,{},{},'bad',{})
            read.assert_not_called()
            self.assertEqual(list(p.iterdir()),[])
