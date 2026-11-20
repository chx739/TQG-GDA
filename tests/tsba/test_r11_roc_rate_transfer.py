from fractions import Fraction
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np

from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import roc_rate as r, roc_score, roc_transfer


class ROCRateTests(TestCase):
    def test_balanced_accuracy_and_exact_endpoints(self):
        v = np.arange(1., 9.)
        y = np.array([0,0,1,1,0,1,0,1])
        roc = r.source_roc(v,y)
        point,q = r.choose(roc,Fraction(1,2))
        self.assertEqual((point,q),(dict(tp=4,fp=2,cut=2.),Fraction(3,4)))
        a = 3*v+1
        h = np.array([-1e6, 7., 8., 1e6])
        action, rec = r.transfer(roc,Fraction(1,2),a,h)
        self.assertEqual(rec['a_cut'],7.)
        np.testing.assert_array_equal(action,[False,False,True,True])
        point,q = r.choose(roc,Fraction(1))
        self.assertEqual((point['cut'],q),(2.,Fraction(1)))
        for prior,expected in ((Fraction(0),False),(Fraction(1),True)):
            action, _ = r.transfer(roc,prior,a,h)
            np.testing.assert_array_equal(action,np.full(len(h),expected))

    def test_monotone_transform_on_a_and_h_preserves_actions(self):
        v = np.array([-4.,-2.,-1.,0.,1.,2.])
        y = np.array([0,1,0,0,1,1])
        roc = r.source_roc(v,y)
        a = np.array([-8.,-3.,-3.,0.,1.,2.,2.])
        h = np.array([-9.,-3.,-.9,0.,1.,2.,13.])
        for prior in [Fraction(0),Fraction(2,5),Fraction(1,2),Fraction(1)]:
            original, before = r.transfer(roc,prior,a,h)
            new, after = r.transfer(roc,prior,2*a+np.arctan(a),2*h+np.arctan(h))
            np.testing.assert_array_equal(original,new)
            self.assertEqual(before['q'],after['q'])

    def test_ties_and_near_half_are_exact_and_single_class_fallback(self):
        equal = np.zeros(5)
        h = np.array([-100.,0.,100.])
        self.assertEqual(r.quantile(equal,Fraction(1,2)),(float('inf'),0))
        above = Fraction.from_float(float(np.nextafter(.5,1.)))
        self.assertEqual(r.quantile(equal,above),(-float('inf'),5))
        roc = r.source_roc(equal,np.array([0,0,1,1,1]))
        self.assertEqual(r.choose(roc,Fraction(1,2))[0]['cut'],'+inf')
        self.assertEqual(r.choose(roc,above)[0]['cut'],'-inf')
        self.assertIsNone(r.source_roc(equal,np.zeros(5)))
        actions, trace = roc_transfer.decisions(None,above,Fraction(1,2),equal,[],h,h,.5,1e-8)
        for arm in roc_transfer.ROC_ARMS:
            np.testing.assert_array_equal(actions[arm],h>0)
        self.assertEqual(trace['status'],'source_V_missing_class')

    def test_two_worlds_share_margin_distribution_prevalence_and_auroc(self):
        scores = np.repeat(np.arange(3,dtype=np.float64),100)
        def analyze(counts):
            y = np.concatenate([np.r_[np.ones(k,dtype=int),np.zeros(100-k,dtype=int)] for k in counts])
            roc = r.source_roc(scores,y)
            points = roc['points']
            area = sum(Fraction((q['fp']-p['fp'])*(p['tp']+q['tp']),2*roc['positives']*roc['negatives'])
                       for p,q in zip(points[:-1],points[1:],strict=True))
            _, q = r.choose(roc,Fraction(int(y.sum()),len(y)))
            return area, q, y.sum()
        self.assertEqual(analyze([12,66,72]),(Fraction(23,30),Fraction(2,3),150))
        self.assertEqual(analyze([28,34,88]),(Fraction(23,30),Fraction(1,3),150))

    def test_correct_source_target_mixture_selects_bayes_decision(self):
        scores = np.repeat(np.arange(3,dtype=np.float64),4)
        labels = np.array([0,0,0,0,0,0,0,1,1,1,1,1])
        roc = r.source_roc(scores,labels)
        prior = Fraction(int(labels.sum()),len(labels))
        action,_ = r.transfer(roc,prior,scores,scores)
        np.testing.assert_array_equal(action,scores>1.)
        self.assertEqual(int((action==labels).sum()),11)

    def test_incomplete_unlabeled_file_prevents_reading_published_a_counts(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            plan = dict(settings=['proteins_edge'],domains=[0,1],seeds=[3405])
            bound = plan,None,None,output
            with mock.patch.object(roc_transfer,'stage') as read_stage, self.assertRaises(FileNotFoundError):
                roc_transfer.diagnose(output,{},bound,{},'wrong')
            read_stage.assert_not_called()

    def test_setting_gate_exact_positive_direction_and_swaps(self):
        settings = ['mutagenicity_edge','mutagenicity_node','proteins_edge']
        domains = [0,1,2,3]
        plan = dict(settings=settings,domains=domains,seeds=[3405])
        args = dict(minimum_accuracy_fraction=[1,200],positive_directions=7)
        measured = {(s,a,b,3405,'RateROC',arm):Fraction(1,200)
                    for s in settings for a,b in permutations(domains,2)
                    for arm in (*roc_transfer.STRONG,*roc_score.GATE_CONTROLS)}
        self.assertTrue(roc_score.gate(plan,args,measured)['extend_to_reserve_seeds'])
        measured['proteins_edge',0,1,3405,'RateROC','WL'] = Fraction(0)
        result = roc_score.gate(plan,args,measured)
        self.assertEqual(result['settings']['proteins_edge']['strongest_source'],'WL')
        self.assertFalse(result['extend_to_reserve_seeds'])
