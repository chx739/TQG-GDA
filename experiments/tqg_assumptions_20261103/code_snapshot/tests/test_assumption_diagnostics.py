import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from tqg_gda.assumption_diagnostics import (ProbeConfig, fit_grid, logistic_objective,
    standardize, split_folds, metrics, ProbeLabelBoundary, support_statistics)


class AssumptionChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rng=np.random.default_rng(11)
        self.x=self.rng.normal(size=(12,3));self.ex=self.rng.normal(size=(4,3))
        self.y=np.arange(12)%2;self.p=np.linspace(.1,.9,12);self.ep=np.full(4,.5)
        self.k=np.exp(-((self.x[:,None]-self.x[None,:])**2).sum(2))
        self.ek=np.exp(-((self.ex[:,None]-self.x[None,:])**2).sum(2))
        self.q=np.array([1.,-1.])/np.sqrt(2)
        self.cfg=replace(ProbeConfig(),steps=5,hidden=4)
    def fit(self,family,ys,penalties,**kw):
        return fit_grid(family,self.x,ys,self.p,self.ex,self.ep,self.k,self.ek,self.q,penalties,71,self.cfg,**kw)
    def test_krr_contrast_probability_recovery(self):
        p,_,_=self.fit('KRR',self.y[None,:],[.1])
        explicit=self.ep+self.ek@np.linalg.solve(self.k+.1*np.eye(12),self.y-self.p)
        np.testing.assert_allclose(p[0,0],explicit,atol=1e-12)
    def test_logistic_gradient_finite_difference(self):
        w=np.array([.1,.2,-.1,.3]);_,g=logistic_objective(w,self.x,self.y,.3)
        h=1e-6;direct=[]
        for i in range(4):
            delta=np.zeros(4);delta[i]=h
            direct.append((logistic_objective(w+delta,self.x,self.y,.3)[0]-logistic_objective(w-delta,self.x,self.y,.3)[0])/(2*h))
        np.testing.assert_allclose(g,direct,atol=1e-8)
    def test_standardization_never_uses_evaluation(self):
        _,_,mean,scale=standardize(self.x,self.ex+1000)
        np.testing.assert_allclose(mean,self.x.mean(0));np.testing.assert_allclose(scale,self.x.std(0))
    def test_batched_mlp_independent(self):
        ys=np.stack([self.y,1-self.y]);p,_,_=self.fit('MLP',ys,[.01,.1])
        for v in range(2):
            for j,penalty in enumerate([.01,.1]):
                single,_,_=self.fit('MLP',ys[v:v+1],[penalty])
                np.testing.assert_allclose(p[v,j],single[0,0],atol=1e-12)
    def test_paired_refits_equal_individual_grid_choices(self):
        ys=np.stack([self.y,1-self.y])
        for family in ['KRR','LR','MLP']:
            full,_,_=self.fit(family,ys,[.01,.1]);paired,_,_=self.fit(family,ys,[.01,.1],paired=True)
            np.testing.assert_allclose(full[0,0],paired[0,0],atol=1e-12)
            np.testing.assert_allclose(full[1,1],paired[1,0],atol=1e-12)
    def test_folds_partition_without_labels(self):
        folds=split_folds(range(196),4,'outer')
        self.assertEqual([len(f) for f in folds],[49]*4)
        self.assertEqual(sorted(sum(folds,[])),list(range(196)))
    def test_permissions_reject_unsealed_B_and_final(self):
        with tempfile.TemporaryDirectory() as d:
            boundary=ProbeLabelBoundary(None,[1],[2],[3],Path(d)/'access.jsonl')
            for role,ids in [('B_evaluation',[2]),('A_train',[3]),('unknown',[1])]:
                with self.assertRaises(PermissionError):boundary.read(role,ids)
            boundary.seal('hash',1)
            with self.assertRaises(PermissionError):boundary.read('B_evaluation',[3])
            events=[json.loads(x) for x in (Path(d)/'access.jsonl').read_text().splitlines()]
            self.assertEqual(sum(x['event']=='access_denied' for x in events),4)
    def test_support_near_and_zero_kernel_not_fake_ess(self):
        ss=np.array([[0.,1.],[1.,0.]])
        sb=np.array([[.5,1000.],[2.,1000.]])
        result=support_statistics(ss,sb,np.array([[.5,0.],[.3,0.]]))
        self.assertEqual(result['near'].tolist(),[True,False]);self.assertTrue(np.isnan(result['ess'][1]))
    def test_metrics_clipping_and_empty(self):
        m=metrics([0,1],[-2.,3.]);self.assertTrue(np.isfinite(m['nll']));self.assertEqual(m['outside'],1.)
        self.assertIsNone(metrics([],[])['nll'])
    def test_evaluation_inputs_do_not_change_fit(self):
        _,states,_=self.fit('LR',self.y[None,:],[.1])
        _,other,_=fit_grid('LR',self.x,self.y[None,:],self.p,self.ex+1000,self.ep,self.k,self.ek,self.q,[.1],71,self.cfg)
        np.testing.assert_array_equal(states[0,0]['weight'],other[0,0]['weight'])


if __name__=='__main__':unittest.main()
