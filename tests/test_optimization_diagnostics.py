import unittest
import numpy as np
import torch

from tqg_gda.optimization_diagnostics import (bandwidth_kernel, effective_lambda, ridge_grid,
    choose_kernel, mean_readout, closest_bank, check_mean_identity)


class OptimizationChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rng=np.random.default_rng(441)
        self.x=self.rng.normal(size=(15,4))
        self.d=((self.x[:,None]-self.x[None,:])**2).sum(2)
        self.k=bandwidth_kernel(self.d,1.)
        self.y=np.array([np.arange(10)%2,1-np.arange(10)%2])
        self.p=np.linspace(.1,.9,10)

    def test_ridge_agrees_with_independent_direct_solve(self):
        fit=ridge_grid(self.k[:10,:10],self.k[10:,:10],self.y,self.p,np.full(5,.3),[.01,.5])
        for i in range(2):
            for j,lam in enumerate([.01,.5]):
                alpha=np.linalg.solve(self.k[:10,:10]+lam*np.eye(10),self.y[i]-self.p)
                np.testing.assert_allclose(fit.alpha[i,j],alpha,atol=1e-11)
                np.testing.assert_allclose(fit.probabilities[i,j],.3+self.k[10:,:10]@alpha,atol=1e-11)
                self.assertAlmostEqual(fit.rkhs_squared[i,j],float(alpha@self.k[:10,:10]@alpha),places=9)

    def test_no_evaluation_information_in_fit_parameters(self):
        a=ridge_grid(self.k[:10,:10],self.k[10:,:10],self.y,self.p,np.full(5,.3),[.01])
        b=ridge_grid(self.k[:10,:10],np.zeros((2,10)),self.y,self.p,np.array([.9,.1]),[.01])
        np.testing.assert_array_equal(a.alpha,b.alpha)

    def test_bandwidth_multiplies_s_not_s_squared(self):
        np.testing.assert_allclose(bandwidth_kernel(self.d,2),self.k**.25)
        with self.assertRaises(ValueError):bandwidth_kernel(self.d,0)

    def test_regularization_reference_is_explicit(self):
        self.assertEqual(effective_lambda(.1,49,147,'FIXED'),.1)
        self.assertAlmostEqual(effective_lambda(.1,49,147,'NORMALIZED'),.1/3)
        self.assertAlmostEqual(effective_lambda(.1,196,196,'NORMALIZED'),.1)
        with self.assertRaises(ValueError):effective_lambda(.1,1,1,'TYPO')

    def test_bank_ties_resolve_by_graph_identity(self):
        idx=closest_bank([90,10,40],np.array([[.3,.4],[.4,.1],[.1,.2]]),1)
        self.assertEqual(idx.tolist(),[1])

    def test_choice_uses_cv_only_and_fixed_tie_rule(self):
        self.assertEqual(choose_kernel(np.ones((3,4)),[.5,1,2],[.001,.01,.1,1]),(2,3))
        scores=np.ones((3,4));scores[0,0]=0
        self.assertEqual(choose_kernel(scores,[.5,1,2],[.001,.01,.1,1]),(0,0))

    def test_multiclass_mean_krr_iso_identity(self):
        for q in [1,2,3]:
            a=torch.from_numpy(self.rng.normal(size=(5,q,6)))
            r=torch.from_numpy(self.rng.normal(size=10*q))
            checks=check_mean_identity(torch.from_numpy(self.k[:10,:10]),torch.from_numpy(self.k[:10,10:]),
                    torch.from_numpy(self.k[10:,10:]),a,r,.03,{})
            self.assertLess(max(checks.values()),1e-10)

    def test_binary_mean_weights_match_pointwise_aggregation(self):
        a=self.rng.normal(size=(5,1,7));r=self.rng.normal(size=10)
        w=mean_readout(self.k[:10,:10],self.k[:10,10:],a,.03)
        point=self.k[10:,:10]@np.linalg.solve(self.k[:10,:10]+.03*np.eye(10),r)
        np.testing.assert_allclose(w@r,a[:,0,:].T@point/5,atol=1e-11)


if __name__=='__main__':unittest.main()
