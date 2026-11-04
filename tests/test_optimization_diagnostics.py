import unittest
import numpy as np
import torch

from tqg_gda.optimization_diagnostics import (bandwidth_kernel, effective_lambda, ridge_grid,
    choose_kernel, mean_readout, closest_bank, check_mean_identity, select_kernel, select_local, teacher_fit,
    feedback_controls,binary_logit_metrics)


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

    def test_kernel_selection_does_not_use_evaluation_rows(self):
        cfg=dict(signatures=['H'],bandwidth_multipliers=[.5,1],lambda_conventions=['FIXED','NORMALIZED'],kernel_lambdas=[.01,.1])
        a,records=select_kernel(list(range(10)),{'H':self.d},self.y,self.p,cfg)
        altered=self.d.copy();altered[10:]=1e6;altered[:,10:]=1e6
        b,_=select_kernel(list(range(10)),{'H':altered},self.y,self.p,cfg)
        self.assertEqual(a,b)
        self.assertTrue(all(set(r['training_ids']).isdisjoint(r['validation_ids']) for r in records))

    def test_local_source_audit_not_in_selected_bank(self):
        cfg=dict(bank_sizes=[3,5],signatures=['H','HA'],lambda_conventions=['FIXED','NORMALIZED'],base_joint_lambda=.01)
        selected,records=select_local(list(range(10)),{'H':self.k,'HA':self.k**2},self.y,self.p,cfg)
        self.assertEqual(len(selected),2)
        self.assertTrue(all(set(r['bank_ids']).isdisjoint(r['validation_ids']) for r in records))

    def test_teacher_nulls_repeat_entire_selection_and_refit(self):
        ids=list(range(10));pred,states,chosen,records=teacher_fit(ids,self.x[:10],self.y,self.x[10:],[.01,.1],1)
        for v in range(2):
            one,one_states,one_choice,_=teacher_fit(ids,self.x[:10],self.y[v:v+1],self.x[10:],[.01,.1],1)
            np.testing.assert_allclose(pred[v],one[0],atol=1e-12)
            np.testing.assert_allclose(states[v,0]['weight'],one_states[0,0]['weight'],atol=1e-12)
            self.assertEqual(chosen[v]['penalty'],one_choice[0]['penalty'])
        self.assertTrue(all(set(r['train_ids']).isdisjoint(r['validation_ids']) for r in records))

    def test_feedback_controls_retain_base_and_match_correct_norm(self):
        basis=torch.eye(4,dtype=torch.float64);full=torch.tensor([1.,2.,3.,4.]);base=torch.tensor([1.,0.,0.,0.])
        full=full.double();base=base.double()
        nulls=[full.roll(i) for i in [1,2,3]]
        vectors,valid=feedback_controls(full,base,nulls,list(reversed(nulls)),basis,77)
        for i in [1,2,3]:
            self.assertAlmostEqual(float((vectors[f'CPL_{i}']-base).norm()),float((full-base).norm()),places=11)
            for name in [f'CPL_M_{i}',f'CRL_M_{i}',f'FULL_PERM_M_{i}',f'FULL_RAND_M_{i}']:
                self.assertAlmostEqual(float(vectors[name].norm()),float(full.norm()),places=11)
        np.testing.assert_allclose(vectors['FULL_PERM_M_1'],nulls[-1],atol=1e-11)

    def test_binary_margin_ce_matches_torch_without_probability_clipping(self):
        z=torch.tensor([[-1.,2.],[4.,-2.],[0.,40.]],dtype=torch.float64);y=torch.tensor([1,0,0])
        measured=binary_logit_metrics((z[:,1]-z[:,0]).numpy(),y.numpy())
        self.assertAlmostEqual(measured['ce'],float(torch.nn.functional.cross_entropy(z,y)),places=12)


if __name__=='__main__':unittest.main()
