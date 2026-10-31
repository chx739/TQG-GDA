"""Acceptance of generated truths, soft gradients and decomposition counterfactuals."""
import unittest
from types import SimpleNamespace
import torch
from tqg_gda.feedback_v2 import ParameterSnapshot,contrast_basis
from tqg_gda.p0_diagnostics import Geometry,unlabeled_geometry
from tqg_gda.mechanism_diagnostics import (graph_pool,linear_pool,LinearBinary,soft_gradient,
    split_residual,decomposition_vectors,decomposition_steps,soft_ce)

class MechanismTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(414)
    def test_paired_scenarios_only_change_intended_quantities(self):
        pools={s:graph_pool(24,7320,target=True,scenario=s) for s in ['S1','S2','S3','S4']}
        for i in range(24):
            a=pools['S1'].graphs[i]
            self.assertIsNone(a.y)
            for s in ['S2','S3','S4']:
                self.assertTrue(torch.equal(a.edge_index,pools[s].graphs[i].edge_index))
            torch.testing.assert_close(a.x,pools['S3'].graphs[i].x,atol=0,rtol=0)
            torch.testing.assert_close(pools['S4'].graphs[i].x[:,1],a.x[:,1]+10)
            t=pools['S1'].latents[i]['t'];eps=pools['S1'].latents[i]['noise']
            self.assertAlmostEqual(float(pools['S2'].graphs[i].x[0,1]),-(2*t-1)+eps)
        torch.testing.assert_close(pools['S3'].eta,1-pools['S1'].eta)
        self.assertTrue(torch.equal(pools['S2'].eta,pools['S1'].eta))
        self.assertEqual({v['n'] for v in pools['S1'].latents},{18,24,30})
    def test_soft_gradient_matches_analytic_binary_and_at_true_theta_is_zero(self):
        x,eta,_=linear_pool(48,27320,target=True)
        model=LinearBinary([.2,-.4]);snap=ParameterSnapshot(model)
        g=soft_gradient(snap,[x[:17],x[17:]],eta)
        expected=((torch.sigmoid(x@model.theta.detach())-eta)[:,None]*x).mean(0)
        torch.testing.assert_close(g,expected,atol=1e-14,rtol=1e-12)
        snap=ParameterSnapshot(LinearBinary([1.,-1.]))
        self.assertLess(float(soft_gradient(snap,[x],eta).norm()),1e-14)
    def test_conditional_gradient_is_inside_full_target_space(self):
        x,e,_=linear_pool(24,250);s,_,_=linear_pool(30,260)
        geo=unlabeled_geometry(LinearBinary([.5,-.5]),s,x)
        gradient=soft_gradient(geo.snapshot,[x],e)
        torch.testing.assert_close(geo.basis@(geo.basis.mT@gradient),gradient,atol=1e-12,rtol=1e-10)
        soft=torch.stack((1-e,e),dim=-1)
        residual=(soft-geo.snapshot.logits(x).detach().softmax(-1))@geo.contrasts
        reconstructed=-geo.basis@torch.einsum('bqr,bq->r',geo.target_response,residual)/len(x)
        torch.testing.assert_close(gradient,reconstructed,atol=1e-12,rtol=1e-10)
    def test_decomposition_additivity_and_permutation_only_preserves_global(self):
        x,_,_=linear_pool(8,265);s,_,y=linear_pool(20,266)
        geo=unlabeled_geometry(LinearBinary([.5,-.5]),s,x)
        v,_,stats=decomposition_vectors(geo,y,321,[9101,9102,9103],[9201,9202,9203],1e-12)
        torch.testing.assert_close(v['FULL'],v['GLOBAL']+v['CENTERED'])
        r,g,c=split_residual(geo.source_probabilities,y,geo.contrasts)
        self.assertLess(float(c.reshape(len(y),-1).mean(0).norm()),1e-14)
        for rep,offset in enumerate([9101,9102,9103],1):
            indices=torch.randperm(len(y),generator=torch.Generator().manual_seed(321+offset))
            permfull,_=geo.gradient(y[indices])
            torch.testing.assert_close(permfull,v['GLOBAL']+v[f'PC-{rep}'])
            self.assertAlmostEqual(float(v[f'RC-{rep}'].norm()),float(v['CENTERED'].norm()),places=12)
        gs=torch.ones_like(v['FULL']);ga=gs.flip(0)
        arms,valid=decomposition_steps(v,gs,ga,beta=.1,epsilon=1e-12)
        for kind in ['PC','RC']:
            for rep in [1,2,3]:
                increment=arms[f'S+GLOBAL+{kind}-{rep}']-arms['S+GLOBAL']
                self.assertAlmostEqual(float(increment.norm()),.1*float(v['CENTERED'].norm()),places=12)
    def test_soft_CE_gradient_predicts_small_step(self):
        x,eta,_=linear_pool(64,272);snap=ParameterSnapshot(LinearBinary([0.,0.]))
        g=soft_gradient(snap,[x],eta);base=soft_ce(snap,snap.flat,[x],eta)
        new=soft_ce(snap,snap.flat-1e-5*g,[x],eta)
        self.assertLess(new,base);self.assertAlmostEqual((new-base)/1e-5,-float(g.square().sum()),places=5)

if __name__=='__main__':unittest.main()
