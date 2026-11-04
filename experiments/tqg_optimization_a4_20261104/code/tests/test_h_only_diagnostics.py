"""H-only acceptance: kernel exclusion and paired-label/random control semantics."""
from types import SimpleNamespace
import unittest
import torch
from tqg_gda.d34_diagnostics import kernel_family,solve_observed
from tqg_gda.p0_diagnostics import Geometry,controls,one_step_arms

class HOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(1042)
        self.sh=torch.randn(6,3,dtype=torch.float64);self.th=torch.randn(4,3,dtype=torch.float64)
        self.sa=torch.randn(6,1,2,dtype=torch.float64);self.ta=torch.randn(4,1,2,dtype=torch.float64)
    def test_H_kernel_excludes_response_distances(self):
        identity=torch.arange(6)
        original,_=kernel_family(self.sh,self.th,self.sa,self.ta,identity)
        changed,_=kernel_family(self.sh,self.th,self.sa*17,self.ta.flip(0),identity)
        for a,b in zip(original['H'],changed['H']):torch.testing.assert_close(a,b,rtol=0,atol=0)
        # The target response still defines the queried functional; only its distance is removed.
        w,info=solve_observed(original['H'],self.ta,'JOINT',.01,{})
        w2,_=solve_observed(changed['H'],self.ta,'JOINT',.01,{})
        torch.testing.assert_close(w,w2,rtol=0,atol=0)
        self.assertTrue(info['converged'])
    def test_nulls_permute_only_labels_and_match_random_and_step_norms(self):
        kernels,_=kernel_family(self.sh,self.th,self.sa,self.ta,torch.arange(6))
        w,_=solve_observed(kernels['H'],self.ta,'JOINT',.01,{})
        basis=torch.linalg.qr(torch.randn(7,2,dtype=torch.float64)).Q
        p=torch.tensor([[.7,.3],[.1,.9],[.6,.4],[.8,.2],[.2,.8],[.4,.6]],dtype=torch.float64)
        q=torch.tensor([[1.],[-1.]],dtype=torch.float64)/2**.5
        geo=Geometry(None,basis,q,self.sa,self.ta,w,p,self.sh,self.th,{})
        y=torch.tensor([0,1,1,0,1,0]);before=w.clone()
        grads,_,_=controls(geo,y,123,[9101,9102,9103],[9201,9202,9203],1e-12)
        for rep,offset in enumerate([9101,9102,9103],1):
            perm=torch.randperm(6,generator=torch.Generator().manual_seed(123+offset))
            residual=((torch.nn.functional.one_hot(y[perm],2)-p)@q).reshape(-1)
            torch.testing.assert_close(grads[f'PERM-{rep}'],-basis@w@residual)
            self.assertAlmostEqual(float(grads[f'RAND-{rep}'].norm()),float(grads['TRUE'].norm()),places=12)
        torch.testing.assert_close(w,before,rtol=0,atol=0)
        gs=torch.randn(7,dtype=torch.float64);ga=torch.randn(7,dtype=torch.float64)
        arms,valid=one_step_arms(grads,ga,gs,.1,1e-12)
        for name in ['S+P-1','S+P-2','S+P-3','S+R-1','S+R-2','S+R-3','S+O-M']:
            self.assertTrue(valid[name]);self.assertAlmostEqual(float((arms[name]-gs).norm()),float(.1*grads['TRUE'].norm()),places=12)

if __name__=='__main__':unittest.main()
