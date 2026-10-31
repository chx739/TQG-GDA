"""Independent check of binary CONST observer; not used to alter formal solves."""
import unittest
import torch
from d34_analysis import constant_binary_readout
from tqg_gda.calibration_v2 import CalibrationConfig,solve_calibration


class ConstantAuditTest(unittest.TestCase):
    def test_closed_form_matches_converged_joint_and_stationarity(self):
        torch.set_num_threads(1)
        n=5;r=3;lam=.2
        u=torch.tensor([1.,.3,-.2],dtype=torch.float64)
        response=u.reshape(1,1,r).repeat(4,1,1)
        exact=constant_binary_readout(response,n,lam)
        approximate,info=solve_calibration(torch.ones(n,n,dtype=u.dtype),torch.ones(n,1,dtype=u.dtype)*u,
            torch.outer(u,u),CalibrationConfig(lambda_w=lam,max_iterations=1000,gap_atol=1e-13,gap_rtol=1e-11))
        self.assertTrue(info['converged'])
        torch.testing.assert_close(approximate,exact,atol=1e-8,rtol=1e-7)
        e=torch.outer(u-exact.sum(1),u-exact.sum(1))
        omega=torch.outer(u,u)/u.square().sum()
        d=torch.eye(r,dtype=u.dtype)+r*omega
        stationarity=d@exact@torch.ones(n,n,dtype=u.dtype)+lam*exact-d@(torch.ones(n,1,dtype=u.dtype)*u).mT
        self.assertLess(float(stationarity.norm()),1e-12)


if __name__=='__main__':unittest.main()
