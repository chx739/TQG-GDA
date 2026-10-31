"""D3/D4 acceptance checks: algebra and counterfactual alignment, not efficacy."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from tqg_gda.calibration_v2 import CalibrationConfig,solve_calibration
from tqg_gda.feedback_v2 import signature_kernels
from tqg_gda.d34_diagnostics import (SourceAuditOracle,kernel_family,query_error,solve_observed)


class D34Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(783)
        self.sh=torch.randn(6,4,dtype=torch.float64);self.th=torch.randn(3,4,dtype=torch.float64)
        self.sa=torch.randn(6,2,3,dtype=torch.float64);self.ta=torch.randn(3,2,3,dtype=torch.float64)
        self.perm=torch.tensor([4,2,0,5,1,3])
        self.kernels,self.stats=kernel_family(self.sh,self.th,self.sa,self.ta,self.perm)

    def test_HA_replays_production_and_all_five_full_grams_are_PSD(self):
        original=signature_kernels(self.sh,self.th,self.sa,self.ta)
        for actual,expected in zip(self.kernels['HA'],original[:3]):
            torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
        for name,(ss,st,tt) in self.kernels.items():
            full=torch.cat((torch.cat((ss,st),1),torch.cat((st.mT,tt),1)),0)
            self.assertGreaterEqual(float(torch.linalg.eigvalsh(full).min()),-1e-10,name)
        for part in self.kernels['CONST']:self.assertTrue(torch.equal(part,torch.ones_like(part)))

    def test_signature_permutation_changes_both_source_gram_axes_and_cross_rows(self):
        ss,st,tt=self.kernels['HA'];ps,pt,pp=self.kernels['HA_PERM']
        torch.testing.assert_close(ps,ss[self.perm][:,self.perm],atol=0,rtol=0)
        torch.testing.assert_close(pt,st[self.perm],atol=0,rtol=0)
        torch.testing.assert_close(pp,tt,atol=0,rtol=0)

    def test_fixed_ISO_matches_MEAN_at_half_lambda_for_multiclass_queries(self):
        mean,_=solve_observed(self.kernels['HA'],self.ta,'MEAN',.005,{})
        iso,info=solve_observed(self.kernels['HA'],self.ta,'ISO',.01,{})
        torch.testing.assert_close(iso,mean,atol=1e-11,rtol=1e-9)
        self.assertAlmostEqual(info['omega_deviation'],0.)
        self.assertEqual(info['delta_min'],2.)

    def test_optional_details_leave_production_weights_and_original_statistics_unchanged(self):
        ss=self.kernels['HA'][0]
        from tqg_gda.feedback_v2 import query_matrices
        hst,htt=query_matrices(*self.kernels['HA'][1:],self.ta)
        for mode in ('joint','mean_only'):
            cfg=CalibrationConfig(mode=mode)
            w,plain=solve_calibration(ss,hst,htt,cfg)
            observed,info=solve_calibration(ss,hst,htt,cfg,return_details=True)
            details=info.pop('_details')
            torch.testing.assert_close(w,observed,atol=0,rtol=0)
            self.assertEqual(info,plain)
            torch.testing.assert_close(details['error'],query_error(ss,hst,htt,w),atol=1e-12,rtol=1e-10)
            if mode=='joint':
                omega=details['omega']
                self.assertAlmostEqual(float(omega.trace()),1.,places=12)
                self.assertGreaterEqual(float(torch.linalg.eigvalsh(omega).min()),-1e-12)
                self.assertFalse(omega.requires_grad)

    def test_zero_rank_is_recorded_without_division_by_zero(self):
        empty=self.ta[:,:,:0]
        for mode in ('MEAN','ISO','JOINT'):
            w,info=solve_observed(self.kernels['HA'],empty,mode,.01,{})
            self.assertEqual(w.shape,(0,12));self.assertIsNone(info['mean_error'])
            self.assertEqual(info['termination'],'zero_rank')

    def test_source_audit_denies_bank_fit_target_and_unsealed_access(self):
        task=SimpleNamespace(query_ids=tuple(range(6)),labels={i:i%2 for i in range(6)})
        with tempfile.TemporaryDirectory() as d:
            oracle=SourceAuditOracle(task,[0,1,2],[3,4,5],Path(d)/'access.jsonl',enabled=True)
            oracle.seal('ok',[3],theta_hash='t',readout_hash='w')
            self.assertEqual(oracle.labels([3,4],token='ok',purpose='test').tolist(),[1,0])
            for ids,token in [([0],'ok'),([7],'ok'),([100],'ok'),([3],'unknown')]:
                with self.assertRaises(PermissionError):oracle.labels(ids,token=token,purpose='negative')
            oracle.enabled=False
            with self.assertRaises(PermissionError):oracle.labels([3],token='ok',purpose='negative')


if __name__=='__main__':unittest.main()
