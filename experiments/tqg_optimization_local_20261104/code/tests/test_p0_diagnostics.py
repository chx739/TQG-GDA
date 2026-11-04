"""Tests of diagnostic semantics, not evidence of domain transfer."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from tqg_gda.feedback_v2 import ParameterSnapshot
from tqg_gda.p0_diagnostics import (DevelopmentOracle, development_pools, gradient_metrics,
    gradient_of_ce, normalized, one_step_arms, sample_ids)


class P0DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_microbatch_gradient_is_graph_weighted_not_batch_weighted(self):
        torch.manual_seed(193)
        model=nn.Sequential(nn.Linear(2,2)).double().eval()
        snapshot=ParameterSnapshot(model)
        x=torch.randn(7,2,dtype=torch.float64)
        y=torch.tensor([0,0,0,1,1,1,1])
        direct=gradient_of_ce(snapshot,[x],y)
        split=gradient_of_ce(snapshot,[x[:3],x[3:6],x[6:]],y)
        torch.testing.assert_close(split,direct,atol=1e-12,rtol=1e-10)

    def test_zero_gradients_have_undefined_cosines_and_ratios(self):
        z=torch.zeros(4,dtype=torch.float64)
        m=gradient_metrics(z,z,z,z,z.reshape(4,1))
        for key in ('c_A','c_B','e_A','a_A','weighted_ratio','norm_ratio'):
            self.assertIsNone(m[key],key)
        self.assertFalse(m['valid_comparison'])

    def test_random_and_permuted_one_step_controls_have_true_feedback_norm(self):
        d=torch.float64
        true=torch.tensor([1.,2.,3.],dtype=d)
        vectors={'TRUE':true}
        for kind in ('PERM','RAND'):
            for rep in (1,2,3): vectors[f'{kind}-{rep}']=torch.tensor([rep*4.,1.,0.],dtype=d)
        source=torch.tensor([2.,0.,1.],dtype=d)
        oracle=torch.tensor([0.,2.,1.],dtype=d)
        arms,valid=one_step_arms(vectors,oracle,source,.1,1e-12)
        for name in ('S+O-M','S+P-1','S+P-2','S+P-3','S+R-1','S+R-2','S+R-3'):
            self.assertTrue(valid[name])
            self.assertAlmostEqual(float((arms[name]-source).norm()),.1*float(true.norm()),places=12)
        torch.testing.assert_close(arms['O-RAW'],oracle)

    def test_zero_reference_never_amplifies_null(self):
        v,valid=normalized(torch.ones(4),0.,1e-12)
        self.assertFalse(valid)
        self.assertTrue(torch.equal(v,torch.zeros_like(v)))

    def test_hash_split_and_sampling_preserve_role_and_no_within_batch_duplicates(self):
        ids=list(range(1000,1542))
        a,b=development_pools(ids,2)
        self.assertEqual(len(a),271)
        self.assertEqual(len(b),271)
        self.assertFalse(set(a)&set(b))
        self.assertEqual(set(a)|set(b),set(ids))
        self.assertEqual((a,b),development_pools(list(reversed(ids)),2))
        sample=sample_ids(a,16,61200000)
        self.assertEqual(len(set(sample)),16)
        self.assertEqual(sample,sample_ids(a,16,61200000))

    def test_oracle_denials_happen_before_any_raw_label_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'access.jsonl'
            task=SimpleNamespace(target_ids=(10,11,12))
            oracle=DevelopmentOracle(task,[10,11],[12],path,enabled=False)
            oracle.seal('ok',[10],theta_hash='x',readout_hash='y')
            with self.assertRaises(PermissionError): oracle.labels([10],token='ok',purpose='test')
            oracle.enabled=True
            with self.assertRaises(PermissionError): oracle.labels([12],token='ok',purpose='test')
            with self.assertRaises(PermissionError): oracle.labels([10],token='absent',purpose='test')
            events=[json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(sum(x['event']=='access_denied' for x in events),3)
            self.assertEqual(oracle.access_count,0)


if __name__=='__main__':
    unittest.main()
