"""Independent checks of research oracles, geometry and passive instrumentation."""
import copy
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
import torch
from tsba.adapt import adapt_classifier
from tsba.boundary import BoundaryBatch, build_boundaries
from tsba.classifier import ParameterMap, ResidualClassifier
from tsba.config import Config
from experiments.common.synthetic.analytic import run_analytic, Threshold
from experiments.common.synthetic.archive import digest, save_tensor, write_json
from experiments.common.synthetic.distributions import GaussianMixtureScore, setup_cell, oracles, ZeroScore
from experiments.common.synthetic.geometry import normal_roots, compare_geometry
from experiments.common.synthetic.metrics import score_audit


class ValidationTests(unittest.TestCase):
    def test_oracle_equals_gradient_of_independent_log_density(self):
        x=torch.tensor([[-.8,.2],[.7,-.5]],dtype=torch.float64,requires_grad=True)
        means=torch.tensor([[-1.,0.],[1.,0.]],dtype=torch.float64)
        variance=torch.tensor([.25,1.],dtype=torch.float64)
        sigma=.31
        score=GaussianMixtureScore(means,variance,sigma)
        mixture=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(torch.ones(2)/2),
            torch.distributions.Independent(torch.distributions.Normal(means,(variance+sigma**2).sqrt()),1))
        expected=torch.autograd.grad(mixture.log_prob(x).sum(),x)[0]
        torch.testing.assert_close(score(x),expected,atol=1e-12,rtol=1e-12)
        empirical=GaussianMixtureScore(means,torch.zeros_like(means),sigma)
        expected=-(x[0]-means[0])/sigma**2
        # A single component has exact linear score even far in the tails.
        one=GaussianMixtureScore(means[:1],torch.zeros_like(means[:1]),sigma)
        torch.testing.assert_close(one(x[:1])[0],expected,atol=1e-12,rtol=1e-12)
        self.assertTrue(torch.isfinite(empirical(1000*x)).all())

    def test_normal_observer_subtracts_old_residual_and_detects_multiple_roots(self):
        b=BoundaryBatch(torch.tensor([[.50001]],dtype=torch.float64),torch.tensor([[0,1]]),
                        torch.tensor([3.],dtype=torch.float64),torch.tensor([.00003],dtype=torch.float64),
                        torch.tensor([-.01],dtype=torch.float64),torch.ones(1,1,dtype=torch.float64),{})
        old,status=normal_roots(Threshold(.5),b,.25)
        new,status2=normal_roots(Threshold(.51),b,.25)
        self.assertEqual(status,['ok']);self.assertEqual(status2,['ok'])
        self.assertAlmostEqual(float(new-old),.01,places=8)
        class TwoRoots(torch.nn.Module):
            def forward(self,x):
                m=(x[:,0]-.48)*(x[:,0]-.52)
                return torch.stack([m,-m],-1)
        root,status=normal_roots(TwoRoots(),b,.25)
        self.assertEqual(status,['ambiguous_multiple_roots']);self.assertTrue(torch.isnan(root).all())

    def test_observer_is_passive_even_if_callback_mutates_payload_and_rng(self):
        cfg=Config(seed=3,adapt_rounds=2,source_adapt_batch=16,target_batch=16)
        cell=setup_cell(2,3,cfg);sigma=.3
        pop,_=oracles(cell['target'],cell['coordinates'],sigma)
        plain=copy.deepcopy(cell['head']);observed=copy.deepcopy(plain)
        kwargs=dict(source=cell['source'],labels=cell['source_y'],target=cell['target'],score=pop,sigma=sigma,cfg=cfg)
        torch.manual_seed(938);rng=torch.random.get_rng_state().clone()
        a=adapt_classifier(plain,**kwargs)
        received=[]
        def observer(payload):
            received.append(copy.deepcopy(payload))
            torch.randn(51)
            payload['after'].zero_();payload['before'].zero_()
            payload['record']['reason']='mutated'
            if payload['boundaries'] is not None:payload['boundaries']['zeta'].fill_(1e6)
        b=adapt_classifier(observed,**kwargs,observer=observer)
        torch.testing.assert_close(ParameterMap(plain).vector(),ParameterMap(observed).vector(),rtol=0,atol=0)
        self.assertEqual(a,b);self.assertEqual(len(received),2)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        for payload in received:
            bb=payload['boundaries']
            self.assertEqual(len(bb['input_indices']),len(bb['points']))
            self.assertTrue((bb['input_indices']<16).all())

    def test_e1_required_risk_signs_and_linear_geometry(self):
        result=run_analytic()
        self.assertEqual(len(result['cases']),15)
        self.assertLessEqual(result['linear_geometry_error'],1e-8)
        self.assertTrue(result['scalar_conflict']['roots'])

    def test_score_sign_metric_does_not_invent_balanced_score(self):
        points=torch.tensor([[.5],[.6]],dtype=torch.float64)
        pop=GaussianMixtureScore(torch.tensor([[0.]],dtype=torch.float64),torch.ones(1,1,dtype=torch.float64),.2)
        row,_=score_audit(points,torch.ones_like(points),dict(learned=pop,empirical=pop,population=pop,zero=ZeroScore()),.2)
        self.assertEqual(row['comparisons']['learned_vs_population']['normal_rmse'],0.)
        self.assertIsNone(row['comparisons']['learned_vs_population']['balanced_sign_agreement'])
        self.assertEqual(row['comparisons']['zero_vs_population']['sign_agreement'],0.)

    def test_archive_is_nonoverwriting_and_checkpoint_reloadable(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'checkpoint.pt'
            model=ResidualClassifier(2,2,2,dtype=torch.float64)
            save_tensor(p,model.state_dict());before=digest(p)
            restored=copy.deepcopy(model);restored.load_state_dict(torch.load(p,weights_only=True))
            self.assertTrue(torch.equal(ParameterMap(model).vector(),ParameterMap(restored).vector()))
            with self.assertRaises(FileExistsError):save_tensor(p,{})
            self.assertEqual(before,digest(p))
            json_path=Path(directory)/'metadata.json';write_json(json_path,{'hash':before})
            with self.assertRaises(FileExistsError):write_json(json_path,{})
