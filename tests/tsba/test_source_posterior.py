import copy
import unittest
import torch
from torch.func import grad,jacrev
from torch.nn import functional as F
from tsba.adapt import adapt_classifier
from tsba.boundary import build_boundaries
from tsba.classifier import ParameterMap
from tsba.config import Config
from tsba.subproblem import GGNSystem
from experiments.synthetic_data.e2_目标请求对照_20261109.src.controls import SourceProblem,SourceGGN,adapt_control
from experiments.synthetic_data.e1_e2_符号几何与适配_20261109.src.distributions import setup_cell,oracles
from experiments.synthetic_data.e2_解析源后验对照_20261110.src.source_posterior import (
    analytic_source_posterior,expected_ce,PosteriorSourceProblem,PosteriorTsbaProblem,
    adapt_posterior,audit_posterior_event,ARMS)


class SourcePosteriorTests(unittest.TestCase):
    def case(self,rounds=2):
        cfg=Config(seed=0,adapt_rounds=rounds,source_adapt_batch=512,target_batch=16)
        cell=setup_cell(2,0,cfg);pop,_=oracles(cell['target'],cell['coordinates'],.3)
        posterior,_=analytic_source_posterior(cell['source'],cell['coordinates'])
        rng=torch.Generator().manual_seed(201)
        schedule=[dict(source_ids=torch.randperm(512,generator=rng),target_ids=torch.randperm(512,generator=rng)[:16]) for _ in range(rounds)]
        return cfg,cell,pop,posterior,schedule

    def test_expected_ce_gradient_and_ggn_curvature(self):
        cfg,cell,pop,p,schedule=self.case()
        problem=PosteriorSourceProblem(cell['head'],cell['source'],p,cfg)
        logits=problem.logits(problem.base)
        torch.testing.assert_close(expected_ce(logits,p),F.cross_entropy(logits,p,reduction='none'),atol=1e-14,rtol=1e-14)
        j=jacrev(problem.logits)(problem.base)
        formula=torch.einsum('bcp,bc->p',j,(logits.softmax(-1)*p.sum(-1,keepdim=True)-p))/len(p)
        torch.testing.assert_close(grad(problem.loss)(problem.base),formula,atol=1e-14,rtol=1e-12)
        self.assertLess(float(formula.norm()),1e-12)
        # Soft labels replace the gradient; the categorical GGN curvature stays.
        hard=SourceProblem(cell['head'],cell['source'],cell['source_y'],cfg)
        torch.testing.assert_close(SourceGGN(problem).dense(),SourceGGN(hard).dense(),atol=0,rtol=0)
        b=build_boundaries(cell['head'],cell['target'][:16],pop,.3,cfg)
        bp=PosteriorTsbaProblem(cell['head'],cell['source'],p,b,cfg)
        torch.testing.assert_close(GGNSystem(bp).gs,formula,atol=1e-14,rtol=1e-12)

    def test_one_hot_variant_matches_archived_algorithms(self):
        cfg,cell,pop,p,schedule=self.case();onehot=F.one_hot(cell['source_y'],2).to(cell['source'])
        for arm in ARMS:
            old=copy.deepcopy(cell['head']);new=copy.deepcopy(cell['head'])
            a=[];b=[]
            if arm=='population':
                adapt_classifier(old,cell['source'],cell['source_y'],cell['target'],pop,.3,cfg,observer=a.append)
            else:adapt_control(old,cell['source'],cell['source_y'],cell['target'],.3,cfg,schedule,arm,observer=a.append)
            adapt_posterior(new,cell['source'],onehot,cell['target'],pop,.3,cfg,schedule,arm,observer=b.append)
            self.assertEqual(len(a),len(b))
            for left,right in zip(a,b):
                self.assertEqual(left['record']['reason'],right['record']['reason'])
                torch.testing.assert_close(left['after'],right['after'],atol=1e-11,rtol=1e-11)
                torch.testing.assert_close(left['source_ids'],right['source_ids'],atol=0,rtol=0)
                torch.testing.assert_close(left['target_ids'],right['target_ids'],atol=0,rtol=0)

    def test_both_controls_stationary_and_all_trial_ce_audited(self):
        cfg,cell,pop,p,schedule=self.case(3)
        for arm in ARMS:
            head=copy.deepcopy(cell['head']);initial=ParameterMap(head).vector();events=[]
            def observe(e):audit_posterior_event(head,cell,p,schedule[e['record']['round']],e);events.append(e)
            adapt_posterior(head,cell['source'],p,cell['target'],pop,.3,cfg,schedule,arm,observer=observe)
            if arm!='population':self.assertLess(float((ParameterMap(head).vector()-initial).abs().max()),1e-10)
            if arm=='population':self.assertGreater(float((ParameterMap(head).vector()-initial).abs().max()),1e-8)
            used=[e for e in events if e['record'].get('trials')]
            self.assertTrue(used)
            corrupt=copy.deepcopy(used[0]);corrupt['record']['trials'][0]['source_ce']+=.01
            with self.assertRaises(ValueError):audit_posterior_event(head,cell,p,schedule[corrupt['record']['round']],corrupt)
