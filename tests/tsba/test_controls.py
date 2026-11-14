"""Controls must remove exactly the declared terms and preserve state semantics."""
import copy
from dataclasses import replace
import unittest
import torch
from torch.func import hessian, jacrev
from tsba.adapt import adapt_classifier
from tsba.boundary import build_boundaries
from tsba.classifier import ParameterMap
from tsba.config import Config
from tsba.problem import TsbaProblem
from tsba.subproblem import GGNSystem
from experiments.synthetic_data.e2_目标请求对照_20261109.src.controls import SourceGGN, SourceProblem, solve_source, adapt_control
from experiments.synthetic_data.e1_e2_符号几何与适配_20261109.src.distributions import setup_cell, ZeroScore, generator


def schedule(cfg, n=512):
    rng=generator(cfg.seed+201)
    return [dict(source_ids=torch.randperm(n,generator=rng)[:cfg.source_adapt_batch],
                 target_ids=torch.randperm(n,generator=rng)[:cfg.target_batch]) for _ in range(cfg.adapt_rounds)]


class ControlTests(unittest.TestCase):
    def test_source_ggn_matches_linear_ce_hessian_and_matrix_free_action(self):
        torch.manual_seed(4)
        model=torch.nn.Linear(3,2,dtype=torch.float64)
        x=torch.randn(12,3,dtype=torch.float64);y=torch.arange(12)%2
        cfg=Config()
        problem=SourceProblem(model,x,y,cfg);system=SourceGGN(problem)
        expected=hessian(problem.loss)(problem.base)+cfg.damping*torch.eye(len(problem.base),dtype=x.dtype)
        torch.testing.assert_close(system.dense(),expected,atol=1e-12,rtol=1e-12)
        direction=torch.randn_like(problem.base)
        torch.testing.assert_close(system.matvec(direction),expected@direction,atol=1e-12,rtol=1e-12)
        delta,info=solve_source(problem)
        self.assertTrue(info['valid']);self.assertLess(info['directional_derivative'],0)
        torch.testing.assert_close(expected@delta,-system.gs,atol=1e-12,rtol=1e-12)

    def test_zero_request_preserves_geometry_curvature_but_runs_source_update(self):
        cfg=Config(seed=0,adapt_rounds=1,source_adapt_batch=32,target_batch=32)
        cell=setup_cell(2,0,cfg);head=cell['head'];ids=schedule(cfg);s=ids[0]['source_ids'];t=ids[0]['target_ids']
        b=build_boundaries(head,cell['target'][t],ZeroScore(),.3,cfg)
        self.assertGreater(len(b),0)
        problem=TsbaProblem(head,cell['source'][s],cell['source_y'][s],b,cfg)
        system=GGNSystem(problem)
        source=SourceGGN(SourceProblem(head,cell['source'][s],cell['source_y'][s],cfg))
        d=jacrev(problem.normalized_margin)(problem.base)
        torch.testing.assert_close(system.dense()-source.dense(),cfg.lambda_boundary/len(b)*d.T@d,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(system.rhs,source.rhs,atol=0,rtol=0)
        before=ParameterMap(head).vector()
        ordinary=copy.deepcopy(head)
        prod=adapt_classifier(ordinary,cell['source'],cell['source_y'],cell['target'],ZeroScore(),.3,cfg)
        self.assertEqual(prod['accepted'],0)
        result=adapt_control(head,cell['source'],cell['source_y'],cell['target'],.3,cfg,ids,'zero_zeta_geometry')
        self.assertEqual(result['accepted'],1)
        self.assertFalse(torch.equal(ParameterMap(head).vector(),before))
        self.assertTrue(torch.equal(ParameterMap(ordinary).vector(),before))

    def test_empty_geometry_skips_and_source_control_ignores_target(self):
        cfg=Config(seed=1,adapt_rounds=1,source_adapt_batch=32,target_batch=32)
        cell=setup_cell(2,1,cfg);ids=schedule(cfg)
        far=torch.full_like(cell['target'],1e3)
        head=copy.deepcopy(cell['head']);before=ParameterMap(head).vector()
        result=adapt_control(head,cell['source'],cell['source_y'],far,1e-4,cfg,ids,'zero_zeta_geometry')
        self.assertEqual(result['history'][0]['reason'],'no_valid_boundary')
        self.assertTrue(torch.equal(before,ParameterMap(head).vector()))
        left=copy.deepcopy(head);right=copy.deepcopy(head)
        a=adapt_control(left,cell['source'],cell['source_y'],far,1e-4,cfg,ids,'source_ce_ggn')
        b=adapt_control(right,cell['source'],cell['source_y'],cell['target'],5.,cfg,ids,'source_ce_ggn')
        self.assertEqual(a,b);self.assertEqual(a['accepted'],1)
        self.assertTrue(torch.equal(ParameterMap(left).vector(),ParameterMap(right).vector()))

    def test_exhausted_zero_request_backtracking_never_commits(self):
        cfg=Config(seed=0,adapt_rounds=1,source_adapt_batch=32,target_batch=32,backtracks=1,execution_multiplier=1e-30)
        cell=setup_cell(2,0,cfg);head=cell['head'];before=ParameterMap(head).vector()
        out=adapt_control(head,cell['source'],cell['source_y'],cell['target'],.3,cfg,schedule(cfg),'zero_zeta_geometry')
        self.assertEqual(out['history'][0]['reason'],'backtracking_exhausted')
        self.assertTrue(torch.equal(before,ParameterMap(head).vector()))
