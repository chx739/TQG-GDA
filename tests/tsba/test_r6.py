"""R6 synthetic state decomposition, label isolation, and paired mechanisms."""
import copy
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from torch.nn.functional import cross_entropy
import test_r4
from tsba.classifier import ParameterMap
from experiments.mutagenicity.r6_源梯度与密度请求分解_20261114.src import protocol,candidates,evaluate,summarize
from experiments.mutagenicity.r6_源梯度与密度请求分解_20261114.src.shared import io,p4,t4,assert_nested_equal
from experiments.mutagenicity.r4_全方向迁移对照_20261112.src import training

REPO=Path(__file__).resolve().parents[2]
R6=REPO/'experiments/mutagenicity/r6_源梯度与密度请求分解_20261114'

class R6Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        request=json.loads((R6/'configs/r6-plan.json').read_text())
        self.plan=protocol.resolve(request,'mutagenicity_node')
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);dep=self.root/'deps'
        fixture=test_r4.R4Test();fixture.setUp()
        fixture.plan['methods']['A0']=dict(group='M00',density='learned')
        fixture.plan['model'].update(score_epochs=2,adapt_rounds=2)
        fixture.plan['score'].update(endpoints=[9,18],recovery_epochs=1)
        fixture.plan['side_attempts']=[1,2]
        self.old=fixture.plan;self.task=p4.Task(2,3,3405)
        fixture.prepare_adapt(dep,self.task)
        training.train_score(dep,self.old,self.task)
        t4.adapt(dep,self.old,(self.task,'A0'),stop_after=1)
        io.write_json(dep/'protocol/plan.json',self.old)
        self.args=protocol.restore(self.root,self.task,1)

    def generate(self,**changes):
        model,pools,y,record,previous,sigma,cfg,old=self.args
        data=candidates.make_candidates(model,pools,y,record,previous,sigma,changes.get('cfg',cfg),old,self.plan)
        data.update(setting='fixture',source=2,target=3,seed=3405,attempt=1)
        return data

    def test_six_arms_and_fixed_steps_do_not_mutate_or_filter(self):
        model,*_=self.args;params=ParameterMap(model);before=params.vector();rng=torch.random.get_rng_state()
        result=self.generate(cfg=replace(self.args[-2],execution_multiplier=1e-12))
        self.assertEqual(len(result['candidates']),18)
        self.assertTrue(any(not c['motion_feasible'] for c in result['candidates']))
        for c in result['candidates']:
            torch.testing.assert_close(c['vector'],result['theta']+c['beta']*(c['a']*result['s']+c['q']*result['u']),rtol=0,atol=0)
            self.assertEqual(c['beta'],c['k']*result['alpha'])
        assert_nested_equal(params.vector(),before);assert_nested_equal(torch.random.get_rng_state(),rng)
        self.assertLess(result['residuals']['s']['residual'],result['residuals']['s']['threshold'])

    def test_zero_request_is_zero_response(self):
        record=self.args[3]
        record['bounds']['zeta'].zero_()
        for q in (-1,1):record['side']['candidates'][q]=copy.deepcopy(record['side']['candidates'][0])
        result=self.generate()
        self.assertFalse(result['u'].any())
        for c in result['candidates']:
            if c['arm'] in ('N','U+','U-'):assert_nested_equal(c['vector'],result['theta'])

    def test_missing_boundary_propagates_without_zero_direction(self):
        self.args[3]['side']={'status':'no_valid_boundary'}
        result=self.generate()
        self.assertEqual(result['status'],'no_valid_boundary');self.assertNotIn('s',result)
        rows=evaluate.evaluate_block(None,None,None,result,self.plan)
        self.assertEqual(len(rows['coverage']),3);self.assertFalse(rows['paired'])

    def test_derivatives_match_finite_difference_and_interaction(self):
        r=self.generate();model,pools,y,*_=self.args
        labels=dict(source_fit=y,source_batch=y[r['batch']['source_positions']],
                    source_validation=torch.arange(len(pools['source_validation']))%2,target=torch.arange(len(pools['target']))%2)
        params=ParameterMap(model);x=pools['target'];target=labels['target']
        d=evaluate.directional(params,r['theta'],x,target,r['s'],r['u'],self.plan)
        eps=1e-4
        fd=float((cross_entropy(params.logits(r['theta']+eps*r['u'],x),target)-cross_entropy(params.logits(r['theta']-eps*r['u'],x),target))/(2*eps))
        self.assertAlmostEqual(fd,d['d_u'],places=8)
        self.assertLess(d['additivity_error'],1e-10)
        values=evaluate.evaluate_block(model,pools,labels,r,self.plan)
        self.assertEqual(len(values['candidates']),144)
        for k in self.plan['steps']['multipliers']:
            selected={v['arm']:v for v in values['candidates'] if v['k']==k and v['pool']=='target'}
            interaction=next(v for v in values['interactions'] if v['k']==k and v['pool']=='target')
            self.assertAlmostEqual(interaction['interaction'],selected['SU+']['nll']-selected['S']['nll']-selected['U+']['nll']+selected['N']['nll'],places=13)
            for groups in (('initially_correct','initially_wrong'),('generators','complement')):
                parts=[v['weighted_delta_nll'] for v in values['candidates'] if v['k']==k and v['arm']=='U+' and v['pool'] in groups and v['n']]
                self.assertAlmostEqual(sum(parts),selected['U+']['delta_nll'],places=12)

    def test_candidate_worker_never_loads_evaluation_labels(self):
        original=io.load
        def guarded(path):
            if 'source_labels' in Path(path).parts and not str(path).endswith('_fit.pt'):
                self.fail('Candidate worker attempted validation-label access')
            return original(path)
        with patch.object(io,'load',side_effect=guarded),patch.object(evaluate.p3,'read_values') as target_reader:
            candidates.worker(self.root,self.plan,(self.task,1))
            target_reader.assert_not_called()
        self.assertEqual(io.load(protocol.block_path(self.root,self.task,1))['status'],'available')

    def test_unsealed_or_partial_observation_cannot_read_labels(self):
        with patch.object(evaluate.p3,'read_values') as reader:
            with self.assertRaises(FileNotFoundError):evaluate.evaluation_labels(self.root,self.plan)
            io.write_json(self.root/'replay.json',{'blocks':120})
            io.seal(self.root,'R6_INPUT',['deps'])
            io.seal(self.root,'R6_OBS',['deps','replay.json','seals/R6_INPUT.json'])
            with self.assertRaises(PermissionError):evaluate.evaluation_labels(self.root,self.plan)
            reader.assert_not_called()

    def test_common_mask_and_missing_seed_have_fixed_denominators(self):
        plan=copy.deepcopy(self.plan);plan.update(seeds=[1,2],directions=[[0,1],[1,0]])
        plan['states']['attempts_before']=[1];plan['steps']['multipliers']=[1.]
        rows=[]
        for seed in (1,2):
            for s,t in plan['directions']:
                rows.append(dict(attempt=1,k=1.,arm='U+−N',pool='target',source=s,target=t,seed=seed,
                    n=7,net_correct=1 if s==0 else -1,nll=.01,accuracy=0.,balanced_accuracy=0.,macro_f1=0.,
                    common_motion_feasible=(seed,s)!=(1,0)))
        seeds,summary=summarize.summaries(plan,dict(paired=rows,interactions=[],derivatives=[]))
        select=lambda subset:next(r for r in summary if r['family']=='paired' and r['arm']=='U+−N' and r['pool']=='target' and r['scope']=='Avg' and r['metric']=='accuracy' and r['subset']==subset)
        self.assertEqual(select('all_finite')['ties'],2)
        self.assertIsNone(select('common_motion_feasible')['mean'])
        self.assertEqual(select('common_motion_feasible')['available_blocks'],3)

    def test_old_candidate_disagreement_is_rejected(self):
        self.args[3]['side']['candidates'][1]['predictions']['target'][0]=torch.tensor([0.,-100.],dtype=torch.float64)
        with self.assertRaises((AssertionError,ValueError)):self.generate()

    def test_evaluation_failure_retains_block_and_does_not_restore_missing(self):
        record=self.generate();record['status']='engineering_failure';record['error']='fixture failure'
        io.atomic_tensor(protocol.block_path(self.root,self.task,1),record)
        with patch.object(evaluate,'restore',side_effect=AssertionError('must not restore')):
            evaluate.worker(self.root,self.plan,(self.task,1,{}))
        rows=json.loads((self.root/'evaluation/blocks'/self.task.key/'before01.json').read_text())
        self.assertEqual(rows['coverage'][0]['status'],'engineering_failure')

    def test_report_builds_all_tables_from_evaluated_fixture(self):
        from experiments.mutagenicity.r6_源梯度与密度请求分解_20261114.src.report import collect
        r=self.generate();model,pools,y,*_=self.args
        labels=dict(source_fit=y,source_batch=y[r['batch']['source_positions']],
                    source_validation=torch.arange(len(pools['source_validation']))%2,target=torch.arange(len(pools['target']))%2)
        result=evaluate.evaluate_block(model,pools,labels,r,self.plan)
        io.write_json(self.root/'evaluation/blocks'/self.task.key/'before01.json',result)
        plan=copy.deepcopy(self.plan);plan.update(seeds=[3405],directions=[[2,3]])
        plan['states']['attempts_before']=[1]
        with patch('experiments.mutagenicity.r6_源梯度与密度请求分解_20261114.src.report.blocks',return_value=[(self.task,1)]):
            collect(self.root,plan)
        self.assertTrue((self.root/'evaluation/summary.csv').is_file())
        self.assertIn('U+−N',(self.root/'evaluation/report.md').read_text())
