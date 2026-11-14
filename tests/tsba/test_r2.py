"""Numerical and protocol checks for the full R2 trajectory runner."""
import copy
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch
import json
from pathlib import Path

import torch

from tsba.adapt import adapt_classifier
from tsba.classifier import ResidualClassifier, ParameterMap
from tsba.score import ScoreNetwork
from tsba.subproblem import GGNSystem
from tsba.problem import TsbaProblem
from tsba.boundary import BoundaryBatch
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.adapt import one_attempt, settings, ZeroScore
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.common import batches, jobs, atomic_tensor, load
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.score import restore, epoch_update, assert_nested_equal, comparison


class R2Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(123)
        self.model=ResidualClassifier(2,2,3,dtype=torch.float64)
        with torch.no_grad():
            self.model.linear.weight.copy_(torch.tensor([[1.,0.],[-1.,0.]]))
            self.model.linear.bias.zero_()
        self.model.eval()
        self.source=torch.randn(150,2,dtype=torch.float64)*.2
        self.target=torch.randn(140,2,dtype=torch.float64)*.2
        self.y=(self.source[:,1]>.03).long()

    def test_production_parity_and_refresh(self):
        score=ScoreNetwork(2,4).double().eval();cfg=settings(8,10,3)
        production=copy.deepcopy(self.model);ours=copy.deepcopy(self.model)
        observed=[]
        adapt_classifier(production,self.source,self.y,self.target,score,1.,cfg,observer=observed.append)
        schedule=batches(list(range(150)),list(range(140)),8,rounds=3)
        records=[]
        for batch,expected in zip(schedule,observed):
            s,t=batch['source_ids'],batch['target_ids']
            p=one_attempt(ours,self.source[s],self.y[s],self.target[t],score,1.,cfg)
            torch.testing.assert_close(p['after_vector'],expected['after'],rtol=0,atol=0)
            self.assertEqual(p['record']['reason'],expected['record']['reason'])
            self.assertEqual(s,expected['source_ids'].tolist());self.assertEqual(t,expected['target_ids'].tolist())
            records.append(p)
        self.assertGreater(len({p['matrix_sha256'] for p in records}),1)
        self.assertFalse(torch.equal(records[0]['boundaries']['points'],records[1]['boundaries']['points']))

    def test_reverse_only_request_and_geometric_zero_solves(self):
        score=ScoreNetwork(2,4).double().eval();cfg=settings(1,10)
        plus=one_attempt(copy.deepcopy(self.model),self.source,self.y,self.target,score,1.,cfg)
        minus=one_attempt(copy.deepcopy(self.model),self.source,self.y,self.target,score,1.,cfg,sign=-1)
        zero=one_attempt(copy.deepcopy(self.model),self.source,self.y,self.target,ZeroScore(),1.,cfg,sign=0,force_geometry=True)
        guard=one_attempt(copy.deepcopy(self.model),self.source,self.y,self.target,ZeroScore(),1.,cfg)
        self.assertEqual(guard['record']['reason'],'zero_requests')
        self.assertIn('solver',zero['record']);self.assertTrue(zero['record']['solver']['valid'])
        self.assertEqual(plus['matrix_sha256'],minus['matrix_sha256']);self.assertEqual(plus['matrix_sha256'],zero['matrix_sha256'])
        torch.testing.assert_close(plus['boundaries']['zeta'],-minus['boundaries']['zeta'],rtol=0,atol=0)
        torch.testing.assert_close(plus['delta']+minus['delta'],2*zero['delta'],rtol=1e-7,atol=1e-9)
        self.assertGreater(float(zero['delta'].norm()),0.)

    def test_lambda_in_matrix_rhs_and_objective(self):
        p=one_attempt(copy.deepcopy(self.model),self.source,self.y,self.target,ScoreNetwork(2,4).double(),1.,settings(1,1))
        b=BoundaryBatch(**p['boundaries'])
        problems=[TsbaProblem(self.model,self.source,self.y,b,settings(1,l)) for l in (1,10)]
        systems=[GGNSystem(q) for q in problems];d=p['D']
        torch.testing.assert_close(systems[1].dense()-systems[0].dense(),9*d.T@d/len(b))
        torch.testing.assert_close(systems[1].rhs-systems[0].rhs,9*d.T@b.zeta/len(b))
        candidate=problems[0].base+.01*p['delta']
        residual=(problems[0].movement(candidate)-b.zeta).square().mean()/2
        torch.testing.assert_close(problems[1].objective(candidate)-problems[0].objective(candidate),9*residual)

    def test_score_resume_exact_and_atomic_checkpoint(self):
        model=ScoreNetwork(2,64).double();opt=torch.optim.Adam(model.parameters(),lr=.001)
        rng=torch.Generator().manual_seed(888)
        epoch_update(model,opt,rng,self.source,.3)
        state=dict(model_state=copy.deepcopy(model.state_dict()),optimizer_state=copy.deepcopy(opt.state_dict()),
            training_rng_state=rng.get_state(),torch_rng_state=torch.random.get_rng_state())
        resumed,optimizer,generator=restore(state)
        a=epoch_update(model,opt,rng,self.source,.3);b=epoch_update(resumed,optimizer,generator,self.source,.3)
        self.assertEqual(a,b);assert_nested_equal(model.state_dict(),resumed.state_dict())
        assert_nested_equal(opt.state_dict(),optimizer.state_dict());assert_nested_equal(rng.get_state(),generator.get_state())
        with tempfile.TemporaryDirectory() as d:
            atomic_tensor(d+'/state.pt',state);assert_nested_equal(load(d+'/state.pt'),state)
            with self.assertRaises(FileExistsError):atomic_tensor(d+'/state.pt',state)

    def test_counts_and_near_zero_stratum(self):
        plan=dict(seeds=list(range(3405,3410)),domains=[1,3],lambdas=[1,10,100],bandwidth_multipliers=[1,2],score_updates=[900,1800,3600,7200])
        grid=jobs(plan);self.assertEqual(len(grid),630)
        self.assertEqual([sum(s['kind']==k for s in grid) for k in ('learned','kde','geometry')],[480,120,30])
        e=torch.tensor([[0.,1.],[2.,1.]],dtype=torch.float64);n=torch.tensor([[1.,0.],[1.,0.]],dtype=torch.float64)
        c=comparison(e,-e,n,1.);self.assertEqual(c['all']['n'],2);self.assertEqual(c['near_zero']['n'],1)
        self.assertEqual(c['away_zero']['sign_agreement'],0.)

    def test_r2_label_gate_and_seal_tampering(self):
        from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.common import Broker, FROZEN
        from experiments.common.runtime import seal
        from experiments.common.io import digest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for d in (*FROZEN,'logs','deps/inputs','deps/roles'):(root/d).mkdir(parents=True,exist_ok=True)
            raw=root/'raw.txt';raw.write_text('0\n1\n'*10)
            (root/'deps/inputs/broker.json').write_text(json.dumps(dict(path=str(raw),sha256=digest(raw),offsets=[[2*i,2] for i in range(20)])))
            (root/'deps/inputs/class_mapping.json').write_text(json.dumps(dict(raw_to_class={'0':0,'1':1})))
            (root/'deps/roles.json').write_text(json.dumps(dict(targets={'1':dict(A=[8,9,10,11],B=[12,13,14,15],final=[16,17,18,19])})))
            (root/'deps/roles/seed1.json').write_text(json.dumps(dict(fit_ids=[0,1,2,3],validation_ids=[4,5],diagnostic_ids=[6,7])))
            broker=Broker(root)
            with self.assertRaises(FileNotFoundError):broker.read([0],'source_fit','test',seed=1)
            seal(root,'INPUT',['deps','batches','protocol'])
            self.assertEqual(broker.read([0,1],'source_fit','test',seed=1),{0:0,1:1})
            with self.assertRaises(PermissionError):broker.read([12],'source_fit','test',seed=1)
            with self.assertRaises(PermissionError):broker.read([8],'target_A','test',domain=1)
            with self.assertRaises(PermissionError):broker.read([16],'final','test',domain=1)
            with self.assertRaises(FileNotFoundError):broker.read([12],'target_B','test',domain=1)
            seal(root,'OBS',list(FROZEN))
            self.assertEqual(broker.read([12,13],'target_B','test',domain=1),{12:0,13:1})
            (root/'trajectories/new-unsealed.pt').write_text('changed')
            with self.assertRaises(PermissionError):broker.read([12],'target_B','test',domain=1)

    def test_first_projection_failure_keeps_unavailable_diagnostic(self):
        from tsba.utils import NumericalError
        from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.diagnostics import first_round
        from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.common import job_key, write_json
        plan=dict(seeds=[1],domains=[1],lambdas=[1],bandwidth_multipliers=[1],score_updates=[900])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);x=self.source[:6]
            atomic_tensor(root/'deps/source/seed1/frozen.pt',dict(x=x,initial_head=self.model.state_dict()))
            write_json(root/'deps/roles.json',dict(targets={'1':dict(B=[2,3])}))
            write_json(root/'deps/roles/seed1.json',dict(fit_ids=[0,1],validation_ids=[4],diagnostic_ids=[5]))
            write_json(root/'source-labels/seed1.json',{'0':0,'1':1})
            for spec in jobs(plan):
                with patch('experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.adapt.build_boundaries',side_effect=NumericalError('injected')):
                    p=one_attempt(copy.deepcopy(self.model),x[:2],self.y[:2],x[2:4],ZeroScore(),1.,settings(1,1))
                self.assertEqual(p['record']['status'],'failed')
                p.update(sigma0=1.,batch=dict(source_ids=[0,1],target_ids=[2,3]))
                atomic_tensor(root/'trajectories'/job_key(spec)/'round001.pt',p)
            result=first_round(root,1,1,1,plan)
            self.assertIsNone(result['alpha']);self.assertEqual(result['reason'],'first_round_numerical_failure')
            self.assertEqual(len(result['files']),5)
            for p in (root/'diagnostics').rglob('*.pt'):
                self.assertFalse(load(p)['available'])

    def test_atomic_json_and_seal_interruption_resume(self):
        from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.common import write_json
        from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.run import replay_and_seal
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'value.json'
            with patch('experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.common.os.link',side_effect=OSError('interruption')):
                with self.assertRaises(OSError):write_json(p,dict(ok=True))
            self.assertFalse(p.exists());write_json(p,dict(ok=True));self.assertTrue(json.loads(p.read_text())['ok'])
            for stage,n in (('scores',20),('adapt',630),('diagnose',30)):write_json(root/f'{stage}-index.json',dict(count=n))
            plan=dict(seeds=[])
            with patch('experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.run.verify'),patch('experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.run.seal',side_effect=RuntimeError('interrupted before OBS')):
                with self.assertRaises(RuntimeError):replay_and_seal(root,plan)
            self.assertTrue((root/'prediction-replay.json').exists())
            with patch('experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.run.verify'),patch('experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.run.seal',return_value=dict(digest='test')):
                replay_and_seal(root,plan)

    def test_evaluation_products_resume_without_overwrite(self):
        from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.evaluate import write_json,write_csv
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_json(root/'labels.json',{'source':{1:0,2:1}})
            before=(root/'labels.json').read_bytes()
            write_json(root/'labels.json',{'source':{1:0,2:1}})
            self.assertEqual(before,(root/'labels.json').read_bytes())
            with self.assertRaises(PermissionError):write_json(root/'labels.json',{'source':{1:1,2:1}})
            write_csv(root/'metrics.csv',[dict(seed=1,nll=.5)])
            before=(root/'metrics.csv').read_bytes()
            write_csv(root/'metrics.csv',[dict(seed=1,nll=.5)])
            self.assertEqual(before,(root/'metrics.csv').read_bytes())

    def test_manifest_recovery_after_publication_before_unlink(self):
        from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.evaluate import write_json,evidence_manifest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'evidence.txt').write_text('frozen evidence')
            (root/'other.partial-kept').write_text('other interrupted evidence')
            expected=evidence_manifest(root);unlink=Path.unlink
            def interrupted(path,*args,**kwargs):
                if path.name.startswith('manifest.json.partial-'):raise InterruptedError('after publication')
                return unlink(path,*args,**kwargs)
            with patch.object(Path,'unlink',interrupted):
                with self.assertRaises(InterruptedError):write_json(root/'manifest.json',expected)
            self.assertTrue((root/'manifest.json').exists())
            self.assertEqual(len(list(root.glob('manifest.json.partial-*'))),1)
            self.assertIn('other.partial-kept',evidence_manifest(root))
            write_json(root/'manifest.json',evidence_manifest(root))
            self.assertEqual(json.loads((root/'manifest.json').read_text()),expected)


if __name__=='__main__':unittest.main()
