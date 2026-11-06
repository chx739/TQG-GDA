"""Synthetic acceptance of the sealed M0/M1/M2 experiment driver."""
import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import json
import numpy as np

import torch
from torch.nn import functional as F

from helpers import fixture_files
from tqg_gda.adapt import Adapter
from tqg_gda.classifier import GraphClassifier
from tqg_gda.config import Config,EncoderConfig,PretrainConfig,AdaptConfig
from tqg_gda.data import load_task
from tqg_gda.p0_diagnostics import DevelopmentOracle,model_hash,gradient_of_ce
from tqg_gda.v1.query import build_feedback,feedback_loss,ProbeConfig
from tqg_gda.v2.query import ParameterSnapshot,build_gradient_feedback,parameter_feedback_loss
from workflows.tqg.minimal_validation.diagnostics import (construct,derived_seed,generator,
    permuted_labels,flat_gradient,step_audit,gradient_rows)
from workflows.tqg.minimal_validation.training import PermutedTask,query_ce_step
from workflows.tqg.minimal_validation.decisions import summarize
from workflows.tqg.minimal_validation.run import run_block


class MinimalValidationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(33)
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        files=fixture_files(Path(self.tmp.name)/'data')
        self.config=Config(cache_path=str(files[0]),source_split=str(files[1]),target_index=str(files[2]),
            dataset='fixture',seed=3405,use_node_attributes=True,fit_fraction=.75,
            encoder=EncoderConfig(hidden=5,num_layers=2,out_dim=3,dropout=.2),
            pretrain=PretrainConfig(epochs=1,batch_size=4),
            adapt=AdaptConfig(steps=2,fit_batch_size=4,query_batch_size=4,target_batch_size=3),
            v1=ProbeConfig(count=2,radius=.1),out_dir=self.tmp.name)
        self.task=load_task(self.config)
        self.model=GraphClassifier(dict(in_dim=2,edge_dim=0,**self.config.to_dict()['encoder']),2,
                                   dtype=torch.float32).eval()
        self.q=self.task.batch(self.task.query_ids,'query',dtype=torch.float32)
        self.labels,self.q.y=self.q.y,None
        self.q.graph_ids=list(self.task.query_ids)
        self.target=self.task.batch(self.task.target_ids[:3],'target',dtype=torch.float32)

    def test_true_gradients_match_canonical_paths_and_fixed_control_geometry(self):
        for version in ('v1','v2'):
            cfg=replace(self.config,version=version)
            item=construct(self.model,self.q,self.labels,self.target,cfg,3405,1,0)
            if version=='v1':
                f=build_feedback(self.model,self.q,self.labels,self.target,cfg.v1,
                                 generator(derived_seed(3405,1,0,'v1/probe')))
                expected=flat_gradient(feedback_loss(self.model,self.target,f),self.model)
            else:
                f=build_gradient_feedback(self.model,self.q,self.labels,self.target,cfg.v2)
                expected=flat_gradient(parameter_feedback_loss(self.model,f),self.model)
            torch.testing.assert_close(item['gradients']['TRUE'],expected,atol=0,rtol=0)
            for rep in (1,2,3):
                changed,_=permuted_labels(self.q.graph_ids,self.labels,3405,rep)
                residual=F.one_hot(changed,2).float()-ParameterSnapshot(self.model).logits(self.q).detach().softmax(-1)
                if version=='v2':
                    expected=-(f.basis.double()@(f.readout@(residual@f.contrasts).double().flatten())).float()
                    torch.testing.assert_close(item['gradients'][f'PERM-{rep}'],expected,atol=0,rtol=0)
            space=item['space'].double()
            for rep in (1,2,3):
                vector=item['gradients'][f'RAND-{rep}'].double()
                torch.testing.assert_close(vector.norm(),item['gradients']['TRUE'].double().norm(),rtol=1e-5,atol=1e-7)
                torch.testing.assert_close(space@(space.T@vector),vector,rtol=2e-4,atol=1e-7)

    def test_mean_y_preserves_prediction_and_centered_label_decomposition(self):
        cfg=replace(self.config,version='v2')
        item=construct(self.model,self.q,self.labels,self.target,cfg,3405,1,0)
        f=item['feedback']
        y=F.one_hot(self.labels,2).float()
        p=ParameterSnapshot(self.model).logits(self.q).detach().softmax(-1)
        expected=-(f.basis.double()@(f.readout@((y.mean(0)-p)@f.contrasts).double().flatten())).float()
        torch.testing.assert_close(item['gradients']['MEAN-Y'],expected,rtol=0,atol=0)
        centered=-(f.basis.double()@(f.readout@((y-y.mean(0))@f.contrasts).double().flatten())).float()
        torch.testing.assert_close(item['gradients']['TRUE']-expected,centered,rtol=1e-4,atol=1e-6)

    def test_permutation_global_mapping_is_independent_of_batch_order(self):
        ids=self.q.graph_ids
        for rep in (1,2,3):
            first,mapping=permuted_labels(ids,self.labels,3405,rep)
            order=torch.arange(len(ids)-1,-1,-1)
            second,other=permuted_labels([ids[i] for i in order],self.labels[order],3405,rep)
            self.assertEqual(mapping,other)
            self.assertEqual(sorted(first.tolist()),sorted(self.labels.tolist()))
            torch.testing.assert_close(first[order],second)

    def test_temporary_steps_restore_snapshot_and_use_actual_fp32_writeback(self):
        before=model_hash(self.model)
        item=construct(self.model,self.q,self.labels,self.target,replace(self.config,version='v1'),3405,1,0)
        snap=ParameterSnapshot(self.model)
        ya=torch.tensor([0,1,0]);bids=self.task.target_ids[3:];yb=torch.tensor([i%2 for i in bids])
        ga=gradient_of_ce(snap,[self.target],ya)
        b=self.task.batch(bids,'target',dtype=torch.float32)
        gb=gradient_of_ce(snap,[b],yb)
        gq=gradient_of_ce(snap,[self.q],self.labels)
        rows,directions,common=step_audit(self.model,self.task,self.task.target_ids[:3],bids,
            ya,yb,ga,gb,gq,gq,item['gradients'],self.config)
        self.assertEqual(len(rows),14)
        self.assertEqual(model_hash(self.model),before)
        self.assertEqual(len(common),3)
        for row in rows:
            self.assertGreaterEqual(row['numeric_floor'],10*torch.finfo(torch.float32).eps)
            self.assertLess(row['rounded_sgd_max_error'],1e-6)
        again,_,_=step_audit(self.model,self.task,self.task.target_ids[:3],bids,
            ya,yb,ga,gb,gq,gq,item['gradients'],self.config,common)
        self.assertEqual(sum(r['reused_shared_step'] for r in again),3)

    def test_query_ce_control_uses_same_ids_and_exact_joint_gradient(self):
        adapter=Adapter(self.model,replace(self.config,version='source_only'))
        original=copy.deepcopy(self.model)
        row=query_ce_step(adapter,self.task)
        fit=self.task.batch(row['batches']['fit'],'fit',dtype=torch.float32)
        query=self.task.batch(row['batches']['query'],'query',dtype=torch.float32)
        yf,fit.y=fit.y,None; yq,query.y=query.y,None
        loss=F.cross_entropy(original(fit),yf)+self.config.adapt.feedback_weight*F.cross_entropy(original(query),yq)
        optimizer=torch.optim.SGD(original.parameters(),lr=self.config.adapt.lr)
        loss.backward();optimizer.step()
        for name,value in original.state_dict().items():
            torch.testing.assert_close(value,self.model.state_dict()[name],atol=0,rtol=0)
        plain=Adapter(copy.deepcopy(original),replace(self.config,version='source_only'))
        self.assertEqual(plain.step(self.task)['batches']['fit'],row['batches']['fit'])

    def test_final_label_access_and_unsealed_access_are_rejected(self):
        oracle=DevelopmentOracle(self.task,self.task.target_ids[:4],self.task.target_ids[4:],
            Path(self.tmp.name)/'access.jsonl',enabled=True)
        with self.assertRaises(PermissionError):
            oracle.labels(self.task.target_ids[:2],token='missing',purpose='test')
        oracle.seal('test',self.task.target_ids[:4],theta_hash='synthetic',readout_hash='synthetic')
        with self.assertRaises(PermissionError):
            oracle.labels(self.task.target_ids[4:],token='test',purpose='must deny final')

    def test_complete_block_seals_before_development_access_and_is_reusable(self):
        root=Path(self.tmp.name)
        files=fixture_files(root/'labeled_synthetic',target_label=None)
        cfg=replace(self.config,cache_path=str(files[0]),source_split=str(files[1]),target_index=str(files[2]))
        task=load_task(cfg)
        dev,final=list(task.target_ids[:-1]),list(task.target_ids[-1:])
        np.save(root/'dev.npy',dev);np.save(root/'final.npy',final)
        manifest={'targets':{'1':{'development':{'path':str(root/'dev.npy')},'final':{'path':str(root/'final.npy')}}}}
        block=dict(seed=3406,domain=1,block=0,fit=list(task.fit_ids[:4]),query=list(task.query_ids),A=dev[:3],B=dev[3:])
        output=root/'block_run';output.mkdir()
        rows=run_block(self.model,task,cfg,block,manifest,output)
        self.assertEqual(len(rows),2)
        access=[json.loads(line) for line in (output/'label_access.jsonl').read_text().splitlines()]
        self.assertEqual(access[0]['event'],'unlabeled_query_sealed')
        self.assertTrue(all(not set(r['ids'])&set(final) for r in access))
        self.assertEqual(run_block(self.model,task,cfg,block,manifest,output),rows)

    def test_permuted_training_changes_only_query_labels_and_preserves_schedule(self):
        _,mapping=permuted_labels(self.task.query_ids,self.labels,3405,1)
        wrapped=PermutedTask(self.task,mapping)
        for role in ('fit','query','target'):
            a,ids=self.task.sample(role,3,torch.Generator().manual_seed(1),dtype=torch.float32)
            b,other=wrapped.sample(role,3,torch.Generator().manual_seed(1),dtype=torch.float32)
            self.assertEqual(ids,other)
            torch.testing.assert_close(a.x,b.x,atol=0,rtol=0)
            if role=='query':
                self.assertEqual(b.y.tolist(),[mapping[i] for i in ids])
            elif role=='fit':
                torch.testing.assert_close(a.y,b.y)
            else:
                self.assertIsNone(b.y)

    def test_numerical_failure_is_retained_as_zero_feedback(self):
        with patch('workflows.tqg.minimal_validation.diagnostics.build_feedback',side_effect=FloatingPointError('fixture')):
            item=construct(self.model,self.q,self.labels,self.target,replace(self.config,version='v1'),3405,1,0)
        self.assertEqual(item['health']['query']['status'],'numerical_fallback_source_only')
        self.assertTrue(all(torch.equal(x,torch.zeros_like(x)) for x in item['gradients'].values()))

    def test_preset_gate_keeps_failed_seeds_and_requires_both_tasks(self):
        blocks=[]
        for version in ('v1','v2'):
            for domain in (1,3):
                for seed in (3405,3406,3407):
                    for block in range(4):
                        gradients=[dict(arm=a,cos_A=.5 if a=='TRUE' else .1) for a in
                            ['TRUE','MEAN-Y']+[f'{n}-{i}' for n in ('PERM','RAND') for i in (1,2,3)]]
                        names=['FROZEN','FIT-CE','FIT+QRY-CE','FIT+TRUE','FIT+MEAN-Y','TRUE-only','ORACLE-only']
                        names += [f'FIT+{n}-{i}' for n in ('PERM','RAND') for i in (1,2,3)]
                        steps=[dict(arm=n,B_ce=.9 if n=='FIT+TRUE' else 1.,B_ce64_actual=.9 if n=='FIT+TRUE' else 1.,numeric_floor=1e-6) for n in names]
                        blocks.append(dict(version=version,domain=domain,seed=seed,block=block,gradients=gradients,
                            steps=steps,health=dict(query=dict(status='gradient_feedback',calibration=dict(converged=True)))))
        source=[dict(source_usable=True) for _ in range(3)]
        good,_=summarize(blocks,source)
        self.assertEqual(good['selected_M2_versions'],['v1','v2'])
        for row in blocks:
            if row['version']=='v2' and row['domain']==3:
                row['gradients'][0]['cos_A']=None
        failed,_=summarize(blocks,source)
        self.assertEqual(failed['selected_M2_versions'],['v1'])
        source[0]['source_usable']=False;source[1]['source_usable']=False
        self.assertEqual(summarize(blocks,source)[0]['selected_M2_versions'],[])


if __name__=='__main__':
    unittest.main()
