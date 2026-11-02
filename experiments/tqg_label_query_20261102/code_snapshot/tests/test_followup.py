import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from helpers import fixture_files, small_config, task_from
from tqg_gda.experiment_io import run_verified_job, checkpoint_metadata
from tqg_gda.models import GraphClassifier
from tqg_gda.training import TQGTrainer, model_spec, run_warmup
from tqg_gda.utils import merge


class VerifiedJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def action(self,path):
        path.mkdir(); (path/'result.json').write_text('{"status":"completed"}')
        return {'result':path/'result.json'}

    def test_identical_job_reuses_but_changed_inputs_are_rejected(self):
        job=self.root/'job'; contract={'config':{'seed':3},'code':'v1','data':'d1'}
        first=run_verified_job(job,contract,self.action)
        second=run_verified_job(job,contract,lambda _:self.fail('Should reuse verified job'))
        self.assertEqual(first,second)
        for key in ('config','code','data'):
            changed=copy.deepcopy(contract); changed[key]='changed'
            with self.assertRaises(ValueError):run_verified_job(job,changed,self.action)

    def test_empty_marker_and_tampered_output_are_rejected(self):
        job=self.root/'empty'; job.mkdir(); (job/'completed.json').write_text('{}')
        with self.assertRaises(ValueError):run_verified_job(job,{},self.action)
        job=self.root/'valid'; outputs=run_verified_job(job,{},self.action)
        outputs['result'].write_text('changed')
        with self.assertRaises(ValueError):run_verified_job(job,{},self.action)

    def test_failed_jobs_never_receive_completed_marker(self):
        job=self.root/'failed'
        def fail(_):raise RuntimeError('planned test failure')
        with self.assertRaises(RuntimeError):run_verified_job(job,{},fail)
        self.assertFalse((job/'completed.json').exists());self.assertTrue((job/'failure.txt').exists())

    def test_metadata_separates_final_checkpoint_and_selected_ancestor(self):
        state={'kind':'tqg_adaptation','step':200,'selection':{'step':10,'ce':.24}}
        meta=checkpoint_metadata(state,.253)
        self.assertEqual(meta['model_step'],200);self.assertEqual(meta['selected_step'],10)
        self.assertEqual(meta['checkpoint_source_validation_ce'],.253)
        self.assertEqual(meta['best_source_validation_ce'],.24)


class FollowupTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.files=fixture_files(self.root/'inputs')
        self.config=small_config();self.task=task_from(self.files,self.config)

    def model(self):return GraphClassifier(model_spec(self.task,self.config),2).double()

    def test_label_permutation_preserves_counts_and_other_rng_streams(self):
        from tqg_gda.feedback import build_feedback
        config=merge(self.config,{'adaptation':{'query_label_mode':'permute_within_batch','query_permutation_seed':113}})
        model=self.model();a=TQGTrainer(copy.deepcopy(model),self.config);b=TQGTrainer(copy.deepcopy(model),config)
        seen=[]
        def capture(*args,**kw):
            seen.append((args[1].y.clone(),args[2].clone()))
            return build_feedback(*args,**kw)
        with patch('tqg_gda.training.build_feedback',side_effect=capture):
            for _ in range(4):
                ra=a.step(self.task);rb=b.step(self.task)
                self.assertEqual(ra['batches'],rb['batches'])
                self.assertTrue(torch.equal(a.probe_rng.get_state(),b.probe_rng.get_state()))
        for original,passed in seen:
            self.assertTrue(torch.equal(original.sort().values,passed.sort().values))
        self.assertTrue(any(not torch.equal(x,y) for x,y in seen))

    def test_permutation_rng_is_restored_with_the_training_state(self):
        config=merge(self.config,{'adaptation':{'query_label_mode':'permute_within_batch','query_permutation_seed':227}})
        first=TQGTrainer(self.model(),config);first.step(self.task);state=copy.deepcopy(first.state())
        expected=first.step(self.task)
        second=TQGTrainer(self.model(),config);second.restore(state);actual=second.step(self.task)
        expected.pop('seconds');actual.pop('seconds');self.assertEqual(expected,actual)
        for n,p in first.model.state_dict().items():self.assertTrue(torch.equal(p,second.model.state_dict()[n]))

    def test_exact_warmup_budget_and_source_roles(self):
        config=merge(self.config,{'method':'source_only','warmup':{'source_pool':'all_source','update_budget':5,'validation_interval_updates':2}})
        task=task_from(self.files,config,target=False)
        with contextlib.redirect_stdout(io.StringIO()):run_warmup(task,config,self.root/'warmup')
        rows=[json.loads(x) for x in (self.root/'warmup/history.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),5);self.assertEqual([r['step'] for r in rows if 'source_validation' in r],[2,4,5])
        self.assertTrue(all(set(r['source_ids'])<=set(task.train_ids) for r in rows))
        result=json.loads((self.root/'warmup/result.json').read_text());self.assertEqual(result['completed_updates'],5)

    def test_heldout_features_reuse_the_original_shared_scale(self):
        from tqg_gda.diagnostics import heldout_response_features
        from tqg_gda.feedback import Snapshot,ProbeConfig,build_feedback
        model=self.model();source=self.task.batch(self.task.query_ids,'query')
        target=self.task.batch(self.task.target_ids[:3],'target')
        for mode in ('odd_even','odd','linear'):
            cfg=ProbeConfig(count=2,responses=mode)
            feedback=build_feedback(model,source,source.y,target,cfg,torch.Generator().manual_seed(13))
            snap=Snapshot(model)
            source_features=heldout_response_features(snap,source,feedback,cfg)
            target_features=heldout_response_features(snap,target,feedback,cfg)
            torch.testing.assert_close(source_features,feedback.source_features,atol=0,rtol=0)
            torch.testing.assert_close(target_features,feedback.target_features,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
