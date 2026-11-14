"""The full-source diagnostic must change the source slice only."""
import copy
import unittest
import torch
from tsba.adapt import adapt_classifier
from tsba.classifier import ParameterMap
from tsba.config import Config
from experiments.synthetic.e2_controls_20261109.src.controls import adapt_control
from experiments.synthetic.e1_e2_20261109.src.distributions import setup_cell, oracles
from experiments.synthetic.e2_full_source_20261110.src.execution import full_source_config, full_source_schedule, audit_full_source_event


class FullSourceTests(unittest.TestCase):
    def make_case(self):
        old=Config(seed=0,adapt_rounds=2,source_adapt_batch=128,target_batch=16)
        cell=setup_cell(2,old.seed,old)
        rng=torch.Generator().manual_seed(old.seed+201)
        prior=[dict(source_ids=torch.randperm(512,generator=rng)[:128],target_ids=torch.randperm(512,generator=rng)[:16]) for _ in range(2)]
        return old,cell,prior

    def test_only_source_batch_changes_and_target_rng_is_preserved(self):
        old,cell,prior=self.make_case();cfg=full_source_config(old,512)
        changed=[k for k,v in cfg.to_dict().items() if v!=old.to_dict()[k]]
        self.assertEqual(changed,['source_adapt_batch'])
        schedule=full_source_schedule(prior,old,512,512)
        for a,b in zip(schedule,prior):
            self.assertTrue(torch.equal(a['source_ids'].sort().values,torch.arange(512)))
            self.assertTrue(torch.equal(a['source_ids'][:128],b['source_ids']))
            self.assertTrue(torch.equal(a['target_ids'],b['target_ids']))
        corrupt=copy.deepcopy(prior);corrupt[0]['target_ids'][0]=-1
        with self.assertRaises(ValueError):full_source_schedule(corrupt,old,512,512)

    def test_three_paths_use_full_source_ce_for_all_trials(self):
        old,cell,prior=self.make_case();cfg=full_source_config(old,512)
        schedule=full_source_schedule(prior,old,512,512)
        pop,_=oracles(cell['target'],cell['coordinates'],.3)
        for arm in ('source_ce_ggn','zero_zeta_geometry','population'):
            head=copy.deepcopy(cell['head']);events=[]
            def observe(event):
                audit_full_source_event(head,cell,schedule[event['record']['round']],event);events.append(event)
            if arm=='population':
                result=adapt_classifier(head,cell['source'],cell['source_y'],cell['target'],pop,.3,cfg,observer=observe)
            else:
                result=adapt_control(head,cell['source'],cell['source_y'],cell['target'],.3,cfg,schedule,arm,observer=observe)
            self.assertEqual(len(events),2);self.assertEqual(result['accepted'],2)
            for event in events:
                self.assertLessEqual(event['full_source_ce_audit']['max_logged_ce_error'],1e-12)
                if arm!='population':self.assertLessEqual(event['full_source_ce_audit']['change'],1e-10)
            bad=copy.deepcopy(events[0]);bad['record']['trials'][0]['source_ce']+=.1
            with self.assertRaises(ValueError):audit_full_source_event(head,cell,schedule[0],bad)
