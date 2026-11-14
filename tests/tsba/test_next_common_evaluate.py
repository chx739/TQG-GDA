"""Synthetic phase isolation and complete-denominator evaluation checks."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.common.io import digest, save_tensor, write_json
from experiments.common.runtime import seal
from experiments.mutagenicity.r1b_密度与单步适配诊断_20261111.src.common import NextLabelBroker, verify_seal
from experiments.mutagenicity.r1b_密度与单步适配诊断_20261111.src.evaluate import evaluate_steps, evaluate_warm


class NextCommonEvaluationTests(unittest.TestCase):
    def test_byte_offset_broker_denies_B_before_OBS_and_never_parses_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'deps/inputs').mkdir(parents=True)
            (root/'logs').mkdir()
            raw = root/'raw.txt'
            raw.write_bytes(b'0\n1\nNOT-A-LABEL\n')
            write_json(root/'deps/roles.json',dict(targets={'1':dict(A=[0],B=[1],final=[2])}))
            write_json(root/'deps/inputs/broker.json',dict(path=str(raw),sha256=digest(raw),offsets=[[0,2],[2,2],[4,12]]))
            write_json(root/'deps/inputs/class_mapping.json',dict(raw_to_class={'0':0,'1':1}))
            seal(root,'INPUT',['deps'])
            broker = NextLabelBroker(root)
            self.assertEqual(broker.read([0],'target_A','fit',domain=1),{0:0})
            raw.write_bytes(b'1\n1\nNOT-A-LABEL\n')
            with self.assertRaises(ValueError):
                broker.read([0],'target_A','changed raw bytes after initialization',domain=1)
            raw.write_bytes(b'0\n1\nNOT-A-LABEL\n')
            with self.assertRaises(FileNotFoundError):
                broker.read([1],'target_B','premature',domain=1)
            with self.assertRaises(PermissionError):
                broker.read([2],'target_A','disguised final',domain=1)
            seal(root,'OBS',['deps','seals/INPUT.json'])
            self.assertEqual(broker.read([1],'target_B','frozen evaluation',domain=1),{1:1})
            (root/'deps/unsealed-added-file.txt').write_text('cannot be consumed after OBS')
            with self.assertRaises(PermissionError):
                broker.read([1],'target_B','added frozen evidence',domain=1)
            (root/'deps/unsealed-added-file.txt').unlink()
            with self.assertRaises(PermissionError):
                broker.read([2],'final','not allowed',domain=1)
            path = root/'seals/OBS.json'
            record = json.loads(path.read_text())
            record['files']['deps/roles.json'] = '0'*64
            path.write_text(json.dumps(record))
            with self.assertRaises(PermissionError):
                broker.read([1],'target_B','changed manifest after cache',domain=1)
            audit = [json.loads(s) for s in (root/'logs/label-access.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['permitted'] for e in audit),2)
            self.assertTrue(all(not e['permitted'] for e in audit if 2 in e['ids']))

    def _metrics_fixture(self,root):
        (root/'deps/source/seed7').mkdir(parents=True)
        (root/'evaluation').mkdir()
        p = torch.tensor([[.8,.2],[.3,.7]]*4,dtype=torch.float64)
        save_tensor(root/'deps/source/seed7/predictions.pt',dict(ids=list(range(8)),probabilities=p,log_probabilities=p.log()))
        write_json(root/'deps/roles.json',dict(targets={'1':dict(B=[0,1])}))
        ids = dict(B1=[0,1],source_fit=[2,3],source_validation=[4,5],source_diagnostic=[6,7],source_all=list(range(2,8)))
        pools = {k:dict(ids=v,probabilities=p[v],log_probabilities=p[v].log()) for k,v in ids.items()}
        return pools,{1:{0:0,1:1}},{7:{i:i%2 for i in range(2,8)}}

    def test_warm_failed_fold_is_NA_with_fixed_denominator_and_all_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pools,b,source=self._metrics_fixture(root)
            for fold in range(4):
                for method,lr in (('warm',.001),('warm',.0001),('random',None)):
                    path=root/'warm'/f'{method}{lr}'/f'fold{fold}'
                    path.mkdir(parents=True)
                    meta=dict(seed=7,domain=1,fold=fold,selected_step=0,status='success')
                    if method=='warm':
                        meta.update(lr=lr,initialization_kind='source_affine')
                    if fold==3 and lr==.001:
                        meta.update(status='failed',failure='synthetic numerical failure')
                    write_json(path/'metadata.json',meta)
                    endpoints=['0','selected','200','800'] if lr else ['0','selected','original_selected']
                    if meta['status']=='success':
                        save_tensor(path/'probabilities.pt',{e:pools for e in endpoints})
                old=root/f'deps/probes/target1/seed7/fold{fold}/H/mlp/true'
                old.mkdir(parents=True)
                write_json(old/'metadata.json',dict(selected_step=20))
                save_tensor(old/'probabilities.pt',dict(selected=dict(B1=pools['B1'])))
            result=evaluate_warm(root,dict(seeds=[7],domains=[1]),b,source)
            row=next(r for r in result['by_seed'] if r['method']=='warm_lr1e-03' and r['endpoint']=='selected' and r['pool']=='B1')
            self.assertEqual((row['available_folds'],row['expected_folds']),(3,4))
            self.assertIsNone(row['nll'])
            paired=next(r for r in result['paired'] if r['method']=='warm_lr1e-03' and r['reference']=='source_initial')
            self.assertIsNone(paired['nll']['mean'])
            self.assertEqual(paired['nll']['expected_count'],1)
            self.assertEqual(paired['nll_improved_seeds'],0)
            good=next(r for r in result['by_seed'] if r['method']=='warm_lr1e-04' and r['endpoint']=='selected' and r['pool']=='B1')
            self.assertIsNotNone(good['nll'])

    def test_failed_step_grid_preserves_all_views_pools_and_NA_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _,b,source=self._metrics_fixture(root)
            (root/'steps').mkdir()
            grid=[dict(seed=7,domain=1,estimator=e,bandwidth_multiplier=h,arm=a,status='failed',failure='synthetic')
                  for e in ('empirical','learned') for h in (1.,2.) for a in ('normal','reverse','zero','matched_source')]
            write_json(root/'steps/index.json',grid)
            result=evaluate_steps(root,dict(seeds=[7],domains=[1]),b,source)
            self.assertEqual(len(result['rows']),160)
            self.assertEqual(len(result['paired']),24)
            self.assertTrue(all(r['nll_difference']['mean'] is None for r in result['paired']))
            self.assertTrue(all(r['nll_difference']['expected_count']==1 for r in result['paired']))


if __name__=='__main__':
    unittest.main()
