"""Frozen E2 ablations with verified jobs and development-only evaluation."""
import argparse
import csv
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from tqg_gda.data import GraphDomainTask
from tqg_gda.experiment_io import run_verified_job,checkpoint_metadata
from tqg_gda.metrics import evaluate_source
from tqg_gda.training import load_model,run_adaptation,run_warmup
from tqg_gda.utils import load_config,sha256,fingerprint,write_json,validate_config


def module(name):
    spec=importlib.util.spec_from_file_location('e2_'+name,ROOT/'scripts'/f'{name}.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


class Followup:
    def __init__(self,plan_path,output):
        self.plan=json.loads(Path(plan_path).read_text());self.output=Path(output).resolve()
        self.plan_hash=sha256(plan_path);self.old=Path(self.plan['previous_run'])
        for rel,digest in self.plan['code_sha256'].items():
            if sha256(ROOT/rel)!=digest:raise ValueError('Execution code changed after plan freeze: '+rel)
        for key in ('cache','source_manifest','target_index','development_index','final_index'):
            item=self.plan['data'][key]
            if sha256(item['path'])!=item['sha256']:raise ValueError('Data role changed: '+key)
        dev=set(np.load(self.path('development_index'),allow_pickle=False).tolist())
        final=set(np.load(self.path('final_index'),allow_pickle=False).tolist())
        if dev&final or dev|final!=set(np.load(self.path('target_index'),allow_pickle=False).tolist()):
            raise ValueError('Invalid target development/final partition')
        self.evaluator=module('evaluate');self.records=[];self.started=time.perf_counter()

    def path(self,key):return self.plan['data'][key]['path']

    def emit(self,stage,**kw):
        value=dict(stage=stage,**kw)
        with (self.output/'events.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
        print(json.dumps(value,allow_nan=False),flush=True)

    def contract(self,stage,**kw):
        return dict(schema=1,experiment=self.plan['experiment'],plan_sha256=self.plan_hash,stage=stage,**kw)

    def task(self,config,target=True):
        return GraphDomainTask(self.path('cache'),self.path('source_manifest'),
            target_index=self.path('target_index') if target else None,
            target_policy='transductive-unlabeled' if target else None,
            fit_fraction=config['split']['fit_fraction'],split_seed=config['split']['seed'],
            use_node_attributes=config['use_node_attributes'],preprocess_pool=config['warmup']['source_pool'])

    def config(self,name,seed):
        config=validate_config(load_config(ROOT/f'configs/{name}.json'));config['seed']=seed
        expected=self.plan['resolved_configs'][name]
        actual=dict(config);actual.pop('seed')
        if actual!=expected:raise ValueError('Resolved config changed: '+name)
        return config

    def initial_checkpoint(self,name,seed,config):
        if name!='source_all_700updates':
            checkpoint=self.old/f'source_fit_{seed}/source/best.pt'
            if sha256(checkpoint)!=self.plan['source_checkpoint_sha256'][str(seed)]:
                raise ValueError('Initial source checkpoint changed')
            return checkpoint
        directory=self.output/f'jobs/warmup_700_{seed}'
        def train(path):
            run_warmup(self.task(config,target=False),config,path)
            return {'selected':path/'best.pt','final':path/'last.pt','result':path/'result.json','history':path/'history.jsonl'}
        outputs=run_verified_job(directory,self.contract('warmup_700',config=config),train)
        self.emit('warmup_completed',seed=seed)
        return outputs['selected']

    def train(self,name,seed):
        config=self.config(name,seed);checkpoint=self.initial_checkpoint(name,seed,config)
        def train(path):
            run_adaptation(self.task(config),config,checkpoint,path)
            return {'source_selected':path/'selected.pt','fixed_final':path/'last.pt',
                    'result':path/'result.json','history':path/'history.jsonl'}
        self.emit('training',method=name,seed=seed)
        outputs=run_verified_job(self.output/f'jobs/{name}_{seed}',
            self.contract('adapt',config=config,source_checkpoint_sha256=sha256(checkpoint)),train)
        result=json.loads(outputs['result'].read_text())
        for role in ('source_selected','fixed_final'):
            self.evaluate(name,seed,role,outputs[role],result,origin='new_training')

    def reference(self,name,seed):
        if name=='source_frozen':
            checkpoint=self.old/f'source_fit_{seed}/source/best.pt'
            self.evaluate(name,seed,'frozen',checkpoint,None,origin='verified_previous_checkpoint')
            return
        folder=self.old/f'{name}_{seed}/adaptation'
        result=json.loads((folder/'result.json').read_text())
        if result['completed_steps']!=200 or result['target_labels_used']:
            raise ValueError('Previous training does not match reference contract')
        for role,file in (('source_selected','selected.pt'),('fixed_final','last.pt')):
            self.evaluate(name,seed,role,folder/file,result,origin='verified_previous_checkpoint')

    def evaluate(self,name,seed,role,checkpoint,training,origin):
        checkpoint=Path(checkpoint)
        expected=self.plan.get('reference_checkpoint_sha256',{}).get(f'{name}/{seed}/{role}')
        if expected is not None and sha256(checkpoint)!=expected:raise ValueError('Reference checkpoint changed')
        contract=self.contract('evaluate',method=name,seed=seed,role=role,checkpoint_sha256=sha256(checkpoint),
                               development_index_sha256=self.plan['data']['development_index']['sha256'])
        def action(path):
            path.mkdir()
            metrics,ids,_,_=self.evaluator.evaluate_target(checkpoint,self.path('cache'),self.path('source_manifest'),self.path('development_index'))
            if set(ids)!=set(np.load(self.path('development_index'),allow_pickle=False).tolist()):
                raise ValueError('Unexpected evaluation graph IDs')
            model,state=load_model(checkpoint)
            source_ce=evaluate_source(model,self.task(state['config'],target=False))['ce']
            metadata=checkpoint_metadata(state,source_ce,training)
            record=dict(config_name=name,seed=seed,checkpoint_role=role,origin=origin,
                **{k:metrics[k] for k in ('accuracy','balanced_accuracy','ce','auroc','graphs')},**metadata,
                checkpoint_sha256=sha256(checkpoint),adaptation_seconds=training['seconds'] if training else 0.,
                event_flags='clean',final_target_labels_read=False)
            write_json(path/'metrics.json',dict(record=record,classification=metrics))
            return {'metrics':path/'metrics.json'}
        outputs=run_verified_job(self.output/f'evaluations/{name}_{seed}_{role}',contract,action)
        self.records.append(json.loads(outputs['metrics'].read_text())['record'])
        self.write_results()

    def write_results(self):
        keys=[(r['config_name'],r['seed'],r['checkpoint_role']) for r in self.records]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate method/seed/checkpoint record')
        with (self.output/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(self.records[0]));writer.writeheader();writer.writerows(self.records)
        grouped={}
        for row in self.records:grouped.setdefault((row['config_name'],row['checkpoint_role']),[]).append(row)
        methods={}
        for (name,role),rows in grouped.items():
            metrics={}
            for key in ('accuracy','balanced_accuracy','ce','auroc'):
                values=[r[key] for r in rows if r[key] is not None]
                metrics[key]=dict(mean=statistics.mean(values) if values else None,
                    seed_std=statistics.stdev(values) if len(values)>1 else None)
            methods[name+'/'+role]=dict(seeds=[r['seed'] for r in rows],metrics=metrics,
                model_steps=[r['model_step'] for r in rows],selected_steps=[r['selected_step'] for r in rows])
        comparisons={}
        tqg_name='tqg_development'
        for name in self.plan['new_methods']+['random_probes','source_only_same_fit','source_only_all_labels']:
            for role in ('fixed_final','source_selected'):
                a={r['seed']:r for r in self.records if r['config_name']==name and r['checkpoint_role']==role}
                b={r['seed']:r for r in self.records if r['config_name']==tqg_name and r['checkpoint_role']==role}
                seeds=sorted(set(a)&set(b));values=[a[s]['accuracy']-b[s]['accuracy'] for s in seeds]
                if values:comparisons[name+'/'+role]=dict(seeds=seeds,accuracy_differences_vs_tqg=values,
                    mean=statistics.mean(values),seed_std=statistics.stdev(values) if len(values)>1 else None,
                    wins=sum(x>0 for x in values),ties=sum(x==0 for x in values),losses=sum(x<0 for x in values))
        # Permutation seeds are nested repeats; average them BEFORE computing variation over model seeds.
        for role in ('fixed_final','source_selected'):
            per_seed=[]
            for seed in self.plan['seeds']:
                rows=[r for r in self.records if r['seed']==seed and r['checkpoint_role']==role and r['config_name'].startswith('permuted_query_')]
                if len(rows)==len(self.plan['permutation_seeds']):
                    per_seed.append(dict(seed=seed,**{key:statistics.mean(r[key] for r in rows) for key in ('accuracy','balanced_accuracy','ce','auroc')}))
            if per_seed:
                methods['permuted_query_mean/'+role]=dict(seeds=[r['seed'] for r in per_seed],
                    metrics={key:dict(mean=statistics.mean(r[key] for r in per_seed),
                        seed_std=statistics.stdev(r[key] for r in per_seed) if len(per_seed)>1 else None) for key in ('accuracy','balanced_accuracy','ce','auroc')},
                    nested_permutation_seeds=self.plan['permutation_seeds'])
        write_json(self.output/'results_summary.json',dict(methods=methods,comparisons_method_minus_tqg=comparisons,
            primary_checkpoint_role='fixed_final',study_type='exploratory_followup_on_same_development_set',final_target_labels_read=False))

    def repair_previous_metadata(self):
        previous=self.old/'results.csv';rows=list(csv.DictReader(previous.open()));changed=0
        receipt=self.output/'previous_metadata_correction.json'
        corrected_path=self.output/'previous_pilot_corrected.csv'
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved['original_sha256']!=sha256(previous) or saved['corrected_sha256']!=sha256(corrected_path):
                raise ValueError('Previous-result correction inputs or outputs changed')
            return
        corrected=[]
        for row in rows:
            row=dict(row);name,seed,role=row['config_name'],row['seed'],row['checkpoint_role']
            best=float(row['source_validation_ce']) if row['source_validation_ce'] else None
            if name=='source_frozen':
                current=json.loads((self.old/f'source_fit_{seed}/source/result.json').read_text())['source_validation']['ce'];step=0
            elif role=='fixed_final':
                last=json.loads((self.old/f'{name}_{seed}/adaptation/history.jsonl').read_text().splitlines()[-1])
                current=last['source_validation']['ce'];step=last['step']
                changed+=abs(current-best)>1e-12
            else:current=best;step=int(row['selected_step'])
            row.update(model_step=step,checkpoint_source_validation_ce=current,best_source_validation_ce=best,source_validation_ce=current)
            corrected.append(row)
        with corrected_path.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(corrected[0]));writer.writeheader();writer.writerows(corrected)
        write_json(receipt,dict(original_csv=str(previous),corrected_sha256=sha256(corrected_path),
            original_sha256=sha256(previous),fixed_final_source_ce_corrected_rows=changed,
            target_metrics_changed=False,original_file_overwritten=False))

    def run(self):
        self.repair_previous_metadata()
        for seed in self.plan['seeds']:
            for name in self.plan['reference_methods']:self.reference(name,seed)
        self.emit('references_re_evaluated',records=len(self.records))
        for name in self.plan['new_methods']:
            for seed in self.plan['seeds']:
                self.train(name,seed)
                self.emit('training_completed',method=name,seed=seed,records=len(self.records))
        write_json(self.output/'completed.json',dict(status='completed',evaluations=len(self.records),
            new_adaptation_runs=len(self.plan['new_methods'])*len(self.plan['seeds']),
            final_target_labels_read=False,elapsed_seconds=time.perf_counter()-self.started))
        self.emit('all_training_completed',records=len(self.records))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();Followup(args.plan,args.output).run()


if __name__=='__main__':main()
