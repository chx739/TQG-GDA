"""Approved paired pilot: explicit train/seal/evaluate phases, no implicit target scoring."""
from __future__ import annotations
import argparse
import contextlib
import copy
import csv
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
sys.path.append(str(Path(__file__).resolve().parents[1]/'runtime_deps'))
from torch.utils.tensorboard import SummaryWriter
from tqg_gda.data import GraphDomainTask
from tqg_gda.experiment_io import run_verified_job, checkpoint_metadata
from tqg_gda.metrics import evaluate_source
from tqg_gda.training import run_warmup, run_adaptation, load_model, environment
from tqg_gda.utils import sha256, fingerprint, validate_config, write_json

CODE = Path(__file__).resolve().parent
STAGE = CODE.parent
RUN = STAGE.parent


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git_commit():
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE, text=True).strip()
    if dirty:
        raise ValueError('Training requires a clean committed code tree: ' + dirty)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()


def hardware():
    cpu = platform.processor()
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            cpu = line.split(':', 1)[1].strip(); break
    return dict(device='cpu', cpu_model=cpu, logical_cpus=os.cpu_count(), threads=1,
                platform=platform.platform(), **environment())


def flatten_numbers(value, prefix=''):
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, dict):
            yield from flatten_numbers(item, name + '/')
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            yield name, item


def mirror_tensorboard(job, output):
    """Mirror every numerical source/training metric; never inspect target labels."""
    with SummaryWriter(str(job/'tensorboard')) as writer:
        writer.add_text('config', json.dumps(read(output/'config.json'), sort_keys=True))
        for line in (output/'history.jsonl').read_text().splitlines():
            row = json.loads(line)
            step = row.get('step', row.get('epoch', 0))
            for name, value in flatten_numbers(row):
                if name.startswith('batches/') or name.endswith('_ids'):
                    continue
                writer.add_scalar(name, value, step)
        writer.flush()


class Pilot:
    def __init__(self, config_path=CODE/'configs/pilot/controller.json'):
        self.config_path = Path(config_path).resolve()
        self.plan = read(self.config_path)
        self.roles_path = RUN/'stage2_method/data_roles/manifest.json'
        self.roles = read(self.roles_path)
        self.commit = git_commit()
        self.hardware = hardware()
        self.plan_sha = sha256(RUN/'stage2_method/experiment_plan.yaml')
        self.data = dict(cache=sha256(self.plan['cache']), source_split=sha256(self.plan['source_split']),
                         roles_manifest=sha256(self.roles_path))
        if self.data['cache'] != 'ba6437e6a8ea4083e1ff21fab96d206cb88ec8e8d4f1804abc7457648c0cf473':
            raise ValueError('Official cache changed')
        for role in self.roles['domains'].values():
            for info in (dict(path=role['index_path'],sha256=role['index_sha256']),role['development'],role['final']):
                if sha256(info['path']) != info['sha256']:
                    raise ValueError('Target label-blind index changed')
        self.events = STAGE/'monitor_events.jsonl'

    def emit(self, event, **kw):
        payload = dict(timestamp=stamp(),event=event,**kw)
        with self.events.open('a') as stream:
            stream.write(json.dumps(payload,allow_nan=False)+'\n')
        print(json.dumps(payload),flush=True)

    def config(self, seed, arm='original_TQG_target_probes'):
        c = copy.deepcopy(self.plan['base']); c['seed'] = seed
        if arm == 'full_source_CE_700plus200':
            c['method'] = 'source_only'
            c['warmup']['source_pool'] = c['adaptation']['source_pool'] = 'all_source'
        elif arm == 'source_fit_CE':
            c['method'] = 'source_only'
        elif arm == 'TQG_random_probes':
            c['probes']['direction'] = 'random'
        elif arm == 'TQG_source_probes':
            c['probes']['direction'] = 'source'
        elif arm == 'confidence_pseudo_labels':
            c['adaptation'].update(feedback_predictor='pseudo_labels',confidence_threshold=.9)
        elif arm != 'original_TQG_target_probes':
            raise ValueError('Unknown arm')
        validate_config(c)
        return c

    def task(self, config, domain=None):
        return GraphDomainTask(self.plan['cache'],self.plan['source_split'],
            target_index=self.roles['domains'][str(domain)]['index_path'] if domain is not None else None,
            target_policy='transductive-unlabeled' if domain is not None else None,
            fit_fraction=config['split']['fit_fraction'],split_seed=config['split']['seed'],
            use_node_attributes=config['use_node_attributes'],preprocess_pool=config['warmup']['source_pool'])

    def specs(self):
        jobs=[]
        for pool,arm in [('fit','original_TQG_target_probes'),('all_source','full_source_CE_700plus200')]:
            for seed in self.plan['seeds']:
                jobs.append(dict(kind='warmup',name=f'warmup_{pool}_{seed}',seed=seed,pool=pool,
                                 domain=None,arm=arm,config=self.config(seed,arm)))
        for domain in self.plan['domains']:
            for arm in self.plan['arms']:
                for seed in self.plan['seeds']:
                    pool='all_source' if arm=='full_source_CE_700plus200' else 'fit'
                    jobs.append(dict(kind='adaptation',name=f'domain{domain}_{arm}_{seed}',seed=seed,
                        pool=pool,domain=domain,arm=arm,config=self.config(seed,arm)))
        return jobs

    def job_path(self,spec):
        return STAGE/'jobs'/spec['name']

    def checkpoint(self,spec):
        return STAGE/'jobs'/f"warmup_{spec['pool']}_{spec['seed']}"/'artifacts/best.pt'

    def contract(self,spec):
        result=dict(schema_version=1,git_commit=self.commit,pilot_config_sha256=sha256(self.config_path),
            plan_sha256=self.plan_sha,hardware=self.hardware,data=self.data,spec=spec)
        if spec['domain'] is not None:
            result['target_index_sha256']=self.roles['domains'][str(spec['domain'])]['index_sha256']
            result['source_checkpoint_sha256']=sha256(self.checkpoint(spec))
        return result

    def prepare(self):
        specs=self.specs()
        if len(specs)!=70 or sum(s['kind']=='warmup' for s in specs)!=10:
            raise ValueError('Plan must have exactly10warmups and60adaptations')
        records=[dict(task=f"0_to_{s['domain']}",seed=s['seed'],arm=s['arm'],warmup_checkpoint=str(self.checkpoint(s)),
            final_checkpoint=str(self.job_path(s)/'artifacts/last.pt'),selected_checkpoint=str(self.job_path(s)/'artifacts/selected.pt'),
            job_directory=str(self.job_path(s))) for s in specs if s['kind']=='adaptation']
        payload=dict(schema_version=1,git_commit=self.commit,code_git_commit=self.commit,pilot_config_sha256=sha256(self.config_path),
            plan_sha256=self.plan_sha,data=self.data,hardware=self.hardware,jobs=specs,records=records,
            warmup_count=10,adaptation_count=60,target_labels_read_for_training=False)
        path=STAGE/'training_manifest.json'
        if path.exists() and read(path)!=payload:
            raise ValueError('Frozen training manifest changed')
        write_json(path,payload)
        return payload

    def approval(self):
        a=read(STAGE/'approval.json')
        if not a.get('approved') or a.get('cpu_wall_hours')!=2 or a.get('gpu_hours')!=0:
            raise PermissionError('Require explicit approved2CPUwallhours/0GPUhours')
        return a

    def remaining(self):
        path=STAGE/'compute_ledger.json'
        if not path.exists():
            self.approval()
            write_json(path,dict(started_at=stamp(),started_epoch=time.time(),hard_cap_seconds=7200,
                deadline_unix=time.time()+7200,gpu_hours=0,cpu_wall_seconds=0,job_seconds=0,status='running'))
        ledger=read(path)
        elapsed=time.time()-ledger['started_epoch']
        ledger['cpu_wall_seconds']=elapsed
        for fraction in (.5,.8):
            key=f'notified_{fraction}'
            if elapsed>=7200*fraction and not ledger.get(key):
                self.emit('budget_warning',fraction=fraction,elapsed_seconds=elapsed)
                ledger[key]=True
        write_json(path,ledger)
        if elapsed>=5760 and not self.approval().get('continue_after_80_percent',False):
            raise TimeoutError('80%budget checkpoint: preserve artifacts and request continuation')
        return max(0,7200-elapsed)

    def run_job(self,spec):
        contract=self.contract(spec)
        job=self.job_path(spec)
        if (job/'completed.json').exists():
            return run_verified_job(job,contract,lambda out:None)
        timeout=self.remaining()
        if timeout<=0:
            raise TimeoutError('Approved2hour wall budget exhausted')
        started=time.perf_counter()
        def action(output):
            write_json(job/'worker.json',dict(spec=spec,contract=contract))
            write_json(job/'RUN_INFO.json',dict(timestamp=stamp(),git_commit=self.commit,seed=spec['seed'],
                config_name=spec['arm'],full_config=spec['config'],hardware=self.hardware,
                data_hashes=self.data,plan_sha256=self.plan_sha,gpu_hours=0,job=spec['name']))
            with (job/'stderr.log').open('x') as err, SummaryWriter(str(job/'tensorboard')) as writer:
                writer.add_text('provenance',json.dumps(contract,sort_keys=True))
                process=subprocess.Popen([sys.executable,str(CODE/'experiment.py'),'worker','--job',str(job)],
                    cwd=CODE,stderr=err,env={**os.environ,'OMP_NUM_THREADS':'1',
                    'PYTHONDONTWRITEBYTECODE':'1'},stdout=sys.stdout)
                consumed=0;best_loss=float('inf');last_improvement=0;warning_emitted=False
                try:
                    while True:
                        history=output/'history.jsonl'
                        if history.exists():
                            text=history.read_text();lines=text.splitlines()
                            if text and not text.endswith('\n'):lines=lines[:-1]
                            for line in lines[consumed:]:
                                row=json.loads(line);step=row.get('step',row.get('epoch',0))
                                for name,value in flatten_numbers(row):
                                    if not math.isfinite(value):raise FloatingPointError('Nonfinite history metric '+name)
                                    writer.add_scalar(name,value,step)
                                loss=row.get('total_loss',row.get('source_ce'))
                                if loss is not None and loss<best_loss:
                                    best_loss=loss;last_improvement=step
                                if step-last_improvement>100 and not warning_emitted:
                                    self.emit('loss_not_improving_warning',job=spec['name'],step=step,action='continue_fixed_budget')
                                    warning_emitted=True
                            consumed=len(lines);writer.flush()
                        status=process.poll()
                        if status is not None:
                            if status:raise subprocess.CalledProcessError(status,process.args)
                            break
                        if time.perf_counter()-started>=timeout:
                            raise subprocess.TimeoutExpired(process.args,timeout)
                        time.sleep(.2)
                finally:
                    if process.poll() is None:
                        process.kill();process.wait()
                for line in (output/'history.jsonl').read_text().splitlines()[consumed:]:
                    row=json.loads(line);step=row.get('step',row.get('epoch',0))
                    for name,value in flatten_numbers(row):writer.add_scalar(name,value,step)
                writer.flush()
            outputs=dict(final=output/'last.pt',best=output/('best.pt' if spec['kind']=='warmup' else 'selected.pt'),
                result=output/'result.json',history=output/'history.jsonl',config=output/'config.json',
                environment=output/'environment.json',data_roles=output/'data_roles.json',worker_metrics=job/'worker_metrics.json',run_info=job/'RUN_INFO.json')
            return outputs
        try:
            result=run_verified_job(job,contract,action)
        except Exception as e:
            flag='timeout' if isinstance(e,subprocess.TimeoutExpired) else 'nan' if 'nonfinite' in str(e).lower() else 'failure'
            self.emit('job_failed',job=spec['name'],flag=flag,error=str(e));raise
        finally:
            seconds=time.perf_counter()-started
            ledger=read(STAGE/'compute_ledger.json');ledger['job_seconds']+=seconds
            ledger['cpu_wall_seconds']=time.time()-ledger['started_epoch']
            write_json(STAGE/'compute_ledger.json',ledger)
        self.emit('job_completed',job=spec['name'],seconds=seconds,flag='clean')
        return result

    def train(self):
        self.approval();self.prepare()
        for spec in self.specs():
            self.run_job(spec)
        self.seal()

    def seal(self):
        entries=[]
        for spec in self.specs():
            if not (self.job_path(spec)/'completed.json').is_file():
                raise PermissionError('Cannot seal incomplete training matrix: '+spec['name'])
            outputs=run_verified_job(self.job_path(spec),self.contract(spec),lambda out:None)
            entries.append(dict(name=spec['name'],outputs={key:dict(path=str(path),sha256=sha256(path)) for key,path in outputs.items()}))
        payload=dict(schema_version=1,git_commit=self.commit,plan_sha256=self.plan_sha,
            manifest_sha256=sha256(STAGE/'training_manifest.json'),jobs=entries,
            all_70_training_jobs_completed=True,target_labels_used=False)
        path=STAGE/'training_seal.json'
        if path.exists() and read(path)!=payload:
            raise ValueError('Training evidence changed after seal')
        write_json(path,payload);self.emit('training_sealed',sha256=sha256(path))

    def event_flags(self,job_name):
        flags=set()
        if self.events.exists():
            for line in self.events.read_text().splitlines():
                event=json.loads(line)
                if event.get('job')==job_name and event.get('flag') in ('oom','nan','restart','timeout'):
                    flags.add(event['flag'])
        return ','.join(sorted(flags)) if flags else 'clean'

    def evaluate(self):
        # Check every job hash again; seal alone is not enough to authorize scoring.
        self.seal()
        evidence=STAGE/'source_evidence.json'
        if not evidence.exists():
            raise PermissionError('Source diagnostic evidence must be sealed before target development scoring')
        spec=importlib.util.spec_from_file_location('pilot_standalone_eval',CODE/'scripts/evaluate.py')
        evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
        rows=[]
        for domain in self.plan['domains']:
            for seed in self.plan['seeds']:
                frozen=dict(kind='warmup',name=f'warmup_fit_{seed}',seed=seed,pool='fit',domain=None,
                    arm='original_TQG_target_probes',config=self.config(seed))
                entries=[('frozen_source','frozen_source',self.checkpoint(frozen),frozen)]
                for arm in self.plan['arms']:
                    s=next(s for s in self.specs() if s['kind']=='adaptation' and s['domain']==domain and s['seed']==seed and s['arm']==arm)
                    entries.extend((arm,role,self.job_path(s)/'artifacts'/file,s) for role,file in [('fixed_final','last.pt'),('source_selected','selected.pt')])
                for arm,role,checkpoint,s in entries:
                    self.remaining()
                    roleinfo=self.roles['domains'][str(domain)];dev=roleinfo['development'];final=roleinfo['final']
                    ids=np.load(dev['path']).tolist()
                    if set(ids)&set(np.load(final['path']).tolist()):
                        raise PermissionError('Target role overlap')
                    contract=dict(kind='target_development_eval',training_seal_sha256=sha256(STAGE/'training_seal.json'),
                        source_evidence_sha256=sha256(evidence),checkpoint_sha256=sha256(checkpoint),domain=domain,
                        development_sha256=dev['sha256'],git_commit=self.commit,role=role)
                    def action(output):
                        output.mkdir()
                        metrics,used,probabilities,labels=evaluator.evaluate_target(checkpoint,self.plan['cache'],self.plan['source_split'],dev['path'])
                        if set(used)!=set(ids):raise ValueError('Wrong evaluation graph IDs')
                        confidence,pred=probabilities.max(-1);correct=(pred==labels).double();ece=0.
                        for bin_index in range(10):
                            mask=(confidence>=bin_index/10)&((confidence<(bin_index+1)/10) if bin_index<9 else (confidence<=1))
                            if mask.any():ece+=float(mask.double().mean()*(confidence[mask].mean()-correct[mask].mean()).abs())
                        model,state=load_model(checkpoint)
                        source=evaluate_source(model,self.task(state['config']))
                        job=self.job_path(s);wm=read(job/'worker_metrics.json') if (job/'worker_metrics.json').exists() else {}
                        row=dict(run_id=self.plan['run_id'],config_name=arm,task=f'PROTEINS_0_to_{domain}',seed=seed,
                            training_job=s['name'],training_wall_seconds=wm.get('wall_seconds',0),
                            checkpoint_role=role,git_commit=self.commit,gpu_hours=0,cpu_wall_hours=wm.get('wall_seconds',0)/3600,
                            hardware=self.hardware['cpu_model'],primary_metric=metrics['balanced_accuracy'],
                            balanced_accuracy=metrics['balanced_accuracy'],accuracy=metrics['accuracy'],cross_entropy=metrics['ce'],
                            calibration_ece=ece,source_validation_cross_entropy=source['ce'],peak_memory_bytes=wm.get('peak_memory_bytes'),
                            event_flags=self.event_flags(s['name']),notes='Exploratory pilot; final target labels excluded; target development post-training only',
                            checkpoint_sha256=sha256(checkpoint),checkpoint_path=str(checkpoint),
                            **checkpoint_metadata(state,source['ce']))
                        write_json(output/'metrics.json',dict(record=row,classification=metrics,final_target_labels_read=False))
                        return dict(metrics=output/'metrics.json')
                    outputs=run_verified_job(STAGE/'evaluations'/f'domain{domain}_{arm}_{seed}_{role}',contract,action)
                    rows.append(read(outputs['metrics'])['record'])
        with (STAGE/'results.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        self.emit('target_development_evaluation_completed',rows=len(rows),final_target_labels_read=False)


def worker(job):
    info=read(job/'worker.json');s=info['spec'];c=s['config']
    started=time.perf_counter()
    # Worker instantiates from frozen config, not matrix discovery or target evaluator.
    roles=read(RUN/'stage2_method/data_roles/manifest.json');p=read(CODE/'configs/pilot/controller.json')
    task=GraphDomainTask(p['cache'],p['source_split'],
        target_index=roles['domains'][str(s['domain'])]['index_path'] if s['domain'] is not None else None,
        target_policy='transductive-unlabeled' if s['domain'] is not None else None,
        fit_fraction=c['split']['fit_fraction'],split_seed=c['split']['seed'],
        use_node_attributes=c['use_node_attributes'],preprocess_pool=c['warmup']['source_pool'])
    if s['kind']=='warmup':
        run_warmup(task,c,job/'artifacts')
    else:
        checkpoint=STAGE/'jobs'/f"warmup_{s['pool']}_{s['seed']}"/'artifacts/best.pt'
        run_adaptation(task,c,checkpoint,job/'artifacts')
    write_json(job/'worker_metrics.json',dict(wall_seconds=time.perf_counter()-started,
        peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        gpu_hours=0,hardware=info['contract']['hardware'],event_flags='clean',
        graph_exposures={role:sum(len(json.loads(line).get('batches',{}).get(role,[])) for line in (job/'artifacts/history.jsonl').read_text().splitlines()) for role in ('fit','query','target')},
        source_graphs_processed=read(job/'artifacts/result.json').get('source_graphs_processed')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','train','seal','evaluate','worker'])
    parser.add_argument('--job',type=Path)
    args=parser.parse_args()
    if args.phase=='worker':
        if args.job is None:parser.error('--job required')
        worker(args.job.resolve());return
    pilot=Pilot()
    getattr(pilot,args.phase)()


if __name__=='__main__':main()
