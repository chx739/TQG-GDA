"""Frozen Mutagenicity target-batch ablation: prepare and per-seed workers, no target scoring."""
import argparse
from collections import deque
import contextlib
import copy
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from tqg_gda.data import GraphDomainTask
from tqg_gda.training import load_model, run_adaptation, environment
from tqg_gda.utils import fingerprint, sha256, validate_config

CODE = Path(__file__).resolve().parent
STAGE = CODE.parent
RUN = STAGE.parent
PLAN = RUN/'stage2_method/experiment_plan.yaml'
ROLES = RUN/'stage2_method/data_roles'


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def event(kind, **values):
    row = dict(timestamp=stamp(), unix=time.time(), type=kind, **values)
    with (STAGE/'monitor_events.jsonl').open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def current_commit():
    if subprocess.check_output(['git','status','--porcelain'], cwd=CODE, text=True).strip():
        raise ValueError('Execution snapshot must be clean and committed')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()


def hardware():
    cpu = next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
               if line.startswith('model name'))
    return dict(cpu=cpu, device='cpu', threads_per_worker=1, logical_cpus=os.cpu_count(), **environment())


class Context:
    def __init__(self):
        self.plan = read(PLAN)
        self.commit = current_commit()
        self.plan_sha = sha256(PLAN)
        if sha256(CODE/'configs/experiment_plan.yaml') != self.plan_sha:
            raise ValueError('External plan differs from the committed plan copy')
        self._tasks = {}
        path = STAGE/'experiment_manifest.json'
        self.manifest = read(path) if path.exists() else None
        if self.manifest:
            if self.manifest['git_commit'] != self.commit or self.manifest['plan_sha256'] != self.plan_sha:
                raise ValueError('Frozen plan or execution commit changed')
            for asset in self.manifest['assets']:
                if sha256(asset['path']) != asset['sha256']:
                    raise ValueError('Frozen data/model asset changed: '+asset['path'])

    def source_path(self, seed, pool='fit'):
        return STAGE/'source_checkpoints'/pool/str(seed)/'best.pt'

    def task(self, domain, pool='fit'):
        key = (domain,pool)
        if key not in self._tasks:
            self._tasks[key] = GraphDomainTask(self.plan['cache'], ROLES/'source_protocol.json',
                target_index=ROLES/f'0_to_{domain}/target.npy', target_policy=self.plan['target_policy'],
                fit_fraction=.8, split_seed=431, use_node_attributes=False, preprocess_pool=pool)
        return self._tasks[key]

    def config(self, seed, arm):
        settings = self.plan['arms'][arm]
        source = torch.load(self.source_path(seed,settings['pool']),map_location='cpu',weights_only=False)
        config = copy.deepcopy(source['config'])
        config.pop('probes',None)
        config['method'] = settings['method']
        config['seed'] = seed
        config['adaptation'] = {key:value for key,value in self.plan['adaptation'].items()
                                if key not in ('momentum','weight_decay')}
        config['adaptation']['source_pool'] = settings['pool']
        config['adaptation']['target_batch_size'] = settings['target_batch_size']
        config['v2'] = copy.deepcopy(self.plan['v2'])
        if 'calibration_mode' in settings:
            config['v2']['calibration']['mode'] = settings['calibration_mode']
        if 'signature' in settings:
            config['v2']['signature'] = settings['signature']
        for key in ('feedback_predictor','confidence_threshold','query_label_mode','query_permutation_seed'):
            if key in settings:
                config['adaptation'][key] = settings[key]
        return validate_config(config)

    def specs(self):
        return [dict(domain=domain,seed=seed,arm=arm) for seed in self.plan['seeds']
                for domain in self.plan['domains'] for arm in self.plan['execution_order']]

    def job(self, spec):
        return STAGE/'jobs'/f"0_to_{spec['domain']}"/str(spec['seed'])/spec['arm']


def prepare():
    ctx = Context()
    if ctx.manifest:
        return ctx.manifest
    if sha256(ctx.plan['cache']) != ctx.plan['cache_sha256']:
        raise ValueError('Mutagenicity cache changed')
    assets = [dict(path=ctx.plan['cache'],sha256=sha256(ctx.plan['cache']))]
    assets += [dict(path=str(path),sha256=sha256(path)) for path in sorted(ROLES.rglob('*')) if path.is_file()]
    archive = Path(ctx.plan['source_archive'])
    with (archive/'job_index.csv').open() as stream:
        origins = {(int(row['seed']),row['arm']):row for row in csv.DictReader(stream)
                   if row['phase']=='source_pretraining'}
    source_models = []
    for pool in ctx.plan['source_pretraining']['reuse_pools']:
        for seed in ctx.plan['seeds']:
            original = archive/'checkpoints/source_pretraining'/pool/str(seed)/'best.pt'
            destination = ctx.source_path(seed,pool)
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():
                if sha256(destination) != sha256(original):
                    raise ValueError('Existing source copy differs')
            else:
                shutil.copyfile(original,destination)
            state = torch.load(destination,map_location='cpu',weights_only=False)
            if state['config']['warmup'] != dict(epochs=100,batch_size=32,lr=.003,source_pool=pool):
                raise ValueError('Source training configuration changed')
            if state['encoder_spec'] != dict(input_dim=14,hidden_dim=32,num_layers=2,edge_dim=3,readout='mean'):
                raise ValueError('Expected the SEGA 2-layer GINE backbone')
            digest = sha256(destination)
            assets.append(dict(path=str(destination),sha256=digest))
            source_models.append(dict(seed=seed,pool=pool,path=str(destination),sha256=digest,
                original_path=str(original),original_training_commit=origins[(seed,pool)]['training_commit'],
                selected_epoch=state['epoch'],config=state['config']))
    domain_roles = {}
    for domain in ctx.plan['domains']:
        task = ctx.task(domain)
        assert (len(task.fit_ids),len(task.query_ids),len(task.validation_ids)) == (779,196,109)
        dev = np.load(ROLES/f'0_to_{domain}/development.npy',allow_pickle=False).tolist()
        final = np.load(ROLES/f'0_to_{domain}/final.npy',allow_pickle=False).tolist()
        assert len(dev)==542 and not set(dev)&set(final) and set(dev)|set(final)==set(task.target_ids)
        assert not set(task.fit_ids)&set(task.query_ids)
        domain_roles[str(domain)] = dict(development_ids=dev,final_ids=final,target_ids=list(task.target_ids),
            source_fit_ids=list(task.fit_ids),source_query_ids=list(task.query_ids),source_validation_ids=list(task.validation_ids))
    jobs = []
    for spec in ctx.specs():
        config = ctx.config(spec['seed'],spec['arm'])
        jobs.append(dict(**spec,config=config,output=str(ctx.job(spec)),
            source_checkpoint=str(ctx.source_path(spec['seed'],ctx.plan['arms'][spec['arm']]['pool']))))
    references = prepare_references(ctx, assets)
    manifest = dict(schema_version=1,prepared_at=stamp(),git_commit=ctx.commit,plan_sha256=ctx.plan_sha,
        hardware=hardware(),jobs=jobs,count=len(jobs),source_models=source_models,assets=assets,
        references=references,
        roles=domain_roles,target_labels_read_for_training=False,target_final_labels_allowed=False)
    assert len(jobs)==30
    atomic_json(STAGE/'experiment_manifest.json',manifest)
    atomic_json(STAGE/'prepared.json',dict(status='ready',adaptations=30,source_models_reused=5,git_commit=ctx.commit))
    print(json.dumps(dict(status='prepared',jobs=len(jobs),git_commit=ctx.commit)),flush=True)
    return manifest



def prepare_references(ctx, assets):
    """Validate and reference immutable old B16/source-only jobs, no retraining."""
    archive = Path(ctx.plan['reference_archive'])
    archive_manifest = read(archive/'manifest.json')
    by_path = {row['path']:row for row in archive_manifest['files']}
    with (archive/'job_index.csv').open(newline='') as stream:
        indexed = {(row['task'],int(row['seed']),row['arm']):row
                   for row in csv.DictReader(stream) if row['phase']=='adaptation'}
    for path in sorted((CODE/'tqg_gda').glob('*.py')):
        if path.read_bytes() != (archive/'code_snapshot/tqg_gda'/path.name).read_bytes():
            raise ValueError('Reused reference numeric code differs: '+path.name)
    records=[]
    for domain in ctx.plan['domains']:
        for seed in ctx.plan['seeds']:
            for arm in ('B16','B_fit'):
                original_arm = ctx.plan['references'][arm]['original_arm']
                old = indexed[(f'0_to_{domain}',seed,original_arm)]
                rel = Path(f'0_to_{domain}')/str(seed)/original_arm
                candidates = list((archive/'checkpoints'/rel).glob('*.pt'))
                candidates += [archive/old['history']]
                candidates += [archive/'metadata/jobs'/rel/name for name in
                               ('config.json','environment.json','data_roles.json','result.json','completion.json','RUN_INFO_attempt_1.json')]
                artifacts={}
                for source in candidates:
                    recorded=by_path[str(source.relative_to(archive))]
                    digest=sha256(source)
                    if digest!=recorded['sha256']:
                        raise ValueError('Reused reference changed: '+str(source))
                    artifacts[source.name]=dict(path=str(source),sha256=digest)
                    assets.append(dict(path=str(source),sha256=digest))
                expected=ctx.config(seed,'B32')
                expected['adaptation']['target_batch_size']=16
                if arm=='B_fit':
                    expected['method']='source_only'
                if read(artifacts['config.json']['path']) != expected:
                    raise ValueError('Reference configuration differs beyond target batch: '+str(rel))
                if read(artifacts['data_roles.json']['path']) != ctx.task(domain).identity:
                    raise ValueError('Reference data roles differ')
                source_sha=sha256(ctx.source_path(seed))
                last=torch.load(artifacts['last.pt']['path'],map_location='cpu',weights_only=False)
                if last['source_checkpoint_sha256']!=source_sha or last['step']!=200:
                    raise ValueError('Reference source/checkpoint budget differs')
                original=read(artifacts['completion.json']['path'])
                records.append(dict(spec=dict(domain=domain,seed=seed,arm=arm),reused=True,
                    original_arm=original_arm,git_commit=old['training_commit'],
                    original_plan_sha256=original['plan_sha256'],artifacts=artifacts,
                    wall_seconds=original['wall_seconds'],event_flags=original['event_flags'],
                    source_checkpoint_sha256=source_sha))
    assert len(records)==30
    return records


def tensorboard_writer(folder, resume_step=None):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        dependencies = Path('/home/hxchen/runs/2026-10-24-tqg-graph-adaptation-research/stage3_execution/runtime_deps')
        sys.path.append(str(dependencies))
        from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(str(folder),purge_step=resume_step+1 if resume_step is not None else None)


def write_scalars(writer,row):
    def visit(prefix,value):
        if isinstance(value,dict):
            for key,item in value.items():
                visit(prefix+'/'+str(key) if prefix else str(key),item)
        elif isinstance(value,(list,tuple)):
            for index,item in enumerate(value):
                visit(prefix+'/'+str(index),item)
        elif isinstance(value,(int,float)) and not isinstance(value,bool):
            writer.add_scalar(prefix,value,row['step'])
    for name,value in row.items():
        if name not in ('batches','query_label_control'):
            visit(name,value)


def check_completion(ctx,spec):
    path = ctx.job(spec)/'completion.json'
    if not path.exists():
        return None
    saved = read(path)
    if saved['git_commit'] != ctx.commit or saved['plan_sha256'] != ctx.plan_sha or saved['spec'] != spec:
        raise ValueError('Completed job contract differs')
    for info in saved['artifacts'].values():
        if sha256(info['path']) != info['sha256']:
            raise ValueError('Completed artifact changed: '+info['path'])
    return saved


def run_job(ctx,spec):
    complete = check_completion(ctx,spec)
    if complete:
        return complete
    job = ctx.job(spec)
    job.mkdir(parents=True,exist_ok=True)
    arm = ctx.plan['arms'][spec['arm']]
    config = ctx.config(spec['seed'],spec['arm'])
    source = ctx.source_path(spec['seed'],arm['pool'])
    task = ctx.task(spec['domain'],arm['pool'])
    previous_attempts = sorted(job.glob('attempt_*'))
    resume = None
    for directory in reversed(previous_attempts):
        if (directory/'last.pt').exists():
            resume = directory/'last.pt'
            break
    attempt = len(list(job.glob('RUN_INFO_attempt_*.json')))+1
    if attempt > ctx.plan['monitoring']['max_failure_attempts']:
        raise RuntimeError('Five consecutive attempts exhausted for '+str(spec))
    output = job/f'attempt_{attempt:02d}'
    resumed_step = int(torch.load(resume,map_location='cpu',weights_only=False)['step']) if resume else None
    info = dict(run_id=ctx.plan['run_id'],config_name=spec['arm'],spec=spec,seed=spec['seed'],
        git_commit=ctx.commit,plan_sha256=ctx.plan_sha,hardware=ctx.manifest['hardware'],config_full=config,
        source_checkpoint=str(source),source_checkpoint_sha256=sha256(source),planned_gpu_hours=0,
        started_at=stamp(),started_unix=time.time(),attempt=attempt,resumed_from=str(resume) if resume else None)
    atomic_json(job/f'RUN_INFO_attempt_{attempt}.json',info)
    heartbeat = dict(spec=spec,status='running',step=resumed_step or 0,unix=time.time(),attempt=attempt)
    atomic_json(STAGE/f'worker_{spec["seed"]}.json',heartbeat)
    losses = deque(maxlen=ctx.plan['monitoring']['loss_window'])
    writer = tensorboard_writer(job/'tensorboard',resumed_step)
    writer.add_text('RUN_INFO',json.dumps(info,ensure_ascii=False))

    def callback(row):
        losses.append(row['total_loss'])
        write_scalars(writer,row)
        query = row['query']
        if query.get('status')=='numerical_fallback_source_only':
            event('numerical_fallback',spec=spec,step=row['step'],reason=query.get('numerical_error'))
        if 'execution' in row and (row['execution']['feedback_gradient_identity_error'] > 1e-10
                or row['execution']['sgd_step_identity_error'] > 1e-10):
            raise ArithmeticError('Parameter feedback execution identity failed')
        if True:  # A heartbeat after every step, including expensive batch64 queries.
            atomic_json(STAGE/f'worker_{spec["seed"]}.json',dict(heartbeat,step=row['step'],unix=time.time(),
                source_ce=row['source_ce'],feedback_loss=row['feedback_loss'],calibration=query.get('calibration')))
        if row['step']==200 and len(losses)==100:
            first = list(losses)[:50]; last = list(losses)[50:]
            if sum(last)/50 >= sum(first)/50:
                event('no_decrease_warning',spec=spec,step=row['step'],
                    reason='Nonstationary stochastic adaptation loss; record only, do not change training')
        if row['step']%20==0:
            writer.flush()

    try:
        with (job/f'stdout_attempt_{attempt}.log').open('x') as log, contextlib.redirect_stdout(log):
            result = run_adaptation(task,config,source,output,step_callback=callback,resume=resume)
        assert result['completed_steps']==200
        artifacts = {name:dict(path=str(output/name),sha256=sha256(output/name))
                     for name in ('last.pt','selected.pt','history.jsonl','config.json','data_roles.json','environment.json','result.json')}
        flags = ['restart'] if attempt>1 else ['clean']
        complete = dict(status='completed',spec=spec,git_commit=ctx.commit,plan_sha256=ctx.plan_sha,
            output=str(output),artifacts=artifacts,event_flags=flags,attempts=attempt,
            gpu_hours=0,wall_seconds=time.time()-info['started_unix'],result=result)
        atomic_json(job/'completion.json',complete)
        event('job_complete',spec=spec,seconds=complete['wall_seconds'],attempt=attempt)
        print(json.dumps(dict(event='job_complete',**spec,seconds=complete['wall_seconds'])),flush=True)
        return complete
    except BaseException as error:
        text = traceback.format_exc()
        (job/f'failure_attempt_{attempt}.txt').write_text(text)
        event('job_failure',spec=spec,attempt=attempt,error=str(error),error_type=type(error).__name__)
        raise
    finally:
        writer.flush();writer.close()


def worker(seed):
    ctx = Context()
    if ctx.manifest is None or seed not in ctx.plan['seeds']:
        raise ValueError('Prepare matrix before launching declared seeds')
    for spec in ctx.specs():
        if spec['seed'] != seed:
            continue
        ledger = read(STAGE/'compute_ledger.json')
        if time.time() >= ledger['deadline_unix']:
            raise TimeoutError('Wall budget exhausted before next job')
        run_job(ctx,spec)
    atomic_json(STAGE/f'worker_{seed}.json',dict(seed=seed,status='completed',unix=time.time(),jobs=6))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','worker'])
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.action=='prepare':
        prepare()
    else:
        worker(args.seed)
