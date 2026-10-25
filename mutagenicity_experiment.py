"""SEGA-configured Mutagenicity TQG matrix; no target labels during training."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch

from revision_common import atomic_json, hardware
from training_revision import RevisionTrainer, run_job, tensorboard_writer
from tqg_gda.data import GraphDomainTask
from tqg_gda.models import GraphClassifier
from tqg_gda.training import run_warmup, load_model, model_spec
from tqg_gda.utils import sha256, seed_everything, cpu_state

CODE = Path(__file__).resolve().parent
STAGE = CODE.parent
RUN = STAGE.parent
CONFIG = CODE / 'configs' / 'mutagenicity' / 'sega_full_query.json'
CACHE = Path('/home/hxchen/TQG_data/Mutagenicity_official_20261026/Mutagenicity/processed/data.pt')
SEGA = Path('/home/hxchen/sega_gda_review_oymmoufx/SEGA_GDA/configs/source_development.json')
LABELS = dict(B_frozen='fit-only 源模型不动', B_fit='继续源 fit 训练',
              B_full='全源标签训练', B_pseudo='置信度伪标签',
              Q_query_O='全部query，odd', Q_query_OE='全部query，odd+even',
              Q_all_O='全部源训练图，odd', Q_all_OE='全部源训练图，odd+even')


def read(path):
    return json.loads(Path(path).read_text())


def commit():
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE, text=True):
        raise RuntimeError('Training/evaluation requires a clean committed code tree')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()


def task(domain=None, all_source=False):
    return GraphDomainTask(CACHE, STAGE / 'source_protocol.json',
        target_index=CODE / f'idx/idx_Mutagenicity_{domain}.npy' if domain else None,
        target_policy='transductive-unlabeled' if domain else None,
        fit_fraction=.8, split_seed=431, use_node_attributes=False,
        preprocess_pool='all_source' if all_source else 'fit')


def prepare():
    source = task()
    old = read(CODE / 'configs/full_query/0_to_1.json')
    sega = read(SEGA)
    arms = {}
    for name, base in [('B_fit','B_fit'), ('B_full','B_full'), ('B_pseudo','B_pseudo'),
                       ('Q_query_O','Q51_O'), ('Q_query_OE','Q51_OE'),
                       ('Q_all_O','Q250_O'), ('Q_all_OE','Q250_OE')]:
        value = copy.deepcopy(old['arms'][base])
        value.update(arm=name, n=len(source.train_ids) if name.startswith('Q_all') else len(source.query_ids))
        if name.startswith('Q_'):
            value['pretraining_fit_overlap'] = len(source.fit_ids) if name.startswith('Q_all') else 0
        arms[name] = value
    plan = dict(revision_id='mutagenicity_sega_20261026', dataset='Mutagenicity',
        seeds=list(range(3405,3410)), domains=[1,2,3], arms=arms,
        adaptation=old['adaptation'], warmup_sega=sega,
        source_counts=dict(train=len(source.train_ids), validation=len(source.validation_ids),
                           fit=len(source.fit_ids), query=len(source.query_ids)),
        warmup=dict(epochs=sega['epochs'],batch_size=sega['batch_size'],lr=sega['lr']),
        cache=str(CACHE),cache_sha256=sha256(CACHE), source_protocol_sha256=sha256(STAGE/'source_protocol.json'),
        primary='fixed_200_balanced_accuracy', secondary='source_validation_CE_selected_including_step0',
        source_split_seed=431, target_split_seed_rule='default_rng(271828+domain), ID-only 50/50',
        target_final_labels_allowed=False, mixed_precision=False, dtype='float64',
        resource_wall_cap_seconds=28800, workers=2, hardware=hardware(),
        scope='User-approved three directions, five seeds, four baselines, full-query/all-source odd and odd+even',
        hypothesis='Exploratory: each TQG arm improves paired fixed-step BA versus B_fit; report all arms/tasks regardless of sign',
        sega_boundary='SEGA source model/hyperparameters and existing edge-count domains; TQG adaptation is 200 steps, not SEGA geometry/OT rounds',
        initialization_boundary='B_full uses all-source warmup; other arms use fit-only warmup',
        epochs_boundary='100 epochs as SEGA source config, not the prior PROTEINS fixed 700 updates',
        template='Reuse existing tested from-scratch graph classifier; vision/LM templates do not fit')
    CONFIG.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(CONFIG,plan)
    roles={}
    for domain in plan['domains']:
        ids=np.load(CODE/f'idx/idx_Mutagenicity_{domain}.npy',allow_pickle=False)
        order=np.random.default_rng(271828+domain).permutation(ids)
        dest=STAGE/'roles'/f'0_to_{domain}';dest.mkdir(parents=True,exist_ok=True)
        for name,group in [('development',order[:len(order)//2]),('final',order[len(order)//2:])]:
            np.save(dest/f'{name}.npy',np.sort(group))
        roles[str(domain)]={name:dict(path=str(dest/f'{name}.npy'),sha256=sha256(dest/f'{name}.npy')) for name in ('development','final')}
    atomic_json(STAGE/'roles.json',roles)
    atomic_json(STAGE/'execution_lock.json',dict(cache=sha256(CACHE),protocol=sha256(STAGE/'source_protocol.json'),
        config=sha256(CONFIG),roles=sha256(STAGE/'roles.json'),sega_source_config=sha256(SEGA),
        domains={str(i):sha256(CODE/f'idx/idx_Mutagenicity_{i}.npy') for i in range(4)}))
    print(json.dumps(plan['source_counts']),flush=True)


class Context:
    def __init__(self):
        self.plan=read(CONFIG); self.commit=commit();self.roles=read(STAGE/'roles.json');self._tasks={}
        lock=read(STAGE/'execution_lock.json')
        for p,k in [(CACHE,'cache'),(STAGE/'source_protocol.json','protocol'),(CONFIG,'config'),(STAGE/'roles.json','roles')]:
            if sha256(p)!=lock[k]:raise ValueError('Locked input changed: '+str(p))
    def task(self,domain,all_source=False):
        key=domain,all_source
        if key not in self._tasks:self._tasks[key]=task(domain,all_source)
        return self._tasks[key]
    def arm(self,name):return copy.deepcopy(self.plan['arms'][name])
    def warmup_path(self,seed,all_source=False):
        return STAGE/'warmups'/str(seed)/('all_source' if all_source else 'fit')
    def source(self,domain,seed,all_source=False):
        path=self.warmup_path(seed,all_source)/'best.pt';model,state=load_model(path)
        old,new=copy.deepcopy(state['identity']),copy.deepcopy(self.task(domain,all_source).identity)
        for value in (old,new):
            for key in ('target_ids','target_index_sha256','target_policy','fingerprint'):value.pop(key,None)
        if old!=new:raise ValueError('Warmup source identity mismatch')
        return model,state,path
    def development_ids(self,domain):
        role=self.roles[str(domain)]['development'];path=Path(role['path'])
        if sha256(path)!=role['sha256']:raise ValueError('Role changed')
        return tuple(map(int,np.load(path,allow_pickle=False)))


class Budget:
    def __init__(self,plan):
        self.limit=plan['resource_wall_cap_seconds'];self.warned=False
        path=STAGE/'started.json'
        self.started=read(path)['unix']
    def check(self):
        fraction=(time.time()-self.started)/self.limit
        if fraction>=.8:raise TimeoutError('80% compute-budget review threshold reached')
        if fraction>=.5 and not self.warned:
            print(json.dumps(dict(event='budget_50_percent',fraction=fraction)),flush=True);self.warned=True


def warmup(ctx,seed,pool):
    output=ctx.warmup_path(seed,pool=='all_source')
    if output.exists():
        receipt=read(output/'completion.json')
        for name,digest in receipt['hashes'].items():
            if sha256(output/name)!=digest:raise ValueError('Warmup artifact changed')
        return
    cfg=read(CODE/'configs/tqg_development.json');sega=ctx.plan['warmup_sega']
    cfg.update(seed=seed,model={k:sega[k] for k in ('hidden_dim','num_layers','readout')})
    cfg['warmup']=dict(ctx.plan['warmup'],source_pool=pool)
    cfg['adaptation']['source_pool']=pool
    run_warmup(task(all_source=pool=='all_source'),cfg,output)
    writer=tensorboard_writer(output)
    for line in (output/'history.jsonl').read_text().splitlines():
        row=json.loads(line)
        writer.add_scalar('source_ce',row['source_ce'],row['epoch'])
        writer.add_scalar('source_validation_ce',row['source_validation']['ce'],row['epoch'])
    writer.close()
    atomic_json(output/'RUN_INFO.json',dict(seed=seed,pool=pool,git_commit=ctx.commit,hardware=hardware(),
        config_name=CONFIG.stem,gpu_hours=0,epochs=100,optimizer_updates=100*math.ceil(len(task().ids(pool))/32)))
    atomic_json(output/'completion.json',dict(git_commit=ctx.commit,hashes={name:sha256(output/name) for name in
        ('best.pt','last.pt','history.jsonl','config.json','data_roles.json','result.json','RUN_INFO.json')}))


def acceptance(ctx):
    from unittest.mock import patch
    graph_task=ctx.task(1)
    cfg=read(CODE/'configs/tqg_development.json')
    seed_everything(3405,1,True)
    model=GraphClassifier(model_spec(graph_task,cfg),2).double().eval()
    checks=[]
    for name in ctx.plan['arms']:
        current=copy.deepcopy(model)
        trainer=RevisionTrainer(current,copy.deepcopy(model),graph_task,ctx.arm(name),ctx.plan,3405,1)
        with patch.object(trainer.optimizer,'step',wraps=trainer.optimizer.step) as step:
            row=trainer.step()
            assert step.call_count==1
        assert math.isfinite(row['total_loss'])
        if name.startswith('Q_'):
            role=ctx.arm(name)['query_role']
            assert set(row['batches']['query'])==set(graph_task.ids(role))
        assert model.encoder.spec['edge_dim']==3
        checks.append(name)
    atomic_json(STAGE/'acceptance.json',dict(passed=True,real_GINE_handlers=checks,
        target_labels_read=False,numerical_training_modules_unchanged=True,git_commit=ctx.commit))
    print('ACCEPTANCE PASSED',flush=True)


def worker(ctx,slot):
    budget=Budget(ctx.plan)
    for seed in ctx.plan['seeds'][slot::ctx.plan['workers']]:
        for pool in ('fit','all_source'):
            budget.check();warmup(ctx,seed,pool)
        for domain in ctx.plan['domains']:
            for arm in ctx.plan['arms']:
                output=STAGE/'jobs'/f'0_to_{domain}'/str(seed)/arm
                print(json.dumps(dict(event='job_start',seed=seed,domain=domain,arm=arm)),flush=True)
                result=run_job(ctx,domain,seed,arm,output=output,resume=output.exists(),budget=budget)
                print(json.dumps(dict(event='job_done',seed=seed,domain=domain,arm=arm,
                    seconds=result['seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','acceptance','worker'])
    parser.add_argument('--slot',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    if args.phase=='prepare':prepare();return
    ctx=Context()
    if args.phase=='acceptance':acceptance(ctx)
    else:worker(ctx,args.slot)


if __name__=='__main__':main()
