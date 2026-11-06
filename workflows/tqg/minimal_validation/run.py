"""Execute the frozen minimal M0/M1 protocol and its conditional M2 stage."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from tqg_gda.config import Config
from tqg_gda.data import load_task
from tqg_gda.metrics import classification_metrics
from tqg_gda.p0_diagnostics import (DevelopmentOracle, append_json, model_hash,
    gradient_of_ce, sample_ids, tensor_hash, development_pools)
from tqg_gda.predict import load_predictor
from tqg_gda.pretrain import pretrain_model
from tqg_gda.provenance import code_identity, runtime_identity, verify_artifacts
from tqg_gda.utils import atomic_save, fingerprint, seed_everything, sha256, write_json
from tqg_gda.v2.query import ParameterSnapshot

from .diagnostics import (construct, derived_seed, gradient_rows, input_batches,
                          permuted_labels, step_audit)
from .decisions import summarize

ROOT = Path(__file__).resolve().parents[3]
PLAN = ROOT/'docs/research_plans/tqg_sega_minimal_20261106'
CONFIGS = ROOT/'configs/sega_style/minimal_20261106'


def write_csv(path, records):
    keys = list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def config_for(seed, domain, cache, output, version='v1'):
    raw = json.loads((CONFIGS/f'{version}_0_to_{domain}_seed{seed}.json').read_text())
    raw.update(cache_path=str(Path(cache).resolve()), out_dir=str(Path(output).resolve()))
    for name in ('source_split','target_index'):
        raw[name] = str(ROOT/raw[name])
    return Config.from_dict(raw)


def verify_inputs(cache):
    manifest = json.loads((PLAN/'input_manifest.json').read_text())
    if sha256(cache)!=manifest['cache']['expected_sha256']:
        raise ValueError('Cache differs from frozen protocol')
    items = [manifest['source']]
    for domain in manifest['targets'].values():
        items.extend(domain[k] for k in ('target','development','final','training_index'))
    for item in items:
        if sha256(ROOT/item['path'])!=item['sha256']:
            raise ValueError('Frozen input identity changed: '+item['path'])
    return manifest


def register_blocks(task, manifest):
    result, perms = [], {}
    labels = torch.tensor([task.labels[i] for i in task.query_ids])
    for seed in (3405,3406,3407):
        perms[str(seed)] = {str(rep):permuted_labels(task.query_ids,labels,seed,rep)[1] for rep in (1,2,3)}
        for domain in (1,3):
            ids = manifest['targets'][str(domain)]
            for block in range(4):
                result.append(dict(seed=seed,domain=domain,block=block,
                    fit=list(sample_ids(task.fit_ids,16,derived_seed(seed,domain,block,'fit'))),
                    query=list(task.query_ids),
                    A=list(sample_ids(ids['diagnostic_A_ids'],16,derived_seed(seed,domain,block,'A'))),
                    B=ids['diagnostic_B_ids']))
    return dict(blocks=result,permutations=perms)


def source_record(config, task, code, runtime):
    started = time.perf_counter()
    model, info, folder = pretrain_model(config,task,code,runtime)
    prior = torch.bincount(torch.tensor([task.labels[i] for i in task.fit_ids]),minlength=len(task.classes)).double()
    prior /= prior.sum()
    labels = torch.tensor([task.labels[i] for i in task.validation_ids])
    baseline = classification_metrics(prior.log().expand(len(labels),-1),labels)
    actual = info['source_validation']
    row = dict(seed=config.seed,selected_epoch=info['selected_epoch'],updates=info['completed_updates'],
        source_validation=actual,constant_prior_validation=baseline,
        source_usable=actual['ce']<baseline['ce'] and actual['balanced_accuracy']>.5,
        checkpoint=str(folder/'best.pt'),checkpoint_sha256=sha256(folder/'best.pt'),
        source_key=info['source_key'],seconds=time.perf_counter()-started,cache_hit=info['cache_hit'])
    return model,row


def run_block(model,task,config,block_record,manifest,output):
    seed,domain,index = (block_record[k] for k in ('seed','domain','block'))
    token = f'seed{seed}_0to{domain}_block{index}'
    folder = output/'M1'/token
    if folder.exists():
        if not (folder/'result.json').exists():
            raise ValueError('Incomplete block retained for audit: '+str(folder))
        completed = json.loads((folder/'result.json').read_text())
        verify_artifacts(folder,completed['artifacts'])
        if completed['source_model_hash']!=model_hash(model):
            raise ValueError('Completed block has a different source checkpoint')
        return completed['versions']
    folder.mkdir(parents=True)
    started = time.perf_counter()
    snapshot = ParameterSnapshot(model)
    fit = task.batch(block_record['fit'],'fit',dtype=snapshot.flat.dtype)
    query = task.batch(block_record['query'],'query',dtype=snapshot.flat.dtype)
    yf,fit.y = fit.y,None
    yq,query.y = query.y,None
    query.graph_ids = block_record['query']
    target = task.batch(block_record['A'],'target',dtype=snapshot.flat.dtype)
    gf,gq = gradient_of_ce(snapshot,[fit],yf),gradient_of_ce(snapshot,[query],yq)
    built = {}
    for version in ('v1','v2'):
        cfg = replace(config,version=version)
        replay = seed==3405 and domain==1 and index==0
        built[version] = construct(model,query,yq,target,cfg,seed,domain,index,replay=replay)
        item = built[version]
        atomic_save(folder/f'{version}_sealed.pt',dict(
            model_hash=model_hash(model),snapshot=snapshot.flat,space=item['space'],
            feedback=vars(item['feedback']) if item['feedback'] is not None else None,
            gradients=item['gradients'],source_fit_gradient=gf,query_ce_gradient=gq,
            batches=block_record,permutations=item['permutations']))
        write_json(folder/f'{version}_health.json',item['health'])
    sealed_hashes = {v:sha256(folder/f'{v}_sealed.pt') for v in built}
    write_json(folder/'seal.json',dict(source_model_hash=model_hash(model),payloads=sealed_hashes,batches=block_record))
    roles = manifest['targets'][str(domain)]
    dev = np.load(ROOT/roles['development']['path'],allow_pickle=False).tolist()
    final = np.load(ROOT/roles['final']['path'],allow_pickle=False).tolist()
    oracle = DevelopmentOracle(task,dev,final,output/'label_access.jsonl',enabled=True)
    oracle.seal(token,block_record['A']+block_record['B'],theta_hash=model_hash(model),readout_hash=fingerprint(sealed_hashes))
    ya = oracle.labels(block_record['A'],token=token,purpose='M1 gradient A after both versions sealed')
    yb = oracle.labels(block_record['B'],token=token,purpose='M1 independent B gradient and one-step evaluation')
    ga = gradient_of_ce(snapshot,[target],ya)
    gb = gradient_of_ce(snapshot,input_batches(task,block_record['B']),yb)
    atomic_save(folder/'oracle_gradients.pt',dict(A=ga,B=gb))
    versions,common = [],{}
    for version,item in built.items():
        cfg = replace(config,version=version)
        gradients = gradient_rows(item['gradients'],ga,gb,gf,item['space'],cfg.adapt.feedback_weight)
        steps,directions,common = step_audit(model,task,block_record['A'],block_record['B'],
            ya,yb,ga,gb,gf,gq,item['gradients'],cfg,common)
        atomic_save(folder/f'{version}_step_directions.pt',directions)
        versions.append(dict(seed=seed,domain=domain,block=index,version=version,
            source_model_hash=model_hash(model),health=item['health'],gradients=gradients,steps=steps))
    if model_hash(model)!=built['v1']['health']['model_hash']:
        raise AssertionError('Block modified original source model')
    files = [p for p in folder.iterdir() if p.is_file()]
    write_json(folder/'result.json',dict(versions=versions,source_model_hash=model_hash(model),
        seconds=time.perf_counter()-started,peak_process_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        target_labels_scope='development_A_B_only',artifacts={p.name:sha256(p) for p in files}))
    print(json.dumps(dict(event='block_complete',seed=seed,domain=domain,block=index,
        seconds=round(time.perf_counter()-started,2),
        status={v:x['health']['query']['status'] for v,x in built.items()})),flush=True)
    return versions


def run(cache,output,resume=False):
    output = Path(output).resolve()
    manifest = verify_inputs(cache)
    code = code_identity()
    plan_files = {str(p.relative_to(ROOT)):sha256(p) for base in (PLAN,CONFIGS) for p in base.rglob('*') if p.is_file()}
    identity = dict(code=code,protocol_files=plan_files,cache_sha256=sha256(cache))
    if output.exists():
        if not resume or not (output/'registration.json').exists():
            raise ValueError('Use a new output directory, or explicit --resume for this sealed run')
        registered = json.loads((output/'registration.json').read_text())
        if registered['identity']!=identity:
            raise ValueError('Code/protocol/input changed since registration')
    else:
        output.mkdir(parents=True)
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
            raise ValueError('Commit the executable protocol before real training')
        write_json(output/'registration.json',dict(identity=identity,commit=commit,date='2026-11-07',
            scope='M0/M1 followed only by passing versions M2',final_labels_allowed=False))
    config = config_for(3405,1,cache,output)
    task = load_task(config)
    for name,ids in (('fit',task.fit_ids),('query',task.query_ids),('validation',task.validation_ids)):
        if list(ids)!=manifest['source'][name+'_ids']:
            raise ValueError('Frozen source roles changed')
    for domain in (1,3):
        row = manifest['targets'][str(domain)]
        dev = np.load(ROOT/row['development']['path'],allow_pickle=False).tolist()
        a,b = development_pools(dev,domain)
        if list(a)!=row['diagnostic_A_ids'] or list(b)!=row['diagnostic_B_ids']:
            raise ValueError('Frozen A/B partition changed')
    schedule = register_blocks(task,manifest)
    write_json(output/'block_manifest.json',schedule)
    configs = [config_for(seed,domain,cache,output,v).to_dict()
               for seed in (3405,3406,3407) for domain in (1,3) for v in ('v1','v2')]
    write_json(output/'resolved_configs.json',configs)
    source_rows,blocks = [],[]
    for seed in (3405,3406,3407):
        cfg = config_for(seed,1,cache,output)
        runtime = runtime_identity(cfg)
        seed_everything(seed,cfg.num_threads,cfg.deterministic)
        model,source = source_record(cfg,task,code,runtime)
        source_rows.append(source)
        write_json(output/'source_models.json',source_rows)
        print(json.dumps(dict(event='source_ready',**source)),flush=True)
        for domain in (1,3):
            domain_config = config_for(seed,domain,cache,output)
            domain_task = load_task(domain_config)
            for item in schedule['blocks']:
                if item['seed']==seed and item['domain']==domain:
                    blocks.extend(run_block(model,domain_task,domain_config,item,manifest,output))
    decision,seeds = summarize(blocks,source_rows)
    write_json(output/'decision.json',decision)
    write_csv(output/'seed_summary.csv',seeds)
    for key,filename in (('gradients','gradient_blocks.csv'),('steps','one_step_blocks.csv')):
        write_csv(output/filename,[dict(seed=b['seed'],domain=b['domain'],block=b['block'],version=b['version'],**r)
                                  for b in blocks for r in b[key]])
    write_json(output/'query_diagnostics.json',[
        dict(seed=b['seed'],domain=b['domain'],block=b['block'],version=b['version'],**b['health']) for b in blocks])
    if decision['selected_M2_versions']:
        from .training import run_m2
        run_m2(cache,output,manifest,source_rows,decision['selected_M2_versions'])
    else:
        write_json(output/'M2_status.json',dict(status='not_started_by_registered_gate',trajectories=0,updates=0))
    write_json(output/'completion.json',dict(status='completed',source_models=len(source_rows),paired_blocks=24,
        version_blocks=len(blocks),decision=decision,final_labels_read=False,
        peak_process_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    print(json.dumps(dict(event='completed',decision=decision)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-path',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    run(args.cache_path,args.output,args.resume)


if __name__=='__main__':
    main()
