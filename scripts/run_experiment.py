"""Portable orchestration of the existing source trainer and EvolutionTrainer.

No target evaluator, alternate distance or additional training objective.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from sega_gda.data import sha256
from sega_gda.train_source import run as train_source,load_source_checkpoint,seed_everything,write_json,atomic_save
from sega_gda.pilot import initialize_real_task
from sega_gda.real_data import RealTaskGraphs
from sega_gda.evolution import EvolutionTrainer
from monitoring import RuntimeMeasurements,source_validation


def read(path):return json.loads(Path(path).read_text())


def domain_index(path):
    ids=np.load(path,allow_pickle=False)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(set(ids.tolist()))!=len(ids) or not len(ids):
        raise ValueError('Domain index must be a nonempty unique integer vector')
    return ids


def adapt(config,source_checkpoint,task_manifest,output,device,resume=None):
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',config)
    seed_everything(config['seed'],config['num_threads'])
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['config']!=config:raise ValueError('Resume changed adaptation settings')
        _,source=load_source_checkpoint(source_checkpoint,'cpu')
        data=RealTaskGraphs(task_manifest,source['preprocessing'])
        identity=saved['data_identity']
        if sha256(source_checkpoint)!=identity['source_checkpoint_sha256']:
            raise ValueError('Resume changed initial source checkpoint')
        old=identity['task']
        for key in ('dataset','target_policy','source_split_sha256','target_index_sha256'):
            if data.manifest[key]!=old[key]:raise ValueError('Resume changed data protocol: '+key)
        if set(data.source_labels)!=set(saved['source_labels']) or data.source_labels!=saved['source_labels'] or tuple(data.target_ids)!=tuple(saved['target_ids']):
            raise ValueError('Resume changed training graph IDs or source labels')
        engine=EvolutionTrainer.load_checkpoint(resume,data.graph,identity,device,config)
        if 'portable_selection' not in saved:raise ValueError('Resume needs a checkpoint created by this portable entry')
        selection=saved['portable_selection']
        best_ce,best_round,best_state=selection['ce'],selection['round'],selection['engine_state']
        initial=source_validation(engine.model,data)
        proof={'resumed_from':str(resume),'completed_rounds':engine.round}
    else:
        engine,data,proof=initialize_real_task(source_checkpoint,task_manifest,config,device)
        initial=source_validation(engine.model,data)
        engine.save_checkpoint(output/'initial.pt')
        best_ce,best_round=initial['ce'],0
        best_state=torch.load(output/'initial.pt',map_location='cpu',weights_only=False)
    if config['rounds']<=engine.round:raise ValueError('No remaining rounds; use a checkpoint from an earlier boundary')
    write_json(output/'initialization.json',proof)
    write_json(output/'data_identity.json',engine.data_identity)
    write_json(output/'relation_memory.json',dict(source=engine.source_memory.to_dict(),target=engine.target_memory.to_dict()))
    monitoring=RuntimeMeasurements(engine,output)
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();rows=[]
    def save_boundary(path):
        engine.save_checkpoint(path)
        state=torch.load(path,map_location='cpu',weights_only=False)
        state['portable_selection']=dict(ce=best_ce,round=best_round,engine_state=copy.deepcopy(best_state))
        atomic_save(path,state)
    with (output/'rounds.jsonl').open('w') as stream:
        while engine.round<config['rounds']:
            start=time.perf_counter();row=engine.run_round()
            if device.startswith('cuda'):torch.cuda.synchronize()
            row['training_seconds']=time.perf_counter()-start
            row['source_validation']=source_validation(engine.model,data)
            row['metric_norm']=float(engine.metric.B.detach().norm())
            row['max_abs_eigenvalue']=float(torch.linalg.eigvalsh(engine.metric.B.detach()).abs().max())
            row['current_scale2']=float(engine.model.distance.scale2().detach())
            row['cuda_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None
            row['cuda_peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20 if device.startswith('cuda') else None
            if row['source_validation']['ce']<best_ce:
                best_ce,best_round=row['source_validation']['ce'],engine.round
                engine.save_checkpoint(output/'best_state.tmp')
                best_state=torch.load(output/'best_state.tmp',map_location='cpu',weights_only=False)
                (output/'best_state.tmp').unlink()
            save_boundary(output/f'round_{engine.round}.pt')
            row['selected_round']=best_round
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append(row)
            print(json.dumps(dict(round=engine.round,source_validation=row['source_validation'],accepted=row['space']['accepted'])),flush=True)
    save_boundary(output/'last.pt')
    selected=copy.deepcopy(best_state)
    selected['portable_selection']=dict(ce=best_ce,round=best_round,engine_state=copy.deepcopy(best_state))
    atomic_save(output/'selected.pt',selected)
    write_json(output/'runtime_measurements.json',dict(events=monitoring.events,distances=monitoring.distances,ot=monitoring.ot,mechanisms=monitoring.mechanisms))
    write_json(output/'result.json',dict(scope='development_only',initial_source_validation=initial,rounds=rows,
        selected_round=best_round,selected_ce=best_ce,seconds=time.perf_counter()-started,target_evaluated=False,
        target_labels_used=False,formal_paper_result=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('source','adapt','all'))
    parser.add_argument('--data-path',required=True,help='Existing trusted native TU/PyG processed/data.pt; never copied')
    parser.add_argument('--idx-dir',default=str(ROOT/'idx'))
    parser.add_argument('--source-split',default=str(ROOT/'data/splits/PROTEINS_idx_0_source.json'))
    parser.add_argument('--dataset',default='PROTEINS')
    parser.add_argument('--source-domain',type=int,default=0);parser.add_argument('--target-domain',type=int,default=1)
    parser.add_argument('--target-policy',choices=('transductive-unlabeled',),help='Required explicit choice for adapt/all')
    parser.add_argument('--source-config',default=str(ROOT/'configs/source_development.json'))
    parser.add_argument('--adapt-config',default=str(ROOT/'configs/adaptation_development.json'))
    parser.add_argument('--source-checkpoint',help='Your newly trained source best.pt; required for adapt')
    parser.add_argument('--resume-source',help='Your own source last.pt; output must be a new directory')
    parser.add_argument('--resume-adapt',help='Your own portable adaptation boundary checkpoint')
    parser.add_argument('--device',default='cpu');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    for key in ('data_path','idx_dir','source_split','source_config','adapt_config','output','source_checkpoint','resume_source','resume_adapt'):
        if getattr(args,key):setattr(args,key,Path(getattr(args,key)).resolve())
    os.chdir(ROOT)
    if args.stage!='source' and not args.target_policy:parser.error('adapt/all requires --target-policy transductive-unlabeled')
    if args.stage=='adapt' and not args.source_checkpoint:parser.error('adapt requires --source-checkpoint from your source run')
    if args.stage=='all' and args.source_checkpoint:parser.error('all trains source from scratch; use adapt for an existing source model')
    if args.resume_adapt and args.stage!='adapt':parser.error('--resume-adapt requires stage adapt')
    if args.device.startswith('cuda') and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError('Expose exactly ONE available GPU with CUDA_VISIBLE_DEVICES')
    manifest=read(args.source_split)
    if manifest['dataset']!=args.dataset:raise ValueError('Dataset and frozen source split differ')
    if sha256(args.data_path)!=manifest['dataset_cache_sha256']:
        raise ValueError('External cache differs from the frozen data version. Do not silently accept a new data ordering or rewrite the split.')
    source_index=args.idx_dir/f'idx_{args.dataset}_{args.source_domain}.npy'
    ids=domain_index(source_index)
    if set(ids)!=set(manifest['source_train_indices']+manifest['source_validation_indices']):
        raise ValueError('Source index is not the original train/validation union')
    if manifest.get('domain_index_sha256') and sha256(source_index)!=manifest['domain_index_sha256']:
        raise ValueError('Source domain index changed')
    if args.output.exists():raise FileExistsError('Use a new output directory: '+str(args.output))
    args.output.mkdir(parents=True)
    source_checkpoint=args.source_checkpoint
    if args.stage in ('source','all'):
        config=read(args.source_config)
        config.update(data_cache=str(args.data_path),split_manifest=str(args.source_split),device=args.device)
        train_source(config,args.output/'source',args.resume_source)
        source_checkpoint=args.output/'source/best.pt'
    if args.stage in ('adapt','all'):
        target_index=args.idx_dir/f'idx_{args.dataset}_{args.target_domain}.npy'
        target_ids=domain_index(target_index)
        if set(ids)&set(target_ids):raise ValueError('Source/target graph IDs overlap')
        _,source=load_source_checkpoint(source_checkpoint,'cpu')
        task=dict(status='LOCKED_DEVELOPMENT',protocol_authority='Explicit CLI transductive-unlabeled development selection',
            dataset=args.dataset,family='idx',source_domain=args.source_domain,target_domain=args.target_domain,
            target_policy='transductive_unlabeled',source_split_manifest=str(args.source_split),source_split_sha256=sha256(args.source_split),
            data_cache=str(args.data_path),use_node_attributes=source['config']['use_node_attributes'],target_index_file=str(target_index),
            target_index_sha256=sha256(target_index),closed_set_class_mapping=source['class_labels'],target_labels='FORBIDDEN_NO_TARGET_EVALUATOR')
        inputs=args.output/'inputs';inputs.mkdir()
        write_json(inputs/'task.json',task)
        adapt(read(args.adapt_config),source_checkpoint,inputs/'task.json',args.output/'adaptation',args.device,args.resume_adapt)


if __name__=='__main__':main()
