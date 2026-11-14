"""R6 phase-isolated execution on copied R5 dependencies."""
import argparse
from collections import Counter
import fcntl
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
import traceback
from uuid import uuid4
import torch
from .shared import io,output,p3,p4,provenance,digest,assert_nested_equal,run_stage
from .protocol import resolve,prepare,blocks,block_path,restore,old_plan
from .candidates import worker,make_candidates


def initialize(root,repo,request,plan):
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise ValueError('Commit reviewed implementation before R6 execution')
    temporary=root.parent/f'.{root.name}.prepare-{uuid4().hex}'
    temporary.mkdir(parents=True)
    io.write_json(temporary/'protocol/requested-plan.json',request)
    io.write_json(temporary/'protocol/plan.json',plan)
    io.write_json(temporary/'protocol/provenance.json',provenance(repo))
    io.write_json(temporary/'protocol/environment.json',{name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','torch-geometric')})
    io.write_json(temporary/'protocol/command.json',dict(argv=[sys.executable,*sys.orig_argv[1:]],cwd=str(repo),
        started_utc=datetime.now(timezone.utc).isoformat(),threads={name:os.environ.get(name) for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    with (temporary/'protocol/source.tar').open('xb') as stream:
        subprocess.run(['git','archive','HEAD'],cwd=repo,stdout=stream,check=True)
    try:
        prepare(temporary,repo,plan)
        io.seal(temporary,'R6_INPUT',['protocol','deps','registry.json','input-manifest.json'])
    except Exception:
        io.write_json(temporary/'preparation-failure.json',dict(error=traceback.format_exc()))
        raise
    temporary.rename(root)


def check_inputs(root,repo,plan):
    io.verify(root,'R6_INPUT')
    if p3.read_json(root/'protocol/plan.json')!=plan:raise ValueError('Registered plan changed')
    for path,sha in p3.read_json(root/'protocol/provenance.json')['file_hashes'].items():
        if path.endswith('.py') and digest(repo/path)!=sha:raise ValueError(f'Registered code changed: {path}')


def replay(root,plan):
    counts=Counter();statuses=Counter()
    expected_paths={block_path(root,t,s) for t,s in blocks(plan)}
    if set((root/'blocks').glob('**/before*.pt'))!=expected_paths:raise ValueError('Block registry incomplete')
    for task,step in blocks(plan):
        record=io.load(block_path(root,task,step))
        if any(record[k]!=v for k,v in dict(setting=plan['setting']['id'],source=task.source,target=task.target,seed=task.seed,attempt=step).items()):
            raise ValueError('Block identity mismatch')
        counts['blocks']+=1;statuses[record['status']]+=1
        if record['status'] in ('engineering_failure','numerical_failure'):
            if not record.get('error'):raise ValueError('Failed block lacks diagnostic')
            continue
        model,pools,y,old_record,previous,sigma,cfg,old=restore(root,task,step)
        rebuilt=make_candidates(model,pools,y,old_record,previous,sigma,cfg,old,plan)
        for key,value in rebuilt.items():assert_nested_equal(value,record[key])
        if record['status']=='available':
            cells=record['candidates']
            expected={(k,a['id']) for k in plan['steps']['multipliers'] for a in plan['factorial']['arms']}
            if {(c['k'],c['arm']) for c in cells}!=expected or len(cells)!=len(expected):raise ValueError('Incomplete candidate grid')
            counts['candidate_cells']+=len(cells);counts['replayed_cells']+=len(cells)
            counts['distinct_parameter_states']+=len({io.tensor_sha(c['vector']) for c in cells})
    counts.setdefault('candidate_cells',0);counts.setdefault('replayed_cells',0)
    return dict(counts,statuses=dict(statuses))


def execute(root,repo,plan,stage):
    check_inputs(root,repo,plan)
    if stage=='prepare':return
    if (root/'seals/R6_FINAL.json').exists():
        io.verify(root,'R6_OBS');io.verify(root,'R6_FINAL');return
    observed=(root/'seals/R6_OBS.json').exists()
    if stage in ('all','candidates') and not observed:
        run_stage(root,plan,'candidates',worker,blocks(plan))
    if stage=='candidates':return
    if stage in ('all','replay') and not observed:
        result=replay(root,plan)
        output.write_json(root/'replay.json',result)
        io.seal(root,'R6_OBS',['seals/R6_INPUT.json','protocol','deps','input-manifest.json','registry.json','blocks','replay.json'])
        print('R6_OBS sealed',plan['setting']['id'],result,flush=True)
    if stage=='replay':return
    if not (root/'seals/R6_OBS.json').exists():raise PermissionError('R6 evaluation requires its own observation seal')
    if stage in ('all','evaluate'):
        from .evaluate import evaluation_labels,worker as evaluate_worker
        targets=evaluation_labels(root,plan)
        items=[]
        for task,step in blocks(plan):
            fit=io.load(p4.labels_path(root/'deps',task.source,task.seed,'fit'))
            validation=io.load(p4.labels_path(root/'deps',task.source,task.seed,'validation'))
            record=io.load(block_path(root,task,step))
            if 'batch' in record:
                batch=record['batch']
            else:
                batch=io.load(p4.batch_path(root/'deps',task))[step-1]
            labels=dict(source_fit=fit,source_validation=validation,target=targets[task.target],source_batch=fit[batch['source_positions']])
            items.append((task,step,labels))
        run_stage(root,plan,'evaluate',evaluate_worker,items)
    if stage=='evaluate':return
    io.verify(root,'R6_OBS')
    from .report import collect
    collect(root,plan)
    io.seal(root,'R6_FINAL',['seals/R6_OBS.json','evaluation'])
    print('R6_FINAL sealed',plan['setting']['id'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--setting',help='Default: all three registered settings')
    parser.add_argument('--archive-root',type=Path,help='Parent for setting/attempt-001; default: registered output paths')
    parser.add_argument('--stage',choices=('all','prepare','candidates','replay','evaluate','report'),default='all')
    args=parser.parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError(f'Set {name}=1')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    repo=Path(__file__).resolve().parents[4];request=p3.read_json(args.plan)
    selected=[s for s in request['settings'] if args.setting is None or s['id']==args.setting]
    if not selected:raise ValueError('Unknown setting')
    ready=[]
    # Prepare all settings at the same committed code version before any candidate execution.
    for spec in selected:
        plan=resolve(request,spec['id'])
        root=(args.archive_root/spec['id']/'attempt-001' if args.archive_root else Path(spec['output_root_suggestion'])).resolve()
        root.parent.mkdir(parents=True,exist_ok=True)
        with root.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not root.exists():initialize(root,repo,request,plan)
        ready.append((root,plan))
    for root,plan in ready:
        with root.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            execute(root,repo,plan,args.stage)
