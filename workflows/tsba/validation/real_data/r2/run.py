"""Registered R2 stages; run --stage init, pilot, scores, adapt, diagnose, seal."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import torch

from ...archive import provenance, digest
from .common import write_json, seal, load, prepare, verify, Broker, jobs, job_key, atomic_tensor, FROZEN, model_from_frozen, predictions, pools
from .score import continue_score, assert_nested_equal
from .adapt import run_trajectory
from .diagnostics import first_round


def configure():
    if os.environ.get('OMP_NUM_THREADS')!='1' or os.environ.get('MKL_NUM_THREADS')!='1':
        raise RuntimeError('Set OMP_NUM_THREADS=1 MKL_NUM_THREADS=1')
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.use_deterministic_algorithms(True)


def worker(stage,root,spec,plan):
    configure()
    if stage=='scores':return continue_score(root,spec,plan)
    if stage=='adapt':return run_trajectory(root,spec,plan)
    if stage=='diagnose':return first_round(root,*spec,plan)
    raise ValueError(stage)


def initialize(repo,root,plan,test_log):
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise RuntimeError('Commit protocol and code before real-data execution')
    if '\nOK\n' not in Path(test_log).read_text():raise ValueError('Passing unittest log required')
    request=Path(plan['request_path'])
    if digest(request)!=plan['request_sha256']:raise ValueError('Request changed')
    root.mkdir(parents=True,exist_ok=False)
    for d in (*FROZEN,'logs','evaluation','failures','pilot'):(root/d).mkdir(exist_ok=True)
    shutil.copyfile(test_log,root/'logs/R0-tests.txt')
    for name in ('r2-plan.json','r2-protocol.md'):
        shutil.copyfile(repo/'experiments/mutagenicity/r2-20261111'/name,root/'protocol'/name)
    shutil.copyfile(request,root/'protocol/user-request.txt')
    write_json(root/'execution-plan.json',plan);write_json(root/'provenance.json',provenance(repo))
    with (root/'protocol/source.tar').open('xb') as f:subprocess.run(['git','archive','--format=tar','HEAD'],cwd=repo,stdout=f,check=True)
    packages=sorted({(d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']})
    (root/'protocol/environment.txt').write_text('OMP_NUM_THREADS=1\nMKL_NUM_THREADS=1\ntorch deterministic=True\n'+'\n'.join(f'{a}=={b}' for a,b in packages)+'\n')
    prepare(root,plan)
    sealed=seal(root,'INPUT',['deps','batches','protocol','execution-plan.json','dependency-manifest.json','input-checks.json','provenance.json','logs/R0-tests.txt'])
    print('INPUT '+sealed['digest'],flush=True)
    broker=Broker(root)
    for seed in plan['seeds']:
        s=json.loads((root/f'deps/roles/seed{seed}.json').read_text())
        labels=broker.read(s['fit_ids'],'source_fit','R2 source batch CE only',seed=seed)
        write_json(root/f'source-labels/seed{seed}.json',labels)
    # Exact 900 endpoints are available for the resource pilot. Continuation
    # verifies them and writes the remaining endpoints in the formal score stage.
    for seed in plan['seeds']:
        for d in plan['domains']:
            for b in plan['bandwidth_multipliers']:
                old=root/f'deps/r1b/scores/seed{seed}_domain{d}/b{b}/epochs/epoch099.pt'
                atomic_tensor(root/f'scores/seed{seed}_domain{d}/b{b}/checkpoints/u00900.pt',load(old))


def pilot(root,plan):
    if (root/'pilot/complete.json').exists():return
    rows=[]
    # Fixed two-cell pilot, not selected by performance. Full 30 attempts give
    # realistic nonlinear projection cost; these are excluded from main counts.
    for domain in plan['domains']:
        spec=dict(seed=3405,domain=domain,lam=1,kind='learned',bandwidth=1,updates=900,sign=1)
        result=run_trajectory(root,spec,plan,output_root=root/'pilot/trajectories')
        rows.append(result);print('PILOT '+json.dumps(result),flush=True)
    # Actual interruption/reload: three attempts + resume must reproduce pilot.
    spec=dict(seed=3405,domain=1,lam=1,kind='learned',bandwidth=1,updates=900,sign=1)
    run_trajectory(root,spec,plan,output_root=root/'pilot/resumed',stop_after=3)
    run_trajectory(root,spec,plan,output_root=root/'pilot/resumed')
    for attempt in range(31):
        a=load(root/'pilot/trajectories'/job_key(spec)/f'round{attempt:03d}.pt')
        b=load(root/'pilot/resumed'/job_key(spec)/f'round{attempt:03d}.pt')
        assert_nested_equal(a['model_state'],b['model_state'])
        if 'predictions' in a:assert_nested_equal(a['predictions'],b['predictions'])
    maximum=max(r['elapsed_seconds'] for r in rows)
    result=dict(rows=rows,resume_exact_all_31_states=True,
        conservative_adaptation_cpu_seconds=maximum*630,
        ideal_parallel_adaptation_seconds=maximum*630/plan['resources']['workers'],
        estimate_note='Two fixed lambda=1 learned900 trajectories; harder lambda/heads and I/O may cost more',
        workers=plan['resources']['workers'],peak_worker_rss_bytes=max(r['peak_rss_bytes'] for r in rows))
    write_json(root/'pilot/complete.json',result);print(json.dumps(result),flush=True)


def parallel(stage,root,plan):
    if (root/'seals/OBS.json').exists():raise PermissionError('Training closed after OBS')
    verify(root,'INPUT')
    if not (root/'pilot/complete.json').exists():raise RuntimeError('Resource pilot must complete first')
    if stage=='scores':specs=[dict(seed=s,domain=d,bandwidth=b) for s in plan['seeds'] for d in plan['domains'] for b in plan['bandwidth_multipliers']]
    elif stage=='adapt':specs=jobs(plan)
    else:specs=[(s,d,l) for s in plan['seeds'] for d in plan['domains'] for l in plan['lambdas']]
    index=root/f'{stage}-index.json'
    if index.exists():return
    started=time.monotonic();results=[];errors=[]
    with ProcessPoolExecutor(max_workers=plan['resources']['workers'],mp_context=multiprocessing.get_context('spawn')) as executor:
        pending={executor.submit(worker,stage,str(root),s,plan):s for s in specs}
        for future in as_completed(pending):
            try:
                value=future.result();results.append(value)
                print(f'{stage} {len(results)}/{len(specs)} elapsed={time.monotonic()-started:.1f}s '+json.dumps({k:value[k] for k in ('seed','domain','status','attempts','counts','lam') if k in value}),flush=True)
            except Exception as exc:
                failure=dict(stage=stage,spec=pending[future],error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                write_json(root/'failures'/f'{stage}-{time.time_ns()}.json',failure);errors.append(failure)
                print(json.dumps(failure),flush=True)
    if errors:raise RuntimeError(f'{len(errors)} execution errors retained; no evaluation opened')
    write_json(index,dict(stage=stage,count=len(results),elapsed_seconds=time.monotonic()-started,results=results))


def replay_and_seal(root,plan):
    if (root/'seals/OBS.json').exists():verify(root,'OBS');return
    for stage,n in (('scores',20),('adapt',630),('diagnose',30)):
        index=json.loads((root/f'{stage}-index.json').read_text());assert index['count']==n
    checked=0
    for seed in plan['seeds']:
        f=load(root/f'deps/source/seed{seed}/frozen.pt');model=model_from_frozen(f)
        for spec in [s for s in jobs(plan) if s['seed']==seed]:
            directory=root/'trajectories'/job_key(spec)
            for attempt in plan['prediction_attempts']:
                path=directory/f'round{attempt:03d}.pt'
                if not path.exists():continue  # Failed trajectories keep their true terminal attempt.
                state=load(path);model.load_state_dict(state['model_state'])
                assert_nested_equal(predictions(model,f['x'],pools(root,seed,spec['domain'])),state['predictions']);checked+=1
        for path in (root/'diagnostics').glob(f'seed{seed}_domain*/lambda*/*.pt'):
            state=load(path);model.load_state_dict(state['model_state'])
            assert_nested_equal(predictions(model,f['x'],pools(root,seed,state['spec']['domain'])),state['predictions']);checked+=1
    replay=dict(checkpoints=checked,exact=True,final_labels_parsed=False)
    if (root/'prediction-replay.json').exists():
        assert json.loads((root/'prediction-replay.json').read_text())==replay
    else:write_json(root/'prediction-replay.json',replay)
    verify(root,'INPUT')
    s=seal(root,'OBS',[*FROZEN,'pilot','prediction-replay.json','scores-index.json','adapt-index.json','diagnose-index.json',
        'seals/INPUT.json','execution-plan.json','dependency-manifest.json','input-checks.json','provenance.json'])
    print('OBS '+s['digest'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['init','pilot','scores','adapt','diagnose','seal','evaluate'])
    parser.add_argument('--root',type=Path);parser.add_argument('--test-log',type=Path)
    args=parser.parse_args();configure()
    repo=Path(__file__).resolve().parents[5]
    plan=json.loads((repo/'experiments/mutagenicity/r2-20261111/r2-plan.json').read_text())
    root=args.root or Path(plan['output_root'])
    if args.stage=='init':initialize(repo,root,plan,args.test_log)
    else:
        assert json.loads((root/'execution-plan.json').read_text())==plan
        # Resume/evaluation uses the same implementation as INPUT. Reporting
        # changes can be committed separately, but training code cannot drift.
        archived=json.loads((root/'provenance.json').read_text())
        for rel,expected in archived['file_hashes'].items():
            if rel.endswith('.py') and digest(repo/rel)!=expected:
                raise RuntimeError('Code changed after INPUT: '+rel)
        if args.stage=='pilot':pilot(root,plan)
        elif args.stage=='seal':replay_and_seal(root,plan)
        elif args.stage=='evaluate':
            from .evaluate import evaluate
            evaluate(root,plan)
        else:parallel(args.stage,root,plan)


if __name__=='__main__':main()
