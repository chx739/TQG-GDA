"""Finalize reproducible metadata for a completed, audited assumption-probe run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def read(p):return json.loads(Path(p).read_text())
def write(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def finalize(stage,repo,analysis_names):
    stage=Path(stage);repo=Path(repo);run=stage.parent
    summary=read(stage/'results_summary.json');manifest=read(stage/'experiment_manifest.json');settings=manifest['settings']
    assert summary['status']=='completed'
    names=['assumptions_summarize.py']+analysis_names
    analysis=stage/'analysis_code';analysis.mkdir(exist_ok=False)
    for name in names:shutil.copy2(repo/name,analysis/name)
    write(analysis/'provenance.json',dict(repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
          scope='read-only aggregation and cross-dataset description; no refits',files=[dict(path=p.name,sha256=sha(p)) for p in analysis.glob('*.py')]))
    roles=[]
    for p in sorted((stage/'features').glob('*.json')):
        r=read(p);roles.append({k:r[k] for k in ['query','fit','validation','A','B','final','target_batch','checkpoint']})
    selections=[dict(path=str(p.relative_to(stage)),sha256=sha(p)) for p in sorted((stage/'logs').glob('340*/*_ids.json'))]
    write(stage/'sample_ids.json',dict(roles=roles,outer_inner_selection_files=selections,labels_used_for_selection=False))
    write(stage/'provenance.json',dict(execution=read(stage/'freeze_formal.json'),method_sha256=sha(run/'stage2_method/method.md'),
          config=settings,dataset_roles_sha256=sha(run/'stage2_method/data_roles/manifest.json'),no_GNN_updates=True))
    with (stage/'label_access.jsonl').open('w') as f:
        for p in [stage/'D0/label_access.jsonl']+sorted((stage/'logs').glob('*/label_access.jsonl')):
            for line in p.read_text().splitlines():
                r=json.loads(line);r['source_file']=str(p.relative_to(stage));f.write(json.dumps(r,allow_nan=False)+'\n')
    with (stage/'history.jsonl').open('w') as f:
        for p in sorted((stage/'logs').glob('*/history.jsonl')):
            f.write(json.dumps(dict(event='history_shard_index',path=str(p.relative_to(stage)),sha256=sha(p),records=sum(1 for _ in p.open())))+'\n')
    cpu=summary['cpu_seconds_formal']+summary['A0']['cpu_seconds']+summary['compute_estimate']['pilot']['cpu_seconds']
    ledger=dict(status='completed',gpu_hours=0,cpu_seconds_formal=summary['cpu_seconds_formal'],cpu_seconds_A0=summary['A0']['cpu_seconds'],
       cpu_seconds_pilot=summary['compute_estimate']['pilot']['cpu_seconds'],measured_worker_cpu_hours_total=cpu/3600,cap_cpu_hours=settings['cpu_hour_cap'],
       within_budget=cpu<settings['cpu_hour_cap']*3600,peak_worker_rss_kib=max(read(stage/f'logs/{s}/done.json')['peak_rss_kib'] for s in settings['seeds']),
       scope='tracked phase workers; import/orchestration/postprocessing overhead separate; 360sec reserve before formal launch')
    write(stage/'compute_ledger.json',ledger)
    marker=dict(stage=3,scope=settings.get('dataset','Mutagenicity')+' A0/A1/A2-a',status='complete',started_at=manifest['prepared_at'],finished_at=time.time(),
                gpu_hours_consumed_so_far=0,cpu_hours_measured=cpu/3600)
    write(run/'stage_3_done',marker);write(stage/'stage_3_done.json',marker)
    state=read(run/'run.yaml');state.update(status='completed_first_batch',finished_at=marker['finished_at']);write(run/'run.yaml',state);write(stage/'run_state.json',state)
    (stage/'requirements_lock.txt').write_text(''.join(f'{k}=={v}\n' for k,v in manifest['versions'].items()))
    (stage/'reproduction_commands.sh').write_text(f'''# Recorded execution; use a NEW run layout, matching cache SHA and data_roles manifest.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Frozen commit {manifest['git_commit']}; original Python /home/hxchen/anaconda3/envs/bfn4jet/bin/python
# assumptions_experiment.py <prepare|a0|pilot|run> --repo <repo> --stage <stage> --config configs/assumptions_proteins.json
# assumptions_regression.py --repo <repo> --output <stage>/D0/mutagenicity_regression.json
# assumptions_summarize.py --stage <stage>
# assumptions_proteins_interpretation.py --stage <stage> --repo <repo>
# scripts/archive_assumptions.py --stage <stage> --repo <repo> --dest <fresh archive>
''')
    return ledger


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--repo',required=True);p.add_argument('--analysis',nargs='+',required=True);a=p.parse_args()
    print(json.dumps(finalize(a.stage,a.repo,a.analysis),ensure_ascii=False))
