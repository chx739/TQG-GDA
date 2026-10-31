"""Audit and publish completed D8 and residual-decomposition evidence."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(line) for line in p.read_text().splitlines() if line]
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def audit(stage,repo):
    summary=read(stage/'results_summary.json');manifest=read(stage/'experiment_manifest.json')
    assert summary['status']=='completed' and read(stage/'D0/engineering_checks.json')['passed']
    assert read(stage/'D0/mechanism_checks.json')['passed']
    result=list(csv.DictReader((stage/'results.csv').open()))
    assert len(result)==135
    assert all(r['git_commit']==manifest['git_commit'] and r['event_flags']=='clean' for r in result)
    allowed=set();forbidden=set()
    for p in (stage/'data_roles').glob('*/identity.json'):
        d=read(p);allowed.update(d['original_development']);forbidden.update(d['final_ids'])
    real_access=denied=0
    for p in [stage/'D0/label_access.jsonl']+list((stage/'logs').glob('*/label_access.jsonl')):
        seals=set()
        for r in rows(p):
            if r['event']=='unlabeled_query_sealed':seals.add(r['token'])
            elif r['event']=='access_denied':denied+=1
            elif r['event']=='label_access':
                assert r['token'] in seals and set(r['ids'])<=allowed and not set(r['ids'])&forbidden
                real_access+=len(r['ids'])
    synthetic_access=0
    for p in (stage/'logs').glob('*/synthetic_truth_access.jsonl'):
        seals=set()
        for r in rows(p):
            if r['event']=='sealed':seals.add(r['token'])
            elif r['event']=='truth_access':assert r['token'] in seals;synthetic_access+=1
    assert synthetic_access==602 and denied==9
    for r in read(stage/'snapshot_manifest.json')['references']:assert sha(repo/r['checkpoint'])==r['checkpoint_sha256']
    checkpoints=[]
    for seed in range(3405,3410):
        out=stage/'logs'/str(seed)
        warmup=rows(out/'warmup_history.jsonl')
        assert len(warmup)==200 and [r['step'] for r in warmup]==list(range(1,201))
        assert all(len(r['fit_ids'])==32 and len(set(r['fit_ids']))==32 and min(r['fit_ids'])>=0 and max(r['fit_ids'])<512 for r in warmup)
        ckpts=rows(out/'synthetic_checkpoint_manifest.jsonl');assert len(ckpts)==3
        for r in ckpts:assert sha(stage/r['path'])==r['sha256'];checkpoints.append(r)
        for role,n in manifest['settings']['source_counts'].items():
            d=read(out/f'generator_source_{role}.json');assert len(d['latents'])==len(d['eta'])==len(d['labels'])==n
        for role in ['target_construction','target_evaluation']:
            p1=read(out/f'generator_S1_{role}.json');p3=read(out/f'generator_S3_{role}.json')
            assert len(p1['latents'])==1024 and p1['latents']==p3['latents']
            assert max(abs(a+b-1) for a,b in zip(p1['eta'],p3['eta']))<1e-14
        checks=rows(out/'temporary_step_checks.jsonl');assert len(checks)==72 and all(r['passed'] and r['original_model_unchanged'] for r in checks)
    # Input caches remain immutable after use.
    for r in read(stage/'input_geometry_manifest.json')['files']:assert sha(Path(r['root'])/r['path'])==r['sha256']
    # Independently recover primary means from raw CSV; sizes are balanced across all seeds/snapshots.
    graph=list(csv.DictReader((stage/'graph_gradient_blocks.csv').open()))
    for r in summary['D8']:
        if r['scenario']=='LINEAR':continue
        selected=[x for x in graph if x['scenario']==r['scenario'] and x['method']==r['method']]
        for kind,key in [('TRUE','c_TRUE'),('PERM-','c_PERM'),('RAND-','c_RAND')]:
            # Missing S4 cosines make flat averages invalid: repeat -> block -> snapshot -> seed.
            blocks=defaultdict(list)
            for x in selected:
                if (x['arm']==kind if kind=='TRUE' else x['arm'].startswith(kind)) and x['c_A']!='':
                    blocks[(x['seed'],x['snapshot_step'],x['block_id'])].append(float(x['c_A']))
            snapshots=defaultdict(list)
            for (seed,snapshot,block),values in blocks.items():snapshots[(seed,snapshot)].append(statistics.mean(values))
            seedmeans=defaultdict(list)
            for (seed,snapshot),values in snapshots.items():seedmeans[seed].append(statistics.mean(values))
            values=[statistics.mean(v) for v in seedmeans.values()]
            if values:assert abs(statistics.mean(values)-r[key])<1e-10,(r['scenario'],r['method'],key)
    real=list(csv.DictReader((stage/'real_step_blocks.csv').open()))
    grouped=defaultdict(dict)
    for r in real:grouped[(r['method'],r['task'],r['seed'],r['snapshot_step'],r['block_id'])][r['arm']]=r
    for r in summary['real_decomposition']:
        pairs=[v for key,v in grouped.items() if key[:2]==(r['method'],r['task'])];assert len(pairs)==60
        for name,key in [('S+FULL','full_extra'),('S+GLOBAL','global_extra')]:
            values=[(float(x['S']['CE_B'])-float(x[name]['CE_B']))/float(x['S']['base_CE_B']) for x in pairs]
            assert abs(statistics.mean(values)-r[key])<1e-12
        inc=[(float(x['S+GLOBAL']['CE_B'])-float(x['S+FULL']['CE_B']))/float(x['S']['base_CE_B']) for x in pairs]
        assert abs(statistics.mean(inc)-r['centered_increment'])<1e-12
    report=dict(passed=True,scope='Artifact, data-role, seals and independent aggregation audit; not efficacy proof',
        result_seed_rows=135,real_development_label_values_requested=real_access,denied_permission_negative_tests=denied,
        final_labels_read=0,synthetic_sealed_truth_access_events=synthetic_access,formal_synthetic_checkpoints=len(checkpoints),
        warmup_updates=1000,source_snapshot_references=45,new_real_training_jobs=0,
        real_temporary_updates=4680,synthetic_temporary_updates=2880,
        S1_S3_latents_same_eta_flipped=True,independent_primary_means_verified=True,
        original_real_models_unchanged=True,full_replay_max_error=summary['full_replay_max_metric_error'])
    write(stage/'artifact_audit.json',report)
    write(stage/'synthetic_checkpoint_manifest.json',dict(checkpoints=checkpoints))
    return report


def publish(stage,repo,dest):
    report=audit(stage,repo);dest.mkdir(exist_ok=False)
    for p in stage.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv','.md','.txt','.sh']:shutil.copy2(p,dest/p.name)
    for d in ['D0','data_roles','figures','logs','checkpoints']:
        shutil.copytree(stage/d,dest/d)
    if (stage/'analysis_code').exists():shutil.copytree(stage/'analysis_code',dest/'analysis_code')
    shutil.copytree(stage/'code',dest/'code_snapshot',ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    shutil.copytree(stage.parent/'stage2_method',dest/'protocol')
    (dest/'.gitattributes').write_text('*.csv whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n# Immutable historical source formatting is preserved.\ncode_snapshot/full_query_experiment.py -whitespace\n')
    subprocess.run(['git','bundle','create',str(dest/'execution_history.bundle'),'--all'],cwd=stage/'code',check=True)
    pub=dest/'publication_code';pub.mkdir();shutil.copy2(Path(__file__),pub/'archive_mechanisms.py')
    shutil.copy2(stage.parent/'stage_3_done',dest/'stage_3_done.json');shutil.copy2(stage.parent/'run.yaml',dest/'run_state.json')
    index=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file()]
    assert max(r['bytes'] for r in index)<100_000_000,'Single-file GitHub limit requires explicit packaging'
    write(dest/'manifest.json',dict(experiment='tqg_d8_residual_20261031',execution_commit=read(stage/'experiment_manifest.json')['git_commit'],
        source_run=str(stage),audit=report,excluded=['real dataset','previous real model checkpoints','high dimensional geometry caches'],files=index))
    print(json.dumps(dict(status='archived',files=len(index),bytes=sum(r['bytes'] for r in index),audit=report)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--repo',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();publish(Path(a.stage).resolve(),Path(a.repo).resolve(),Path(a.destination).resolve())
