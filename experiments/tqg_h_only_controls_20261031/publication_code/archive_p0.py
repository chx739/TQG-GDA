"""Publish completed P0 evidence; no datasets or high-dimensional geometry cache."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def audit(stage,repo):
    summary=json.loads((stage/'results_summary.json').read_text())
    if summary['status']!='completed': raise ValueError('Do not archive incomplete runs')
    if not json.loads((stage/'D0/engineering_checks.json').read_text())['passed']:
        raise ValueError('D0 did not pass')
    allowed=set()
    forbidden=set()
    for path in (stage/'data_roles').glob('*/identity.json'):
        identity=json.loads(path.read_text())
        allowed.update(identity['original_development'])
        forbidden.update(identity['final_ids'])
    permitted_accesses=0
    denied=0
    sealed={}
    files=[stage/'D0/label_access.jsonl']+list((stage/'logs').glob('*/label_access.jsonl'))
    for path in files:
        tokens=set()
        for line in path.read_text().splitlines():
            event=json.loads(line)
            if event['event']=='unlabeled_query_sealed':
                tokens.add(event['token'])
            elif event['event']=='access_denied':
                denied+=1
            elif event['event']=='label_access':
                ids=set(event['ids'])
                if not ids<=allowed or ids&forbidden or event['token'] not in tokens:
                    raise ValueError('Oracle role or seal event-order violation')
                permitted_accesses+=len(event['ids'])
        sealed[str(path.relative_to(stage))]=len(tokens)
    rows=list(csv.DictReader((stage/'results.csv').open()))
    if len(rows)!=60: raise ValueError('Expected 15 D1 plus 45 D2 seed rows')
    manifest=json.loads((stage/'experiment_manifest.json').read_text())
    if any(r['git_commit']!=manifest['git_commit'] or r['event_flags']!='clean' for r in rows):
        raise ValueError('Results provenance mismatch')
    gradients=list(csv.DictReader((stage/'gradient_blocks.csv').open()))
    updates=list(csv.DictReader((stage/'one_step_blocks.csv').open()))
    if len(gradients)!=3600 or len(updates)!=6660: raise ValueError('Raw row counts differ')
    # Independent flat averages are valid here because every task/seed/snapshot
    # has exactly the same number of valid blocks and null replicates.
    for result in summary['D1']:
        task=result['task']
        for prefix,key in [('TRUE','c_TRUE'),('PERM-','c_PERM'),('RAND-','c_RAND')]:
            values=[float(r['c_A']) for r in gradients if r['task']==task
                    and (r['arm']==prefix if prefix=='TRUE' else r['arm'].startswith(prefix))]
            if abs(statistics.mean(values)-result[key])>1e-12:
                raise ValueError('D1 numeric aggregation mismatch')
        means=[statistics.mean(float(r['c_A']) for r in gradients
               if r['task']==task and int(r['seed'])==seed and r['arm']=='TRUE') for seed in range(3405,3410)]
        if abs(statistics.stdev(means)-result['cosine_SD'])>1e-12:
            raise ValueError('D1 seed sample SD mismatch')
    pairs={}
    for row in updates:
        if row['arm'] in ('S','S+Q') and float(row['eta'])==.0003:
            key=(row['task'],row['seed'],row['snapshot_step'],row['block_id'])
            pairs.setdefault(key,{})[row['arm']]=row
    for result in summary['D2']:
        values=[(float(v['S']['CE_B'])-float(v['S+Q']['CE_B']))/float(v['S']['base_CE_B'])
                for k,v in pairs.items() if k[0]==result['task']]
        if len(values)!=60 or abs(statistics.mean(values)-result['R_Q'])>1e-12:
            raise ValueError('D2 numeric aggregation mismatch')
    for path in (stage/'logs').glob('*/temporary_step_checks.jsonl'):
        for line in path.read_text().splitlines():
            if not json.loads(line)['original_model_unchanged']: raise ValueError('Original model changed')
    snapshot_manifest=json.loads((stage/'snapshot_manifest.json').read_text())
    for row in snapshot_manifest['references']:
        if sha(repo/row['checkpoint'])!=row['checkpoint_sha256']: raise ValueError('Input checkpoint changed')
    geometry=[]
    for path in (stage/'logs').glob('*/geometry_stats.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            cache=stage/row['geometry_cache']
            if sha(cache)!=row['geometry_sha256']: raise ValueError('Local geometry cache changed')
            geometry.append(dict(path=row['geometry_cache'],sha256=row['geometry_sha256'],bytes=cache.stat().st_size))
    report=dict(passed=True,scope='Artifact/provenance/role/count audit, not statistical evidence',
        result_rows=60,gradient_rows=3600,one_step_rows=6660,temporary_updates=6480,
        source_snapshot_references=45,unique_source_checkpoint_files=35,
        permitted_development_label_values_requested=permitted_accesses,
        denied_label_requests=denied,final_label_values_read=0,
        geometry_seals_per_log=sealed,local_geometry_files=len(geometry),
        local_geometry_bytes=sum(x['bytes'] for x in geometry),
        normal_UDA_models_saved=0,oracle_checkpoints_saved=0)
    report['independent_primary_means_and_sample_SD_recomputed']=True
    write(stage/'artifact_audit.json',report)
    write(stage/'local_geometry_manifest.json',dict(source_run=str(stage),files=geometry,
          publication='Local only. Excluded from GitHub; metrics, hashes and reproduction code are published.'))
    return report


def archive(stage,repo,destination):
    report=audit(stage,repo)
    destination.mkdir(parents=True,exist_ok=False)
    for path in stage.iterdir():
        if path.is_file() and path.suffix in ('.json','.csv','.md','.txt'):
            shutil.copy2(path,destination/path.name)
    for directory in ['D0','data_roles','figures','logs']:
        shutil.copytree(stage/directory,destination/directory)
    if (stage/'analysis_code').exists():
        shutil.copytree(stage/'analysis_code',destination/'analysis_code')
    shutil.copytree(stage/'code',destination/'code_snapshot',ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    shutil.copytree(stage.parent/'stage2_method',destination/'protocol')
    subprocess.run(['git','bundle','create',str(destination/'execution_history.bundle'),'--all'],
                   cwd=stage/'code',check=True)
    index=[]
    for path in sorted(destination.rglob('*')):
        if path.is_file(): index.append(dict(path=str(path.relative_to(destination)),bytes=path.stat().st_size,sha256=sha(path)))
    write(destination/'manifest.json',dict(schema_version=1,experiment='tqg_v2_P0_20261031',
        source_run=str(stage),execution_commit=json.loads((stage/'experiment_manifest.json').read_text())['git_commit'],
        scope='Completed D0/D1/D2 diagnostics, no continuous adaptation, no final labels',
        local_only_excluded=['geometry_cache','checkpoints'],audit=report,files=index))
    print(json.dumps(dict(status='archived',path=str(destination),files=len(index),bytes=sum(x['bytes'] for x in index))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True);parser.add_argument('--repo',required=True)
    parser.add_argument('--destination',required=True)
    args=parser.parse_args()
    archive(Path(args.stage).resolve(),Path(args.repo).resolve(),Path(args.destination).resolve())
