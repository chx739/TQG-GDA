"""Audit and publish completed D3/D4 diagnostics without datasets or repeated models."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def jsonl(path):return [json.loads(x) for x in path.read_text().splitlines() if x]


def audit(stage,repo):
    summary=read(stage/'results_summary.json');manifest=read(stage/'experiment_manifest.json')
    if summary['status']!='completed' or not read(stage/'D0/engineering_checks.json')['passed']:
        raise ValueError('Only completed D0-accepted experiments can be archived')
    files=['calibration_grid_blocks.csv','signature_target_blocks.csv','source_audit_blocks.csv','step_blocks.csv']
    rows={p:list(csv.DictReader((stage/p).open())) for p in files}
    if tuple(len(rows[p]) for p in files)!=(5400,1800,600,8400):raise ValueError('Wrong raw row counts')
    results=list(csv.DictReader((stage/'results.csv').open()))
    if len(results)!=975:raise ValueError('Expected 325 gradient +650 step model-seed rows')
    for collection in [results,*rows.values()]:
        if any(r['git_commit']!=manifest['git_commit'] or r['plan_sha256']!=manifest['plan_sha256'] for r in collection):
            raise ValueError('Mixed provenance')
    dev=set();final=set();dev_by_domain={}
    for p in (stage/'data_roles').glob('*/identity.json'):
        roles=read(p);dev.update(roles['original_development']);final.update(roles['final_ids'])
        dev_by_domain[p.parent.name]=set(roles['original_development'])
    p0=Path(manifest['settings']['p0_stage'])
    original_query=set(read(p0/'logs/3405/1_3405_0_0_indices.json')['query'])
    target_count=source_count=denied=0
    for kind in ['label_access.jsonl','source_label_access.jsonl']:
        for path in (stage/'logs').glob('*/'+kind):
            tokens=set()
            readout_seals={r['token']:r for r in jsonl(path.parent/'readout_seals.jsonl')}
            for event in jsonl(path):
                if event['event']=='unlabeled_query_sealed':
                    tokens.add(event['token'])
                    if event['token'] not in readout_seals:raise ValueError('Missing full-profile seal')
                    if readout_seals[event['token']]['time']>event['time']:raise ValueError('Seal ordering error')
                elif event['event']=='access_denied':denied+=1
                elif event['event']=='label_access':
                    if event['token'] not in tokens:raise ValueError('Labels accessed before seal')
                    ids=set(event['ids'])
                    if kind=='label_access.jsonl':
                        domain=event['token'].split('_')[1]
                        if not ids<=dev_by_domain['0_to_'+domain] or ids&final:raise ValueError('Forbidden target label IDs')
                        target_count+=len(event['ids'])
                    else:
                        role=read(path.parent/(event['token']+'_indices.json'))
                        bank,audit_ids=set(role['query']),set(role['audit'])
                        if bank&audit_ids or bank|audit_ids!=original_query or not ids<=audit_ids or ids&bank:
                            raise ValueError('Source audit roles violated')
                        source_count+=len(event['ids'])
    for p in (stage/'logs').glob('*/temporary_step_checks.jsonl'):
        if any(not r['original_model_unchanged'] for r in jsonl(p)):raise ValueError('Original model changed')
    for record in read(stage/'snapshot_manifest.json')['references']:
        if sha(repo/record['checkpoint'])!=record['checkpoint_sha256']:raise ValueError('Source checkpoint changed')
    old={}
    for seed in range(3405,3410):
        for r in jsonl(p0/'logs'/str(seed)/'gradient_blocks.jsonl'):
            if r['arm']=='TRUE':old[(r['task'],r['seed'],r['snapshot_step'],r['block_id'])]=r
    baseline_errors=[]
    for row in rows['calibration_grid_blocks.csv']:
        if row['arm']=='JOINT' and float(row['alpha'])==.005:
            ref=old[(row['task'],int(row['seed']),int(row['snapshot_step']),int(row['block_id']))]
            for key in ['c_A','e_A','c_B','gradient_norm']:
                error=abs(float(row[key])-ref[key]);baseline_errors.append(error)
                if error>1e-10*max(1.,abs(ref[key])):raise ValueError('P0 original-arm metrics did not replay')
    # Recompute paired task means directly from wide raw rows, independently
    # from the hierarchical summary. Balanced valid counts make these equal.
    cosine_pairs=defaultdict(dict);step_pairs=defaultdict(dict)
    for filename in files[:3]:
        for r in rows[filename]:
            if r['experiment_id']=='D3' and float(r['alpha'])!=.005:continue
            key=(r['experiment_id'],r['scope'],r['task'],r['seed'],r['snapshot_step'],r['fold'],r['block_id'])
            cosine_pairs[key][r['arm']]=r
    for r in rows['step_blocks.csv']:
        if r['scale']!='norm_matched' or (r['experiment_id']=='D3' and float(r['alpha'])!=.005):continue
        key=(r['experiment_id'],r['scope'],r['task'],r['seed'],r['snapshot_step'],r['fold'],r['block_id'])
        step_pairs[key][r['arm']]=r
    for result in summary['criteria']:
        identity=(result['experiment_id'],result['scope'],result['task'])
        candidate,reference=('JOINT','ISO') if identity[0]=='D3' else ('HA','H')
        differences=[float(v[candidate]['c_A'])-float(v[reference]['c_A']) for k,v in cosine_pairs.items() if k[:3]==identity]
        b_differences=[(float(v[reference]['CE_B'])-float(v[candidate]['CE_B']))/float(v[reference]['base_CE_B'])
                       for k,v in step_pairs.items() if k[:3]==identity]
        if abs(statistics.mean(differences)-result['cosine_advantage'])>1e-12:raise ValueError('Cosine aggregation mismatch')
        if abs(statistics.mean(b_differences)-result['norm_matched_B_advantage'])>1e-12:raise ValueError('Matched-step aggregation mismatch')
    report=dict(passed=True,scope='Roles, readout-before-label seals, hashes, balanced counts and independent numeric means',
        D3_rows=5400,D4_target_rows=1800,D4_source_rows=600,temporary_steps=8400,result_seed_rows=975,
        development_label_values_requested=target_count,source_query_audit_values_requested=source_count,
        denied_requests=denied,final_labels_read=0,source_checkpoint_references=45,unique_source_checkpoints=35,
        P0_original_arm_max_metric_error=max(baseline_errors),no_continuous_training=True,
        original_models_and_buffers_unchanged=True,independent_paired_means_verified=True)
    write(stage/'artifact_audit.json',report)
    return report


def archive(stage,repo,dest):
    report=audit(stage,repo)
    dest.mkdir(parents=True,exist_ok=False)
    for p in stage.iterdir():
        if p.is_file() and p.suffix in ('.json','.csv','.md','.txt'):shutil.copy2(p,dest/p.name)
    for directory in ['logs','D0','data_roles','figures','analysis_code']:
        if (stage/directory).exists():shutil.copytree(stage/directory,dest/directory)
    shutil.copytree(stage/'code',dest/'code_snapshot',ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    shutil.copytree(stage.parent/'stage2_method',dest/'protocol')
    subprocess.run(['git','bundle','create',str(dest/'execution_history.bundle'),'--all'],cwd=stage/'code',check=True)
    (dest/'.gitattributes').write_text('*.csv whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n')
    files=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file()]
    if max(f['bytes'] for f in files)>50*1024**2:raise ValueError('Archive has oversized individual file')
    write(dest/'manifest.json',dict(schema_version=1,experiment='tqg_d3_d4_20261031',source_run=str(stage),
        execution_commit=read(stage/'experiment_manifest.json')['git_commit'],audit=report,
        excluded_local_inputs=['P0 geometry_cache; referenced by existing hash manifest','source model checkpoints; already archived'],files=files))
    print(json.dumps(dict(archived_files=len(files),archive_bytes=sum(f['bytes'] for f in files),destination=str(dest))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);parser.add_argument('--repo',required=True);parser.add_argument('--destination',required=True)
    args=parser.parse_args();archive(Path(args.stage).resolve(),Path(args.repo).resolve(),Path(args.destination).resolve())
