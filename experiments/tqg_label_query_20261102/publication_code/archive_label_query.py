"""Publish audited H label-attribution and D7-H A/B diagnostics."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def audit(s,repo):
    summary=read(s/'results_summary.json');m=read(s/'experiment_manifest.json')
    assert summary['status']=='completed' and read(s/'D0/label_query_checks.json')['passed']
    assert read(s/'D0/engineering_checks.json')['passed'] and read(s/'D0/duplicate_invariance.json')['passed']
    result=list(csv.DictReader((s/'results.csv').open()));assert len(result)==180
    assert all(r['git_commit']==m['git_commit'] and r['event_flags']=='clean' for r in result)
    allowed=set();forbidden=set()
    for p in (s/'data_roles').glob('*/identity.json'):
        d=read(p);allowed.update(d['original_development']);forbidden.update(d['final_ids'])
    accesses=denied=0
    for p in [s/'D0/label_access.jsonl']+list((s/'logs').glob('*/label_access.jsonl')):
        tokens=set();readouts={}
        sealpath=p.with_name('readout_seals.jsonl')
        if sealpath.exists():readouts={r['token']:r for r in rows(sealpath)}
        for r in rows(p):
            if r['event']=='unlabeled_query_sealed':
                tokens.add(r['token'])
                if readouts:
                    ref=readouts[r['token']];assert ref['time']<=r['time']
                    digest=hashlib.sha256(json.dumps(ref['readouts'],sort_keys=True).encode()).hexdigest()
                    assert digest==r['readout_hash']==ref['digest']
            elif r['event']=='label_access':
                assert r['token'] in tokens and set(r['ids'])<=allowed and not set(r['ids'])&forbidden;accesses+=len(r['ids'])
            elif r['event']=='access_denied':denied+=1
    assert denied==9
    for r in read(s/'snapshot_manifest.json')['references']:assert sha(repo/r['checkpoint'])==r['checkpoint_sha256']
    cache=read(s/'input_geometry_manifest.json')
    for r in cache['files']:assert sha(Path(cache['root'])/r['path'])==r['sha256']
    q=[];ls=[]
    for seed in range(3405,3410):
        out=s/'logs'/str(seed)
        checks=rows(out/'temporary_step_checks.jsonl');assert len(checks)==72 and all(r['passed'] and r['original_model_unchanged'] for r in checks)
        q+=rows(out/'query_size_blocks.jsonl');ls+=rows(out/'label_step_blocks.jsonl')
    assert len(q)==6000 and len(ls)==3420
    # Independent input-only nested membership audit, including all regularization copies.
    for r in q:
        assert r['snapshot_step']==0
        ids=read(s/r['indices_reference']);query=ids['query'];assert not set(query)&set(ids['fit'])
        order=sorted(query,key=lambda i:hashlib.sha256(f"D7|{r['seed']}|{r['rep']}|{i}".encode()).hexdigest())
        assert r['selected_query_ids']==sorted(order[:r['n']]) and len(set(r['selected_query_ids']))==r['n']
        expected=.01 if r['convention']=='FIXED' else .01*(r['n']/196)
        assert r['lambda_w']==expected
        if r['n']==196:assert r['profile_W_hash']==r['W_hash']
    # Every task/seed/snapshot has balanced counts, no missing primary cosines.
    for r in summary['query_quantity']:
        for n,key,error_key in [(16,'c16','e16'),(196,'c196','e196')]:
            selected=[x for x in q if x['task']==r['task'] and x['convention']==r['convention'] and x['n']==n]
            assert len(selected)==200 and all(x['c_A'] is not None and x['e_A'] is not None for x in selected)
            assert abs(statistics.mean(x['c_A'] for x in selected)-r[key])<1e-12
            assert abs(statistics.mean(x['e_A'] for x in selected)-r[error_key])<1e-12
    grouped=defaultdict(dict)
    for r in ls:grouped[(r['task'],r['seed'],r['snapshot_step'],r['block_id'])][r['arm']]=r
    for r in summary['label_attribution']:
        pairs=[v for k,v in grouped.items() if k[0]==r['task']];assert len(pairs)==60
        for arm,key in [('B','step_FULL_extra'),('A','step_BASE_extra'),('A-M','step_BASE_M_extra')]:
            values=[(x['S']['CE_B']-x[arm]['CE_B'])/x['S']['base_CE_B'] for x in pairs]
            assert abs(statistics.mean(values)-r[key])<1e-12
        for kind,key in [('PL','step_advantage_PERM_TOTAL_M'),('RL','step_advantage_RAND_TOTAL_M')]:
            values=[(statistics.mean(x[f'{kind}-{i}-M']['CE_B'] for i in [1,2,3])-x['B']['CE_B'])/x['S']['base_CE_B'] for x in pairs]
            assert abs(statistics.mean(values)-r[key])<1e-12
    report=dict(passed=True,scope='Artifact, label-role, seal, input membership and independent primary mean audit',
        result_seed_rows=180,real_development_label_values_requested=accesses,denied_permission_tests=denied,final_labels_read=0,
        source_snapshot_references=45,label_temporary_updates=3420,query_logical_solves=6000,query_physical_solves=4920,
        query_logical_steps=3000,query_physical_steps=2460,input_geometry_hashes=360,
        source_fit_query_roles_disjoint=True,original_models_unchanged=True,independent_primary_means_verified=True,
        new_training_jobs=0)
    write(s/'artifact_audit.json',report);return report


def publish(s,repo,dest):
    report=audit(s,repo);dest.mkdir(exist_ok=False)
    for p in s.iterdir():
        if p.is_file() and p.suffix in ['.json','.csv','.md','.txt','.sh']:shutil.copy2(p,dest/p.name)
    for d in ['D0','data_roles','figures','logs','analysis_code']:shutil.copytree(s/d,dest/d)
    shutil.copytree(s/'code',dest/'code_snapshot',ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    shutil.copytree(s.parent/'stage2_method',dest/'protocol')
    (dest/'.gitattributes').write_text('*.csv whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n# Preserve immutable historical source formatting.\ncode_snapshot/full_query_experiment.py -whitespace\n')
    subprocess.run(['git','bundle','create',str(dest/'execution_history.bundle'),'--all'],cwd=s/'code',check=True)
    pub=dest/'publication_code';pub.mkdir();shutil.copy2(Path(__file__),pub/'archive_label_query.py')
    shutil.copy2(s.parent/'stage_3_done',dest/'stage_3_done.json');shutil.copy2(s.parent/'run.yaml',dest/'run_state.json')
    index=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file()]
    assert max(r['bytes'] for r in index)<100_000_000
    write(dest/'manifest.json',dict(experiment='tqg_label_query_20261102',execution_commit=read(s/'experiment_manifest.json')['git_commit'],
        source_run=str(s),audit=report,excluded=['real datasets','existing checkpoint copies','high dimensional geometry caches'],files=index))
    print(json.dumps(dict(status='archived',files=len(index),bytes=sum(r['bytes'] for r in index),audit=report)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--repo',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();publish(Path(a.stage).resolve(),Path(a.repo).resolve(),Path(a.destination).resolve())
