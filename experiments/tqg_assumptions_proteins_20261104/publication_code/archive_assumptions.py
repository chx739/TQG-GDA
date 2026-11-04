"""Independently audit and publish the completed A0/A1/A2-a study."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def audit(stage,repo):
    result=read(stage/'results_summary.json');manifest=read(stage/'experiment_manifest.json')
    assert result['status']=='completed' and result['selection_units']==2880 and result['logical_fits']==34560
    assert '\nOK' in (stage/'D0/formal_unit_tests.log').read_text()
    assert read(stage/'D0/replay.json')['passed']
    refs=read(stage/'snapshot_manifest.json')['references']
    for ref in refs:assert sha(repo/ref['checkpoint'])==ref['checkpoint_sha256']
    allowed_a=set();allowed_b=set();forbidden=set();models=predictions=cv_count=target_requests=0
    for p in (stage/'features').glob('*.json'):
        role=read(p);allowed_a.update(role['A']);allowed_b.update(role['B']);forbidden.update(role['final'])
        assert sha(p.with_suffix('.npz'))==role['feature_sha256'] and role['model_unchanged']
    assert not (allowed_a|allowed_b)&forbidden and not allowed_a&allowed_b
    for seed in manifest['settings']['seeds']:
        d=stage/f'logs/{seed}'
        groups=defaultdict(list)
        for r in lines(d/'inner_cv.jsonl'):
            key=tuple(r[k] for k in ['task','fold','signature','train_domain','family','variant'])
            groups[key].append(r);cv_count+=1
        assert len(groups)==576
        for key,records in groups.items():
            task,fold,sig,td,fam,variant=key
            by_penalty=defaultdict(list)
            for r in records:by_penalty[r['penalty']].append(r['nll'])
            assert all(len(values)==3 for values in by_penalty.values())
            best=min(by_penalty,key=lambda p:(sum(x/3 for x in by_penalty[p]),-p))
            path=d/f'models/{task[-1]}_{fold}_{sig}_{td}_{fam}_{variant}.npz'
            model=np.load(path)
            assert float(model['penalty'])==best
            assert len(model['training_y'])==len(model['train_ids'])
            ids=read(d/f'{task[-1]}_{fold}_{sig}_{td}_ids.json')
            assert model['train_ids'].tolist()==ids['train'] and len(ids['train'])==len(ids['source_train']);models+=1
        seals={};artifact_events={}
        for r in lines(d/'label_access.jsonl'):
            if r['event']=='probe_artifacts_sealed':artifact_events[r['domain']]=r
            if r['event']=='prediction_sealed':seals[r['sha256']]=r
            if r['event']=='label_access':
                allowed=allowed_a if r['role']=='A_train' else allowed_b
                assert set(r['ids'])<=allowed and not set(r['ids'])&forbidden
                if r['role']=='B_evaluation':
                    seal=seals[r['prediction_sha256']];assert seal['models']==192 and seal['time']<=r['time']
                target_requests+=len(r['ids'])
        for domain in [1,2,3]:
            file=d/f'{domain}_sealed_artifacts.json';payload=read(file)
            assert artifact_events[domain]['sha256']==sha(file)
            predfile=d/f'{domain}_sealed_predictions.npz';assert payload['prediction_sha256']==sha(predfile)
            seal=seals[sha(predfile)];assert artifact_events[domain]['time']<=seal['time']
            assert sha(d/f'{domain}_support.json')==payload['support_sha256']
            prefix=(d/'inner_cv.jsonl').read_bytes()[:payload['inner_cv_prefix_bytes']]
            assert hashlib.sha256(prefix).hexdigest()==payload['inner_cv_prefix_sha256']
            for row in payload['models']:assert sha(d/row['path'])==row['sha256']
        sealed={domain:dict(np.load(d/f'{domain}_sealed_predictions.npz')) for domain in [1,2,3]}
        membership={}
        for domain in [1,2,3]:
            for fold in range(4):
                for sig in ['H','HA']:
                    for td in ['S','T']:
                        ids=read(d/f'{domain}_{fold}_{sig}_{td}_ids.json')
                        membership[domain,fold,sig,td]=ids['source_audit']+ids['B'] if td=='S' else ids['B']
        with (d/'probe_predictions.csv').open() as f:
            for row in csv.DictReader(f):
                domain=int(row['task'][-1]);fold=int(row['fold']);sig=row['signature'];td=row['train_domain'];family=row['family'];variant=int(row['variant'])
                key=f'{domain}_{fold}_{sig}_{td}_{family}_{variant}'
                position=membership[domain,fold,sig,td].index(int(row['graph_id']))
                assert float(row['probability'])==float(sealed[domain][key][position]);predictions+=1
    assert models==2880 and cv_count==31680 and predictions==manifest['settings'].get('expected_predictions',851040)
    pilot=stage/'logs/pilot_3405/label_access.jsonl'
    assert not any(r.get('role')=='B_evaluation' and r['event']=='label_access' for r in lines(pilot))
    assert target_requests==manifest['settings'].get('expected_target_label_requests',12885)
    # Recompute every true-label seed aggregate directly from raw fold metrics.
    by_seed=defaultdict(list)
    for seed in manifest['settings']['seeds']:
        for r in lines(stage/f'logs/{seed}/probe_metrics.jsonl'):
            if r['variant']==0:by_seed[seed,r['task'],r['signature'],r['family'],r['train_domain'],r['pool']].append(r)
    for r in result['triangle']:
        for td,pool,name in [('S','S_audit','S_S_NLL'),('S','B','S_T_NLL'),('T','B','T_T_NLL')]:
            means=[sum(x['nll'] for x in by_seed[s,r['task'],r['signature'],r['family'],td,pool])/4 for s in manifest['settings']['seeds']]
            assert abs(sum(means)/5-r[name])<1e-12
    audit=dict(passed=True,inner_cv_fits=cv_count,selected_models=models,total_probe_fits=cv_count+models,
        per_graph_predictions=predictions,formal_target_label_values_requested=target_requests,final_labels_read=0,
        source_snapshot_references=len(refs),all_selected_parameters_inner_cv_only=True,
        predictions_exactly_match_prelabel_seals=True,all_model_and_support_hashes_match_seals=True,
        pilot_B_labels_read=0,independent_primary_mean_recomputation=True)
    write(stage/'publication_audit.json',audit)
    return audit


def publish(stage,repo,dest):
    proof=audit(stage,repo);dest.mkdir(exist_ok=False)
    for p in stage.iterdir():
        if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.md','.txt','.sh']:shutil.copy2(p,dest/p.name)
    for name in ['D0','features','logs','figures','analysis_code']:
        if (stage/name).exists():shutil.copytree(stage/name,dest/name)
    shutil.copytree(stage/'code_formal',dest/'code_snapshot',ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    shutil.copytree(stage.parent/'stage2_method',dest/'protocol')
    pub=dest/'publication_code';pub.mkdir();shutil.copy2(Path(__file__),pub/'archive_assumptions.py')
    subprocess.run(['git','bundle','create',str(dest/'execution_history.bundle'),'--all'],cwd=stage/'code_formal',check=True)
    (dest/'.gitattributes').write_text('*.csv whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\ncode_snapshot/full_query_experiment.py -whitespace\n')
    files=[dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file()]
    assert max(r['bytes'] for r in files)<100_000_000
    write(dest/'manifest.json',dict(experiment=read(stage/'config.json').get('publication_name','tqg_assumptions_20261103'),execution_commit=read(stage/'experiment_manifest.json')['git_commit'],
        source_run=str(stage),audit=proof,files=files,excluded=['real graph cache','duplicated existing GNN checkpoints']))
    print(json.dumps(dict(status='archived',files=len(files),bytes=sum(r['bytes'] for r in files),audit=proof)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--dest',type=Path)
    a=p.parse_args()
    if a.dest:publish(a.stage,a.repo,a.dest)
    else:print(json.dumps(audit(a.stage,a.repo)))
