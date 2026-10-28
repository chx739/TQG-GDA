"""Independent artifact/summary audit; no target-label reader or model inference."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics

import torch

RUN=Path(__file__).resolve().parent
STAGE=RUN/'stage3_execution'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    marker=read(RUN/'stage_3_done')
    manifest=read(STAGE/'experiment_manifest.json')
    seal=read(STAGE/'training_seal.json')
    plan=read(RUN/'stage2_method/experiment_plan.yaml')
    summary=read(STAGE/'results_summary.json')
    assert marker['new_adaptations']==15 and marker['reused_adaptations']==60
    assert digest(RUN/'stage2_method/experiment_plan.yaml')==manifest['plan_sha256']
    assert digest(STAGE/'code/configs/experiment_plan.yaml')==manifest['plan_sha256']
    for asset in manifest['assets']:assert digest(asset['path'])==asset['sha256']
    assert manifest['references']==seal['references']
    references={(r['spec']['domain'],r['spec']['seed'],r['spec']['arm']):r for r in seal['references']}
    histories={};checked_models=set();total_new_steps=0
    for record in [*seal['jobs'],*seal['references']]:
        spec=record['spec'];key=(spec['domain'],spec['seed'],spec['arm'])
        for info in record['artifacts'].values():assert digest(info['path'])==info['sha256']
        config=read(record['artifacts']['config.json']['path'])
        previous=read(references[(spec['domain'],spec['seed'],'B16')]['artifacts']['config.json']['path'])
        expected=json.loads(json.dumps(previous))
        if spec['arm']=='B_fit':expected['method']='source_only'
        else:expected['adaptation']['target_batch_size']=int(spec['arm'][1:])
        assert config==expected
        rows=[json.loads(line) for line in Path(record['artifacts']['history.jsonl']['path']).read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(1,201));histories[key]=rows
        if spec['arm']=='B196':total_new_steps+=len(rows)
        checkpoint=Path(record['artifacts']['last.pt']['path'])
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['step']==200
        source=next(s for s in manifest['source_models'] if s['seed']==spec['seed'])
        assert state['source_checkpoint_sha256']==source['sha256']
        for group in state['optimizer']['param_groups']:
            assert group['momentum']==0 and group['weight_decay']==0 and 'betas' not in group
        assert all(bool(torch.isfinite(v).all()) for v in state['model'].values())
    assert len(histories)==75 and total_new_steps==3000
    paired_steps=0
    for domain in (1,2,3):
        for seed in range(3405,3410):
            for step in range(200):
                groups={arm:histories[(domain,seed,arm)][step]['batches'] for arm in ('B_fit','B16','B32','B64','B196')}
                assert all(g['fit']==groups['B16']['fit'] for g in groups.values())
                for arm in ('B16','B32','B64','B196'):
                    assert len(groups[arm]['query'])==196 and len(set(groups[arm]['query']))==196
                    assert not set(groups[arm]['fit'])&set(groups[arm]['query'])
                    assert groups[arm]['query']==groups['B16']['query']
                    assert groups[arm]['target']==groups['B196']['target'][:int(arm[1:])]
                    assert len(set(groups[arm]['target']))==int(arm[1:])
                paired_steps+=1
    with (STAGE/'results.csv').open() as stream:results=list(csv.DictReader(stream))
    assert len(results)==165
    keys=[(r['task'],r['seed'],r['arm'],r['selection']) for r in results]
    assert len(set(keys))==165
    for row in results:
        checkpoint=Path(row['checkpoint'])
        assert digest(checkpoint)==row['checkpoint_sha256']
        assert row['evaluation_commit']==manifest['git_commit'] and row['target_final_labels_evaluated']=='False'
        if checkpoint not in checked_models:
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert all(bool(torch.isfinite(v).all()) for v in state['model'].values())
            checked_models.add(checkpoint)
    for task,arms in summary['metrics'].items():
        for arm,selections in arms.items():
            for selection,metrics in selections.items():
                selected=[r for r in results if (r['task'],r['arm'],r['selection'])==(task,arm,selection)]
                assert len(selected)==5
                for name in ('balanced_accuracy','accuracy','ce','auroc'):
                    values=[float(r[name]) for r in selected]
                    assert abs(statistics.mean(values)-metrics[name]['mean'])<1e-12
                    assert abs(statistics.stdev(values)-metrics[name]['sd'])<1e-12
    prior=Path(plan['reference_archive'])/'results/results.csv'
    with prior.open() as stream:old_results=list(csv.DictReader(stream))
    old_index={(r['task'],r['seed'],r['arm'],r['selection']):r for r in old_results}
    for row in results:
        if row['arm'] not in ('B_frozen','B_fit','B16','B32','B64'):continue
        old_arm=row['arm']
        old=old_index[(row['task'],row['seed'],old_arm,row['selection'])]
        for name in ('balanced_accuracy','accuracy','ce','auroc'):
            assert abs(float(row[name])-float(old[name]))<1e-10
    accesses=[json.loads(line) for line in (STAGE/'evaluation_label_audit.jsonl').read_text().splitlines()]
    assert len(accesses)==3 and sum(r['labels_read'] for r in accesses)==1626
    sealed_at=dt.datetime.fromisoformat(seal['sealed_at'])
    for row in accesses:
        assert row['training_sealed'] and not row['final_labels_read']
        assert dt.datetime.fromisoformat(row['timestamp'])>=sealed_at
        assert set(row['ids'])==set(manifest['roles'][str(row['domain'])]['development_ids'])
        assert not set(row['ids'])&set(manifest['roles'][str(row['domain'])]['final_ids'])
    coverage=[]
    for (domain,seed,arm),history_rows in histories.items():
        if arm=='B_fit':continue
        unique=set(i for row in history_rows for i in row['batches']['target'])
        pool=set(manifest['roles'][str(domain)]['target_ids'])
        assert unique<=pool
        coverage.append(dict(domain=domain,seed=seed,arm=arm,unique_target_graphs=len(unique),
                             target_pool_size=len(pool),coverage_fraction=len(unique)/len(pool),
                             target_uses=sum(len(row['batches']['target']) for row in history_rows)))
    numeric=read(STAGE/'training_diagnostics.json')
    checkpoints={}
    for item in [*seal['jobs'],*seal['references']]:
        spec=item['spec']
        checkpoints[(spec['domain'],spec['seed'],spec['arm'])]=item['artifacts']['last.pt']['path']
    parameter_differences=[]
    for domain in (1,2,3):
        for seed in range(3405,3410):
            base=torch.load(checkpoints[(domain,seed,'B_fit')],map_location='cpu',weights_only=False)['model']
            for arm in ('B16','B32','B64','B196'):
                candidate=torch.load(checkpoints[(domain,seed,arm)],map_location='cpu',weights_only=False)['model']
                norm=sum(float((candidate[name]-base[name]).square().sum()) for name in base)**.5
                parameter_differences.append(dict(domain=domain,seed=seed,arm=arm,l2_from_source_sgd=norm))
    report=dict(passed=True,git_commit=manifest['git_commit'],plan_sha256=manifest['plan_sha256'],
                new_adaptations=15,reused_adaptations=60,new_logged_steps=3000,reused_logged_steps=12000,
                results_rows=165,source_checkpoints_reused=5,all_finite_checkpoints=True,
                summaries_recomputed=True,reused_metrics_match_prior=True,
                paired_batch_steps_verified=paired_steps,only_target_batch_changed=True,
                target_development_labels_read=1626,target_final_labels_read=0,
                numerical_fallbacks={arm:sum(r['numerical_fallbacks'] for r in numeric['jobs'] if r['arm']==arm) for arm in ('B16','B32','B64','B196')},
                nonconverged_queries={arm:sum(r['nonconverged'] for r in numeric['jobs'] if r['arm']==arm) for arm in ('B16','B32','B64','B196')},
                max_gradient_identity_error=max(r['max_gradient_identity_error'] for r in numeric['jobs']),
                max_sgd_step_identity_error=max(r['max_sgd_step_identity_error'] for r in numeric['jobs']),
                target_coverage=coverage,parameter_differences=parameter_differences,
                all_query_models_differ_from_source_sgd=all(r['l2_from_source_sgd']>0 for r in parameter_differences),
                this_audit_reads_target_labels=False,this_audit_trains_models=False)
    (STAGE/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
