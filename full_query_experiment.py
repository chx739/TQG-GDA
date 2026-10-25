"""Full-query global TQG ablation; explicit new user-requested scope."""
import time
BOOT = time.perf_counter()
import argparse
import copy
import csv
import json
from pathlib import Path
import statistics
from unittest.mock import patch

import torch

from revision_common import Context, Budget, CODE, STAGE, RUN, atomic_json, read
from training_revision import RevisionTrainer, paired_ids, run_job
from tqg_gda.revision import ResponseDictionary, ResidualPredictor
from tqg_gda.training import load_model
from tqg_gda.utils import sha256, seed_everything

V1 = RUN / 'stage3_revision_v1'
PLAN = CODE / 'configs/full_query_20261025.json'
LABELS = dict(B_frozen='fit-only 源模型不动', B_fit='继续源 fit 训练',
              B_full='全源标签训练', B_pseudo='简单置信度伪标签',
              Q51_OE='全部 query 51，odd+even', Q51_O='全部 query 51，odd',
              Q250_OE='全部源训练图 250，odd+even', Q250_O='全部源训练图 250，odd')


class FullContext(Context):
    def __init__(self):
        super().__init__()
        self.full_plan = read(PLAN)
        self.plan['revision_id'] = self.full_plan['revision_id']
        self.plan['adaptation'] = copy.deepcopy(self.full_plan['adaptation'])
        self.plan['seeds'] = self.full_plan['seeds']
        self.plan['approval'] = dict(training_requested_in_this_turn=True,
            basis='User explicitly requested n51/n250 odd/odd+even plus four baselines',
            internal_resource_ceiling_seconds=self.full_plan['resource_cap_seconds'])
        self.plan['status'] = 'USER_REQUESTED_EXECUTION'
        self.arms = self.full_plan['arms']
        self.roles['domains']['1'] = read(STAGE/'roles_domain1.json')
        self.plan['datasets'][0]['tasks'] = ['0_to_1']
        # Warmups trained only on domain0; target metadata is checked separately.
        reusable=[]
        for record in self.old_manifest['records']:
            if record['task']=='0_to_2' and record['arm'] in ('original_TQG_target_probes','full_source_CE_700plus200'):
                reusable.append(dict(task='0_to_1',seed=record['seed'],arm=record['arm'],
                    warmup_checkpoint=record['warmup_checkpoint'],
                    **({'warmup_checkpoint_sha256':record['warmup_checkpoint_sha256']} if 'warmup_checkpoint_sha256' in record else {})))
        self.old_manifest['records'].extend(reusable)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b, (a, b)


def trainer(ctx, domain, seed, arm):
    cfg = ctx.arm(arm)
    all_source = cfg['initialization'] == 'all_source_warmup'
    seed_everything(seed, 1, True)
    model, _, _ = ctx.source(domain, seed, all_source)
    fixed = copy.deepcopy(model).eval()
    for p in fixed.parameters():
        p.requires_grad_(False)
    return RevisionTrainer(model, fixed, ctx.task(domain, all_source),
                           cfg, ctx.plan, seed, domain)


def acceptance(ctx):
    checks = []
    for seed in ctx.plan['seeds']:
        for arm in ('B_fit','B_full'):
            item=trainer(ctx,1,seed,arm)
            expected=copy.deepcopy(item.model)
            settings=ctx.plan['adaptation'];adam=settings['adam']
            opt=torch.optim.Adam(expected.parameters(),lr=settings['learning_rate'],
                betas=(adam['beta1'],adam['beta2']),eps=adam['epsilon'],
                weight_decay=adam['weight_decay'],amsgrad=adam['amsgrad'])
            schedule,_=paired_ids(item.task,seed,1,1)
            role='all_source' if arm=='B_full' else 'fit'
            batch=item.task.batch(schedule[role],role)
            loss=torch.nn.functional.cross_entropy(expected(batch),batch.y)
            loss.backward();opt.step()
            item.step()
            assert_tree_equal(item.model.state_dict(),expected.state_dict())
            assert_tree_equal(item.optimizer.state_dict(),opt.state_dict())
    checks.append('10 fresh baseline handlers match independent ordinary source CE update')
    for arm in ('Q51_OE', 'Q51_O', 'Q250_OE', 'Q250_O'):
        item = trainer(ctx, 1, 3405, arm)
        before = copy.deepcopy(item.model.state_dict())
        schedule, _ = paired_ids(item.task, 3405, 1, 1)
        cfg = ctx.arm(arm)
        expected = set(item.task.ids(cfg['query_role']))
        assert len(expected) == cfg['n'] and not expected & set(item.task.validation_ids)
        assert len(expected & set(item.task.fit_ids)) == cfg['pretraining_fit_overlap']
        q = item.task.batch(tuple(sorted(expected)), cfg['query_role'])
        t = item.task.batch(schedule['target'], 'target')
        assert getattr(t, 'y', None) is None
        dictionary = ResponseDictionary.build(item.model, q, t,
            __import__('revision_common').generator(910000,3405,1,1,2),
            K=4,response=cfg['response'],direction='target')
        pred = ResidualPredictor(dictionary,q,cfg,torch.arange(len(expected))).predict(
            t,phi=dictionary.feedback.target_features)
        for value in (pred.reference,pred.correction,pred.coefficients,
                      dictionary.feedback.source_features,dictionary.feedback.directions[0]['encoder.layers.0.eps']):
            assert not value.requires_grad and value.grad_fn is None
        assert_tree_equal(item.model.state_dict(), before)
        assert dictionary.feedback.source_features.shape[0] == cfg['n']
        assert dictionary.feedback.source_features.shape[-1] == (8 if cfg['response']=='odd_even' else 4)
        with patch.object(item.optimizer,'step',wraps=item.optimizer.step) as step:
            row=item.step()
            assert step.call_count==1
        assert set(row['batches']['query'])==expected
        assert row['batches']['fit']==list(schedule['fit'])
        assert row['batches']['target']==list(schedule['target'])
        assert all(p.requires_grad for p in item.model.parameters())
        checks.append(arm+' full source roles, stopped feedback, paired batches, single update')
    # Deterministic query-set extension must also preserve exact recovery.
    a=trainer(ctx,1,3405,'Q250_OE')
    a.step()
    state=copy.deepcopy(a.state())
    a.step()
    b=trainer(ctx,1,3405,'Q250_OE')
    b.restore(state)
    b.step()
    assert_tree_equal(a.model.state_dict(),b.model.state_dict())
    assert_tree_equal(a.optimizer.state_dict(),b.optimizer.state_dict())
    checks.append('Q250 odd+even exact recovery including Adam')
    out=dict(passed=True,checks=checks,git_commit=ctx.commit,
             target_labels_used=False,new_full_training_runs=0)
    atomic_json(STAGE/'acceptance.json',out)
    print(json.dumps(dict(event='acceptance_passed',checks=len(checks))),flush=True)


def keys(ctx):
    return [(d,s,a) for d in ctx.full_plan['tasks'] for s in ctx.plan['seeds']
            for a in ctx.full_plan['new_training_arms']]


def verify_result(folder):
    info=read(folder/'result.json')
    assert info['status']=='completed' and info['completed_updates']==200
    for name,digest in info['artifact_hashes'].items():
        assert sha256(folder/name)==digest,(folder,name)
    return info


def baseline_records(ctx):
    rows=[]
    for seed in ctx.plan['seeds']:
        _,_,source=ctx.source(1,seed)
        rows.append(dict(domain=1,seed=seed,arm='B_frozen',reused=True,
            training_commit=ctx.old_manifest['git_commit'],output=str(source.parent),
            final_checkpoint=str(source),selected_checkpoint=None))
    return rows


def train(ctx,budget,profile=False):
    assert read(STAGE/'acceptance.json')['passed']
    registry=STAGE/'training_manifest.json'
    previous=read(registry) if registry.exists() else dict(records=[])
    rows={(x['domain'],x['seed'],x['arm']):x for x in previous['records'] if not x['reused']}
    selected=keys(ctx)[:1] if profile else keys(ctx)
    for i,(domain,seed,arm) in enumerate(selected,1):
        budget.check()
        folder=STAGE/'jobs'/f'0_to_{domain}'/str(seed)/arm
        print(json.dumps(dict(event='job_started',job=i,total=len(selected),
                             domain=domain,seed=seed,arm=arm)),flush=True)
        result=run_job(ctx,domain,seed,arm,output=folder,resume=folder.exists(),
                       stop_after=10 if profile else None,budget=budget)
        rows[(domain,seed,arm)]=dict(domain=domain,seed=seed,arm=arm,reused=False,
            output=str(folder),status=result['status'],completed_updates=result['completed_updates'],
            result_sha256=sha256(folder/'result.json'),training_commit=ctx.commit,
            final_checkpoint=str(folder/'last.pt'),selected_checkpoint=str(folder/'selected.pt'))
        combined=list(rows.values())+baseline_records(ctx)
        atomic_json(registry,dict(commit=ctx.commit,expected_new_jobs=35,
            new_jobs_completed=sum(x.get('status')=='completed' for x in rows.values()),
            records=combined,target_labels_used=False))
        print(json.dumps(dict(event='job_finished',job=i,total=len(selected),arm=arm,
                             status=result['status'],seconds=result['seconds'])),flush=True)
    if profile:
        return
    assert len(rows)==35 and all(x['status']=='completed' for x in rows.values())
    manifest=read(registry)
    hashes={str(registry):sha256(registry)}
    for row in manifest['records']:
        if row['arm']=='B_frozen':
            path=Path(row['final_checkpoint']);hashes[str(path)]=sha256(path)
        else:
            folder=Path(row['output']);info=verify_result(folder)
            hashes[str(folder/'result.json')]=sha256(folder/'result.json')
            for name,value in info['artifact_hashes'].items():
                hashes[str(folder/name)]=value
    atomic_json(STAGE/'training_seal.json',dict(complete=True,new_training_jobs=35,
        reused_training_jobs=0,frozen_references=5,records=manifest['records'],
        artifact_hashes=hashes,target_labels_used=False))
    print(json.dumps(dict(event='all_training_sealed',new_jobs=35)),flush=True)


def evaluate(ctx):
    # Import target-label access only after all training is complete and sealed.
    seal=read(STAGE/'training_seal.json')
    assert seal['complete'] and seal['new_training_jobs']==35
    for file,digest in seal['artifact_hashes'].items():
        assert sha256(file)==digest,file
    from revision_evaluation import DevelopmentLabels,full_metrics
    predictions=[]
    out=STAGE/'evaluations';out.mkdir(exist_ok=True)
    for row in seal['records']:
        domain,seed,arm=row['domain'],row['seed'],row['arm']
        task=ctx.task(domain,arm=='B_full')
        ids=ctx.development_ids(domain)
        labels=DevelopmentLabels(ctx,domain,task,STAGE/'development_label_access.jsonl')(ids)
        for selection,key in [('fixed_200','final_checkpoint'),('source_selected','selected_checkpoint')]:
            if row[key] is None:continue
            path=Path(row[key]);model,state=load_model(path);model.eval()
            with torch.no_grad():
                logits=torch.cat([model(task.batch(ids[j:j+32],'target'))
                                  for j in range(0,len(ids),32)])
            metrics=full_metrics(logits,labels)
            # Integer counts -> Python float division, no float32 recall rounding.
            cm=metrics['confusion']
            recalls=[cm[c][c]/sum(cm[c]) for c in range(len(cm)) if sum(cm[c])]
            metrics['balanced_accuracy']=statistics.mean(recalls)
            metrics['class_recall']=recalls
            report=dict(task=f'0_to_{domain}',domain=domain,seed=seed,arm=arm,
                selection=selection,reused_training=row['reused'],
                training_commit=row['training_commit'],evaluation_commit=ctx.commit,
                checkpoint=str(path),checkpoint_sha256=sha256(path),target_ids=list(ids),
                target_final_labels_used=False,metrics=metrics)
            atomic_json(out/f'{domain}_{seed}_{arm}_{selection}.json',report)
            predictions.append(report)
    assert len(predictions)==75
    fields=['task','seed','arm','method','selection','balanced_accuracy','accuracy','CE',
            'class_recall','predicted_class_proportions','reused_training',
            'training_commit','evaluation_commit','checkpoint','checkpoint_sha256']
    with (STAGE/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for x in predictions:
            m=x['metrics'];writer.writerow(dict(task=x['task'],seed=x['seed'],arm=x['arm'],
                method=LABELS[x['arm']],selection=x['selection'],
                balanced_accuracy=m['balanced_accuracy'],accuracy=m['accuracy'],CE=m['ce'],
                class_recall=json.dumps(m['class_recall']),
                predicted_class_proportions=json.dumps(m['predicted_class_proportions']),
                **{k:x[k] for k in ['reused_training','training_commit','evaluation_commit',
                                    'checkpoint','checkpoint_sha256']}))
    groups=[]
    for domain in ctx.full_plan['tasks']:
        for arm in ctx.arms:
            for selection in ['fixed_200','source_selected']:
                values=[x for x in predictions if (x['domain'],x['arm'],x['selection'])==
                        (domain,arm,selection)]
                if not values:continue
                assert len(values)==5
                summary={}
                for metric in ['balanced_accuracy','accuracy','ce']:
                    v=[x['metrics'][metric] for x in values]
                    summary[metric]=dict(mean=statistics.mean(v),sd=statistics.stdev(v))
                groups.append(dict(domain=domain,arm=arm,selection=selection,metrics=summary,
                    per_seed=[dict(seed=x['seed'],balanced_accuracy=x['metrics']['balanced_accuracy'])
                              for x in sorted(values,key=lambda x:x['seed'])]))
    paired=[]
    pairs=[(a,b) for a in ('Q51_OE','Q51_O','Q250_OE','Q250_O')
           for b in ('B_frozen','B_fit','B_full','B_pseudo')]
    pairs += [('Q250_OE','Q51_OE'),('Q250_O','Q51_O'),('Q51_OE','Q51_O'),('Q250_OE','Q250_O')]
    for domain in ctx.full_plan['tasks']:
        for left,right in pairs:
            l=next(x for x in groups if (x['domain'],x['arm'],x['selection'])==(domain,left,'fixed_200'))
            r=next(x for x in groups if (x['domain'],x['arm'],x['selection'])==(domain,right,'fixed_200'))
            delta=[a['balanced_accuracy']-b['balanced_accuracy'] for a,b in zip(l['per_seed'],r['per_seed'])]
            paired.append(dict(domain=domain,left=left,right=right,mean_difference_pp=statistics.mean(delta)*100,
                wins=sum(v>1e-12 for v in delta),ties=sum(abs(v)<=1e-12 for v in delta),
                losses=sum(v< -1e-12 for v in delta),per_seed_difference_pp=[v*100 for v in delta]))
    atomic_json(STAGE/'results_summary.json',dict(groups=groups,paired=paired,
        new_training_jobs=35,reused_training_jobs=0,frozen_references=5,
        evaluations=75,final_labels_evaluated=False))
    report='# 全量查询补充：0→1\n\n'
    report+='使用原有0→1开发/保留ID，源预训练只来自源0；本轮35个新适配，5个冻结参考。四个TQG组及三个训练基线全部按本轮配对调度重新运行。\n\n'
    report+='n51/n250与odd/odd+even为主变量；源fit分类损失、相同源模型、K4、rho=.05、ridge=.01、beta=.1、Adam lr=.0003、200步固定。n250包含预训练用过的199张源fit图。\n\n'
    report+='| 方法 | 0→1 balanced accuracy（%，均值±SD） |\n|---|---:|\n'
    for arm in ['B_frozen','B_fit','B_full','B_pseudo','Q51_OE','Q51_O','Q250_OE','Q250_O']:
        metric=next(x for x in groups if (x['domain'],x['arm'],x['selection'])==(1,arm,'fixed_200'))['metrics']['balanced_accuracy']
        report+=f"| {LABELS[arm]} | {metric['mean']*100:.2f} ± {metric['sd']*100:.2f} |\n"
    report+='\n完整每种子结果及配对差见results.csv/results_summary.json。目标final标签未评价，仍为开发探索。\n'
    (STAGE/'run_report.md').write_text(report)
    print(json.dumps(dict(event='evaluation_complete',records=len(predictions))),flush=True)


def audit(ctx):
    seal=read(STAGE/'training_seal.json')
    for path,digest in seal['artifact_hashes'].items():
        assert sha256(path)==digest,path
    for row in seal['records']:
        if row['arm']=='B_frozen':continue
        folder=Path(row['output']);verify_result(folder)
        hist=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
        assert [x['step'] for x in hist]==list(range(1,201))
        task=ctx.task(row['domain'],row['arm']=='B_full')
        for h in hist:
            schedule,_=paired_ids(task,row['seed'],row['domain'],h['step'])
            assert h['batches']['fit']==list(schedule['all_source' if row['arm']=='B_full' else 'fit'])
            if row['arm'] in ('B_fit','B_full'):continue
            assert h['batches']['target']==list(schedule['target'])
            if row['arm'].startswith('Q'):
                cfg=ctx.arm(row['arm'])
                assert h['batches']['query']==sorted(task.ids(cfg['query_role']))
                assert not set(h['batches']['query']) & set(task.validation_ids)
        checkpoint=torch.load(folder/'last.pt',weights_only=False)
        assert checkpoint['step']==200
        assert all(int(x['step'])==200 for x in checkpoint['optimizer']['state'].values())
    requests=[json.loads(l) for l in (STAGE/'development_label_access.jsonl').read_text().splitlines()]
    assert len(requests)==40
    allowed=set(ctx.development_ids(1))
    assert all(set(x['ids'])<=allowed for x in requests)
    summary=read(STAGE/'results_summary.json')
    assert summary['evaluations']==75
    atomic_json(STAGE/'completion_audit.json',dict(passed=True,new_jobs=35,reused_jobs=0,
        frozen_references=5,evaluations=75,all_updates=200,
        source_roles_and_paired_batches_verified=True,Adam_step_counts_verified=True,
        final_labels_evaluated=False,commit=ctx.commit))
    print(json.dumps(dict(event='completion_audit_passed')),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['acceptance','profile','train','evaluate','audit'])
    args=parser.parse_args()
    ctx=FullContext()
    budget=Budget('full_query_'+args.phase)
    status='failed'
    try:
        if args.phase=='acceptance':acceptance(ctx)
        elif args.phase in ('profile','train'):train(ctx,budget,profile=args.phase=='profile')
        elif args.phase=='evaluate':evaluate(ctx)
        else:audit(ctx)
        status='completed'
    finally:
        measured=budget.finish(status)
        # Account for imports and context validation excluded by Budget's timer.
        ledger=read(STAGE/'compute_ledger.json')
        overhead=max(0.,time.perf_counter()-BOOT-measured)
        ledger['consumed_seconds']+=overhead
        ledger['records'].append(dict(phase=args.phase+'_startup_overhead',seconds=overhead,status=status))
        atomic_json(STAGE/'compute_ledger.json',ledger)


if __name__=='__main__':
    main()

