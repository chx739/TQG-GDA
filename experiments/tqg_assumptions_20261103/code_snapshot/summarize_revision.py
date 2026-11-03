"""Seed-level summaries, predeclared D2 progression gate, and readable reports."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics

import torch

from revision_common import Context, STAGE, Budget, read, atomic_json, hardware
from revision_evaluation import DevelopmentLabels, full_metrics
from tqg_gda.training import load_model
from tqg_gda.utils import sha256


def average(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def aggregate(rows, grouping):
    groups={}
    for row in rows:
        key=tuple(row[name] for name in grouping)
        groups.setdefault(key,{}).setdefault(row['seed'],[]).append(row['metrics'])
    result=[]
    for key,seeds in sorted(groups.items()):
        per_seed=[]
        for seed,blocks in sorted(seeds.items()):
            names=sorted(set().union(*(set(b) for b in blocks)))
            means={name:average([b.get(name) for b in blocks]) for name in names}
            valid={name:sum(b.get(name) is not None for b in blocks) for name in names}
            per_seed.append(dict(seed=seed,blocks=len(blocks),metrics=means,valid_counts=valid))
        summary={}
        for name in per_seed[0]['metrics']:
            values=[r['metrics'][name] for r in per_seed if r['metrics'][name] is not None]
            summary[name]=dict(mean=average(values),seed_std=statistics.stdev(values) if len(values)>1 else None,
                               model_seeds=len(values))
        result.append(dict(zip(grouping,key),per_seed=per_seed,metrics=summary))
    return result


def summarize_diagnostics(ctx):
    d1_source=read(STAGE/'diagnostics/D1/source_summary.json')
    expected_D1={STAGE/f'diagnostics/D1/0_to_{d}'/str(s)/f'target_{b:02d}.json'
                 for d in (2,3) for s in ctx.plan['seeds'] for b in range(20)}
    expected_D2={STAGE/f'diagnostics/D2/0_to_{d}'/str(s)/state/f'target_{b:02d}.json'
                 for d in (2,3) for s in ctx.plan['seeds']
                 for state,count in [('warmup',20),('historical_final',3)] for b in range(count)}
    if set((STAGE/'diagnostics/D1').glob('0_to_*/*/target_*.json'))!=expected_D1:
        raise ValueError('All 200 D1 offline blocks must complete before aggregation')
    if set((STAGE/'diagnostics/D2').glob('0_to_*/*/*/target_*.json'))!=expected_D2:
        raise ValueError('All 230 D2 offline blocks must complete before progression decision')
    d1rows=[]
    for file in sorted((STAGE/'diagnostics/D1').glob('0_to_*/*/target_*.json')):
        data=read(file)
        if len(data['rows'])!=74 or len({r['key'] for r in data['rows']})!=74:
            raise ValueError('Incomplete or duplicated D1 configuration rows')
        for row in data['rows']:
            d1rows.append(dict(task=f"0_to_{data['domain']}",seed=data['seed'],configuration=row['key'],metrics=row['metrics']))
    d1groups=aggregate(d1rows,['task','configuration'])
    atomic_json(STAGE/'diagnostics/D1/target_summary.json',dict(groups=d1groups,rows=len(d1rows)))
    h1=[]
    for task in ('0_to_2','0_to_3'):
        def source_group(n):
            return next(g for g in d1_source['groups'] if g['task']==task and
                        g['configuration']==f'fixed__target__n{n}__M4__ridge0.01')
        small,large=source_group(8),source_group(32)
        ms=small['metrics']['source_audit_residual_mse']['mean']
        ml=large['metrics']['source_audit_residual_mse']['mean']
        paired=[dict(seed=a['seed'],small_mse=a['metrics']['source_audit_residual_mse'],
                     large_mse=b['metrics']['source_audit_residual_mse'])
                for a,b in zip(small['per_seed'],large['per_seed'])]
        wins=sum(p['large_mse']<p['small_mse'] for p in paired)
        reduction=(ms-ml)/ms if ms>0 else 0.
        h1.append(dict(task=task,relative_MSE_reduction=reduction,wins=wins,passed=reduction>=.2 and wins>=4,
                       per_seed=paired))
    rows=[]
    expected_arm_conditions={(a,c) for a in ['T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15','T16','zero','global_mean']
                             for c in ('true','perm_113','perm_227','perm_349')}
    for file in sorted((STAGE/'diagnostics/D2').glob('0_to_*/*/*/target_*.json')):
        data=read(file)
        if len(data['rows'])!=56 or {(r['arm'],r['label_condition']) for r in data['rows']}!=expected_arm_conditions:
            raise ValueError('Incomplete or duplicated D2 arm/label controls')
        for row in data['rows']:
            metrics=dict(row['metrics'])
            for name in ('ESS_mean','max_weight','L1_from_uniform','gate_rate','raw_correction_rms','correction_rms'):
                metrics['source_'+name]=row['source_statistics'].get(name)
            metrics['global_oracle_residual_mse']=data['oracle']['residual_mse']
            rows.append(dict(task=f"0_to_{data['domain']}",seed=data['seed'],state=data['state'],
                             arm=row['arm'],label_condition=row['label_condition'],metrics=metrics))
    groups=aggregate(rows,['task','state','arm','label_condition'])
    if any(len(g['per_seed'])!=5 or any(r['blocks']!=(20 if g['state']=='warmup' else 3)
            for r in g['per_seed']) for g in groups):
        raise ValueError('Every D2 group must contain all prescribed blocks and model seeds')
    atomic_json(STAGE/'diagnostics/D2/target_summary.json',dict(groups=groups,rows=len(rows)))
    decisions=[]
    for task in ('0_to_2','0_to_3'):
        def group(arm):
            return next(g for g in groups if (g['task'],g['state'],g['arm'],g['label_condition'])==
                        (task,'warmup',arm,'true'))
        local,shuffle=group('T6'),group('T11')
        tests=[]
        for metric in ('gradient_cosine','actual_Adam_CE_gain_vs_source_only'):
            pairs=[]
            for l,r in zip(local['per_seed'],shuffle['per_seed']):
                lv,rv=l['metrics'][metric],r['metrics'][metric]
                valid=(lv is not None and rv is not None)
                if metric=='gradient_cosine':
                    valid=valid and l['valid_counts'][metric]>=10 and r['valid_counts'][metric]>=10
                pairs.append(dict(seed=l['seed'],local=lv,shuffle=rv,delta=lv-rv if valid else None,valid=valid))
            deltas=[p['delta'] for p in pairs if p['valid']]
            wins=sum(d>0 for d in deltas)
            lm=average([p['local'] for p in pairs if p['valid']])
            dm=average(deltas)
            passed=len(deltas)==5 and lm>0 and dm>0 and wins>=4
            tests.append(dict(metric=metric,local_mean=lm,paired_delta_mean=dm,wins=wins,
                              valid_seeds=len(deltas),passes_progression=passed,per_seed=pairs))
        decisions.append(dict(task=task,passes_progression=any(t['passes_progression'] for t in tests),tests=tests))
    allow=any(d['passes_progression'] for d in decisions)
    strong=[]
    for task in ('0_to_2','0_to_3'):
        def get(arm,condition):
            return next(g for g in groups if (g['task'],g['state'],g['arm'],g['label_condition'])==
                        (task,'warmup',arm,condition))
        true=get('T6','true')['per_seed']
        shuffled=get('T11','true')['per_seed']
        permutations=[get('T6',c)['per_seed'] for c in ('perm_113','perm_227','perm_349')]
        values=[]
        for index,entry in enumerate(true):
            tv=entry['metrics']['gradient_cosine']
            sv=shuffled[index]['metrics']['gradient_cosine']
            pv=average([p[index]['metrics']['gradient_cosine'] for p in permutations])
            values.append(dict(seed=entry['seed'],true=tv,shuffled=sv,permuted_labels=pv,
                               vs_shuffle=tv-sv if tv is not None and sv is not None else None,
                               vs_permuted=tv-pv if tv is not None and pv is not None else None))
        criteria={}
        for field in ('vs_shuffle','vs_permuted'):
            ds=[v[field] for v in values if v[field] is not None]
            criteria[field]=dict(mean_delta=average(ds),wins=sum(d>0 for d in ds),
                                 passed=len(ds)==5 and average(ds)>=.05 and sum(d>0 for d in ds)>=4)
        positive=[v['true'] for v in values if v['true'] is not None]
        criteria['positive_direction']=dict(mean=average(positive),positive_seeds=sum(v>0 for v in positive),
            passed=len(positive)==5 and average(positive)>0 and sum(v>0 for v in positive)>=4)
        strong.append(dict(task=task,criteria=criteria,per_seed=values,
                           strong_mechanism_passed=all(c['passed'] for c in criteria.values())))
    decision=dict(commit=ctx.commit,allow_B=allow,tasks=decisions,strong_mechanism=strong,
                  scope='exploratory_stage_progression_not_H2_success',
                  heterogeneity_only=sum(d['passes_progression'] for d in decisions)==1,
                  keep_both_tasks=True,source_H1=h1)
    atomic_json(STAGE/'diagnostics/D2/decision.json',decision)
    atomic_json(STAGE/'diagnostics/target_evidence_seal.json',dict(commit=ctx.commit,D1_blocks=200,D2_blocks=230,
        artifacts=[dict(path=str(p),sha256=sha256(p)) for p in sorted(expected_D1|expected_D2)],
        complete=True,final_labels_used=False))
    # Flatten numerical diagnostic rows for later traceability.
    all_rows=[dict(stage='D1',**r) for r in d1rows]+[dict(stage='D2',**r) for r in rows]
    keys=sorted(set().union(*(set(r['metrics']) for r in all_rows)))
    columns=['stage','task','seed','state','arm','label_condition','configuration']+keys
    with (STAGE/'diagnostics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({**{k:row.get(k) for k in columns[:7]},**row['metrics']})
    lines=['# 改进版诊断阶段结果','', '以下为开发阶段诊断，不使用目标 final 标签。','',
           '## D1：查询监督数量','', '| 任务 | n32 相比 n8 的源 audit MSE 降幅 | 配对改善种子 | H1 判据 |',
           '|---|---:|---:|---|']
    for result in h1:
        lines.append(f"| {result['task']} | {result['relative_MSE_reduction']*100:.2f}% | {result['wins']}/5 | {'通过' if result['passed'] else '未通过'} |")
    lines += ['', '## D2：是否进入局部匹配 B 组','',f"预定推进门槛：{'通过，开展完整 B 组' if allow else '未通过，暂不开展 B 组'}。",'',
              '| 任务 | 指标 | 局部法均值 | 相对排列权重差值 | 胜出种子 |','|---|---|---:|---:|---:|']
    for task in decisions:
        for test in task['tests']:
            fm=lambda v:f'{v:.6g}' if v is not None else '未定义'
            lines.append(f"| {task['task']} | {test['metric']} | {fm(test['local_mean'])} | {fm(test['paired_delta_mean'])} | {test['wins']}/5 |")
    lines += ['', '该门槛仅用于决定研究是否值得继续，不是分类有效或广泛迁移成功的证明。',
              '完整数值见 D1/D2 target_summary.json、decision.json 和 diagnostics.csv。']
    (STAGE/'diagnostic_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(decision,ensure_ascii=False),flush=True)


def evaluate_training(ctx, phase):
    """All jobs in the requested phase must be sealed before target scoring."""
    seal=read(STAGE/f'phase_{phase}_training_seal.json')
    if not seal.get('complete') or seal.get('commit')!=ctx.commit:
        raise PermissionError('Need complete compatible training seal')
    for job in seal['jobs']:
        for artifact in job['outputs'].values():
            if sha256(artifact['path'])!=artifact['sha256']:
                raise ValueError('Sealed training output changed')
    root=STAGE/'jobs'
    arms=ctx.plan['phases']['training_'+phase]['arms']
    jobs=[(domain,seed,arm,root/f'0_to_{domain}'/str(seed)/arm)
          for domain in (2,3) for seed in ctx.plan['seeds'] for arm in arms]
    if any(not (path/'result.json').exists() or read(path/'result.json').get('status')!='completed'
           for _,_,_,path in jobs):
        raise PermissionError('Complete and seal all predeclared phase jobs before evaluation')
    out=STAGE/'evaluations'
    out.mkdir(exist_ok=True)
    rows=[]
    for domain,seed,arm,path in jobs:
        all_source=arm=='B_full'
        task=ctx.task(domain,all_source)
        dev=ctx.development_ids(domain)
        label_reader=DevelopmentLabels(ctx,domain,task,out/'target_label_access.jsonl')
        labels=label_reader(dev)
        for selection,name in [('fixed_200','last.pt'),('source_selected','selected.pt')]:
            checkpoint=path/name
            result_path=out/f'0_to_{domain}_{seed}_{arm}_{selection}.json'
            if result_path.exists():
                row=read(result_path)
                if row['checkpoint_sha256']!=sha256(checkpoint):
                    raise ValueError('Evaluated checkpoint changed')
            else:
                model,state=load_model(checkpoint)
                with torch.no_grad():
                    logits=torch.cat([model(task.batch(dev[i:i+32],'target')) for i in range(0,len(dev),32)])
                metrics=full_metrics(logits,labels)
                row=dict(task=f'0_to_{domain}',seed=seed,arm=arm,selection=selection,metrics=metrics,
                         checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                         commit=ctx.commit,event_flags=read(path/'result.json').get('event_flags','clean'))
                atomic_json(result_path,row)
            rows.append(row)
    # Frozen reference, counted as evaluations rather than training jobs.
    for domain in (2,3):
        task=ctx.task(domain)
        dev=ctx.development_ids(domain)
        labels=DevelopmentLabels(ctx,domain,task,out/'target_label_access.jsonl')(dev)
        for seed in ctx.plan['seeds']:
            result_path=out/f'0_to_{domain}_{seed}_B_frozen_fixed_200.json'
            if not result_path.exists():
                model,state,path=ctx.source(domain,seed)
                with torch.no_grad():
                    z=torch.cat([model(task.batch(dev[i:i+32],'target')) for i in range(0,len(dev),32)])
                atomic_json(result_path,dict(task=f'0_to_{domain}',seed=seed,arm='B_frozen',selection='fixed_200',
                    metrics=full_metrics(z,labels),checkpoint=str(path),checkpoint_sha256=sha256(path),
                    commit=ctx.commit,event_flags='clean'))
    print(json.dumps(dict(event='phase_evaluated',phase=phase,rows=len(rows))),flush=True)


def final_report(ctx):
    decision=read(STAGE/'diagnostics/D2/decision.json')
    phases=['A','B'] if decision['allow_B'] else ['A']
    expected=sum(ctx.plan['phases']['training_'+p]['jobs'] for p in phases)
    jobs=list((STAGE/'jobs').glob('0_to_*/*/*/result.json'))
    completed=[p for p in jobs if read(p).get('status')=='completed']
    if len(completed)!=expected:
        raise ValueError(f'Expected {expected} completed training jobs, found {len(completed)}')
    evaluations=[read(p) for p in sorted((STAGE/'evaluations').glob('*.json'))]
    expected_evaluations=2*expected+10
    if len(evaluations)!=expected_evaluations:
        raise ValueError(f'Expected {expected_evaluations} evaluations, found {len(evaluations)}')
    hw=hardware()
    result_rows=[]
    for row in evaluations:
        job=STAGE/'jobs'/row['task']/str(row['seed'])/row['arm']
        training=read(job/'result.json') if row['arm']!='B_frozen' else {}
        result_rows.append(dict(run_id='tqg_revision_v1',task=row['task'],seed=row['seed'],
            config_name=row['arm'],selection=row['selection'],git_commit=row['commit'],
            gpu_hours=0,hardware=hw['cpu'],cpu_wall_seconds=training.get('seconds',0),
            primary_metric=row['metrics']['balanced_accuracy'],accuracy=row['metrics']['accuracy'],
            cross_entropy=row['metrics']['ce'],calibration_ece=row['metrics']['calibration_ece'],
            brier=row['metrics']['brier'],class_recall=json.dumps(row['metrics']['class_recall']),
            predicted_class_proportions=json.dumps(row['metrics']['predicted_class_proportions']),
            event_flags=row['event_flags'],checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256']))
    with (STAGE/'results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result_rows[0]))
        writer.writeheader();writer.writerows(result_rows)
    scalar_rows=[]
    for row in evaluations:
        metrics={k:v for k,v in row['metrics'].items() if isinstance(v,(int,float)) or v is None}
        scalar_rows.append(dict(task=row['task'],seed=row['seed'],arm=row['arm'],selection=row['selection'],metrics=metrics))
    groups=aggregate(scalar_rows,['task','arm','selection'])
    lookup={(g['task'],g['arm'],g['selection']):g for g in groups}
    classification=[]
    if decision['allow_B']:
        for task in ('0_to_2','0_to_3'):
            candidate=lookup[(task,'T6','fixed_200')]
            tests=[]
            for baseline,threshold in [('T5',.01),('T9',.01),('T10',.01),('T12',.005),('T13',.005),('T16',.005),
                                       ('T14',.005),('T15',.005),('B_frozen',0.),('B_full',0.)]:
                control=lookup[(task,baseline,'fixed_200')]
                deltas=[a['metrics']['balanced_accuracy']-b['metrics']['balanced_accuracy']
                        for a,b in zip(candidate['per_seed'],control['per_seed'])]
                mean=average(deltas)
                wins=sum(d>0 for d in deltas)
                passed=mean>=threshold and (wins>=4 if baseline in ('T5','T9','T10') else True)
                tests.append(dict(baseline=baseline,threshold=threshold,mean_delta=mean,wins=wins,passed=passed))
            classification.append(dict(task=task,tests=tests,
                                       H2_passed=all(t['passed'] for t in tests[:3]),
                                       practical_floor_passed=all(t['passed'] for t in tests[-2:])))
    atomic_json(STAGE/'results_summary.json',dict(groups=groups,classification=classification,
        D2_gate=decision,stage_B_executed=decision['allow_B'],independent_confirmation_complete=False))
    lines=['# TQG 改进版执行报告','',
           '本轮按照 v1 固定方案执行；目标开发集用于事后诊断和评价，目标 final 标签未评价。',
           '',f'完成 D0、D1、D2 和 {"A/B 两组" if decision["allow_B"] else "A 组"}训练，共 {expected} 个适配作业。',
           f'B 组推进门槛：{"通过" if decision["allow_B"] else "未通过，因此按方案暂不开展 B 组"}。','',
           '## 固定 200 步 balanced accuracy','', '| 方法 | 0→2，均值±种子SD | 0→3，均值±种子SD |','|---|---:|---:|']
    for arm in ['B_frozen']+ctx.plan['phases']['training_A']['arms']+(ctx.plan['phases']['training_B']['arms'] if decision['allow_B'] else []):
        values=[]
        for task in ('0_to_2','0_to_3'):
            metric=lookup[(task,arm,'fixed_200')]['metrics']['balanced_accuracy']
            values.append(f"{100*metric['mean']:.2f} ± {100*metric['seed_std']:.2f}")
        lines.append(f'| {arm} | {values[0]} | {values[1]} |')
    lines+=['','## 结论范围','',
            '各单项假设按固定判据评价，不能只选最优方法或最优训练步作为主结果。',
            'D1 只检验组合权重查询样本量；不代表整个数据集规模的因果作用。',
            'D2 的 B 门槛只是探索推进标准；不等于局部匹配已有效。',
            'NCI1、近期基线复现、额外骨干与论文级独立确认不在此次小规模执行范围。',
            '', '## 可复现入口','',
            '配置与版本：execution_lock.json、runs_manifest.json、各作业 config.json/RUN_INFO.json。',
            '原始数值：results.csv、diagnostics.csv；分类与机制汇总：results_summary.json 和 diagnostics/*/target_summary.json。',
            '真实计时：compute_ledger.json；所有脚本命令见 command_log.jsonl。']
    (STAGE/'run_report.md').write_text('\n'.join(lines)+'\n')
    (STAGE/'hand_off.md').write_text(
        '# 交接\n\nv1 本地实验包已完成。先读 run_report.md 和 diagnostic_report.md；'
        '完整分类结果在 results.csv，所有失败/告警保留。'
        '本轮是开发研究，不进入论文结果包装；独立确认和额外机制需要新方案。\n')
    # Audit actual accesses, rather than relying on a declarative flag.
    count=0
    for path in STAGE.rglob('target_label_access.jsonl'):
        for line in path.read_text().splitlines():
            entry=json.loads(line)
            if entry['role']!='target_development_only':
                raise PermissionError('Unexpected target-label role')
            if not any(set(entry['ids'])<=set(ctx.development_ids(d)) for d in (2,3)):
                raise PermissionError('A label access is outside development roles')
            count+=len(entry['ids'])
    audit=dict(expected_training_jobs=expected,completed_training_jobs=len(completed),
               expected_evaluations=expected_evaluations,completed_evaluations=len(evaluations),
               D1_source_blocks=read(STAGE/'diagnostics/D1/source_seal.json')['completed_blocks'],
               D2_source_blocks=read(STAGE/'diagnostics/D2/source_seal.json')['blocks'],
               audited_development_label_requests=count,final_labels_used=False,commit=ctx.commit,
               resource_ledger=read(STAGE/'compute_ledger.json'))
    atomic_json(STAGE/'completion_audit.json',audit)
    print(json.dumps(dict(event='local_revision_package_completed',jobs=expected,evaluations=len(evaluations))),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['diagnostics','evaluate_A','evaluate_B','final_report'],required=True)
    args=parser.parse_args()
    ctx=Context()
    budget=Budget('aggregate_'+args.phase)
    status='failed'
    try:
        if args.phase=='diagnostics':
            summarize_diagnostics(ctx)
        elif args.phase=='final_report':
            final_report(ctx)
        else:
            evaluate_training(ctx,args.phase[-1])
        status='completed'
    finally:
        budget.finish(status)


if __name__=='__main__':
    main()
