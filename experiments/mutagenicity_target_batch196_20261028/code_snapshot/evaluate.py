"""Sealed batch-size comparison; fixed development labels only."""
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data.separate import separate

from experiment import Context, RUN, STAGE, ROLES, atomic_json, check_completion, read, stamp
from tqg_gda.metrics import classification_metrics, evaluate_source
from tqg_gda.training import load_model
from tqg_gda.utils import sha256

ARMS=['B_frozen','B_fit','B16','B32','B64','B196']


def seal(ctx):
    jobs=[]
    for spec in ctx.specs():
        completed=check_completion(ctx,spec)
        if completed is None or completed['result']['completed_steps']!=200:
            raise ValueError('Cannot evaluate before all 15 new jobs finish: '+str(spec))
        jobs.append(completed)
    references=ctx.manifest['references']
    for ref in references:
        for artifact in ref['artifacts'].values():
            if sha256(artifact['path'])!=artifact['sha256']:
                raise ValueError('Reused reference artifact changed')
    payload=dict(sealed_at=stamp(),git_commit=ctx.commit,plan_sha256=ctx.plan_sha,
                 jobs=jobs,count=15,references=references,reused_adaptations=60,
                 target_development_labels_read=False,target_final_labels_allowed=False)
    path=STAGE/'training_seal.json'
    if path.exists():
        old=read(path)
        if old['jobs']!=jobs or old['references']!=references or old['git_commit']!=ctx.commit:
            raise ValueError('Sealed matrix changed')
    else:
        atomic_json(path,payload)
    return jobs,references


def development_labels(ctx,domain,task):
    sealed=read(STAGE/'training_seal.json')
    if sealed['count']!=15 or sealed['git_commit']!=ctx.commit:
        raise PermissionError('No complete training seal')
    ids=tuple(int(i) for i in np.load(ROLES/f'0_to_{domain}/development.npy',allow_pickle=False))
    expected=set(ctx.manifest['roles'][str(domain)]['development_ids'])
    forbidden=set(ctx.manifest['roles'][str(domain)]['final_ids'])
    if set(ids)!=expected or set(ids)&forbidden or not set(ids)<=set(task.target_ids):
        raise PermissionError('Only frozen development labels may be read')
    labels=[]
    for index in ids:
        graph=separate(task._raw.__class__,task._raw,index,task.slices,decrement=False)
        labels.append(task.class_map[int(graph.y.item())])
    with (STAGE/'evaluation_label_audit.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(timestamp=stamp(),domain=domain,role='development',
                     ids=ids,labels_read=len(labels),training_sealed=True,final_labels_read=False))+'\n')
    return ids,torch.tensor(labels,dtype=torch.long)


def records(ctx,jobs,references):
    out=[]
    for item in jobs:
        out.append(dict(**item['spec'],reused=False,git_commit=item['git_commit'],
                        artifacts=item['artifacts'],output=item['output'],
                        wall_seconds=item['wall_seconds'],event_flags=item['event_flags']))
    for item in references:
        out.append(dict(**item['spec'],reused=True,git_commit=item['git_commit'],
                        artifacts=item['artifacts'],output=None,
                        wall_seconds=item['wall_seconds'],event_flags=item['event_flags']))
    return out


def path_for(record,name):
    return Path(record['artifacts'][name]['path'])


def history(record):
    return [json.loads(line) for line in path_for(record,'history.jsonl').read_text().splitlines()]


def diagnostics(ctx,all_records):
    result=[];batch_checks=0
    indexed={(r['domain'],r['seed'],r['arm']):r for r in all_records}
    for domain in ctx.plan['domains']:
        for seed in ctx.plan['seeds']:
            histories={arm:history(indexed[(domain,seed,arm)]) for arm in ARMS[1:]}
            for arm,rows in histories.items():
                assert [r['step'] for r in rows]==list(range(1,201))
            for step in range(200):
                base=histories['B16'][step]['batches']
                for arm in ['B_fit','B32','B64','B196']:
                    other=histories[arm][step]['batches']
                    assert other['fit']==base['fit']
                    if arm!='B_fit':
                        assert other['query']==base['query'] and other['target'][:16]==base['target']
                        assert len(other['target'])==int(arm[1:]) and len(set(other['target']))==len(other['target'])
                assert histories['B64'][step]['batches']['target'][:32]==histories['B32'][step]['batches']['target']
                assert histories['B196'][step]['batches']['target'][:64]==histories['B64'][step]['batches']['target']
                batch_checks+=1
            for arm in ['B16','B32','B64','B196']:
                record=indexed[(domain,seed,arm)];rows=histories[arm]
                execution=[r['execution'] for r in rows]
                norms=np.array([e['feedback_gradient_norm'] for e in execution])
                ratios=[.1*e['feedback_gradient_norm']/e['source_gradient_norm']
                        for e in execution if e['source_gradient_norm']>0]
                queries=[r['query'] for r in rows]
                cal=[q['calibration'] for q in queries if 'calibration' in q]
                rank_counts={}
                for q in queries:rank_counts[str(q.get('rank'))]=rank_counts.get(str(q.get('rank')),0)+1
                result.append(dict(domain=domain,seed=seed,arm=arm,reused=record['reused'],
                    steps=len(rows),wall_seconds=record['wall_seconds'],
                    step_seconds_median=float(np.median([r['seconds'] for r in rows])),
                    feedback_norm_mean=float(norms.mean()),feedback_norm_sd=float(norms.std(ddof=1)),
                    feedback_norm_cv=float(norms.std(ddof=1)/norms.mean()) if norms.mean()>0 else None,
                    feedback_norm_p95=float(np.quantile(norms,.95)),
                    feedback_source_ratio_median=float(np.median(ratios)) if ratios else None,
                    rank_counts=rank_counts,calibration_converged=sum(c['converged'] for c in cal),
                    calibration_calls=len(cal),nonconverged=sum(not c['converged'] for c in cal),
                    numerical_fallbacks=sum(q.get('status')=='numerical_fallback_source_only' for q in queries),
                    zero_rank_steps=sum(q.get('status')=='zero_rank_source_only' for q in queries),
                    max_gap=max(c['gap'] for c in cal) if cal else None,
                    max_iterations=max(c['iterations'] for c in cal) if cal else None,
                    max_gradient_identity_error=max(e['feedback_gradient_identity_error'] for e in execution),
                    max_sgd_step_identity_error=max(e['sgd_step_identity_error'] for e in execution),
                    kernel_cross_mean=float(np.mean([q['source_target_kernel_mean'] for q in queries if 'source_target_kernel_mean' in q]))))
    output=dict(jobs=result,paired_batch_steps_verified=batch_checks,
                source_fit_sequences_identical=True,source_query_sequences_identical=True,
                target_permutation_prefixes_identical=True,
                stability_scope='magnitude variation over a nonstationary trajectory; gradient direction angles not recorded')
    atomic_json(STAGE/'training_diagnostics.json',output)
    return output


def summarize(rows):
    metrics={};paired={}
    for domain in (1,2,3):
        task=f'0_to_{domain}';metrics[task]={};paired[task]={}
        for arm in ARMS:
            metrics[task][arm]={}
            for selection in (['frozen'] if arm=='B_frozen' else ['fixed_200','source_selected']):
                group=[r for r in rows if r['task']==task and r['arm']==arm and r['selection']==selection]
                assert len(group)==5
                metrics[task][arm][selection]={key:dict(mean=statistics.mean(r[key] for r in group),
                    sd=statistics.stdev(r[key] for r in group)) for key in ('balanced_accuracy','accuracy','ce','auroc')}
        for reference in ('B16','B_fit'):
            paired[task][reference]={}
            for arm in ('B16','B32','B64','B196'):
                changes=[]
                for seed in range(3405,3410):
                    selected=[r for r in rows if r['task']==task and r['seed']==seed and r['selection']=='fixed_200']
                    a=next(r for r in selected if r['arm']==arm)
                    b=next(r for r in selected if r['arm']==reference)
                    changes.append(100*(a['balanced_accuracy']-b['balanced_accuracy']))
                paired[task][reference][arm]=dict(mean_difference_pp=statistics.mean(changes),
                    seed_differences_pp=changes,wins=sum(x>1e-10 for x in changes),
                    ties=sum(abs(x)<=1e-10 for x in changes),losses=sum(x< -1e-10 for x in changes))
    return metrics,paired


def write_report(ctx,summary,paired,diag):
    text='# Mutagenicity：目标batch 16／32／64／196对照\n\n'
    text+='只改变目标batch；源fit每步16、query全部196、相同五个fit预训练模型、200步、SGD lr0.0003、反馈权重0.1、lambda_W0.01、FP64。0→1/2/3各五种子。batch16/32/64及源SGD复用已审计实验；batch196新增15次训练。\n\n'
    for selection,title in [('fixed_200','固定200步：主结果'),('source_selected','源验证CE选点：次要结果')]:
        text+='## '+title+'\n\nBA%，五种子均值±样本标准差。\n\n| 方法 | 0→1 | 0→2 | 0→3 |\n|---|---:|---:|---:|\n'
        for arm in ARMS:
            cells=[]
            for d in (1,2,3):
                m=summary[f'0_to_{d}'][arm]['frozen' if arm=='B_frozen' else selection]['balanced_accuracy']
                cells.append(f"{100*m['mean']:.2f} ± {100*m['sd']:.2f}")
            text+='| '+arm+' | '+' | '.join(cells)+' |\n'
        text+='\n'
    text+='## 相对batch16的配对差\n\n单位为百分点；括号为五个种子胜／平／负。\n\n| 方法 | 0→1 | 0→2 | 0→3 |\n|---|---|---|---|\n'
    successes={}
    for arm in ('B196',):
        cells=[];successes[arm]=[]
        for d in (1,2,3):
            p=paired[f'0_to_{d}']['B16'][arm]
            cells.append(f"{p['mean_difference_pp']:+.3f} ({p['wins']}/{p['ties']}/{p['losses']})")
            if p['mean_difference_pp']>0 and p['wins']>=3:successes[arm].append(d)
        text+='| '+arm+' | '+' | '.join(cells)+' |\n'
    text+='\n预设条件：某档在至少2/3任务平均差为正且至少3/5种子获胜。\n'
    for arm,domains in successes.items():text+=f'- {arm}：{len(domains)}/3任务满足，'+('达到' if len(domains)>=2 else '未达到')+'预设条件。\n'
    text+='\n## 反馈幅度、数值状态和开销\n\n| batch | 每步秒数中位数的均值 | 反馈范数CV均值 | 加权反馈/源梯度比中位数的均值 | 未收敛次数 | 数值回退 |\n|---|---:|---:|---:|---:|---:|\n'
    for arm in ('B16','B32','B64','B196'):
        group=[r for r in diag['jobs'] if r['arm']==arm]
        text+=f"| {arm} | {statistics.mean(r['step_seconds_median'] for r in group):.3f} | {statistics.mean(r['feedback_norm_cv'] for r in group):.3f} | {statistics.mean(r['feedback_source_ratio_median'] for r in group):.3f} | {sum(r['nonconverged'] for r in group)} | {sum(r['numerical_fallbacks'] for r in group)} |\n"
    text+='\n范数CV仅描述随训练变化的幅度波动，不是梯度方向稳定性、正确性或统计显著性。耗时受历史/本轮机器负载影响；本实验固定更新步数，不等计算预算。方向数量、目标使用次数、核尺度及校准难度可能随batch共同变化。\n'
    text+='\n## Executed as planned\n\n- 15次新适配，60次复用适配，5个共享源模型；全部封存后才评价开发标签。\n- 相同种子各档的fit/query样本序列逐步相同，目标样本按相同随机排列取不同长度前缀；每批不重复。\n- 源模型和数据划分不变；不读取final标签，不按开发结果改变超参数。\n'
    text+='\n## Deviations from plan\n\n- 沿用已验证PyG自定义模板；HF/图像训练模板不能表达完整参数查询与自定义梯度接口。FP64为既定算法，不启用混合精度。GPU不可用，使用五个单线程CPU工作进程。\n'
    events=[json.loads(line) for line in (STAGE/'monitor_events.jsonl').read_text().splitlines()]
    failures=[e for e in events if e['type']=='job_failure']
    text+='\n## Failed runs\n\n'+('无训练异常退出。\n' if not failures else '\n'.join(str(e) for e in failures)+'\n')
    ledger=read(STAGE/'compute_ledger.json')
    text+=f"\n## Compute consumed\n\nGPU 0小时；本轮CPU墙钟约{(time.time()-ledger['started_unix'])/60:.1f}分钟；历史复用开销另列各记录，不计入本轮。\n"
    text+='\n## Headline number\n\n'
    text+='；'.join(f"{arm}相对batch16在0→{d}为{paired[f'0_to_{d}']['B16'][arm]['mean_difference_pp']:+.3f}个百分点" for arm in ('B196',) for d in (1,2,3))+'。\n'
    text+=f"\n## Reproducibility manifest\n\n- 执行代码commit：`{ctx.commit}`；计划SHA256：`{ctx.plan_sha}`。\n- 种子3405–3409，CPU单线程，保留模型、优化器、采样RNG、配置、JSONL、TensorBoard和哈希。\n- 复用作业保留原训练commit，results.csv另列本轮evaluation_commit。\n"
    text+='\n## 解释边界\n\n开发集已在历史实验中观察，本轮为探索性敏感性分析。五个种子和一次固定数据划分不支持普遍有效性或显著性结论。保持源fit16是为了隔离变量，不代表它最优。没有执行按epoch洗牌对照，也没有匹配等目标样本数、等方向数或等墙钟。\n'
    (STAGE/'run_report.md').write_text(text)
    (STAGE/'hand_off.md').write_text('只比较目标batch16/32/64/196，主结果固定200步。保留所有方向和种子，不选取每域最优batch。复用记录保留原提交。final标签未读。以配对差、范数波动、求解状态和开销共同解释。\n')
    return successes


def main():
    torch.set_num_threads(1)
    ctx=Context();jobs,references=seal(ctx)
    all_records=records(ctx,jobs,references)
    diag=diagnostics(ctx,all_records)
    indexed={(r['domain'],r['seed'],r['arm']):r for r in all_records}
    rows=[]
    for domain in ctx.plan['domains']:
        task=ctx.task(domain);ids,labels=development_labels(ctx,domain,task)
        for seed in ctx.plan['seeds']:
            for arm in ARMS:
                record=indexed.get((domain,seed,arm))
                for selection in (['frozen'] if arm=='B_frozen' else ['fixed_200','source_selected']):
                    if record is None:
                        path=ctx.source_path(seed);reused=True
                        training_commit=next(s['original_training_commit'] for s in ctx.manifest['source_models'] if s['seed']==seed)
                    else:
                        path=path_for(record,'last.pt' if selection=='fixed_200' else 'selected.pt')
                        reused=record['reused'];training_commit=record['git_commit']
                    model,state=load_model(path)
                    with torch.no_grad():
                        logits=torch.cat([model(task.batch(ids[start:start+64],'target')) for start in range(0,len(ids),64)])
                        metric=classification_metrics(logits,labels);source=evaluate_source(model,task)
                    selected=state.get('selection',{}).get('step',0)
                    if selection=='source_selected' and abs(source['ce']-state['selection']['ce'])>1e-10:
                        raise ValueError('Source-selected checkpoint metric mismatch')
                    row=dict(task=f'0_to_{domain}',seed=seed,arm=arm,selection=selection,
                             target_batch_size=int(arm[1:]) if arm in ('B16','B32','B64','B196') else 0,
                             git_commit=training_commit,evaluation_commit=ctx.commit,config_name=arm,
                             gpu_hours=0,hardware=ctx.manifest['hardware']['cpu'],reused=reused,
                             event_flags=','.join(record['event_flags']) if record else 'clean',
                             checkpoint=str(path),checkpoint_sha256=sha256(path),
                             model_step=state.get('step',selected),selected_step=selected,
                             source_validation_ce=source['ce'],target_final_labels_evaluated=False,**metric)
                    rows.append(row)
                    print(json.dumps(dict(event='evaluated',task=row['task'],seed=seed,arm=arm,
                                         selection=selection,balanced_accuracy=row['balanced_accuracy'])),flush=True)
    assert len(rows)==165
    with (STAGE/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    summary,paired=summarize(rows)
    atomic_json(STAGE/'results_summary.json',dict(metrics=summary,paired=paired,git_commit=ctx.commit,
        plan_sha256=ctx.plan_sha,event_flags=sorted(set(r['event_flags'] for r in rows)),target_final_labels_evaluated=False))
    successes=write_report(ctx,summary,paired,diag)
    from plot import plot
    plot(summary,STAGE/'figures')
    ledger=read(STAGE/'compute_ledger.json')
    marker=dict(stage=3,started_at=ledger['started_at'],finished_at=stamp(),gpu_hours_consumed_so_far=0,
                git_commit=ctx.commit,new_adaptations=15,reused_adaptations=60,source_models_reused=5,
                evaluations=165,hypothesis_tasks=successes,target_final_labels_evaluated=False)
    atomic_json(RUN/'stage_3_done',marker)
    atomic_json(STAGE/'completion_audit.json',dict(status='completed',**marker,
                all_job_hashes_verified=True,all_histories_complete=True,target_labels_read_only_after_seal=True))
    print(json.dumps(dict(event='evaluation_complete',**marker)),flush=True)


if __name__=='__main__':main()
