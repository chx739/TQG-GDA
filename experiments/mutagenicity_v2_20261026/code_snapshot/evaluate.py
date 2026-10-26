"""Seal the complete matrix, then evaluate only the fixed development labels."""
import csv
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch_geometric.data.separate import separate

from experiment import Context, RUN, STAGE, ROLES, atomic_json, check_completion, read, stamp
from tqg_gda.metrics import classification_metrics, evaluate_source
from tqg_gda.training import load_model
from tqg_gda.utils import sha256


def seal(ctx):
    jobs = []
    for spec in ctx.specs():
        completed = check_completion(ctx,spec)
        if completed is None or completed['result']['completed_steps'] != 200:
            raise ValueError('Cannot evaluate before all 105 jobs complete: '+str(spec))
        jobs.append(completed)
    payload = dict(sealed_at=stamp(),git_commit=ctx.commit,plan_sha256=ctx.plan_sha,
        jobs=jobs,count=len(jobs),target_development_labels_read=False,target_final_labels_allowed=False)
    path = STAGE/'training_seal.json'
    if path.exists():
        saved = read(path)
        if saved['jobs'] != payload['jobs'] or saved['git_commit'] != ctx.commit:
            raise ValueError('Sealed matrix changed')
    else:
        atomic_json(path,payload)
    return jobs


def development_labels(ctx,domain,task):
    sealed = read(STAGE/'training_seal.json')
    if sealed['count'] != 105 or sealed['git_commit'] != ctx.commit:
        raise PermissionError('No valid complete training seal')
    ids = tuple(int(i) for i in np.load(ROLES/f'0_to_{domain}/development.npy',allow_pickle=False))
    expected = set(ctx.manifest['roles'][str(domain)]['development_ids'])
    forbidden = set(ctx.manifest['roles'][str(domain)]['final_ids'])
    if set(ids) != expected or set(ids)&forbidden or not set(ids)<=set(task.target_ids):
        raise PermissionError('Only frozen development label IDs may be read')
    labels = []
    for index in ids:
        graph = separate(task._raw.__class__,task._raw,index,task.slices,decrement=False)
        if graph.y.numel()!=1 or int(graph.y.item()) not in task.class_map:
            raise ValueError('Invalid development label')
        labels.append(task.class_map[int(graph.y.item())])
    with (STAGE/'evaluation_label_audit.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(timestamp=stamp(),domain=domain,role='development',ids=ids,
            labels_read=len(labels),training_sealed=True,final_labels_read=False))+'\n')
    return ids,torch.tensor(labels,dtype=torch.long)


def flags_for(spec,events):
    result = set()
    for event in events:
        if event.get('spec') != spec:
            continue
        if event['type']=='job_failure':
            result.add('restart')
            error = event.get('error','').lower()
            if 'nonfinite' in error or 'nan' in error:
                result.add('nan')
            if 'out of memory' in error:
                result.add('oom')
            if 'timeout' in error or 'hang' in error:
                result.add('timeout')
        if event['type']=='hang_timeout':
            result.update(['timeout','restart'])
    return ','.join(sorted(result)) if result else 'clean'


def summarize(rows,ctx):
    grouped,paired = {},{}
    for domain in ctx.plan['domains']:
        task = f'0_to_{domain}'
        grouped[task],paired[task] = {},{}
        for arm in ['B_frozen',*ctx.plan['arms']]:
            grouped[task][arm] = {}
            for selection in (['frozen'] if arm=='B_frozen' else ['fixed_200','source_selected']):
                group = [r for r in rows if r['task']==task and r['arm']==arm and r['selection']==selection]
                assert len(group)==5
                grouped[task][arm][selection] = {
                    metric:dict(mean=statistics.mean(r[metric] for r in group),
                                sd=statistics.stdev(r[metric] for r in group))
                    for metric in ['balanced_accuracy','accuracy','ce','auroc']}
                grouped[task][arm][selection]['event_flags'] = sorted(set(r['event_flags'] for r in group))
        for baseline in ['B_fit','KRR','V2_repr','V2_permute','B_full']:
            paired[task][baseline] = {}
            for arm in ctx.plan['arms']:
                differences=[]
                for seed in ctx.plan['seeds']:
                    query=dict(task=task,seed=seed,selection='fixed_200')
                    candidate=next(r for r in rows if all(r[k]==v for k,v in query.items()) and r['arm']==arm)
                    reference=next(r for r in rows if all(r[k]==v for k,v in query.items()) and r['arm']==baseline)
                    differences.append(candidate['balanced_accuracy']-reference['balanced_accuracy'])
                paired[task][baseline][arm]=dict(mean_difference_pp=100*statistics.mean(differences),
                    seed_differences_pp=[100*x for x in differences],wins=sum(x>1e-12 for x in differences),
                    ties=sum(abs(x)<=1e-12 for x in differences),losses=sum(x< -1e-12 for x in differences))
    return grouped,paired


def diagnostics(ctx,completions):
    records=[]
    for completed in completions:
        spec=completed['spec']; folder=Path(completed['output'])
        history=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
        assert [r['step'] for r in history]==list(range(1,201))
        queries=[r['query'] for r in history if r['query'].get('version')=='v2']
        cal=[q['calibration'] for q in queries if 'calibration' in q]
        execution=[r['execution'] for r in history if 'execution' in r]
        row=dict(**spec,history_rows=len(history),v2_steps=len(queries),
            numerical_fallbacks=sum(q.get('status')=='numerical_fallback_source_only' for q in queries),
            zero_rank_steps=sum(q.get('status')=='zero_rank_source_only' for q in queries),
            calibration_converged_fraction=sum(c['converged'] for c in cal)/len(cal) if cal else None,
            calibration_max_gap=max(c['gap'] for c in cal) if cal else None,
            calibration_median_iterations=statistics.median(c['iterations'] for c in cal) if cal else None,
            calibration_max_iterations=max(c['iterations'] for c in cal) if cal else None,
            ranks=sorted(set(q['rank'] for q in queries if 'rank' in q)),
            feedback_gradient_norm_mean=statistics.mean(r['feedback_gradient_norm'] for r in execution) if execution else None,
            gradient_identity_error_max=max(r['feedback_gradient_identity_error'] for r in execution) if execution else None,
            sgd_step_identity_error_max=max(r['sgd_step_identity_error'] for r in execution) if execution else None,
            kernel_cross_mean=statistics.mean(q['source_target_kernel_mean'] for q in queries if 'source_target_kernel_mean' in q) if cal else None)
        records.append(row)
    differences=[]
    for domain in ctx.plan['domains']:
        for seed in ctx.plan['seeds']:
            base=check_completion(ctx,dict(domain=domain,seed=seed,arm='B_fit'))
            reference=torch.load(Path(base['output'])/'last.pt',map_location='cpu',weights_only=False)['model']
            for arm in ['V2_joint','KRR','V2_repr','V2_permute']:
                candidate=check_completion(ctx,dict(domain=domain,seed=seed,arm=arm))
                values=torch.load(Path(candidate['output'])/'last.pt',map_location='cpu',weights_only=False)['model']
                norm=sum(float((values[name]-reference[name]).square().sum()) for name in reference)**.5
                differences.append(dict(domain=domain,seed=seed,arm=arm,parameter_difference_from_source_sgd=norm))
    payload=dict(jobs=records,parameter_differences=differences,
        all_histories_complete=True,total_logged_steps=105*200)
    atomic_json(STAGE/'training_diagnostics.json',payload)
    return payload


def write_report(ctx,summary,paired,diag,rows):
    labels={'B_frozen':'fit-only 源模型不动',**{a:v['label'] for a,v in ctx.plan['arms'].items()}}
    text='# Mutagenicity：TQG v2 完整实验验证\n\n'
    text+='源模型沿用 SEGA 通用配置：2层 GINE、hidden32、mean readout、14维节点标签、3维边标签；源预训练100 epoch，fit/all-source 分别2500/3100次更新。复用原有10个源模型，没有重复预训练。\n\n'
    text+='源域1084张：975训练＝779 fit＋196独立 query，另109源验证。主版本每步使用全部196张 query、16张目标图，完整有效目标梯度空间；适配200 step，无动量SGD、lr0.0003、反馈权重0.1、FP64。核尺度由源query确定，lambda_W=0.01；校准最多200次内部迭代，绝对/相对gap容差1e-8/1e-6。\n\n'
    text+='四个TQG相关设置共享fit预训练起点和源fit主损失；全源监督基线另用975张图预训练和继续训练。伪标签阈值0.9。目标输入采用transductive设置；开发集各542张，final标签始终未评价。\n\n'
    for selection,title in [('fixed_200','固定200步：主结果'),('source_selected','源验证CE选点：次要结果')]:
        text+='## '+title+'\n\n指标为balanced accuracy（%），均值±五个模型种子的样本标准差。\n\n| 方法 | 0→1 | 0→2 | 0→3 |\n|---|---:|---:|---:|\n'
        for arm,label in labels.items():
            cells=[]
            for domain in ctx.plan['domains']:
                metric=summary[f'0_to_{domain}'][arm]['frozen' if arm=='B_frozen' else selection]['balanced_accuracy']
                cells.append(f"{100*metric['mean']:.2f} ± {100*metric['sd']:.2f}")
            text+='| '+label+' | '+' | '.join(cells)+' |\n'
        text+='\n'
    text+='## v2主版本的配对比较\n\n正数表示v2更好；胜/平/负按同种子比较。\n\n| 对照 | 0→1 | 0→2 | 0→3 |\n|---|---|---|---|\n'
    for baseline in ['B_fit','KRR','V2_repr','V2_permute','B_full']:
        cells=[]
        for domain in ctx.plan['domains']:
            p=paired[f'0_to_{domain}'][baseline]['V2_joint']
            cells.append(f"{p['mean_difference_pp']:+.2f} pp；{p['wins']}/{p['ties']}/{p['losses']}")
        text+='| '+labels[baseline]+' | '+' | '.join(cells)+' |\n'
    successes=[d for d in ctx.plan['domains'] if paired[f'0_to_{d}']['B_fit']['V2_joint']['mean_difference_pp']>0
               and paired[f'0_to_{d}']['B_fit']['V2_joint']['wins']>=3]
    krr_wins=[d for d in ctx.plan['domains'] if paired[f'0_to_{d}']['KRR']['V2_joint']['mean_difference_pp']>0]
    text+='\n## 预设问题的回答\n\n'
    text+=f'- 相对继续源fit训练，满足“平均差为正且至少3/5种子获胜”的任务数：{len(successes)}/3；预设门槛为至少2个任务，'+('达到。' if len(successes)>=2 else '未达到。')+'\n'
    text+=f'- 相对普通KRR，主版本平均分更高的任务数：{len(krr_wins)}/3；'+('达到预设的至少2个任务门槛，但仍需结合增幅和种子波动。' if len(krr_wins)>=2 else '未达到至少2个任务的增量门槛。')+'\n'
    v2_records=[r for r in diag['jobs'] if r['v2_steps']]
    text+=f"- 四个查询设置共{sum(r['v2_steps'] for r in v2_records)}步，数值回退{sum(r['numerical_fallbacks'] for r in v2_records)}步；零秩回退{sum(r['zero_rank_steps'] for r in v2_records)}步。完整收敛率、秩和执行等式误差见training_diagnostics.json。\n"
    text+='\n## Executed as planned\n\n- 三个迁移方向、五个种子、七个训练设置，共105次适配和15个冻结参考；主/次评价合计225行。\n- 所有适配完成并封存checkpoint哈希后才读取固定开发集标签。主结果固定200步，次结果只按源验证CE选点。\n- 真实源标签、表示核、平均误差KRR等对照独立报告，不根据开发集效果改配置。\n'
    text+='\n## Deviations from plan\n\n- 使用经过验证的PyG框架作为from-scratch模板，不使用不匹配的HF/图像训练器。保持FP64，未使用会改变既定数值方法的混合精度。\n- GPU驱动不可用，按冻结计划使用5个单线程CPU工作进程。\n'
    events=[json.loads(line) for line in (STAGE/'monitor_events.jsonl').read_text().splitlines()]
    failures=[e for e in events if e['type']=='job_failure']
    text+='\n## Failed runs\n\n'+('无训练异常退出。\n' if not failures else '\n'.join('- '+str(e) for e in failures)+'\n')
    ledger=read(STAGE/'compute_ledger.json')
    seconds=time.time()-ledger['started_unix']
    text+='\n## Compute consumed\n\n'+f'- GPU：0小时。CPU运行墙钟约{seconds/60:.1f}分钟；逐作业耗时见完成记录。复用源预训练的历史耗时不计入本轮。\n- 上限：8小时墙钟；没有改变科学超参数以追求目标开发结果。\n'
    text+='\n## Headline number\n\n'
    text+='；'.join(f"0→{d} 相对源fit SGD {paired[f'0_to_{d}']['B_fit']['V2_joint']['mean_difference_pp']:+.2f}个百分点" for d in ctx.plan['domains'])+'。\n'
    text+='\n## Reproducibility manifest\n\n'+f"- 执行代码commit：`{ctx.commit}`。\n- 计划SHA256：`{ctx.plan_sha}`。\n- 种子：3405–3409；CPU、单进程单线程；模型、优化器、采样与标签打乱RNG完整保存。\n- 配置：stage2_method/experiment_plan.yaml、每个作业config.json和RUN_INFO；源模型及所有数据索引有校验值。\n"
    text+='\n## 解释边界\n\n开发集此前已被旧版实验观察，本轮是预设配置下的探索性验证，不是独立盲测。五个种子不直接支持统计显著性或普遍有效性。普通CE梯度与balanced accuracy优化目标不同，CE/accuracy/AUROC和类别召回另存results.csv。全源基线包含额外标签和不同预训练更新数；没有完成外部强基线、等墙钟预算或更多学习率搜索。\n'
    (STAGE/'run_report.md').write_text(text)
    (STAGE/'hand_off.md').write_text('Mutagenicity v2 的三方向五种子矩阵完成。以固定200步及配对源SGD/KRR差为主；源验证选点单独展示。开发集在完整封存后评价，final标签未读；不得将探索性比较写成显著性或跨数据集结论。\n')
    return successes,krr_wins


def main():
    torch.set_num_threads(1)
    ctx=Context(); completions=seal(ctx)
    events=[json.loads(line) for line in (STAGE/'monitor_events.jsonl').read_text().splitlines()]
    rows=[]
    for domain in ctx.plan['domains']:
        ids,labels=development_labels(ctx,domain,ctx.task(domain))
        for seed in ctx.plan['seeds']:
            specs=[('B_frozen','frozen',ctx.source_path(seed),'fit',None)]
            for arm,definition in ctx.plan['arms'].items():
                spec=dict(domain=domain,seed=seed,arm=arm); completed=check_completion(ctx,spec)
                for selection,name in [('fixed_200','last.pt'),('source_selected','selected.pt')]:
                    specs.append((arm,selection,Path(completed['output'])/name,definition['pool'],completed))
            for arm,selection,path,pool,completed in specs:
                model,state=load_model(path); task=ctx.task(domain,pool)
                with torch.no_grad():
                    logits=torch.cat([model(task.batch(ids[start:start+64],'target')) for start in range(0,len(ids),64)])
                    metrics=classification_metrics(logits,labels)
                    source=evaluate_source(model,task)
                selection_step=state.get('selection',{}).get('step',0)
                if selection=='source_selected' and abs(source['ce']-state['selection']['ce'])>1e-10:
                    raise ValueError('Selected checkpoint/source metric mismatch')
                spec=dict(domain=domain,seed=seed,arm=arm)
                row=dict(task=f'0_to_{domain}',seed=seed,arm=arm,selection=selection,
                    git_commit=ctx.commit,config_name=arm,gpu_hours=0,hardware=ctx.manifest['hardware']['cpu'],
                    event_flags='clean' if arm=='B_frozen' else flags_for(spec,events),
                    checkpoint=str(path),checkpoint_sha256=sha256(path),model_step=state.get('step',selection_step),
                    selected_step=selection_step,source_validation_ce=source['ce'],
                    source_model_reused=True,target_final_labels_evaluated=False,**metrics)
                rows.append(row)
                print(json.dumps(dict(event='evaluated',task=row['task'],seed=seed,arm=arm,
                    selection=selection,balanced_accuracy=row['balanced_accuracy'])),flush=True)
    assert len(rows)==225
    with (STAGE/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    summary,paired=summarize(rows,ctx)
    atomic_json(STAGE/'results_summary.json',dict(metrics=summary,paired=paired,git_commit=ctx.commit,
        plan_sha256=ctx.plan_sha,event_flags=sorted(set(r['event_flags'] for r in rows)),
        target_final_labels_evaluated=False))
    diag=diagnostics(ctx,completions)
    successes,krr_wins=write_report(ctx,summary,paired,diag,rows)
    from plot import plot
    plot(summary,STAGE/'figures')
    ledger=read(STAGE/'compute_ledger.json')
    marker=dict(stage=3,started_at=ledger['started_at'],finished_at=stamp(),gpu_hours_consumed_so_far=0,
        git_commit=ctx.commit,adaptations=105,source_models_reused=10,evaluations=225,
        main_hypothesis_tasks=successes,joint_over_krr_tasks=krr_wins,target_final_labels_evaluated=False)
    atomic_json(RUN/'stage_3_done',marker)
    atomic_json(STAGE/'completion_audit.json',dict(status='completed',**marker,all_job_hashes_verified=True,
        all_histories_complete=True,target_labels_read_only_after_seal=True))
    print(json.dumps(dict(event='evaluation_complete',**marker)),flush=True)


if __name__=='__main__':
    main()
