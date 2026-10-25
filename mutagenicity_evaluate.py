"""Offline development-only evaluation AFTER sealing the complete training matrix."""
import csv
import json
from pathlib import Path
import statistics
import time

import torch

from mutagenicity_experiment import Context, CODE, STAGE, RUN, LABELS, read
from revision_common import atomic_json, hardware
from revision_evaluation import DevelopmentLabels, full_metrics
from tqg_gda.training import load_model
from tqg_gda.utils import sha256


def main():
    ctx=Context();records=[]
    for seed in ctx.plan['seeds']:
        for pool in ('fit','all_source'):
            folder=ctx.warmup_path(seed,pool=='all_source');receipt=read(folder/'completion.json')
            for name,digest in receipt['hashes'].items():
                if sha256(folder/name)!=digest:raise ValueError('Warmup changed')
        for domain in ctx.plan['domains']:
            for arm in ctx.plan['arms']:
                folder=STAGE/'jobs'/f'0_to_{domain}'/str(seed)/arm
                result=read(folder/'result.json')
                if result['status']!='completed' or result['completed_updates']!=200:
                    raise ValueError('Training matrix not complete')
                for name,digest in result['artifact_hashes'].items():
                    if sha256(folder/name)!=digest:raise ValueError('Job artifact changed')
                records.append(dict(seed=seed,domain=domain,arm=arm,output=str(folder),result=result))
    atomic_json(STAGE/'training_seal.json',dict(git_commit=ctx.commit,
        jobs=records,count=len(records),source_warmups=10,target_labels_read=False,
        target_final_labels_allowed=False,sealed_at=time.time()))
    rows=[]; machine=hardware()
    for domain in ctx.plan['domains']:
        task=ctx.task(domain); ids=ctx.development_ids(domain)
        reader=DevelopmentLabels(ctx,domain,task,STAGE/'evaluation_label_audit.jsonl')
        labels=reader(ids)
        for seed in ctx.plan['seeds']:
            specifications=[('B_frozen','frozen',ctx.warmup_path(seed)/'best.pt')]
            for arm in ctx.plan['arms']:
                folder=STAGE/'jobs'/f'0_to_{domain}'/str(seed)/arm
                specifications.extend([(arm,'fixed_200',folder/'last.pt'),
                                       (arm,'source_selected',folder/'selected.pt')])
            for arm,selection,path in specifications:
                model,state=load_model(path)
                with torch.no_grad():
                    outputs=[model(task.batch(ids[i:i+64],'target')) for i in range(0,len(ids),64)]
                    metrics=full_metrics(torch.cat(outputs),labels)
                flags='clean';warnings=0
                monitor=path.parent/'monitor_events.jsonl'
                if monitor.exists():warnings=len(monitor.read_text().splitlines())
                row=dict(task=f'0_to_{domain}',seed=seed,arm=arm,method=LABELS[arm],selection=selection,
                    git_commit=ctx.commit,config_name='sega_full_query',gpu_hours=0,
                    hardware=machine['cpu'],event_flags=flags,monitor_warnings=warnings,
                    checkpoint=str(path),checkpoint_sha256=sha256(path),**metrics)
                rows.append(row)
                print(json.dumps(dict(event='evaluation',task=row['task'],seed=seed,arm=arm,
                    selection=selection,balanced_accuracy=metrics['balanced_accuracy'])),flush=True)
    columns=list(rows[0])
    with (STAGE/'results.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    summary={}; paired={}
    for domain in ctx.plan['domains']:
        task=f'0_to_{domain}';summary[task]={};paired[task]={}
        for arm in LABELS:
            summary[task][arm]={}
            for selection in (['frozen'] if arm=='B_frozen' else ['fixed_200','source_selected']):
                group=[r for r in rows if r['task']==task and r['arm']==arm and r['selection']==selection]
                assert len(group)==5
                summary[task][arm][selection]={name:dict(mean=statistics.mean(r[name] for r in group),
                    sd=statistics.stdev(r[name] for r in group)) for name in
                    ('balanced_accuracy','accuracy','ce','auroc')}
        baseline={r['seed']:r['balanced_accuracy'] for r in rows if r['task']==task and r['arm']=='B_fit' and r['selection']=='fixed_200'}
        for arm in ctx.plan['arms']:
            values=[r['balanced_accuracy']-baseline[r['seed']] for r in rows if r['task']==task and r['arm']==arm and r['selection']=='fixed_200']
            paired[task][arm]=dict(mean_difference_pp=100*statistics.mean(values),
                wins=sum(v>1e-12 for v in values),ties=sum(abs(v)<=1e-12 for v in values),losses=sum(v< -1e-12 for v in values))
    atomic_json(STAGE/'results_summary.json',dict(metrics=summary,paired_vs_fit=paired,
        git_commit=ctx.commit,source_counts=ctx.plan['source_counts'],target_final_labels_evaluated=False))
    report='# Mutagenicity：SEGA骨干与预训练配置下的TQG实验\n\n'
    report+='采用原SEGA边数域索引，GINE（14维节点标签、3维边标签）、hidden32、2层、mean readout、float64。源预训练100 epoch、batch32、lr0.003；TQG200 step、Adam lr0.0003、K4、rho0.05、ridge0.01、反馈权重0.1。\n\n'
    report+='源域1084张：975训练＝779 fit＋196 query，109验证。全源查询为975张，包含预训练使用的779张fit，因此不只是增加样本数。B_full有自己的全源预训练起点。\n\n'
    report+='**SEGA配置边界：**包中没有Mutagenicity专用训练配置；本轮沿用通用源配置和域索引，使用TQG机制替代SEGA几何/OT模块。适配200步为用户确认的TQG预算，未采用SEGA两轮开发示例。100 epoch在fit/all-source分别为2500/3100次更新，不沿用PROTEINS的700步预算。\n\n'
    for selection in ('fixed_200','source_selected'):
        report+='## '+('固定200步（主结果）' if selection=='fixed_200' else '源验证CE选点（次要结果）')+'\n\n'
        report+='指标为balanced accuracy（%），均值±五个模型种子的样本标准差。\n\n| 方法 | 0→1 | 0→2 | 0→3 |\n|---|---:|---:|---:|\n'
        for arm,label in LABELS.items():
            cells=[]
            for domain in ctx.plan['domains']:
                m=summary[f'0_to_{domain}'][arm]['frozen' if arm=='B_frozen' else selection]['balanced_accuracy']
                cells.append(f"{100*m['mean']:.2f} ± {100*m['sd']:.2f}")
            report+='| '+label+' | '+' | '.join(cells)+' |\n'
        report+='\n'
    report+='## 相对继续fit源监督训练的配对变化\n\n| 方法 | 0→1 | 0→2 | 0→3 |\n|---|---|---|---|\n'
    for arm in [a for a in LABELS if a.startswith('Q_')]:
        cells=[]
        for domain in ctx.plan['domains']:
            p=paired[f'0_to_{domain}'][arm]
            cells.append(f"{p['mean_difference_pp']:+.2f}个百分点，{p['wins']}胜{p['ties']}平{p['losses']}负")
        report+='| '+LABELS[arm]+' | '+' | '.join(cells)+' |\n'
    report+='\n## 证据范围\n\n本轮105次适配、10次源预训练、15个冻结参考；225次开发集评价。每个目标域开发集542张；final分别542/542/543张，仅输入用于无标签transductive适配，final标签未评价。源验证用于选模，目标开发指标在全矩阵训练封存后才读取，未据此调参。\n\n'
    report+='相同100 epoch不等于相同更新次数；源图规模、预训练和GINE与此前PROTEINS实验不同，不能将跨数据集差异单独归因于样本量。未复现SEGA方法本身、外部强基线或匹配墙钟预算。五种子探索性结果不代表统计显著或方法普遍有效。\n'
    (STAGE/'run_report.md').write_text(report)
    (STAGE/'hand_off.md').write_text('完整三方向五种子实验已完成；以run_report.md的固定200步表和配对差为主，源验证选点结果单独报告。目标final标签未评价，未根据目标开发结果调整配置。\n')
    from mutagenicity_plot import plot
    plot(summary,STAGE/'figures')
    marker=dict(stage=3,started_at=read(STAGE/'started.json')['unix'],finished_at=time.time(),
        gpu_hours_consumed_so_far=0,adaptations=105,source_warmups=10,evaluations=225,git_commit=ctx.commit)
    atomic_json(RUN/'stage_3_done',marker)
    print(json.dumps(dict(event='stage_complete',**marker)),flush=True)


if __name__=='__main__':main()
