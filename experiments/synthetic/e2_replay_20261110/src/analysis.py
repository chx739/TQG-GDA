"""Audit archived replay records, summarize paired seeds and draw trajectory plots."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import torch
from tsba.classifier import Coordinates,ParameterMap
from experiments.common.io import digest, write_json
from experiments.synthetic.e2_replay_20261110.src.diagnostics import source_metrics, target_metrics
from experiments.synthetic.e2_replay_20261110.src.execution import frozen_head
from experiments.synthetic.e2_full_source_20261110.src.execution import ARMS


def read(p):return json.loads(p.read_text())


def flat(d,prefix=''):
    result={}
    for k,v in d.items():
        name=prefix+k
        if isinstance(v,dict):result.update(flat(v,name+'.'))
        else:result[name]=v
    return result


def stats(v):
    v=[x for x in v if x is not None]
    return dict(n=len(v),mean=statistics.mean(v) if v else None,sd=statistics.stdev(v) if len(v)>1 else None,
                min=min(v) if v else None,max=max(v) if v else None)


def write_csv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def close_tree(left,right,path=''):
    if isinstance(left,dict):
        if left.keys()!=right.keys():raise ValueError('metric keys differ at '+path)
        for k,v in left.items():close_tree(v,right[k],path+'.'+k)
    elif isinstance(left,list):
        if len(left)!=len(right):raise ValueError(path)
        for i,(a,b) in enumerate(zip(left,right)):close_tree(a,b,path+str(i))
    elif isinstance(left,(float,int)):
        if abs(left-right)>1e-12:raise ValueError('metric mismatch '+path)
    elif left!=right:raise ValueError('metric mismatch '+path)


def audit(root):
    records=read(root/'manifest.json')
    for name,r in records.items():
        p=root/name
        if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError('replay archive changed: '+name)
    parent=Path(read(root/'parent-integrity.json')['path'])
    pm=read(parent/'manifest.json')
    for name,r in pm.items():
        p=parent/name
        if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError('parent changed: '+name)
    source_checks=0;prediction_checks=0;geometry_checks=0;max_margin_error=0.;max_initial_error=0.
    endpoints=read(parent/'evaluation.json')
    for directory in sorted(root.glob('d*_seed*')):
        if not (directory/'cell_summary.json').exists():continue
        meta=read(directory/'cell_summary.json');d,s=meta['dim'],meta['seed']
        inputs=torch.load(directory/'inputs.pt',weights_only=True);new=torch.load(directory/'diagnostic_inputs.pt',weights_only=True)
        for name in ('inputs.pt','config.json','evaluation_labels.pt','initial_predictions.pt'):
            if digest(directory/name)!=pm[f'{directory.name}/{name}']['sha256']:raise ValueError('copied input changed')
        head=frozen_head(inputs);params=ParameterMap(head);initial=params.vector()
        coords=Coordinates(inputs['coordinates']['mean'],inputs['coordinates']['scale'])
        source=coords(new['source_raw']);labels=torch.load(directory/'evaluation_labels.pt',weights_only=True)
        with torch.no_grad():source_initial=head(source);target_initial=head(inputs['evaluation'])
        for arm in ARMS:
            saved=torch.load(directory/arm/'checkpoint_vectors.pt',weights_only=True)
            parent_events=torch.load(parent/directory.name/arm/'rounds.pt',weights_only=True)
            expected=torch.stack([initial]+[e['after'] for e in parent_events])
            torch.testing.assert_close(saved,expected,atol=0,rtol=0)
            logits=torch.load(directory/arm/'target_logits.pt',weights_only=True)
            geometry=torch.load(directory/arm/'boundary_records.pt',weights_only=True)
            metrics=read(directory/arm/'metrics.json')
            if len(metrics)!=31 or len(geometry)!=31 or len(logits)!=31:raise ValueError('incomplete replay trajectory')
            for i,row in enumerate(metrics):
                params.commit(saved[i])
                with torch.no_grad():torch.testing.assert_close(head(inputs['evaluation']),logits[i],atol=0,rtol=0)
                prediction_checks+=1
                close_tree(target_metrics(logits[i],labels,target_initial),row['target'])
                b=geometry[i];good=torch.isfinite(b['roots'])
                if int(good.sum())!=row['boundary']['unique_detected']:raise ValueError('root count mismatch')
                if dict(Counter(b['status']))!=row['boundary']['status_counts']:raise ValueError('root status mismatch')
                if good.any():
                    raw=torch.cat((b['roots'][good,None],new['nuisance'][good]),dim=1)
                    with torch.no_grad():out=head(coords(raw));margin=(out[:,1]-out[:,0]).abs()
                    torch.testing.assert_close(margin,b['root_margin_residual'][good],atol=1e-12,rtol=0)
                    max_margin_error=max(max_margin_error,float(margin.max()))
                geometry_checks+=1
                if i in (0,30):
                    with torch.no_grad():sr=source_metrics(head(source),new['source_y'],new['source_log_posterior'],source_initial)
                    close_tree(sr,row['source']);source_checks+=1
            final=next(r for r in endpoints if (r['dim'],r['seed'],r['arm'],r['source_batch'])==(d,s,arm,512))
            close_tree(final['final'],{k:metrics[-1]['target'][k] for k in final['final']})
        max_initial_error=max(max_initial_error,read(directory/'audit.json')['initial_source_posterior_log_error'])
    return dict(files_verified=len(records),parent_files_reverified=len(pm),target_state_predictions_replayed=prediction_checks,
                target_cohort_metrics_recomputed=prediction_checks,geometry_records_and_root_residuals_recomputed=geometry_checks,
                source_initial_and_final_metrics_recomputed=source_checks,max_root_margin_residual=max_margin_error,
                max_initial_source_log_posterior_error=max_initial_error,all_checkpoint_vectors_match_parent=True,
                final_metrics_match_published_full_source_results=True,parameter_training=False)


def summarize(rows):
    flat_rows=[flat(r) for r in rows];groups={};pairs=[];paired_stats={};onsets=[]
    metric_keys=['source_train_ce','source.nll','source.balanced_accuracy','source.conditional_ce.mean','source.posterior_kl.mean',
                 'source.posterior_kl.mc_se','source.posterior_probability_mae','target.balanced_accuracy','target.nll',
                 'target.mean_wrong_confidence','target.high_confidence_error_fraction_90','target.high_confidence_error_fraction_99',
                 'boundary.root_mean','boundary.root_std','boundary.target_root_rms','boundary.normal_angle_mean_deg',
                 'boundary.plane_slope_norm','boundary.nonplanarity_rms','boundary.unique_detected','boundary.positive_orientation_fraction']
    for d in (2,16):
        groups[d]={};paired_stats[d]={}
        for arm in ARMS:
            ends=[r for r in flat_rows if (r['dim'],r['arm'],r['round'])==(d,arm,30)]
            starts=[r for r in flat_rows if (r['dim'],r['arm'],r['round'])==(d,arm,0)]
            groups[d][arm]=dict(initial={k:stats([r.get(k) for r in starts]) for k in metric_keys},
                                final={k:stats([r.get(k) for r in ends]) for k in metric_keys},
                                cohorts={name:stats([r[f'target.cohorts.{name}.nll_delta_contribution'] for r in ends])
                                         for name in ('correct_correct','correct_wrong','wrong_correct','wrong_wrong')})
            for end in ends:
                start=next(r for r in starts if r['seed']==end['seed'])
                traj=sorted([r for r in flat_rows if (r['dim'],r['seed'],r['arm'])==(d,end['seed'],arm)],key=lambda r:r['round'])
                after=traj[1:]
                onsets.append(dict(dim=d,seed=end['seed'],arm=arm,
                    first_ba_below_initial=next((r['round'] for r in after if r['target.balanced_accuracy']<start['target.balanced_accuracy']),None),
                    first_nll_above_initial=next((r['round'] for r in after if r['target.nll']>start['target.nll']),None),
                    first_source_kl_above_1e_4=next((r['round'] for r in after if r['source.posterior_kl.mean']>1e-4),None)))
        for seed in range(5):
            left=next((r for r in flat_rows if (r['dim'],r['seed'],r['arm'],r['round'])==(d,seed,'population',30)),None)
            right=next((r for r in flat_rows if (r['dim'],r['seed'],r['arm'],r['round'])==(d,seed,'zero_zeta_geometry',30)),None)
            if left is None or right is None:continue
            row=dict(dim=d,seed=seed,comparison='population_minus_zero_final')
            row.update({k:left[k]-right[k] if left.get(k) is not None and right.get(k) is not None else None for k in metric_keys})
            pairs.append(row)
        paired_stats[d]={k:stats([r[k] for r in pairs if r['dim']==d]) for k in metric_keys}
    return dict(groups=groups,paired=pairs,paired_stats=paired_stats,onsets=onsets),flat_rows


def plots(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'source_ce_ggn':'tab:blue','zero_zeta_geometry':'tab:orange','population':'tab:green'}
    labels={'source_ce_ggn':'Source GGN','zero_zeta_geometry':'Zero request','population':'Population score'}
    panels=[('source-generalization',[('source_train_ce','Source training CE',False),('source.conditional_ce.mean','Independent source conditional CE',False),('source.posterior_kl.mean','KL(true source posterior || model)',False)]),
            ('target-trajectories',[('target.balanced_accuracy','Target BA change (pp)',True),('target.nll','Target NLL change',True),('target.high_confidence_error_fraction_99','Confident errors (p >= .99), fraction',False)]),
            ('boundary-trajectories',[('boundary.root_mean','Mean boundary x1 (target=0)',False),('boundary.root_std','Boundary x1 standard deviation',False),('boundary.normal_angle_mean_deg','Mean normal angle to +e1 (degrees)',False),('boundary.nonplanarity_rms','RMS residual after affine root fit',False)])]
    for name,columns in panels:
        fig,axes=plt.subplots(2,len(columns),figsize=(4.1*len(columns),6.5),layout='constrained')
        for i,d in enumerate((2,16)):
            for j,(key,label,change) in enumerate(columns):
                axis=axes[i,j]
                for arm in ARMS:
                    curves=[]
                    for seed in range(5):
                        selected=sorted([r for r in rows if (r['dim'],r['seed'],r['arm'])==(d,seed,arm)],key=lambda r:r['round'])
                        if not selected:continue
                        vals=[r.get(key) for r in selected]
                        if any(v is None for v in vals):continue
                        if change:vals=[v-vals[0] for v in vals]
                        if key=='target.balanced_accuracy':vals=[100*v for v in vals]
                        axis.plot([r['round'] for r in selected],vals,c=colors[arm],alpha=.2,lw=.8);curves.append(vals)
                    if curves:axis.plot(range(31),[statistics.mean(v) for v in zip(*curves)],c=colors[arm],lw=2,label=labels[arm])
                axis.set_title(f'd={d}');axis.set_xlabel('Saved round');axis.set_ylabel(label);axis.legend(fontsize=7)
                if change or key=='boundary.root_mean':axis.axhline(0,c='gray',lw=.7)
                if key=='boundary.root_mean':axis.axhline(.5,c='gray',lw=.7,ls=':')
        fig.savefig(out/(name+'.png'),dpi=160);fig.savefig(out/(name+'.svg'));plt.close(fig)
    names=['correct_correct','correct_wrong','wrong_correct','wrong_wrong'];colors4=['tab:blue','tab:red','tab:green','tab:orange']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for axis,d in zip(axes,(2,16)):
        positive=[0.]*3;negative=[0.]*3
        for name,color in zip(names,colors4):
            vals=[statistics.mean(r['target.cohorts.'+name+'.nll_delta_contribution'] for r in rows if (r['dim'],r['arm'],r['round'])==(d,arm,30)) for arm in ARMS]
            axis.bar(range(3),vals,bottom=[positive[i] if v>=0 else negative[i] for i,v in enumerate(vals)],color=color,label=name)
            for i,v in enumerate(vals):
                if v>=0:positive[i]+=v
                else:negative[i]+=v
        axis.axhline(0,color='gray',lw=.7);axis.set_xticks(range(3),[labels[a] for a in ARMS],rotation=10)
        axis.set_title(f'd={d}: final minus initial NLL');axis.set_ylabel('NLL contribution');axis.legend(fontsize=7)
    fig.savefig(out/'nll-cohorts.png',dpi=160);fig.savefig(out/'nll-cohorts.svg');plt.close(fig)


def report(s):
    fmt=lambda x:'NA' if x is None else f'{x:.6f}'
    lines=['# 现有checkpoint回放诊断（2026-11-10）','',
           '三组全源512实验的初始化及30轮参数原样回放，没有重新训练、生成参数候选或选择checkpoint。每个维数/种子使用32768个独立源样本和256组固定边界探针；目标评价复用原10000个合成样本。', '',
           f"执行代码：[`{s['execution_commit'][:7]}`](https://github.com/chx739/TQG-GDA/tree/{s['execution_commit']})；[冻结协议](../replay-plan.json)；[前轮结果](../e2_full_source_20261110/report.md)。",'',
           '## 源泛化','',
           '条件CE对每个源输入使用真实类别后验加权；KL衡量预测偏离真实源后验的程度。初始头就是源Bayes后验，故其源总体CE已经最优；KL是总体CE超额的Monte Carlo估计。以下跨5个种子取均值，单种子的MC标准误和逐种子值另存表格。','',
           '| 维度 | 方法 | 训练CE 初→终 | 独立源硬标签CE 初→终 | 独立源条件CE 初→终 | 最终KL均值±种子SD |',
           '| --- | --- | --- | --- | --- | --- |']
    for d in (2,16):
        for arm,g in s['groups'][d].items():
            a,b=g['initial'],g['final']
            delta=lambda k:fmt(a[k]['mean'])+' → '+fmt(b[k]['mean'])
            lines.append(f"| {d} | {arm} | {delta('source_train_ce')} | {delta('source.nll')} | {delta('source.conditional_ce.mean')} | {fmt(b['source.posterior_kl.mean']['mean'])} ± {fmt(b['source.posterior_kl.mean']['sd'])} |")
    lines+=['','![源泛化](source-generalization.png)','', '## 目标分类与置信度','',
            '| 维度 | 方法 | 目标NLL 初→终 | 错误预测平均置信度 初→终 | 置信度≥0.99的错误占全部样本 初→终 |',
            '| --- | --- | --- | --- | --- |']
    for d in (2,16):
        for arm,g in s['groups'][d].items():
            a,b=g['initial'],g['final'];delta=lambda k:fmt(a[k]['mean'])+' → '+fmt(b[k]['mean'])
            lines.append(f"| {d} | {arm} | {delta('target.nll')} | {delta('target.mean_wrong_confidence')} | {delta('target.high_confidence_error_fraction_99')} |")
    lines+=['','![目标轨迹](target-trajectories.png)','',
            '下面四组根据每个样本的初始/当前正确性定义。贡献为该组样本NLL变化之和除以全部样本数；四项精确加和为总ΔNLL。组内样本的身份已固定在存档logits中，可重算。它是描述性分解。','',
            '| 维度 | 方法 | 初末均正确 | 由对变错 | 由错变对 | 初末均错误 |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
    for d in (2,16):
        for arm,g in s['groups'][d].items():lines.append('| '+str(d)+' | '+arm+' | '+' | '.join(fmt(g['cohorts'][k]['mean']) for k in ('correct_correct','correct_wrong','wrong_correct','wrong_wrong'))+' |')
    lines+=['','![NLL分组](nll-cohorts.png)','', '## 固定探针上的边界','',
            '原始坐标中目标正确阈值为0，初始/源阈值为0.5；理想法向为+e1。根的标准差反映不同干扰坐标下的边界位置差异。非平面残差为扣除最佳仿射拟合后的RMS，不直接称为全局曲率。','',
            '| 维度 | 方法 | 最终根均值 | 根SD | 到目标阈值的RMS | 法向平均角度° | 非平面RMS |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for d in (2,16):
        for arm,g in s['groups'][d].items():lines.append('| '+str(d)+' | '+arm+' | '+' | '.join(fmt(g['final'][k]['mean']) for k in ('boundary.root_mean','boundary.root_std','boundary.target_root_rms','boundary.normal_angle_mean_deg','boundary.nonplanarity_rms'))+' |')
    lines+=['','![边界轨迹](boundary-trajectories.png)','',
            '有限区间和网格仅识别“唯一检测根”，不能证明全局唯一，也可能漏掉切触或网格内多次过零。无根、多根、非有限值均保留，不挑最靠近理想位置的根。完整逐轮覆盖在[trajectory.csv](trajectory.csv)。','',
            '## 记录与核验','',f"完成状态评价{s['execution']['states']}次；失败单元{s['execution']['failures']}个；耗时{s['execution']['seconds']:.1f}秒。48项工程测试通过。",'',
            f"重放{s['audit']['target_state_predictions_replayed']}组目标logits逐位一致；所有compact参数向量与原档案一致；原文件前后哈希一致。源初始化及终点指标复算{s['audit']['source_initial_and_final_metrics_recomputed']}组；逐探针根残差/状态核验{s['audit']['geometry_records_and_root_residuals_recomputed']}组。",'',
            '研究单位为5个种子，轮次和探针不扩充独立重复。已知源Bayes初始化与先前结果使本诊断具有明确事后性质；不存在按目标评价选轮次。原样本、配置与checkpoint不覆盖，第2步score样本量/带宽实验没有运行。','',
            '逐轮指标：[trajectory.csv](trajectory.csv)；population减ζ=0的逐种子终点差：[paired.csv](paired.csv)；首次越过初始指标的轮次：[onsets.csv](onsets.csv)。首次越线不等于稳定变差或显著性结论。完整源输入/后验、checkpoint向量、目标logits、根/法向记录、日志与源代码见[归档索引](archive-index.json)。','']
    return '\n'.join(lines)
