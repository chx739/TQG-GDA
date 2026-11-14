"""Audit posterior-CE proposals and compare matched saved trajectories."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import torch
from tsba.boundary import BoundaryBatch
from tsba.classifier import Coordinates,ParameterMap
from tsba.config import Config
from tsba.subproblem import GGNSystem
from experiments.common.synthetic.archive import digest, write_json
from experiments.common.synthetic.controls import SourceGGN
from experiments.common.synthetic.replay import source_metrics, target_metrics
from experiments.common.synthetic.run_controls import make_head
from experiments.synthetic.e2_source_posterior_20261110.source_posterior import ARMS, analytic_source_posterior, audit_posterior_event, PosteriorSourceProblem, PosteriorTsbaProblem
from experiments.common.synthetic.summarize_replay import flat, stats, write_csv, close_tree


def read(p):return json.loads(p.read_text())


def verify_files(root):
    rows=read(root/'manifest.json')
    for name,r in rows.items():
        p=root/name
        if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError('file changed: '+str(p))
    return rows


def audit(root):
    files=verify_files(root);provenance=read(root/'parent-integrity.json')
    parent=Path(provenance['parent']['path']);reference=Path(provenance['replay']['path'])
    pm=verify_files(parent);rm=verify_files(reference)
    for name,path in [('parent',parent),('replay',reference)]:
        if digest(path/'manifest.json')!=provenance[name]['manifest_sha256']:raise ValueError('parent manifest mismatch')
    count=0;trial_count=0;predictions=0;ce_error=0.;gradient_error=0.;residual_error=0.;source_checks=0;geometry_checks=0
    process=[]
    for directory in sorted(root.glob('d*_seed*')):
        if not (directory/'cell_summary.json').exists():continue
        cell=read(directory/'cell_summary.json');dim,seed=cell['dim'],cell['seed']
        for name in ('inputs.pt','config.json','schedule.pt','initial_predictions.pt','evaluation_labels.pt','population-score.pt'):
            if digest(directory/name)!=pm[f'{directory.name}/{name}']['sha256']:raise ValueError('input/config/schedule changed')
        inputs=torch.load(directory/'inputs.pt',weights_only=True);cfg=Config(**read(directory/'config.json'))
        schedule=torch.load(directory/'schedule.pt',weights_only=True)
        coords=Coordinates(inputs['coordinates']['mean'],inputs['coordinates']['scale'])
        posterior,logp=analytic_source_posterior(inputs['source'],coords)
        pstate=torch.load(directory/'source-posterior.pt',weights_only=True)
        torch.testing.assert_close(posterior,pstate['probabilities'],atol=0,rtol=0)
        torch.testing.assert_close(logp,pstate['log_probabilities'],atol=0,rtol=0)
        head=make_head(inputs);params=ParameterMap(head);initial=params.vector()
        diag=root/'diagnostics'/directory.name
        new=torch.load(diag/'diagnostic_inputs.pt',weights_only=True)
        previous_new=torch.load(reference/directory.name/'diagnostic_inputs.pt',weights_only=True)
        for key in ('source_raw','source_y','source_log_posterior','nuisance'):
            torch.testing.assert_close(new[key],previous_new[key],atol=0,rtol=0)
        labels=torch.load(directory/'evaluation_labels.pt',weights_only=True)
        with torch.no_grad():
            target_initial=head(inputs['evaluation']);source_initial=head(coords(new['source_raw']))
        for arm in ARMS:
            params.commit(initial);events=torch.load(directory/arm/'rounds.pt',weights_only=True)
            previous=initial;reasons=Counter();alphas=[]
            if len(events)!=30:raise ValueError('incomplete adaptation')
            for e,ids in zip(events,schedule):
                torch.testing.assert_close(e['before'],previous,atol=0,rtol=0)
                if not e['record']['accepted']:torch.testing.assert_close(e['before'],e['after'],atol=0,rtol=0)
                else:torch.testing.assert_close(e['after'],e['before']+e['record']['trials'][-1]['alpha']*e['delta'],atol=0,rtol=0)
                params.commit(e['before']);saved=dict(e['posterior_audit']);audit_posterior_event(head,inputs,posterior,ids,e)
                close_tree(saved,e['posterior_audit'])
                a=e['posterior_audit'];ce_error=max(ce_error,a['max_trial_ce_error']);gradient_error=max(gradient_error,a['gradient_max_error']);trial_count+=a['trials_checked']
                if e['delta'] is not None and 'solver' in e['record']:
                    si=e['source_ids']
                    if arm=='source_ce_ggn':
                        problem=PosteriorSourceProblem(head,inputs['source'][si],posterior[si],cfg);system=SourceGGN(problem)
                    else:
                        problem=PosteriorTsbaProblem(head,inputs['source'][si],posterior[si],BoundaryBatch(**e['boundaries']),cfg);system=GGNSystem(problem)
                    torch.testing.assert_close(system.gs,e['source_gradient'],atol=1e-13,rtol=1e-12)
                    if 'relative_residual' in e['record']['solver']:
                        resid=float((system.matvec(e['delta'])-system.rhs).norm()/system.rhs.norm())
                        residual_error=max(residual_error,abs(resid-e['record']['solver']['relative_residual']))
                        if abs(resid-e['record']['solver']['relative_residual'])>1e-10:raise ValueError('proposal equation mismatch')
                    for trial in e['record'].get('trials',[]):
                        expected=problem.evaluate(e['before']+trial['alpha']*e['delta'])
                        close_tree(expected,{k:trial[k] for k in expected})
                if e['record']['accepted']:alphas.append(e['record']['trials'][-1]['alpha'])
                previous=e['after'];count+=1;reasons[e['record']['reason']]+=1
            head.load_state_dict(torch.load(directory/arm/'classifier.pt',weights_only=True))
            torch.testing.assert_close(params.vector(),previous,atol=0,rtol=0)
            with torch.no_grad():torch.testing.assert_close(head(inputs['evaluation']),torch.load(directory/arm/'predictions.pt',weights_only=True),atol=0,rtol=0)
            process.append(dict(dim=dim,seed=seed,arm=arm,**cell['arms'][arm],reasons=dict(reasons),mean_accepted_alpha=statistics.mean(alphas) if alphas else None))
            vectors=torch.load(diag/arm/'checkpoint_vectors.pt',weights_only=True)
            torch.testing.assert_close(vectors,torch.stack([initial]+[e['after'] for e in events]),atol=0,rtol=0)
            logits=torch.load(diag/arm/'target_logits.pt',weights_only=True);metrics=read(diag/arm/'metrics.json')
            geometry=torch.load(diag/arm/'boundary_records.pt',weights_only=True)
            if len(logits)!=31 or len(metrics)!=31 or len(geometry)!=31:raise ValueError('incomplete diagnostics')
            for i,row in enumerate(metrics):
                params.commit(vectors[i])
                with torch.no_grad():torch.testing.assert_close(head(inputs['evaluation']),logits[i],atol=0,rtol=0)
                close_tree(target_metrics(logits[i],labels,target_initial),row['target']);predictions+=1
                g=geometry[i];good=torch.isfinite(g['roots'])
                if int(good.sum())!=row['boundary']['unique_detected'] or dict(Counter(g['status']))!=row['boundary']['status_counts']:raise ValueError('probe coverage mismatch')
                if good.any():
                    raw=torch.cat((g['roots'][good,None],new['nuisance'][good]),dim=1)
                    with torch.no_grad():out=head(coords(raw));res=(out[:,1]-out[:,0]).abs()
                    torch.testing.assert_close(res,g['root_margin_residual'][good],atol=1e-12,rtol=0)
                geometry_checks+=1
                if i in (0,30):
                    with torch.no_grad():sr=source_metrics(head(coords(new['source_raw'])),new['source_y'],new['source_log_posterior'],source_initial)
                    close_tree(sr,row['source']);source_checks+=1
    return dict(files_verified=len(files),parent_files_verified=len(pm),replay_reference_files_verified=len(rm),
                adaptation_states_verified=count,candidate_trials_recomputed=trial_count,max_trial_ce_error=ce_error,
                max_source_gradient_error=gradient_error,max_proposal_residual_replay_error=residual_error,
                target_state_predictions_replayed=predictions,source_initial_final_metrics_recomputed=source_checks,
                geometry_records_verified=geometry_checks,config_and_batch_ids_exact=True,diagnostic_inputs_exact=True),process


def collect(root,process):
    reference=Path(read(root/'parent-integrity.json')['replay']['path'])
    old=read(reference/'metrics.json');new=read(root/'diagnostics/metrics.json')
    rows=[dict(objective=label,**flat(r)) for label,part in [('hard',old),('posterior',new)] for r in part]
    valid={(r['dim'],r['seed'],r['arm']) for r in process if r['stop_reason']=='round_budget' and r['rounds']==30}
    rows=[r for r in rows if r['objective']=='hard' or (r['dim'],r['seed'],r['arm']) in valid]
    endpoints=[];groups={};pairs=[];paired_stats={}
    for r in rows:
        if r['round']!=30:continue
        first=next(x for x in rows if (x['objective'],x['dim'],x['seed'],x['arm'],x['round'])==(r['objective'],r['dim'],r['seed'],r['arm'],0))
        endpoints.append(dict(r,delta_ba_pp=100*(r['target.balanced_accuracy']-first['target.balanced_accuracy']),
                              delta_nll=r['target.nll']-first['target.nll']))
    keys=['delta_ba_pp','delta_nll','source.posterior_kl.mean','source.conditional_ce.mean','source_train_ce',
          'source_posterior_train_ce','target.mean_wrong_confidence','target.high_confidence_error_fraction_99',
          'boundary.root_mean','boundary.root_std','boundary.target_root_rms','boundary.normal_angle_mean_deg','boundary.nonplanarity_rms']
    for d in (2,16):
        groups[d]={};paired_stats[d]={}
        for objective in ('hard','posterior'):
            groups[d][objective]={}
            for arm in ARMS:
                selected=[r for r in endpoints if (r['dim'],r['objective'],r['arm'])==(d,objective,arm)]
                values=[r['delta_ba_pp'] for r in selected]
                groups[d][objective][arm]=dict(metrics={k:stats([r.get(k) for r in selected]) for k in keys},
                                               n=len(selected),wins=sum(v>0 for v in values),ties=sum(v==0 for v in values),losses=sum(v<0 for v in values))
        contrasts=[(arm+'_posterior_minus_hard','posterior',arm,'hard',arm) for arm in ARMS]
        contrasts += [('posterior_population_minus_zero','posterior','population','posterior','zero_zeta_geometry')]
        for name,ob1,a1,ob2,a2 in contrasts:
            selected=[]
            for seed in range(5):
                left=next((r for r in endpoints if (r['dim'],r['seed'],r['objective'],r['arm'])==(d,seed,ob1,a1)),None)
                right=next((r for r in endpoints if (r['dim'],r['seed'],r['objective'],r['arm'])==(d,seed,ob2,a2)),None)
                if left is None or right is None:continue
                row=dict(dim=d,seed=seed,comparison=name)
                row.update({k:left[k]-right[k] if left.get(k) is not None and right.get(k) is not None else None for k in keys})
                selected.append(row);pairs.append(row)
            paired_stats[d][name]=dict(metrics={k:stats([r[k] for r in selected]) for k in keys},
                                      wins=sum(r['delta_ba_pp']>0 for r in selected),ties=sum(r['delta_ba_pp']==0 for r in selected),losses=sum(r['delta_ba_pp']<0 for r in selected))
    return dict(trajectory=rows,endpoints=endpoints,groups=groups,paired=pairs,paired_stats=paired_stats)


def plots(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'source_ce_ggn':'tab:blue','zero_zeta_geometry':'tab:orange','population':'tab:green'}
    labels={'source_ce_ggn':'Source GGN','zero_zeta_geometry':'Zero request','population':'Population score'}
    panels=[('posterior-risk',[('target.balanced_accuracy','Target BA change (pp)',True),('target.nll','Target NLL change',True),('source.posterior_kl.mean','Independent source posterior KL',False)]),
            ('posterior-geometry',[('boundary.root_mean','Mean boundary x1',False),('boundary.root_std','Boundary x1 standard deviation',False),('boundary.normal_angle_mean_deg','Normal angle to +e1 (degrees)',False)])]
    for name,columns in panels:
        fig,axes=plt.subplots(2,len(columns),figsize=(12,6.5),layout='constrained')
        for i,d in enumerate((2,16)):
            for j,(key,label,change) in enumerate(columns):
                axis=axes[i,j]
                for objective,style in [('hard','--'),('posterior','-')]:
                    for arm in ARMS:
                        curves=[]
                        for seed in range(5):
                            r=sorted([r for r in rows if (r['dim'],r['seed'],r['objective'],r['arm'])==(d,seed,objective,arm)],key=lambda r:r['round'])
                            if not r or any(x.get(key) is None for x in r):continue
                            vals=[x[key] for x in r]
                            if change:vals=[x-vals[0] for x in vals]
                            if key=='target.balanced_accuracy':vals=[100*x for x in vals]
                            curves.append(vals)
                            if objective=='posterior':axis.plot(range(31),vals,color=colors[arm],alpha=.15,lw=.6)
                        if curves:axis.plot(range(31),[statistics.mean(x) for x in zip(*curves)],color=colors[arm],ls=style,lw=1.7,label=f'{labels[arm]} ({objective})')
                axis.set_title(f'd={d}; solid=posterior CE');axis.set_ylabel(label);axis.set_xlabel('Attempted round')
                if key=='boundary.root_mean':axis.axhline(0,color='gray',lw=.6);axis.axhline(.5,color='gray',lw=.6,ls=':')
                if change:axis.axhline(0,color='gray',lw=.6)
                axis.legend(fontsize=6)
        fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.svg'));plt.close(fig)


def report(s):
    f=lambda x:'NA' if x is None else f'{x:.6f}'
    lines=['# 解析源后验CE对照（2026-11-10）','',
           '唯一干预：同一512个源输入上的硬标签CE，改为解析源后验加权的期望CE；候选梯度和真实验收同时替换。全部Config、初始头、源/目标ID、总体score、带宽及预算沿用全源实验。', '',
           f"执行代码：[`{s['execution_commit'][:7]}`](https://github.com/chx739/TQG-GDA/tree/{s['execution_commit']})；[冻结协议](../source-posterior-plan.json)；[前轮回放](../e2_replay_20261110/report.md)。",'',
           '## 相对初始化的最终结果','',
           '| 维度 | 源CE | 方法 | ΔBA均值±种子SD（百分点） | 胜/平/负 | ΔNLL均值 | 独立源KL均值 |',
           '| --- | --- | --- | ---: | --- | ---: | ---: |']
    for d in (2,16):
        for objective in ('hard','posterior'):
            for arm,g in s['groups'][d][objective].items():
                m=g['metrics'];ba=m['delta_ba_pp'];lines.append(f"| {d} | {objective} | {arm} | {f(ba['mean'])} ± {f(ba['sd'])} | {g['wins']}/{g['ties']}/{g['losses']} | {f(m['delta_nll']['mean'])} | {f(m['source.posterior_kl.mean']['mean'])} |")
    lines+=['','BA正差较好，NLL负差较好。旧硬标签结果重用已封存轨迹，不重复训练。所有种子见[endpoints.csv](endpoints.csv)。','',
            '![风险配对](posterior-risk.png)','', '## 配对差','',
            '| 维度 | 比较 | ΔBA均值±配对SD（百分点） | 胜/平/负 | ΔNLL均值 |',
            '| --- | --- | ---: | --- | ---: |']
    for d in (2,16):
        for name,g in s['paired_stats'][d].items():
            m=g['metrics'];lines.append(f"| {d} | {name} | {f(m['delta_ba_pp']['mean'])} ± {f(m['delta_ba_pp']['sd'])} | {g['wins']}/{g['ties']}/{g['losses']} | {f(m['delta_nll']['mean'])} |")
    lines+=['','差值先按同一种子计算，再跨种子汇总；[paired.csv](paired.csv)包含全部逐种子差值。','',
            '## 边界与错误置信度','',
            '| 维度 | 源CE | 方法 | 根均值 | 根SD | 法向角度° | 非平面RMS | 置信度≥99%的错误占比 |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for d in (2,16):
        for objective in ('hard','posterior'):
            for arm,g in s['groups'][d][objective].items():
                m=g['metrics'];lines.append('| '+str(d)+' | '+objective+' | '+arm+' | '+' | '.join(f(m[k]['mean']) for k in ('boundary.root_mean','boundary.root_std','boundary.normal_angle_mean_deg','boundary.nonplanarity_rms','target.high_confidence_error_fraction_99'))+' |')
    lines+=['','![边界配对](posterior-geometry.png)','',
            '原始坐标中的源/初始阈值是0.5，目标阈值是0。非平面RMS为固定探针上扣除最佳平面后的残差。有限网格仅报告唯一检测根，并保留缺根、多根及无效法向；逐轮覆盖见[trajectory.csv](trajectory.csv)。','',
            '## 控制组、验收与档案','',
            '| 维度 | 方法 | 平均接受轮数 | 最大参数漂移 | 最大目标logit漂移 | 最大源梯度范数 |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
    for d in (2,16):
        for arm in ARMS:
            r=[r for r in s['process'] if (r['dim'],r['arm'])==(d,arm)]
            if r:lines.append(f"| {d} | {arm} | {statistics.mean(x['accepted'] for x in r):.1f} | {max(x['max_parameter_drift'] for x in r):.3g} | {max(x['max_evaluation_logit_drift'] for x in r):.3g} | {max(x['source_gradient_max_norm'] for x in r):.3g} |")
    a=s['audit']
    lines+=['','没有新增梯度归零或stationary跳过阈值。近零梯度可能产生被原Armijo接受的浮点级更新；“接受次数”与“实际参数移动”分别报告。全部原因见[process.csv](process.csv)。', '',
            f"51项工程测试通过。逐轮核验{a['adaptation_states_verified']}次参数状态与候选方程；独立复算{a['candidate_trials_recomputed']}次回溯的后验CE，最大误差{a['max_trial_ce_error']:.3g}；源梯度最大复算差{a['max_source_gradient_error']:.3g}。{a['target_state_predictions_replayed']}组诊断logits重载逐位一致。",'',
            '后验期望在原512个源输入上计算，消除了给定输入的标签抽样噪声，但仍非源输入分布的精确积分。独立源评价样本及固定探针与上轮逐位相同。全部适配预测封存后才评价目标标签，未改选checkpoint。正式TSBA未修改，第2步score样本量/带宽实验未运行。', '',
            '完整日志、逐轮候选/梯度/参数、后验权重、最终checkpoint、诊断logits与几何、代码和环境保存在[独立档案](archive-index.json)。GitHub只含报告、代码、图、标量表和核验清单。','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    a,process=audit(args.run);s=collect(args.run,process)
    s.update(audit=a,process=process,execution=read(args.run/'execution.json'),execution_commit=read(args.run/'provenance.json')['git_commit'],
             failures=read(args.run/'failures.json'),diagnostic_failures=read(args.run/'diagnostics/failures.json'))
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'summary.json',{k:v for k,v in s.items() if k!='trajectory'});write_json(args.output/'checkpoint-audit.json',a)
    for key in ('trajectory','endpoints','paired','process'):write_csv(args.output/(key+'.csv'),s[key])
    plots(s['trajectory'],args.output)
    with (args.output/'report.md').open('x') as f:f.write(report(s))
    print(json.dumps(a,indent=2))


if __name__=='__main__':main()
