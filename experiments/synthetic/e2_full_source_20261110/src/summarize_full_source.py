"""Audit full-source checkpoints and pair them with archived minibatch endpoints."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import torch
from torch.nn import functional as F
from tsba.classifier import ParameterMap, Coordinates
from experiments.common.io import digest, write_json
from experiments.synthetic.e1_e2_20261109.src.distributions import oracles
from experiments.synthetic.e2_controls_20261109.src.execution import make_head
from experiments.synthetic.e2_full_source_20261110.src.execution import ARMS, full_source_config, audit_full_source_event
from tsba.config import Config
from experiments.synthetic.e1_e2_20261109.src.analysis import stats, write_csv


def read(p):return json.loads(p.read_text())


def audit_and_summarize(root):
    files=read(root/'manifest.json')
    for name,r in files.items():
        p=root/name
        if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError(f'archive corruption: {name}')
    provenance=read(root/'parent-integrity.json')
    original=Path(provenance['original']['path']);controls=Path(provenance['controls']['path'])
    parent_manifests={original:read(original/'manifest.json'),controls:read(controls/'manifest.json')}
    for name,parent in [('original',original),('controls',controls)]:
        if digest(parent/'manifest.json')!=provenance[name]['manifest_sha256']:raise ValueError('parent manifest changed')
    process=[];curves=[];cells=[];checkpoints=0;states=0;trials=0;max_ce_error=0.;oracle_checks=0
    for directory in sorted(root.glob('d*_seed*')):
        if not (directory/'cell_summary.json').exists():continue
        cell=read(directory/'cell_summary.json');cells.append(cell);dim,seed=cell['dim'],cell['seed']
        inputs=torch.load(directory/'inputs.pt',weights_only=True)
        oldcfg=Config(**read(directory/'parent-config.json'));cfg=full_source_config(oldcfg,len(inputs['source']))
        if cfg.to_dict()!=read(directory/'config.json'):raise ValueError('configuration mismatch')
        schedule=torch.load(directory/'schedule.pt',weights_only=True)
        head=make_head(inputs);params=ParameterMap(head);initial=params.vector()
        with torch.no_grad():torch.testing.assert_close(head(inputs['evaluation']),torch.load(directory/'initial_predictions.pt',weights_only=True),atol=0,rtol=0)
        c=inputs['coordinates'];pop,_=oracles(inputs['target'],Coordinates(c['mean'],c['scale']),inputs['sigma'])
        saved=torch.load(directory/'population-score.pt',weights_only=True)
        for k,v in pop.state_dict().items():torch.testing.assert_close(v,saved['state_dict'][k],atol=0,rtol=0)
        oracle_checks+=1
        for arm in ARMS:
            events=torch.load(directory/arm/'rounds.pt',weights_only=True)
            if len(events)!=cell['arms'][arm]['rounds'] or len(events)>len(schedule):raise ValueError('round count mismatch')
            previous=initial;anchor_counts=[];alphas=[];trial_counts=[];residuals=[];movements=[]
            for e,schedule_row in zip(events,schedule):
                torch.testing.assert_close(e['before'],previous,atol=0,rtol=0)
                if not e['record']['accepted']:torch.testing.assert_close(e['before'],e['after'],atol=0,rtol=0)
                else:torch.testing.assert_close(e['after'],e['before']+e['record']['trials'][-1]['alpha']*e['delta'],atol=0,rtol=0)
                previous=e['after'];states+=1
                expected=dict(e['full_source_ce_audit'])
                audit_full_source_event(head,inputs,schedule_row,e)
                if expected!=e['full_source_ce_audit']:raise ValueError('source CE audit not reproducible')
                audit=e['full_source_ce_audit'];trials+=audit['trials_checked'];max_ce_error=max(max_ce_error,audit['max_logged_ce_error'])
                r=e['record'];trial_counts.append(len(r.get('trials',[])))
                curves.append(dict(dim=dim,seed=seed,arm=arm,source_batch=512,round=r['round']+1,
                                   before=audit['before'],after=audit['after'],change=audit['change'],
                                   change_from_initial=audit['after']-cell['initial_source_ce'],accepted=r['accepted']))
                if e['boundaries'] is not None:anchor_counts.append(len(e['boundaries']['points']))
                if r['accepted']:
                    alphas.append(r['trials'][-1]['alpha'])
                    if 'max_movement' in r['trials'][-1]:movements.append(r['trials'][-1]['max_movement']/inputs['sigma'])
                if r.get('solver',{}).get('relative_residual') is not None:residuals.append(r['solver']['relative_residual'])
            head.load_state_dict(torch.load(directory/arm/'classifier.pt',weights_only=True))
            torch.testing.assert_close(params.vector(),previous,atol=0,rtol=0)
            with torch.no_grad():torch.testing.assert_close(head(inputs['evaluation']),torch.load(directory/arm/'predictions.pt',weights_only=True),atol=0,rtol=0)
            checkpoints+=1
            row=dict(dim=dim,seed=seed,arm=arm,source_batch=512,**cell['arms'][arm],
                     source_ce_change=cell['arms'][arm]['source_ce']-cell['initial_source_ce'],
                     coverage=sum(anchor_counts)/sum(len(e['target_ids']) for e in events) if anchor_counts else None,
                     mean_trials=statistics.mean(trial_counts) if trial_counts else None,
                     mean_accepted_alpha=statistics.mean(alphas) if alphas else None,
                     max_solver_residual=max(residuals) if residuals else None,
                     max_movement_sigma=max(movements) if movements else None,
                     reasons=dict(Counter(e['record']['reason'] for e in events)))
            process.append(row)
            # Passive recomputation of OLD full-source CE, from the old round snapshots.
            parent=original if arm=='population' else controls
            relative=f'{directory.name}/{arm}/rounds.pt';path=parent/relative
            if digest(path)!=parent_manifests[parent][relative]['sha256']:raise ValueError('old snapshot changed')
            old_events=torch.load(path,weights_only=True)
            if len(old_events)!=len(schedule):raise ValueError('incomplete old trajectory')
            for e,new_ids in zip(old_events,schedule):
                if not torch.equal(e['target_ids'],new_ids['target_ids']):raise ValueError('target ID changed')
                if not torch.equal(e['source_ids'],new_ids['source_ids'][:128]):raise ValueError('source prefix changed')
                with torch.no_grad():
                    before=float(F.cross_entropy(params.logits(e['before'],inputs['source']),inputs['source_y']))
                    after=float(F.cross_entropy(params.logits(e['after'],inputs['source']),inputs['source_y']))
                curves.append(dict(dim=dim,seed=seed,arm=arm,source_batch=128,round=e['record']['round']+1,
                                   before=before,after=after,change=after-before,
                                   change_from_initial=after-cell['initial_source_ce'],accepted=e['record']['accepted']))
    evaluation=read(root/'evaluation.json')
    successful={(r['dim'],r['seed'],r['arm']) for r in process if r['stop_reason']=='round_budget' and r['rounds']==30}
    valid=[r for r in evaluation if r['source_batch']==128 or (r['dim'],r['seed'],r['arm']) in successful]
    grouped={};pairs=[];paired_stats={}
    contrasts=[('population_vs_zero',512,'population',512,'zero_zeta_geometry'),
               ('zero_vs_source',512,'zero_zeta_geometry',512,'source_ce_ggn')]
    contrasts += [(f'{arm}_512_minus_128',512,arm,128,arm) for arm in ARMS]
    for dim in (2,16):
        grouped[dim]={};paired_stats[dim]={}
        for batch in (128,512):
            grouped[dim][batch]={}
            for arm in ARMS:
                rows=[r for r in valid if (r['dim'],r['source_batch'],r['arm'])==(dim,batch,arm)]
                v=[r['delta_ba_pp'] for r in rows]
                grouped[dim][batch][arm]=dict(n=len(rows),delta_ba_pp=stats(v),delta_nll=stats([r['delta_nll'] for r in rows]),
                                              wins=sum(x>0 for x in v),ties=sum(x==0 for x in v),losses=sum(x<0 for x in v))
        for name,b1,a1,b2,a2 in contrasts:
            rows=[]
            for seed in range(5):
                l=next((r for r in valid if (r['dim'],r['seed'],r['source_batch'],r['arm'])==(dim,seed,b1,a1)),None)
                r=next((r for r in valid if (r['dim'],r['seed'],r['source_batch'],r['arm'])==(dim,seed,b2,a2)),None)
                if l is None or r is None:continue
                row=dict(dim=dim,seed=seed,comparison=name,left_arm=a1,left_batch=b1,right_arm=a2,right_batch=b2,
                         ba_difference_pp=100*(l['final']['balanced_accuracy']-r['final']['balanced_accuracy']),
                         nll_difference=l['final']['nll']-r['final']['nll'])
                rows.append(row);pairs.append(row)
            v=[r['ba_difference_pp'] for r in rows]
            paired_stats[dim][name]=dict(n=len(rows),ba_difference_pp=stats(v),nll_difference=stats([r['nll_difference'] for r in rows]),
                                         wins=sum(x>0 for x in v),ties=sum(x==0 for x in v),losses=sum(x<0 for x in v))
    audit=dict(files_verified=len(files),final_classifiers_replayed=checkpoints,oracle_states_verified=oracle_checks,
               state_chains_verified=states,all_source_ids_complete=True,all_target_ids_match_parents=True,
               configuration_change='source_adapt_batch only: 128 -> 512',source_ce_trials_recomputed=trials,
               max_logged_source_ce_error=max_ce_error,prediction_replay_atol=0,prediction_replay_rtol=0,
               old_source_ce_curves='recomputed after sealing; no effect on any update')
    return dict(run_path=str(root),execution_commit=read(root/'provenance.json')['git_commit'],post_hoc=True,
                failures=read(root/'failures.json'),cells=cells,process=process,source_ce_curves=curves,
                evaluation=evaluation,grouped=grouped,paired=pairs,paired_stats=paired_stats,audit=audit)


def plots(s,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'source_ce_ggn':'tab:blue','zero_zeta_geometry':'tab:orange','population':'tab:green'}
    labels={'source_ce_ggn':'Source GGN','zero_zeta_geometry':'Zero request + geometry','population':'Population score'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for axis,dim in zip(axes,[2,16]):
        for i,arm in enumerate(ARMS):
            for batch,offset,marker in [(128,-.18,'x'),(512,.18,'o')]:
                rows=[r for r in s['evaluation'] if (r['dim'],r['arm'],r['source_batch'])==(dim,arm,batch)]
                vals=[r['delta_ba_pp'] for r in rows]
                axis.scatter([i+offset+(r['seed']-2)*.035 for r in rows],vals,c=colors[arm],marker=marker,s=27)
                if vals:axis.plot([i+offset-.13,i+offset+.13],[statistics.mean(vals)]*2,c='black',lw=1.5)
        axis.set_xticks(range(3),[labels[a] for a in ARMS],rotation=15)
        axis.axhline(0,c='gray',lw=.7);axis.set_ylabel('Target BA change vs initialization (pp)')
        axis.set_title(f'd={dim}; x=batch128, circle=batch512')
    fig.savefig(out/'full-source-risk.png',dpi=180);fig.savefig(out/'full-source-risk.svg');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,dim in enumerate([2,16]):
        for j,batch in enumerate([128,512]):
            axis=axes[i,j]
            for arm in ARMS:
                for seed in range(5):
                    rows=sorted([r for r in s['source_ce_curves'] if (r['dim'],r['source_batch'],r['arm'],r['seed'])==(dim,batch,arm,seed)],key=lambda r:r['round'])
                    axis.plot([0]+[r['round'] for r in rows],[0]+[r['change_from_initial'] for r in rows],
                              c=colors[arm],alpha=.3,lw=1,label=labels[arm] if seed==0 else None)
            axis.axhline(0,c='gray',lw=.7);axis.set_title(f'd={dim}, source batch={batch}')
            axis.set_xlabel('Attempted round');axis.set_ylabel('Full-source CE change vs initial')
            axis.legend(fontsize=7)
    fig.savefig(out/'source-ce-curves.png',dpi=180);fig.savefig(out/'source-ce-curves.svg');plt.close(fig)


def fmt(x,n=4):return 'NA' if x is None else f'{x:.{n}f}'


def report(s):
    lines=['# 全源512样本诊断结果（2026-11-10）','',
           '本轮仅把源适配批次从128改为全部512个既有源样本。候选的源梯度/GGN和验收目标中的源CE都使用全源均值；没有增加硬性源风险约束。三组各自在原目标和预算下运行30轮。', '',
           f"执行代码：[`{s['execution_commit'][:7]}`](https://github.com/chx739/TQG-GDA/tree/{s['execution_commit']})；[冻结协议](../full-source-plan.json)。复用原输入、初始化、源坐标、带宽、目标批次ID和全部数值配置，新增2维数×5种子×3臂=30条轨迹。原128批次结果由已有封存预测复用。",'',
           '## 三组的定义','',
           '- 纯源GGN：全源CE，矩阵F_S+μI，源CE Armijo，无目标位移项/执行预算。',
           '- ζ=0几何控制：全源CE＋零请求位移平方损失，保留DᵀD与执行预算，每轮按自己的当前头重建边界。',
           '- 总体真实score：生产TSBA目标与验收，仅源批次扩到512，使用同一个解析总体score。',
           '- 仅改源排列的取用长度，原source-then-target随机调用不变；每轮完整源ID的前128项和全部目标ID均与旧档案一致。','',
           '## 相对初始化的结果','',
           '| 维度 | 源批次 | 方法 | ΔBA均值±样本SD（百分点） | 胜/平/负 | ΔNLL均值 |',
           '| --- | ---: | --- | ---: | --- | ---: |']
    for dim in (2,16):
        for batch in (128,512):
            for arm in ARMS:
                r=s['grouped'][dim][batch][arm]
                lines.append(f"| {dim} | {batch} | {arm} | {fmt(r['delta_ba_pp']['mean'])} ± {fmt(r['delta_ba_pp']['sd'])} | {r['wins']}/{r['ties']}/{r['losses']} | {fmt(r['delta_nll']['mean'],6)} |")
    lines+=['','BA正差较好，NLL负差较好；全部逐种子结果见[evaluation.csv](evaluation.csv)。','',
            '![结果对比](full-source-risk.png)','', '## 配对比较','',
            '| 维度 | 比较 | ΔBA均值±配对SD（百分点） | 胜/平/负 | ΔNLL均值 |',
            '| --- | --- | ---: | --- | ---: |']
    for dim in (2,16):
        for name,r in s['paired_stats'][dim].items():
            lines.append(f"| {dim} | {name} | {fmt(r['ba_difference_pp']['mean'])} ± {fmt(r['ba_difference_pp']['sd'])} | {r['wins']}/{r['ties']}/{r['losses']} | {fmt(r['nll_difference']['mean'],6)} |")
    lines+=['','population_vs_zero与zero_vs_source均指512批次；其余为同一方法512减128。差值先在同种子内计算，再跨种子汇总。完整配对表见[paired.csv](paired.csv)。','',
            '## 全源CE与过程记录','',
            '| 维度 | 方法（512批次） | 平均接受轮数 | 平均全源CE变化 | 全源CE上升轮数 |',
            '| --- | --- | ---: | ---: | ---: |']
    for dim in (2,16):
        for arm in ARMS:
            rows=[r for r in s['process'] if r['dim']==dim and r['arm']==arm]
            if rows:lines.append(f"| {dim} | {arm} | {statistics.mean(r['accepted'] for r in rows):.1f} | {statistics.mean(r['source_ce_change'] for r in rows):.6f} | {sum(r['full_source_ce_increase_rounds'] for r in rows)} / {sum(r['rounds'] for r in rows)} |")
    a=s['audit']
    lines+=['','上升轮数以1e−10为数值容差；它是记录指标，没有加入接受条件。曲线每条细线对应一个种子；128批次的全源CE在封存后用旧参数快照重新计算。','',
            '![全源CE轨迹](source-ce-curves.png)','',
            f"文件校验{a['files_verified']}项；重载{a['final_classifiers_replayed']}个最终分类器，核对{a['state_chains_verified']}轮状态链和全部源/目标ID；独立复算{a['source_ce_trials_recomputed']}次回溯候选的全源CE，最大日志差异{a['max_logged_source_ce_error']:.3g}。预测重载要求逐位一致。",'',
            '## 范围与档案','',
            '这是根据前两轮结果追加的合成机制诊断。固定最终30轮状态，未依据目标指标选轮次或更改配置。失败/异常保留并与正常完成均值分开。源批次增加使每轮源计算量增大，本轮没有等计算量主张。未运行真实图实验。', '',
            f"运行目录：`{s['run_path']}`。结果、完整逐轮日志、参数/锚点、checkpoint、代码快照和校验信息另存压缩档案，见[归档索引](archive-index.json)。原两轮档案不覆盖。",'']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    s=audit_and_summarize(args.run.resolve());args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'summary.json',s);write_json(args.output/'checkpoint-audit.json',s['audit'])
    for key in ('evaluation','process','paired','source_ce_curves'):write_csv(args.output/(key+'.csv'),s[key])
    plots(s,args.output)
    with (args.output/'report.md').open('x') as f:f.write(report(s))
    print(json.dumps(s['paired_stats'],indent=2))
