"""Replay supplemental checkpoints and compare every seed against frozen E2 arms."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import torch
from tsba.classifier import ParameterMap
from .archive import digest, write_json
from .controls import ARMS
from .run_controls import PARENT_ARMS, make_head
from .summarize import stats, write_csv

PAIRS = [('learned','zero_zeta_geometry'), ('population','zero_zeta_geometry'),
         ('empirical','zero_zeta_geometry'), ('zero_zeta_geometry','source_ce_ggn'),
         ('learned','source_ce_ggn')]


def read(path):return json.loads(path.read_text())


def summarize(root):
    files=read(root/'manifest.json')
    for relative,record in files.items():
        path=root/relative
        if path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:
            raise ValueError(f'archive integrity failure: {relative}')
    values=read(root/'evaluation.json');cells=[];process=[];reference_audits=[]
    checkpoints=0;chains=0;paired=0
    for directory in sorted(root.glob('d*_seed*')):
        if not (directory/'cell_summary.json').exists():continue
        cell=read(directory/'cell_summary.json');cells.append(cell)
        dim,seed=cell['dim'],cell['seed']
        inputs=torch.load(directory/'inputs.pt',weights_only=True)
        head=make_head(inputs);params=ParameterMap(head);initial=params.vector()
        schedule=torch.load(directory/'schedule.pt',weights_only=True)
        masks=read(directory/'parent_attempt_masks.json')
        reference_audits.append(dict(dim=dim,seed=seed,**read(directory/'first_round_audit.json'),
                                     parent_nonzero_attempts={a:sum(masks[a]) for a in PARENT_ARMS}))
        for arm in ARMS:
            events=torch.load(directory/arm/'rounds.pt',weights_only=True)
            previous=initial;anchors=[];trial_counts=[];alpha=[];residual=[];moves=[]
            for event,ids in zip(events,schedule):
                torch.testing.assert_close(event['before'],previous,atol=0,rtol=0)
                if not event['record']['accepted']:
                    torch.testing.assert_close(event['before'],event['after'],atol=0,rtol=0)
                previous=event['after'];chains+=1
                for key in ('source_ids','target_ids'):
                    if not torch.equal(event[key],ids[key]):raise ValueError('batch schedule changed')
                paired+=1
                if event['boundaries'] is not None:
                    b=event['boundaries'];assert torch.count_nonzero(b['zeta'])==0
                    assert len(b['input_indices'])==len(b['points'])
                    anchors.append(len(b['points']))
                record=event['record'];trials=record.get('trials',[]);trial_counts.append(len(trials))
                if record.get('solver',{}).get('relative_residual') is not None:residual.append(record['solver']['relative_residual'])
                if record['accepted']:
                    alpha.append(trials[-1]['alpha'])
                    if 'max_movement' in trials[-1]:moves.append(trials[-1]['max_movement']/inputs['sigma'])
            head.load_state_dict(torch.load(directory/arm/'classifier.pt',weights_only=True))
            torch.testing.assert_close(params.vector(),previous,atol=0,rtol=0)
            with torch.no_grad():
                torch.testing.assert_close(head(inputs['evaluation']),torch.load(directory/arm/'predictions.pt',weights_only=True),atol=0,rtol=0)
            checkpoints+=1
            summary=cell['arms'][arm]
            process.append(dict(dim=dim,seed=seed,arm=arm,**summary,
                                source_ce_change=summary['source_ce']-cell['source_initial_ce'],
                                coverage=sum(anchors)/sum(len(e['target_ids']) for e in events) if anchors else None,
                                anchors_per_round=stats(anchors),trial_counts=stats(trial_counts),accepted_alpha=stats(alpha),
                                max_solver_relative_residual=max(residual) if residual else None,
                                max_accepted_movement_sigma=max(moves) if moves else None,
                                reasons=dict(Counter(e['record']['reason'] for e in events))))
    # Exclude numerical failures from successful endpoint means; retain all raw rows.
    normal={(r['dim'],r['seed'],r['arm']) for r in process if r['stop_reason']=='round_budget' and r['rounds']==30}
    valid=[]
    for row in values:
        if row['arm'] in PARENT_ARMS or (row['dim'],row['seed'],row['arm']) in normal:valid.append(row)
    paired_rows=[];grouped={};comparison_groups={}
    for dim in [2,16]:
        grouped[dim]={};comparison_groups[dim]={}
        for arm in (*PARENT_ARMS,*ARMS):
            rows=[r for r in valid if r['dim']==dim and r['arm']==arm]
            changes=[r['delta_ba_pp'] for r in rows]
            grouped[dim][arm]=dict(n=len(rows),delta_ba_pp=stats(changes),delta_nll=stats([r['delta_nll'] for r in rows]),
                                  wins=sum(v>0 for v in changes),ties=sum(v==0 for v in changes),losses=sum(v<0 for v in changes))
        for left,right in PAIRS:
            rows=[]
            for seed in range(5):
                a=next((r for r in valid if (r['dim'],r['seed'],r['arm'])==(dim,seed,left)),None)
                b=next((r for r in valid if (r['dim'],r['seed'],r['arm'])==(dim,seed,right)),None)
                if a is None or b is None:continue
                row=dict(dim=dim,seed=seed,left=left,right=right,
                         ba_difference_pp=100*(a['final']['balanced_accuracy']-b['final']['balanced_accuracy']),
                         nll_difference=a['final']['nll']-b['final']['nll'])
                rows.append(row);paired_rows.append(row)
            differences=[r['ba_difference_pp'] for r in rows]
            comparison_groups[dim][left+' minus '+right]=dict(n=len(rows),ba_difference_pp=stats(differences),
                                                            nll_difference=stats([r['nll_difference'] for r in rows]),
                                                            wins=sum(v>0 for v in differences),ties=sum(v==0 for v in differences),losses=sum(v<0 for v in differences))
    audit=dict(files_verified=len(files),final_classifier_checkpoints_replayed=checkpoints,
               round_state_chains_verified=chains,paired_round_id_comparisons=paired,
               prediction_replay_atol=0,prediction_replay_rtol=0,first_round_audits=reference_audits)
    return dict(run_path=str(root),post_hoc=True,execution_commit=read(root/'provenance.json')['git_commit'],
                parent_integrity=read(root/'parent-integrity.json'),failures=read(root/'failures.json'),
                cells=cells,evaluation=values,process=process,grouped=grouped,paired=paired_rows,
                comparison_groups=comparison_groups,audit=audit)


def plots(s,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    names=['source_ce_ggn','zero_zeta_geometry','learned','empirical','population','zero']
    labels=['Source GGN','Zero request\n+ geometry','Learned','Empirical','Population','No update']
    for axis,dim in zip(axes,[2,16]):
        for i,arm in enumerate(names):
            rows=[r for r in s['evaluation'] if r['dim']==dim and r['arm']==arm]
            values=[r['delta_ba_pp'] for r in rows]
            axis.scatter([i+(r['seed']-2)*.045 for r in rows],values,s=25)
            if values:axis.plot([i-.18,i+.18],[statistics.mean(values)]*2,c='black')
        axis.axhline(0,c='gray',lw=.7);axis.set_xticks(range(6),labels,rotation=25,ha='right')
        axis.set_title(f'd={dim}; paired seeds, post-hoc controls')
        axis.set_ylabel('Target BA change vs initialization (pp)')
    fig.savefig(out/'controls-risk.png',dpi=180);fig.savefig(out/'controls-risk.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    pairs=PAIRS[:4]
    labels=['Learned − zero request','Population − zero request','Empirical − zero request','Zero request − source GGN']
    for axis,dim in zip(axes,[2,16]):
        for i,(a,b) in enumerate(pairs):
            rows=[r for r in s['paired'] if r['dim']==dim and r['left']==a and r['right']==b]
            vals=[r['ba_difference_pp'] for r in rows]
            axis.scatter([i+(r['seed']-2)*.045 for r in rows],vals,s=25)
            if vals:axis.plot([i-.18,i+.18],[statistics.mean(vals)]*2,c='black')
        axis.axhline(0,c='gray',lw=.7);axis.set_xticks(range(4),labels,rotation=25,ha='right')
        axis.set_title(f'd={dim}');axis.set_ylabel('Paired BA difference (pp)')
    fig.savefig(out/'paired-differences.png',dpi=180);fig.savefig(out/'paired-differences.svg');plt.close(fig)


def fmt(x,n=4):return 'NA' if x is None else f'{x:.{n}f}'


def report(s):
    lines=['# E2 补充对照结果（2026-11-09）','',
           '**这是查看首轮结果后追加的诊断，不是新的盲测。** 原40条轨迹保留，新增Source-CE-GGN与zero_zeta_geometry各10条，共20条；输入、源初始化、坐标、带宽、种子、逐轮ID和30次尝试预算均与原档案匹配。真实图数据和标签没有读取。','',
           f"执行代码：[`{s['execution_commit'][:7]}`](https://github.com/chx739/TQG-GDA/tree/{s['execution_commit']})。冻结协议见[controls-plan.json](../controls-plan.json)；原报告见[首轮结果](../e1-e2-20261109/report.md)。",'',
           '## 两组对照的含义','',
           '- Source-CE-GGN：仅源CE，A=F_S+μI，rhs=−g_S；同阻尼和Armijo预算，没有位移损失或目标执行预算。',
           '- zero_zeta_geometry：每轮在自己的当前头上重建锚点，保留DᵀD、位移平方损失和执行预算，强制ζ=0仍求解；空锚点跳过。它不是原zero-score跳过更新臂。',
           '- 首轮参考、源批次和矩阵项逐项核验。轨迹分开后锚点/矩阵也随状态变化；长期配对结果是完整消融比较，不是每轮共享矩阵的单步因果分解。','',
           '## 六臂相对源初始化','',
           '| 维度 | 臂 | ΔBA均值±样本SD（百分点） | 胜/平/负 | ΔNLL均值 |',
           '| --- | --- | ---: | --- | ---: |']
    for dim in [2,16]:
        for arm in ['source_ce_ggn','zero_zeta_geometry','learned','empirical','population','zero']:
            r=s['grouped'][dim][arm]
            lines.append(f"| {dim} | {arm} | {fmt(r['delta_ba_pp']['mean'])} ± {fmt(r['delta_ba_pp']['sd'])} | {r['wins']}/{r['ties']}/{r['losses']} | {fmt(r['delta_nll']['mean'],6)} |")
    lines+=['','BA正差较好，NLL负差较好。原四臂由封存logits重新计算，未重训或重跑适配。完整逐种子结果见[evaluation.csv](evaluation.csv)。','',
            '![六臂结果](controls-risk.png)','', '## 关键配对差','',
            '| 维度 | 左臂 − 右臂 | ΔBA均值±配对SD（百分点） | 胜/平/负 | ΔNLL均值 |',
            '| --- | --- | ---: | --- | ---: |']
    for dim in [2,16]:
        for pair,r in s['comparison_groups'][dim].items():
            lines.append(f"| {dim} | {pair} | {fmt(r['ba_difference_pp']['mean'])} ± {fmt(r['ba_difference_pp']['sd'])} | {r['wins']}/{r['ties']}/{r['losses']} | {fmt(r['nll_difference']['mean'],6)} |")
    lines+=['','![配对差](paired-differences.png)','',
            '差值先在同一种子内计算，再跨5个种子汇总；没有把轮次或锚点当作独立重复。完整配对数值见[paired.csv](paired.csv)。','',
            '## 过程与归档','',
            '| 维度 | 臂 | 平均接受轮数 | 平均源训练CE变化 | 平均锚点覆盖 | 平均回溯尝试次数/轮 |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
    for dim in [2,16]:
        for arm in ARMS:
            rows=[r for r in s['process'] if r['dim']==dim and r['arm']==arm]
            if not rows:continue
            coverage=[r['coverage'] for r in rows if r['coverage'] is not None]
            lines.append(f"| {dim} | {arm} | {fmt(statistics.mean(r['accepted'] for r in rows),1)} | {fmt(statistics.mean(r['source_ce_change'] for r in rows),6)} | {fmt(statistics.mean(coverage) if coverage else None)} | {fmt(statistics.mean(r['trial_counts']['mean'] for r in rows),2)} |")
    audit=s['audit']
    lines+=['',f"执行失败记录：{len(s['failures'])}。已复核{audit['files_verified']}个文件、{audit['final_classifier_checkpoints_replayed']}个最终头、{audit['round_state_chains_verified']}轮状态链及批次，重载预测逐位相同。首轮矩阵/参考检查见[checkpoint-audit.json](checkpoint-audit.json)。",'',
            '源GGN没有边界执行预算，不能称与TSBA等计算量或只差一个score。固定30轮也不代表各方法均已收敛。当前合成设置的结果不能推广为真实图GDA结论，也不用于挑选或改写首轮checkpoint。','',
            f"本地run目录：`{s['run_path']}`。完整日志、逐轮参数/锚点和checkpoint保存在本地目录及独立压缩档案；GitHub发布报告、表格与[归档索引](archive-index.json)，不提交模型运行二进制。",'']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    s=summarize(args.run.resolve());args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'summary.json',s);write_json(args.output/'checkpoint-audit.json',s['audit'])
    for name in ['evaluation','paired','process']:write_csv(args.output/(name+'.csv'),s[name])
    plots(s,args.output)
    with (args.output/'report.md').open('x') as f:f.write(report(s))
    print(json.dumps(s['comparison_groups'],indent=2))


if __name__=='__main__':main()
