"""Render sealed R2 evidence as a report, full CSV grids and scientific figures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.common.statistics import write_csv; from experiments.mutagenicity.r1_20261111.src.evaluate import seed_summary
from experiments.common.io import digest
from experiments.mutagenicity.r2_20261111.src.common import load, verify, write_json


def fmt(x,digits=5):return 'NA' if x is None else f'{x:.{digits}f}'
def ms(x,scale=1.,digits=5):
    return 'NA' if x['mean'] is None else f"{x['mean']*scale:.{digits}f} ± {x['sample_sd']*scale:.{digits}f}"


def publish(root,out,archive):
    root=Path(root);out=Path(out);verify(root,'OBS');verify(root,'FINAL')
    out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    plan=json.loads((root/'execution-plan.json').read_text());completion=json.loads((root/'completion.json').read_text())
    summaries=json.loads((root/'evaluation/summary.json').read_text());metrics=json.loads((root/'evaluation/metrics.json').read_text())
    mechanisms=json.loads((root/'evaluation/mechanisms.json').read_text())
    for name in ('metrics.csv','paired-comparisons.csv','summary.csv','mechanisms.csv','first-round.csv','prior-target-reference.json'):
        shutil.copyfile(root/'evaluation'/name,out/name)
    shutil.copyfile(root/'execution-plan.json',out/'execution-plan.json')
    shutil.copyfile(root/'completion.json',out/'completion.json')
    static=[];curves=[]
    for seed in plan['seeds']:
        for domain in plan['domains']:
            for b in plan['bandwidth_multipliers']:
                directory=root/f'scores/seed{seed}_domain{domain}/b{b}'
                history=json.loads((directory/'history.json').read_text())
                curves.extend(dict(seed=seed,domain=domain,bandwidth=b,updates=9*(r['epoch']+1),**r) for r in history)
                for updates in plan['score_updates']:
                    s=load(directory/f'static/u{updates:05d}.pt')
                    row=dict(seed=seed,domain=domain,bandwidth=b,updates=updates,
                        primary_noise_loss=s['primary_noise_loss'],second_noise_loss=s['second_noise_loss'],
                        adjacent_request_rmse_sigma=s['adjacent_request_rmse_sigma'])
                    row.update({stratum+'_'+k:v for stratum in ('all','near_zero','away_zero') for k,v in s[stratum].items()})
                    static.append(row)
    write_csv(out/'score-endpoints.csv',static);write_csv(out/'score-curves.csv',curves)
    # Every summary first averages attempts within one trajectory, then seeds.
    grouped=defaultdict(list)
    for r in mechanisms:grouped[tuple(r[k] for k in ('seed','domain','kind','bandwidth','updates','lam','sign'))].append(r)
    mech=[]
    fields=('anchors','request_rms_sigma','request_clipping_rate','source_proposal_norm','target_proposal_norm',
        'target_source_norm_ratio','proposal_cosine','actual_margin_rms_sigma','actual_margin_max_sigma',
        'executed_target_margin_rms_sigma','alpha','current_score_request_rmse_sigma')
    for key,rows in sorted(grouped.items()):
        row=dict(zip(('seed','domain','kind','bandwidth','updates','lam','sign'),key))
        row.update(attempts=len(rows),accepted=sum(r['accepted'] for r in rows),accepted_fraction_of_30=sum(r['accepted'] for r in rows)/30)
        for field in fields:
            values=[r[field] for r in rows if r.get(field) is not None]
            row[field]=float(np.mean(values)) if values else None;row[field+'_attempts']=len(values)
        mech.append(row)
    write_csv(out/'trajectory-mechanisms.csv',mech)
    for field in ('executed_target_margin_rms_sigma','actual_margin_rms_sigma'):
        fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
        matrices=[]
        for d in plan['domains']:
            for b in plan['bandwidth_multipliers']:
                matrix=np.full((4,3),np.nan)
                for i,u in enumerate(plan['score_updates']):
                    for j,lam in enumerate(plan['lambdas']):
                        rows=[r for r in mech if (r['domain'],r['bandwidth'],r['updates'],r['lam'],r['kind'],r['sign'])==(d,b,u,lam,'learned',1)]
                        value=seed_summary([r[field] for r in rows])['mean']
                        if value is not None:matrix[i,j]=value
                matrices.append((d,b,matrix))
        vmax=max((float(np.nanmax(m)) for _,_,m in matrices),default=1.) or 1.
        for ax,(d,b,matrix) in zip(axes.flat,matrices):
            im=ax.imshow(matrix,cmap='viridis',vmin=0,vmax=vmax,aspect='auto')
            ax.set(xticks=range(3),xticklabels=plan['lambdas'],yticks=range(4),yticklabels=plan['score_updates'],
                xlabel='Boundary weight lambda',ylabel='Score updates',title=f'0 -> {d}, h = {b} sigma0')
            for i in range(4):
                for j in range(3):ax.text(j,i,f'{matrix[i,j]:.5f}',ha='center',va='center',fontsize=9,color='white' if matrix[i,j]<.6*vmax else 'black')
        fig.colorbar(im,ax=axes,label=field+' (attempt mean, then 5-seed mean)')
        fig.savefig(out/'figures'/f'{field}.png',dpi=160);plt.close(fig)
    # Four primary heatmaps, including the required original-head comparison.
    for measure in ('delta_nll_vs_geometry','delta_nll_vs_original','delta_balanced_accuracy_vs_geometry'):
        fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
        subset=[r for r in summaries if r['attempt']==30 and r['pool'].startswith('B') and r['kind']=='learned']
        values=[r[measure]['mean'] for r in subset if r[measure]['mean'] is not None];vmax=max(map(abs,values),default=1.) or 1.
        for i,d in enumerate(plan['domains']):
            for j,b in enumerate(plan['bandwidth_multipliers']):
                matrix=np.full((4,3),np.nan)
                for u,updates in enumerate(plan['score_updates']):
                    for l,lam in enumerate(plan['lambdas']):
                        row=next(r for r in subset if (r['domain'],r['bandwidth'],r['updates'],r['lam'])==(d,b,updates,lam))
                        if row[measure]['mean'] is not None:matrix[u,l]=row[measure]['mean']
                ax=axes[i,j];im=ax.imshow(matrix,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
                ax.set(xticks=range(3),xticklabels=plan['lambdas'],yticks=range(4),yticklabels=plan['score_updates'],
                    xlabel='Boundary weight lambda',ylabel='Score updates',title=f'0 -> {d}, h = {b} sigma0')
                for u in range(4):
                    for l in range(3):ax.text(l,u,f'{matrix[u,l]:+.4f}',ha='center',va='center',fontsize=9)
        fig.colorbar(im,ax=axes,label=measure+' (5-seed mean)');fig.savefig(out/'figures'/f'{measure}.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for i,d in enumerate(plan['domains']):
        for j,b in enumerate(plan['bandwidth_multipliers']):
            ax=axes[i,j]
            for seed in plan['seeds']:
                rows=[r for r in curves if (r['seed'],r['domain'],r['bandwidth'])==(seed,d,b)]
                ax.plot([r['updates'] for r in rows],[r['noise_validation_loss'] for r in rows],label=str(seed))
            for u in plan['score_updates']:ax.axvline(u,color='grey',alpha=.2,lw=.8)
            ax.set(title=f'0 -> {d}, h = {b} sigma0',xlabel='Actual optimizer updates',ylabel='Fixed-noise DSM validation loss');ax.legend(fontsize=7)
    fig.savefig(out/'figures/score-curves.png',dpi=160);plt.close(fig)
    # Main report: every requested cell, no B-based winner selection.
    lines=['# Mutagenicity R2：完整30次适配结果','',
        '固定 R1 的五份16维 GNN 表示，比较 score 精确更新预算900/1800/3600/7200与边界权重1/10/100；两个任务、两个带宽全部执行。',
        '',f"主轨迹 **{completion['trajectories']}条**，数值失败 **{completion['trajectory_failures']}条**；实际记录 **{completion['mechanism_attempts']}次尝试**。主终点为第30次尝试，主指标为目标 B271 的 NLL。所有均值±SD以五个种子为单位。",'',
        '## 1. score 是否学得更好','',
        '经验 KDE 是相同有限目标池、相同带宽的精确平滑密度参考，不是总体真实 score。两个独立固定噪声上的DSM只用于诊断，分类适配使用精确端点。完整逐种子值见 [score-endpoints.csv](score-endpoints.csv)。',
        '', '| 任务 | h/σ0 | 更新 | 主噪声DSM | 第二噪声DSM | 法向RMSE·σ0 | 请求RMSE/σ0 | 非近零符号一致率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in plan['domains']:
        for b in plan['bandwidth_multipliers']:
            for u in plan['score_updates']:
                rows=[r for r in static if (r['domain'],r['bandwidth'],r['updates'])==(d,b,u)]
                vals=[ms(seed_summary([r[k] for r in rows]),digits=4) for k in ('primary_noise_loss','second_noise_loss','all_normal_rmse_sigma','all_request_rmse_sigma','away_zero_sign_agreement')]
                lines.append(f'| 0→{d} | {b} | {u} | '+' | '.join(vals)+' |')
    lines+=['','![完整DSM曲线](figures/score-curves.png)','',
        '## 2. 完整30次尝试的目标分类结果','',
        'Δ均为正常组减对应对照。NLL差值<0表示改善；BA差值>0表示改善。优于几何源对照但仍差于原始源头，只表示增量，不能称整体改善。',
        '', '| 任务 | 对照 | NLL | BA（%） |', '|---|---|---:|---:|']
    for d in plan['domains']:
        for label,attempt,lam in [('原始源头',0,1),*[(f'几何源 λ={l}',30,l) for l in plan['lambdas']]]:
            rows=[r for r in metrics if r['kind']=='geometry' and r['domain']==d and r['pool']==f'B{d}' and r['attempt']==attempt and r['lam']==lam]
            rows.sort(key=lambda r:r['seed'])
            lines.append(f'| 0→{d} | {label} | '+ms(seed_summary([r.get('nll') for r in rows]))+' | '+ms(seed_summary([r.get('balanced_accuracy') for r in rows]),100,2)+' |')
    lines += [
        '', '![正常减几何源对照NLL](figures/delta_nll_vs_geometry.png)','',
        '![正常减原始源头NLL](figures/delta_nll_vs_original.png)','']
    for d in plan['domains']:
        for b in plan['bandwidth_multipliers']:
            lines += [f'### 0→{d}，h={b}σ0','',
                '| 估计器/更新 | λ | NLL | ΔNLL vs 几何源 | ΔNLL vs 原始源 | ΔNLL vs 反向 | BA（%） |',
                '|---|---:|---:|---:|---:|---:|---:|']
            rows=[r for r in summaries if r['attempt']==30 and r['pool']==f'B{d}' and r['domain']==d and r['bandwidth']==b]
            rows.sort(key=lambda r:(r['kind']=='kde',r['updates'],r['lam']))
            for r in rows:
                label='KDE' if r['kind']=='kde' else str(r['updates'])
                vals=[ms(r[k],digits=5) for k in ('nll','delta_nll_vs_geometry','delta_nll_vs_original','delta_nll_vs_reverse')]
                lines.append(f"| {label} | {r['lam']} | "+' | '.join(vals)+' | '+ms(r['balanced_accuracy'],100,2)+' |')
            lines.append('')
    lines += ['## 3. 密度请求执行量与源侧稳定性','',
        '每条正常轨迹保存每轮 δS、δT、D、矩阵哈希、求解与回溯记录；δS+δT 与生产候选数值核验。参数范数比只是推动量描述，不是分类收益比例。',
        '', '下图第一项为已接受步长下的密度分量 `αDδT/σ0` 的一阶RMS；第二项为真实前向计算的总归一化margin变化RMS。后者同时含源项、密度项和非线性效应，二者均不是重新定位后的精确边界距离。',
        '', '![已接受密度分量的一阶RMS](figures/executed_target_margin_rms_sigma.png)','',
        '![真实总margin变化RMS](figures/actual_margin_rms_sigma.png)',
        '', '完整逐轨迹统计见 [trajectory-mechanisms.csv](trajectory-mechanisms.csv)，逐轮值见 [mechanisms.csv](mechanisms.csv)。每个种子的尝试均值先单独计算，不能把18,900轮当作独立重复。源fit/validation/diagnostic的全部NLL、BA及差值见 [summary.csv](summary.csv) 和 [paired-comparisons.csv](paired-comparisons.csv)。',
        '', '首轮共同小步是旁路，未接入任何主轨迹；共同α按同种子/任务/λ下所有估计器/带宽/预算和方向确定。对应全部评价见 [first-round.csv](first-round.csv)。',
        '', '## 4. 解释边界与复现','',
        '- B是已有开发集；本轮没有用B挑选score端点、中间轮、种子或新增配置。final标签关闭。',
        '- 本轮源/目标batch均为128，边界逐轮刷新；R1b为全源779、固定旧64查询单步，两轮差异不能归因于训练步数一个因素。',
        '- 原始源头是共同起点。旧目标监督参考仅用于报告对照，见 [prior-target-reference.json](prior-target-reference.json)；没有新增目标监督训练。',
        '- 所有模型、预测、无标签诊断先封OBS，再读取B。拒绝/跳过计入30次尝试；失败不替换种子。',
        '', f"执行代码：`{json.loads((root/'provenance.json').read_text())['git_commit']}`。",'',
        f"完整档案：`{archive['archive_path']}`；SHA256 `{archive['archive_sha256']}`；逐成员核验{archive['files_verified']}个文件。",'',
        f"OBS：`{json.loads((root/'seals/OBS.json').read_text())['digest']}`。",'',
        '执行前协议：[r2-protocol.md](../r2-protocol.md)。原始结果：[metrics.csv](metrics.csv)；完整配对网格：[paired-comparisons.csv](paired-comparisons.csv)。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'archive.json',{k:v for k,v in archive.items() if k!='member_manifest'})
    write_json(out/'report-provenance.json',dict(source_root=str(root),OBS=digest(root/'seals/OBS.json'),FINAL=digest(root/'seals/FINAL.json'),
        renderer_sha256=digest(Path(__file__)),note='Read-only rendering after FINAL; does not change predictions or select a configuration'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--archive-record',type=Path,required=True)
    args=p.parse_args();publish(args.root,args.output,json.loads(args.archive_record.read_text()))
