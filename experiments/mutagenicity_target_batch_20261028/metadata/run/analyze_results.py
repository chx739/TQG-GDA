"""Descriptive postprocessing only; no graph cache or target-label reader."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN=Path(__file__).resolve().parent
STAGE=RUN/'stage3_execution'


def main():
    assert (RUN/'stage_3_done').exists(), 'Do not analyze before sealed evaluation'
    summary=json.loads((STAGE/'results_summary.json').read_text())
    diagnostic=json.loads((STAGE/'training_diagnostics.json').read_text())
    rows=[]
    names=['feedback_norm_mean','feedback_norm_sd','feedback_norm_cv','feedback_norm_p95',
           'feedback_source_ratio_median','step_seconds_median','wall_seconds']
    for domain in (1,2,3):
        for arm in ('B16','B32','B64'):
            group=[r for r in diagnostic['jobs'] if r['domain']==domain and r['arm']==arm]
            assert len(group)==5
            row=dict(task=f'0_to_{domain}',arm=arm,seeds=5)
            for name in names:
                values=[r[name] for r in group]
                assert all(v is not None for v in values)
                row[name+'_mean']=statistics.mean(values)
                row[name+'_sd']=statistics.stdev(values)
            row['calibration_calls']=sum(r['calibration_calls'] for r in group)
            row['nonconverged']=sum(r['nonconverged'] for r in group)
            row['numerical_fallbacks']=sum(r['numerical_fallbacks'] for r in group)
            rows.append(row)
    with (STAGE/'diagnostics_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    text='# 目标batch对照：结果解读\n\n'
    text+='**结论：增大目标batch让反馈幅度波动更小、目标图覆盖更充分，但未带来跨任务稳定的分类收益，计算开销明显增加。当前证据不足以支持仅靠增大目标batch解决TQG的效果问题。**\n\n'
    text+='本轮只改变目标batch16/32/64；源fit每步16、query全部196，复用同一组源预训练模型，适配200步，SGD lr0.0003、反馈权重0.1、FP64。0→1/2/3各五种子。batch16和源SGD为已核对配置、数据、模型哈希及数值代码的历史复用；32/64为30次新适配。\n\n'
    text+='B_frozen表示源模型不动，B_fit表示继续源fit监督，B16/B32/B64表示TQG的目标batch大小。\n\n'
    text+='**主指标：固定200步的balanced accuracy（%），均值±五个模型种子的样本标准差。**\n\n| 方法 | 0→1 | 0→2 | 0→3 |\n|---|---:|---:|---:|\n'
    for arm in ('B_frozen','B_fit','B16','B32','B64'):
        cells=[]
        for domain in (1,2,3):
            m=summary['metrics'][f'0_to_{domain}'][arm]['frozen' if arm=='B_frozen' else 'fixed_200']['balanced_accuracy']
            cells.append(f"{100*m['mean']:.2f} ± {100*m['sd']:.2f}")
        text+='| '+arm+' | '+' | '.join(cells)+' |\n'
    text+='\n## 增大batch是否更好？\n\n'
    conclusions={}
    for arm in ('B32','B64'):
        diffs=[];supported=[]
        for domain in (1,2,3):
            p=summary['paired'][f'0_to_{domain}']['B16'][arm]
            diffs.append(p['mean_difference_pp'])
            if p['mean_difference_pp']>0 and p['wins']>=3:supported.append(domain)
        conclusions[arm]=dict(mean_differences_pp=diffs,supported_tasks=supported,
                              preset_criterion_met=len(supported)>=2)
        text+=f"- {arm}相对batch16，在0→1/2/3分别为"+'／'.join(f'{x:+.3f}' for x in diffs)+'个百分点。'
        text+=f"{len(supported)}/3任务同时满足平均差为正、至少3/5种子获胜，"+('达到' if len(supported)>=2 else '未达到')+'本轮预设支持条件。\n'
    text+='\n这只是五种子、固定划分、固定超参数下的探索性证据，不等于统计显著性或跨数据集有效性。不得按每个任务挑最好的batch组成一个新方法结果。源验证选点是次要结果，见run_report.md。\n'
    text+='\n## CE和AUROC的变化\n\n估计对象是普通CE梯度，下面同时列出CE和AUROC的五种子均值。0→3有很小改善，其他任务没有一致改善；这些混合源CE的训练终点也不能替代离线真实梯度方向诊断。\n\n| 任务 | CE16 | CE32 | CE64 | AUROC16 | AUROC32 | AUROC64 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for domain in (1,2,3):
        cells=[]
        for metric in ('ce','auroc'):
            for arm in ('B16','B32','B64'):
                value=summary['metrics'][f'0_to_{domain}'][arm]['fixed_200'][metric]['mean']
                cells.append(f'{value:.6f}')
        text+=f'| 0→{domain} | '+' | '.join(cells)+' |\n'
    text+='\n## 反馈幅度是否更平稳？\n\n先对每个种子的200步反馈梯度范数计算CV=标准差/均值，再报告五种子的均值±样本标准差。CV低只表示沿非平稳训练轨迹的相对幅度波动小，不证明方向正确；应同时看平均范数与相对源梯度的比例。\n\n| 任务 | batch16 CV | batch32 CV | batch64 CV |\n|---|---:|---:|---:|\n'
    for domain in (1,2,3):
        cells=[]
        for arm in ('B16','B32','B64'):
            row=next(r for r in rows if r['task']==f'0_to_{domain}' and r['arm']==arm)
            cells.append(f"{row['feedback_norm_cv_mean']:.3f} ± {row['feedback_norm_cv_sd']:.3f}")
        text+=f'| 0→{domain} | '+' | '.join(cells)+' |\n'
    text+='\n## 每步计算开销\n\n表中为每个作业的训练步骤耗时中位数，再对五个种子求平均，单位秒；不含所有数据采样、验证、checkpoint和日志开销。batch16来自历史运行，机器负载差异使墙钟比较仅供参考。\n\n| 任务 | batch16 | batch32 | batch64 |\n|---|---:|---:|---:|\n'
    for domain in (1,2,3):
        cells=[]
        for arm in ('B16','B32','B64'):
            row=next(r for r in rows if r['task']==f'0_to_{domain}' and r['arm']==arm)
            cells.append(f"{row['step_seconds_median_mean']:.3f}")
        text+=f'| 0→{domain} | '+' | '.join(cells)+' |\n'
    text+='\n固定200步时，目标图使用次数含重复分别为3200/6400/12800，最多有效方向分别为16/32/64；源fit使用次数均为3200。数量、方向空间、核及校准结构可能共同变化，不能把效果单独归因于更低的采样噪声。\n'
    audit=json.loads((STAGE/'independent_audit.json').read_text())
    assert audit['passed']
    text+='\n## 实际目标图覆盖\n\n以下为各种子200步中见过的不同目标图占目标池的比例，再对五个种子取平均。重复使用次数与不同图数量分别解释；覆盖高不等于某一步覆盖整个目标梯度空间。\n\n| 任务 | batch16 | batch32 | batch64 |\n|---|---:|---:|---:|\n'
    for domain in (1,2,3):
        cells=[]
        for arm in ('B16','B32','B64'):
            values=[r['coverage_fraction'] for r in audit['target_coverage'] if r['domain']==domain and r['arm']==arm]
            cells.append(f'{100*statistics.mean(values):.2f}%')
        text+=f'| 0→{domain} | '+' | '.join(cells)+' |\n'
    text+='\n## 数值与数据边界\n\n'
    for arm in ('B16','B32','B64'):
        group=[r for r in diagnostic['jobs'] if r['arm']==arm]
        text+=f"- {arm}：{sum(r['calibration_calls'] for r in group)}次校准，{sum(r['nonconverged'] for r in group)}次未收敛，{sum(r['numerical_fallbacks'] for r in group)}次数值回退。\n"
    text+='- 独立审计记录配置仅改变目标batch、逐步fit/query序列相同、目标采样前缀一致及checkpoint来源。执行等式通过只证明按构造的反馈更新，不证明估计接近真实目标梯度。\n- 开发集此前已被历史实验观察；本轮训练全部封存后才评价开发标签，final标签未读。\n- 没有执行目标按epoch洗牌、源fit batch变化、等计算预算或额外正则搜索。\n'
    (STAGE/'结果解读.md').write_text(text)
    (STAGE/'descriptive_analysis.json').write_text(json.dumps(dict(primary=conclusions,
        diagnostic_aggregation='per seed over 200 steps, then mean/sample SD over five seeds',
        no_target_label_reader=True,no_new_training=True),indent=2)+'\n')
    for metric,title,ylabel in [('feedback_norm_cv','Feedback magnitude variation','Norm coefficient of variation'),
                                ('step_seconds_median','Per-step runtime (historical batch16)','Step seconds')]:
        fig,axes=plt.subplots(1,3,figsize=(11,3.8))
        for domain,ax in zip((1,2,3),axes):
            group=[next(r for r in rows if r['task']==f'0_to_{domain}' and r['arm']==arm) for arm in ('B16','B32','B64')]
            ax.bar(np.arange(3),[r[metric+'_mean'] for r in group],
                   yerr=[r[metric+'_sd'] for r in group],capsize=3,color=['#6b9bc1','#448aab','#df8a42'])
            ax.set_xticks(np.arange(3),['16','32','64']);ax.set_xlabel('Target batch size')
            ax.set_title(f'0 → {domain}');ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle(title);fig.tight_layout()
        for suffix in ('png','pdf'):fig.savefig(STAGE/'figures'/f'{metric}.{suffix}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(conclusions,ensure_ascii=False))


if __name__=='__main__':main()
