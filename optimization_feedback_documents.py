"""Create bounded-phase reports and plots from audited optimization artifacts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def fmt(v,scale=1):return f"{v['mean']*scale:.5f} ± {v['std']*scale:.5f} ({v['positive_seeds']}/5)"


def build(root):
    started=time.process_time();summary=read(root/'results_summary.json');audit=read(root/'audit.json');sa=read(root/'summary_audit.json')
    assert audit['passed'] and sa['passed'];phase=root.name
    figs=root/'figures';figs.mkdir(exist_ok=True)
    note=('Mutagenicity，0→1/2/3，五模型种子3405—3409；fit779/query196，源fit16/目标A16，B271。'
          '各快照都是原源模型的独立诊断；未重新预训练GNN，未进行连续TQG训练。'
          '±为五模型种子标准差，括号为正值种子数；不是五次独立数据集划分。'
          '开发A/B历史已观察，不能称盲测；final标签读取为0。\n\n')
    if phase=='A6':
        rows=[]
        for profile in ['OLD_H','OLD_HA','MEAN_H','MEAN_HA','LOCAL','KERNEL','KERNEL_JOINT','TEACHER_H','TEACHER_HA','TEACHER_LOCAL']:
            chosen=[next(x for x in summary['results'] if x['profile']==profile and x['task']==f'0_to_{d}') for d in [1,2,3]]
            rows.append([profile]+[fmt(x['step']['FULL'],100) for x in chosen])
        primary=table(['方法','0→1 相对CE收益%','0→2 相对CE收益%','0→3 相对CE收益%'],rows)
        attribution=[];direction=[];later=[]
        for r in summary['results']:
            if r['profile'] not in ['LOCAL','KERNEL']:continue
            attribution.append([r['task'],r['profile']]+[fmt(r['step'][m],100) for m in ['DELTA_BASE','DELTA_BASE_M','DELTA_CPL_AVG','DELTA_CRL_AVG','DELTA_FULL_PERM_M_AVG']])
            direction.append([r['task'],r['profile']]+[fmt(r['gradient'][m]) for m in ['FULL','DELTA_FULL_PERM_M_AVG','DELTA_FULL_RAND_M_AVG']])
            for record in r['later']:later.append([r['task'],r['profile'],record['snapshot'],record['nonpositive_direction'],record['nonpositive_step']])
        # Block comparisons first, equal snapshot weighting second, seed last.
        m=pd.read_csv(root/'mechanism_paired_blocks.csv')
        ms=m.groupby(['task','candidate','metric','seed','snapshot_step'])[['teacher_delta','mean_delta']].mean().groupby(['task','candidate','metric','seed']).mean().reset_index()
        ms.to_csv(root/'mechanism_paired_seeds.csv',index=False)
        mechanism=[]
        for (task,candidate,metric),group in ms.groupby(['task','candidate','metric']):
            scale=100 if metric=='R_Q' else 1
            mechanism.append([task,candidate,metric]+[f'{group[k].mean()*scale:.5f} ± {group[k].std()*scale:.5f} ({int((group[k]>0).sum())}/5)' for k in ['teacher_delta','mean_delta']])
        teachers=pd.concat([pd.read_json(root/str(s)/'teacher_prediction_metrics.jsonl',lines=True) for s in range(3405,3410)])
        ts=teachers.groupby(['task','seed','profile','variant','pool','snapshot_step'])[['nll','brier','ba','residual_mse']].mean().groupby(['task','seed','profile','variant','pool']).mean().reset_index()
        ts.to_csv(root/'teacher_prediction_seeds.csv',index=False)
        teacherrows=[]
        for (task,profile),group in ts[(ts.pool=='B')&(ts.variant==0)].groupby(['task','profile']):
            null=ts[(ts.task==task)&(ts.profile==profile)&(ts.pool=='B')&(ts.variant>0)].groupby('seed').nll.mean()
            true=group.set_index('seed').nll;delta=null-true
            teacherrows.append([task,profile,f'{true.mean():.5f} ± {true.std():.5f}',f'{delta.mean():.5f} ({int((delta>0).sum())}/5)'])
        choices=[]
        for seed in range(3405,3410):
            for p in sorted((root/str(seed)/'blocks').glob('*/selection.json')):
                d,_,snapshot,block=map(int,p.parent.name.split('_'));c=read(p)
                for route,key in [('LOCAL','local'),('KERNEL','kernel')]:
                    for variant,item in enumerate(c[key]):choices.append(dict(task=f'0_to_{d}',seed=seed,snapshot=snapshot,block=block,route=route,variant=variant,**item))
        pd.DataFrame(choices).to_csv(root/'selection_choices.csv',index=False)
        frequencies=Counter((r['route'],json.dumps({k:v for k,v in r.items() if k not in ['task','seed','snapshot','block','variant','route','cv_nll']},sort_keys=True)) for r in choices if r['variant']==0)
        write(root/'selection_frequencies.json',[dict(route=k[0],parameters=json.loads(k[1]),blocks=v) for k,v in sorted(frequencies.items())])
        report='# A6：局部选择、核读取与简单教师的最终反馈验收\n\n'+note
        report+='**结论：LOCAL、KERNEL 均为0/3任务通过完整门槛。** 不触发组合、连续200步适配或本轮新增PROTEINS复验；不是因为命令失败或计算预算用尽。\n\n'
        report+='## 问题与改动\n\nLOCAL只根据源内验证选择近邻bank，保持H/JOINT读取；KERNEL用源三折选择H/HA、带宽、正则，使用MEAN。每条算法的乱标签版本都重做源端选择。LR教师仅用源query标签训练，分别匹配H、HA和LOCAL源bank。\n\n'
        report+='## 独立一步收益\n\n`R_Q=(CE_B(源SGD后)−CE_B(源SGD+反馈后))/CE_B(更新前)`；正值为好。表中乘100显示百分比，门槛1e−4对应0.01%，不是准确率百分点。eta=.0003、beta=.1。先平均块，再平均0/50/200快照，再汇总模型种子。\n\n'+primary+'\n\n'
        report+='## 标签归因\n\n以下都是FULL减对应对照的额外相对CE收益，单位%。CPL/CRL保留BASE并仅替换标签大小的项；BASE_M匹配完整反馈范数；FULL_PERM_M重做全流程置乱并匹配完整范数。条件W分解不代表整个源端选择程序标签盲。\n\n'
        report+=table(['任务','候选','对BASE','对BASE_M','对CPL','对CRL','对全流程置乱'],attribution)+'\n\n'
        report+='LOCAL 0→3平均收益较大，但对BASE仅3/5种子改善，对BASE_M只有1/5；方向对置乱优势0.04546低于0.05，且50步快照3/5方向非正。不能据平均收益宣布稳定标签迁移。KERNEL在0→2/3的真实标签反馈弱于其BASE；这不等于BASE已被验证成新的适配方法。\n\n'
        report+='## 方向与后期稳定性\n\n'+table(['任务','候选','方向余弦','对全流程置乱优势','对等幅随机优势'],direction)+'\n\n'+table(['任务','候选','源监督快照','方向非正种子','单步收益非正种子'],later)+'\n\n'
        report+='## 相对于简单教师／MEAN\n\n'+table(['任务','候选','指标','候选−同bank/签名教师','候选−固定MEAN参照'],mechanism)+'\n\n'
        report+='R_Q配对差仍为百分比；方向为余弦差。LOCAL匹配TEACHER_LOCAL，KERNEL匹配当块选中签名的全196教师。LOCAL与MEAN_H的比较改变了bank，不是严格同bank消融。方法间此表没有额外强制等幅，不能把一步幅度优势归为查询机制优势。所有方法对null的预登记等幅对照另行保留。\n\n'
        report+='## 教师直接预测\n\n'+table(['任务','教师','B NLL（越低越好）','乱标签NLL−真标签NLL'],teacherrows)+'\n\n'
        report+='教师在同一个A16上的未投影梯度已落在V张成空间内，完整／投影数值一致，不应当作两个独立算法。教师LABEL仅表示输出相对类比例教师的偏移，不是线性源标签分解。\n\n'
        report+='教师直接预测在0→1/2上稳定胜过重训乱标签对照（各5/5种子），说明源标签并非完全无法迁移；但这没有自动转化为可靠的梯度增量。0→3的局部教师直接预测反而弱于乱标签教师（0/5），进一步提醒：方向余弦、预测NLL、参数一步收益是不同问题。此处局部教师的直接预测置乱保持所选bank，全流程重新选bank的控制另在反馈指标中报告。\n\n'
        report+='源端选择记录见selection_choices.csv：360块中LOCAL有359块选128张；KERNEL有339块选λ=1。这支持“源端验证偏好更多局部监督及强正则”的描述，不能说明该选择在目标域最优；块共享数据和模型，不能当360个独立证据。\n\n'
        report+='## 原因与下一步\n\n已排除本次路径中可审计的求解、符号、控制构造、选参标签边界和实际SGD执行不一致；没有排除所有可能的实现错误。A4源内读得更好却未变成稳定目标反馈，说明单纯调核或近邻筛源不能解决当前迁移障碍。条件关系失配、有限源监督、签名任务信息和模型状态依赖仍可能共同作用，本轮不能唯一归因。\n\n建议下一轮先明确更可检验的迁移条件或寻找跨状态稳定的签名，再另行登记实验；若研究更多独立源标签，应与骨干重训、标签预算和划分变化分开。暂不增加适配轮数、放大反馈系数或堆相似度网络。\n\n'
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for profile in ['LOCAL','KERNEL','TEACHER_H']:
            r=[next(x for x in summary['results'] if x['profile']==profile and x['task']==f'0_to_{d}') for d in [1,2,3]]
            for ax,kind,scale in [(axes[0],'gradient',1),(axes[1],'step',100)]:
                ax.errorbar([1,2,3],[x[kind]['FULL']['mean']*scale for x in r],yerr=[x[kind]['FULL']['std']*scale for x in r],marker='o',capsize=3,label=profile)
        axes[0].set_ylabel('Cosine with empirical A gradient');axes[1].set_ylabel('Extra relative CE benefit (%)')
        axes[1].axhline(.01,color='grey',linestyle=':',label='Step magnitude threshold')
    else:
        rows=[]
        alternatives={(x['task'],x['profile']):x for x in sa['alternate_A2_error_aggregation']}
        for x in summary['results']:
            alt=alternatives[x['task'],x['profile']]
            rows.append([x['task'],x['profile'],fmt(x['c_delta']),fmt(x['error_reduction'],100),f"{alt['ratio_of_seed_mean_errors']*100:.3f} ({alt['positive_seeds']}/5)",x['passed'],alt['passed']])
        report='# A2-b：目标附近的源图是否更有用\n\n'+note
        report+='**结论：没有任何局部配置达到至少2/3任务通过的要求。** 不能从“距离与困难相关”直接推出“只保留近邻就能得到可靠反馈”。\n\n'
        report+='## 实验方案\n\n保持原JOINT读取器；H/HA读取核各比较64/128张、H近邻/HA近邻/五个均匀排列，保留全196。公共带宽固定196输入；FIXED使用λ=.01，NORMALIZED使用λ=.01×n/196。只用目标A输入选bank，不用B。120目标块、6960配置；前4块共3480配置进行独立一步。\n\n'
        report+='## 全部局部配置与同数量均匀源图比较\n\n五个均匀排列先在同一块平均，再做模型种子配对。误差降低存在原协议未明确的聚合口径：主列先逐块算比例，敏感性列先在种子内平均误差再算比例。两列完整公开，不按哪列更有利决定结论。\n\n'
        report+=table(['任务','配置','余弦提升','逐块误差降低%','种子均值误差降低%','主口径通过','敏感性通过'],rows)+'\n\n'
        report+='后一口径只有0→2的H_128_NEAR_H_FIXED通过一个任务（10.295%误差下降，4/5同向）；其余任务没有共同支持。前一口径所有配置均0/3。两种口径都不支持扩大局部路线。最终候选LOCAL按源端规则进入独立A6，其0/3结论另见A6报告，不按此处目标结果挑配置。\n\n'
        report+='## 解释与边界\n\n近邻共享输入特征，不保证共享标签条件规律；删除远源图同时减少监督约束。当前结果未证明距离完全无用，也未证明更多独立源标签无用。反而说明不能把旧支持分层相关性当成局部选择的因果收益。\n\n'
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for task,d in [(f'0_to_{i}',i) for i in [1,2,3]]:
            rr=[x for x in summary['results'] if x['task']==task]
            axes[0].scatter([d]*len(rr),[x['c_delta']['mean'] for x in rr],alpha=.65)
            axes[1].scatter([d]*len(rr),[x['error_reduction']['mean']*100 for x in rr],alpha=.65)
        axes[0].set_ylabel('Cosine gain over matched uniform bank');axes[1].set_ylabel('Mean block relative error reduction (%)')
        axes[0].axhline(.05,color='grey',linestyle=':');axes[1].axhline(10,color='grey',linestyle=':')
    for ax in axes:ax.axhline(0,color='black',lw=.7);ax.set_xticks([1,2,3],['0→1','0→2','0→3']);ax.set_xlabel('Task')
    if phase=='A6':axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(figs/f'{phase}_summary.{ext}',dpi=180)
    plt.close(fig)
    report+='## 审计与限制\n\n'+f"独立逐条审计通过，聚合复核{sa['checks']}项，最大差{sa['max_absolute_error']:.3g}。普通模型副本重放{audit['counts']['ordinary_SGD_forwards']}次FULL/BASE参数步；null的系数与指标全部重放，但没有再次独立前向每一个null参数步。\n\n"
    report+=f"保留主读出未收敛{audit['counts']['finite_nonconverged_primary']}条；教师内折未收敛{audit['counts'].get('teacher_inner_not_converged',0)}条，选中教师重拟合未收敛{audit['counts'].get('teacher_refit_not_converged',0)}条。不删除失败条目，不将工程通过等同研究通过。原始未收敛标记在各块profiles/教师元数据，summary中的覆盖率没有把有限未收敛结果偷偷丢弃。\n\n"
    report+='[完整逐种子/快照指标](results.csv) · [门槛与配对汇总](results_summary.json) · [数值与权限审计](audit.json) · [独立聚合审计](summary_audit.json) · [冻结协议](protocol/method.md)\n'
    (root/'结果解读.md').write_text(report)
    (root/'run_report.md').write_text(report+'\n## Reproduction and export issues\n\n'+
        f'Execution provenance: manifest.json; checkpoints/sample IDs are sealed in each worker blocks/*/ids.json. Recompute summary: `python optimization_feedback_report.py --root <stage>/{phase}`; independently verify: `python optimization_summary_audit.py --root <stage>/{phase}`. Numeric audit uses optimization_feedback_audit.py.\n\n'+
        'The audit script had a syntax error corrected before execution. The first summary export failed on missing control_valid in step logs; fixed by joining sealed gradient validity on token/profile/arm. No fitting, labels, coefficients or outcomes were changed. Two aggregation checks passed via direct invocation because pytest is not installed. Existing invalid or finite nonconverged records are retained.\n')
    (root/'hand_off.md').write_text(f'{phase} is numerically audited and independently aggregation-audited. No candidate meets the multi-task continuation rule. Do not start continuous adaptation, combination, or new PROTEINS runs. Preserve the A2 error-aggregation ambiguity and all null/teacher results. Whole-goal completion still requires publication and objective audit.\n')
    blockids=[]
    for seed in range(3405,3410):
        for p in sorted((root/str(seed)/'blocks').glob('*/ids.json')):blockids.append(dict(path=str(p.relative_to(root)),**read(p)))
    write(root/'sample_ids.json',blockids)
    for name in ['history.jsonl','label_access.jsonl']:
        sources=sorted(root.glob('*/'+name))
        if name=='label_access.jsonl':sources+=sorted((root/'audit').glob('*_label_access.jsonl'))
        with (root/name).open('w') as output:
            for p in sources:
                for line in p.read_text().splitlines():
                    if line.strip():output.write(json.dumps(dict(log_source=str(p.relative_to(root)),**json.loads(line)),ensure_ascii=False)+'\n')
    done=read(root/'done.json');manifest=read(root/'manifest.json')
    write(root/'publication_compute_ledger.json',dict(execution_cpu_seconds_including_pilot=done['cpu_seconds_including_pilot'],
        numerical_audit_cpu_seconds=audit['cpu_seconds'],summary_cpu_seconds=summary['summary_cpu_seconds'],
        summary_audit_cpu_seconds=sa['cpu_seconds'],documents_cpu_seconds=time.process_time()-started,gpu_hours=0,
        budget_cpu_core_hours=24,worker_peak_rss_kib=max(x['peak_rss_kib'] for x in done['workers']),
        scope='CPU process measurements; imports, failed report export, orchestration, filesystem/Git publication overhead not completely metered. Do not call this exact total project CPU.'))
    write(root/'stage_3_done',dict(stage=phase,finished_at=time.time(),goal_completed=False,gpu_hours_consumed_so_far=0,
        started_at=manifest.get('created_at',manifest.get('time','see manifest and history timestamps'))))
    print(json.dumps(dict(phase=phase,reports_written=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();build(Path(a.root))
