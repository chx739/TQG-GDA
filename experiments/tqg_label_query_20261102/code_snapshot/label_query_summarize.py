"""Paired model-seed aggregation for label attribution and D7-H, frozen gates."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time
import numpy as np
from p0_summarize import mean,deviation,balanced_snapshot_mean,export_csv,save,load_jsonl


def average(rows,key):return mean([r[key] for r in rows])
def snap(rows,key):return balanced_snapshot_mean(rows,key)


def summarize(stage):
    s=Path(stage);manifest=json.loads((s/'experiment_manifest.json').read_text());settings=manifest['settings']
    files={'label_gradient_blocks':6120,'label_step_blocks':3420,'label_components':360,'query_size_blocks':6000,'query_step_blocks':3000}
    allrows={name:[] for name in files}
    for seed in range(3405,3410):
        out=s/'logs'/str(seed);done=json.loads((out/'done.json').read_text());assert done['blocks']==72 and done['git_commit']==manifest['git_commit']
        for name in files:allrows[name]+=load_jsonl(out/(name+'.jsonl'))
    for name,rows in allrows.items():assert len(rows)==files[name],(name,len(rows));export_csv(s/(name+'.csv'),rows)
    groups=defaultdict(dict)
    for r in allrows['label_gradient_blocks']:groups[(r['task'],r['seed'],r['snapshot_step'],r['block_id'])][r['arm']]=r
    blocks=[]
    for (task,seed,t,b),arms in groups.items():
        f=arms['FULL'];p=mean([arms[f'CPL-{i}']['c_A'] for i in [1,2,3]]);rand=mean([arms[f'CRL-{i}']['c_A'] for i in [1,2,3]])
        valid=f['c_A'] is not None and all(arms[f'{kind}-{i}']['c_A'] is not None for kind in ['CPL','CRL'] for i in [1,2,3])
        blocks.append(dict(task=task,seed=seed,snapshot_step=t,block_id=b,c_FULL=f['c_A'],c_BASE=arms['BASE']['c_A'],
            c_LABEL=arms['LABEL']['c_A'],c_PRED=arms['PRED']['c_A'],c_COMPOSITE_PERM=p,c_COMPOSITE_RAND=rand,
            advantage_PERM=f['c_A']-p if valid else None,advantage_RAND=f['c_A']-rand if valid else None,coverage=float(valid)))
    stepgroups=defaultdict(dict)
    for r in allrows['label_step_blocks']:stepgroups[(r['task'],r['seed'],r['snapshot_step'],r['block_id'])][r['arm']]=r
    steps=[]
    for (task,seed,t,b),arms in stepgroups.items():
        base=arms['S']['base_CE_B'];ce=lambda a:arms[a]['CE_B'];full=ce('B')
        valid=all(arms[f'{kind}-{i}{suffix}']['valid_comparison'] for kind in ['PL','RL'] for i in [1,2,3] for suffix in ['','-M'])
        steps.append(dict(task=task,seed=seed,snapshot_step=t,block_id=b,FULL_extra=(ce('S')-full)/base,
            BASE_extra=(ce('S')-ce('A'))/base,label_increment=(ce('A')-full)/base,
            advantage_PERM=(mean([ce(f'PL-{i}') for i in [1,2,3]])-full)/base if valid else None,
            advantage_RAND=(mean([ce(f'RL-{i}') for i in [1,2,3]])-full)/base if valid else None,
            advantage_PERM_TOTAL_M=(mean([ce(f'PL-{i}-M') for i in [1,2,3]])-full)/base if valid else None,
            advantage_RAND_TOTAL_M=(mean([ce(f'RL-{i}-M') for i in [1,2,3]])-full)/base if valid else None,
            BASE_M_extra=(ce('S')-ce('A-M'))/base,oracle_extra=(ce('S')-ce('ORACLE-M'))/base,
            pure_LABEL_extra=(ce('S')-ce('LABEL'))/base,pure_PRED_extra=(ce('S')-ce('PRED'))/base,coverage=float(valid)))
    labelseed=[];stepseed=[];snapshots=[]
    gkeys=[k for k in blocks[0] if k not in ['task','seed','snapshot_step','block_id']]
    skeys=[k for k in steps[0] if k not in ['task','seed','snapshot_step','block_id']]
    for task in ['0_to_1','0_to_2','0_to_3']:
        for seed in range(3405,3410):
            g=[r for r in blocks if r['task']==task and r['seed']==seed];st=[r for r in steps if r['task']==task and r['seed']==seed]
            labelseed.append(dict(task=task,seed=seed,**{k:snap(g,k) for k in gkeys}))
            stepseed.append(dict(task=task,seed=seed,**{k:snap(st,k) for k in skeys}))
            for t in [0,50,200]:
                gg=[r for r in g if r['snapshot_step']==t];ss=[r for r in st if r['snapshot_step']==t]
                snapshots.append(dict(task=task,seed=seed,snapshot_step=t,**{k:average(gg,k) for k in gkeys},
                    **{k:average(ss,k) for k in skeys if k not in ['coverage','advantage_PERM','advantage_RAND']},
                    step_advantage_PERM=average(ss,'advantage_PERM'),step_advantage_RAND=average(ss,'advantage_RAND')))
    export_csv(s/'label_direction_seed_summary.csv',labelseed);export_csv(s/'label_step_seed_summary.csv',stepseed)
    export_csv(s/'label_snapshot_seed_summary.csv',snapshots);export_csv(s/'label_direction_paired_blocks.csv',blocks);export_csv(s/'label_step_paired_blocks.csv',steps)
    tasks=[]
    for task in ['0_to_1','0_to_2','0_to_3']:
        g=[r for r in labelseed if r['task']==task];st=[r for r in stepseed if r['task']==task]
        r=dict(task=task,**{k:average(g,k) for k in gkeys},**{'step_'+k:average(st,k) for k in skeys})
        r['direction_PERM_wins']=sum(x['advantage_PERM'] is not None and x['advantage_PERM']>0 for x in g)
        r['direction_RAND_wins']=sum(x['advantage_RAND'] is not None and x['advantage_RAND']>0 for x in g)
        r['direction_passed']=r['c_FULL'] is not None and r['c_FULL']>0 and r['advantage_PERM'] is not None and r['advantage_PERM']>=.05 and r['advantage_RAND'] is not None and r['advantage_RAND']>=.05 and r['direction_PERM_wins']>=3 and r['direction_RAND_wins']>=3 and r['coverage']>=.8
        win_keys=['FULL_extra','label_increment','advantage_PERM','advantage_RAND','advantage_PERM_TOTAL_M','advantage_RAND_TOTAL_M']
        for key in win_keys:r[key+'_wins']=sum(x[key] is not None and x[key]>0 for x in st)
        r['step_passed']=r['step_FULL_extra']>=1e-4 and r['step_coverage']>=.8 and all(r['step_'+k] is not None and r['step_'+k]>0 and r[k+'_wins']>=3 for k in win_keys)
        r['joint_passed']=r['direction_passed'] and r['step_passed'];tasks.append(r)
    # D7: first average five nested input permutations within each target block, then blocks within seed.
    qgroups=defaultdict(list)
    for r in allrows['query_size_blocks']:qgroups[(r['task'],r['seed'],r['block_id'],r['n'],r['convention'])].append(r)
    qblocks=[]
    for (task,seed,b,n,conv),rs in qgroups.items():
        assert len(rs)==5
        coefficients=np.array([r['coefficients'] for r in rs]);reference=np.array(next(r['coefficients'] for r in allrows['query_size_blocks'] if r['task']==task and r['seed']==seed and r['block_id']==b and r['n']==196 and r['convention']==conv))
        norm=float(reference@reference);variance=float(((coefficients-coefficients.mean(0))**2).sum(1).mean())
        qblocks.append(dict(task=task,seed=seed,block_id=b,n=n,convention=conv,c_A=mean([r['c_A'] for r in rs]),e_A=mean([r['e_A'] for r in rs]),
            c_B=mean([r['c_B'] for r in rs]),norm_ratio=mean([r['norm_ratio'] for r in rs]),
            cosine_sample_SD=deviation([r['c_A'] for r in rs]),relative_vector_dispersion=variance/norm if norm>1e-24 else None,
            coverage=sum(r['c_A'] is not None and r['e_A'] is not None for r in rs)/5,
            nonconverged_fraction=sum(not r['solver']['converged'] for r in rs)/5))
    qs_groups=defaultdict(list)
    for r in allrows['query_step_blocks']:qs_groups[(r['task'],r['seed'],r['block_id'],r['n'],r['convention'])].append(r)
    qstepblocks={key:mean([r['R_Q'] for r in rs]) for key,rs in qs_groups.items()}
    qkeys=['c_A','e_A','c_B','norm_ratio','cosine_sample_SD','relative_vector_dispersion','coverage','nonconverged_fraction']
    qseed=[]
    for task in ['0_to_1','0_to_2','0_to_3']:
        for seed in range(3405,3410):
            for conv in ['FIXED','NORMALIZED']:
                for n in [16,32,64,128,196]:
                    rs=[r for r in qblocks if (r['task'],r['seed'],r['convention'],r['n'])==(task,seed,conv,n)]
                    qseed.append(dict(task=task,seed=seed,convention=conv,n=n,**{k:average(rs,k) for k in qkeys},
                        R_Q=mean([v for k,v in qstepblocks.items() if (k[0],k[1],k[3],k[4])==(task,seed,n,conv)])))
    export_csv(s/'query_size_seed_summary.csv',qseed);export_csv(s/'query_size_paired_blocks.csv',qblocks)
    curves=[];judgments=[]
    for task in ['0_to_1','0_to_2','0_to_3']:
        for conv in ['FIXED','NORMALIZED']:
            for n in [16,32,64,128,196]:
                rs=[r for r in qseed if (r['task'],r['convention'],r['n'])==(task,conv,n)]
                curves.append(dict(task=task,convention=conv,n=n,**{k:average(rs,k) for k in qkeys+['R_Q']},c_A_SD=deviation([r['c_A'] for r in rs])))
            small=next(r for r in curves if (r['task'],r['convention'],r['n'])==(task,conv,16));large=next(r for r in curves if (r['task'],r['convention'],r['n'])==(task,conv,196))
            seedwins=0;paired=[]
            for seed in range(3405,3410):
                a=next(r for r in qseed if (r['task'],r['seed'],r['convention'],r['n'])==(task,seed,conv,16));b=next(r for r in qseed if (r['task'],r['seed'],r['convention'],r['n'])==(task,seed,conv,196))
                dc=b['c_A']-a['c_A'];de=(a['e_A']-b['e_A'])/a['e_A'];seedwins+=dc>0 and de>0
                paired.append(dict(seed=seed,cosine_gain=dc,relative_error_reduction=de))
            gain=large['c_A']-small['c_A'];drop=(small['e_A']-large['e_A'])/small['e_A']
            judgments.append(dict(task=task,convention=conv,c16=small['c_A'],c196=large['c_A'],cosine_gain=gain,
                e16=small['e_A'],e196=large['e_A'],relative_error_reduction=drop,paired_seed_wins=seedwins,
                paired_seed_differences=paired,passed=gain>=.03 and drop>=.1 and seedwins>=3 and min(small['coverage'],large['coverage'])>=.8))
    query_supported=sum(all(r['passed'] for r in judgments if r['task']==task) for task in ['0_to_1','0_to_2','0_to_3'])>=2
    export_csv(s/'query_curve_summary.csv',curves)
    # FULL and n196 must match prior H at the same snapshots, not the previous mixed-snapshot mean.
    prior={}
    for seed in range(3405,3410):
        for r in load_jsonl(Path(settings['h_stage'])/'logs'/str(seed)/'gradient_blocks.jsonl'):
            if r['arm']=='TRUE':prior[(r['task'],r['seed'],r['snapshot_step'],r['block_id'])]=r
    errors=[]
    for r in allrows['label_gradient_blocks']:
        if r['arm']=='FULL':
            old=prior[(r['task'],r['seed'],r['snapshot_step'],r['block_id'])];errors.extend(abs(r[k]-old[k]) for k in ['c_A','c_B','e_A','gradient_norm'])
    for r in allrows['query_size_blocks']:
        if r['n']==196:
            old=prior[(r['task'],r['seed'],0,r['block_id'])];errors.extend(abs(r[k]-old[k]) for k in ['c_A','c_B','e_A','gradient_norm'])
    assert max(errors)<1e-10,max(errors)
    physical=sum(not r['physical_solve_reused'] for r in allrows['query_size_blocks']);physicalsteps=sum(not r['physical_step_reused'] for r in allrows['query_step_blocks'])
    assert physical==4920 and physicalsteps==2460
    summary=dict(status='completed',label_attribution=tasks,label_supported=sum(r['joint_passed'] for r in tasks)>=2,
        query_quantity=judgments,query_quantity_supported=query_supported,query_curves=curves,
        counts=dict(files,label_blocks=360,query_blocks=120,query_physical_solves=physical,query_physical_steps=physicalsteps),
        FULL_and_n196_replay_max_error=max(errors),max_label_additivity_error=max(r['additivity_error'] for r in allrows['label_components']),
        query_nonconverged_logical=sum(not r['solver']['converged'] for r in allrows['query_size_blocks']),
        duplicate_invariance_passed=json.loads((s/'D0/duplicate_invariance.json').read_text())['passed'],final_labels_read=0,new_training_jobs=0)
    save(s/'results_summary.json',summary)
    results=[]
    for name,rows,key in [('LABEL_DIRECTION',labelseed,'c_FULL'),('LABEL_STEP',stepseed,'FULL_extra'),('D7_H_A',qseed,'c_A')]:
        for r in rows:results.append(dict(experiment_id=name,git_commit=manifest['git_commit'],plan_sha256=manifest['plan_sha256'],
            execution_protocol_sha256=manifest['execution_protocol_sha256'],config_name=settings['config_name'],hardware=manifest['hardware'],gpu_hours=0,event_flags='clean',metric_name=key,metric_value=r[key],**r))
    export_csv(s/'results.csv',results)
    plots(s,tasks,curves)
    text=render(summary,manifest)
    for n in ['结果解读.md','run_report.md','decision.md']:(s/n).write_text(text)
    (s/'hand_off.md').write_text(f"# 交接\n\nlabel_supported={summary['label_supported']}，query_quantity_supported={query_supported}。本轮只验证H-only、未做D7-C或完整训练，final标签未读。\n")
    ledger=json.loads((s/'compute_ledger.json').read_text());save(s.parent/'stage_3_done',dict(stage=3,scope='H label attribution and D7-H A/B',status='complete',started_at=ledger['started_at'],finished_at=time.time(),gpu_hours_consumed_so_far=0))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def render(summary,manifest):
    label='\n'.join(f"| {r['task'].replace('_to_','→')} | {r['c_FULL']:.4f} | {100*r['step_FULL_extra']:.6f} | {100*r['step_BASE_extra']:.6f} | {100*r['step_advantage_PERM_TOTAL_M']:.6f} | {r['advantage_PERM_TOTAL_M_wins']}/5 | {'通过' if r['joint_passed'] else '未通过'} |" for r in summary['label_attribution'])
    query='\n'.join(f"| {r['task'].replace('_to_','→')} | {r['convention']} | {r['c16']:.4f} | {r['c196']:.4f} | {r['cosine_gain']:.4f} | {r['relative_error_reduction']:.1%} | {r['paired_seed_wins']}/5 | {'通过' if r['passed'] else '未通过'} |" for r in summary['query_quantity'])
    return f'''# H-only标签项／预测项与query数量：结果解读

**稳定标签增量获支持：{summary['label_supported']}；query数量瓶颈判据获支持：{summary['query_quantity_supported']}。** 两组为冻结快照诊断，不是完整适配效果。

## 标签项与预测项：实验如何拆开

每图残差(Y_i−p_i)Q分为GLOBAL=(平均Y−平均p)Q、LABEL=(Y_i−平均Y)Q、PRED=−(p_i−平均p)Q。沿同一V和W分别读取。BASE=GLOBAL+PRED，对固定query标签的对应置乱不变；FULL=BASE+LABEL恰好还原原H-only反馈。BASE仍使用query标签比例，骨干也经源fit预训练，不能称完全无监督模型。

- A：源梯度＋BASE；B：源梯度＋FULL；C/D：保留BASE，只将LABEL换成等幅三次乱标签／同V空间随机LABEL。
- 再把C/D的完整反馈匹配到FULL范数，辅助检查大项抵消与步长的影响；该辅助组会缩放BASE，故不能与主组混称同一个控制。
- 另报单独LABEL、PRED、匹配FULL范数的BASE与Oracle。每次从同一原θ开始，19臂×180块=3420独立临时更新。
- 标签项和预测项可能很大、互相抵消。B胜A可能只是抵消了有害基线；必须同时看B胜源SGD及两类完整等幅null，不把抵消当作新目标监督。

Mutagenicity 0→1/2/3，五模型种子，源fit779／query196／validation109；每块fit16、A16、B完整271。沿用0/50/200源监督快照、每快照8块，方向共360块，单步取前4块。β0.1、η0.0003、FP64、H-JOINT λ0.01。

下表为相对基准CE降低百分比，非准确率百分点。正数更好；FULL／BASE相对源SGD；TOTAL-M表示完整反馈等幅null。

| 任务 | FULL方向余弦 | FULL单步收益 | BASE单步收益 | FULL对乱标签TOTAL-M优势 | 配对胜种子 | 联合门槛 |
|---|---:|---:|---:|---:|---:|---|
{label}

执行前判据：方向FULL c_A>0，对两类保留BASE的null各优势≥0.05、各≥3/5种子正、覆盖≥80%；单步FULL相对源SGD收益≥1e-4，FULL对BASE、主乱／随机及TOTAL-M乱／随机均值优势>0，所有比较各≥3/5种子为正、覆盖≥80%。至少2/3任务两项均过才称稳定标签增量。不修改旧D1/D2判据，不作显著性声明。LABEL/PRED范数、夹角、抵消比例及分种子／快照结果全部保留。

## D7-H：源query变多是否更稳定

本轮是原D7-A/B的H-only候选版本：核由原主HA改为H，范围明确登记，不能称已完成原HA版D7。仅五个初始fit模型×三个任务×八块=120目标块；不混入50/200快照。

196个query输入均可用于固定带宽和签名；每模型种子五个输入无关SHA256(D7|seed|rep|source_id)排列，嵌套取16/32/64/128/196张标签，核用196全池公共带宽。改变的是读取的监督数，输入信息范围及目标V固定。query不混入已预训练的fit图。源标签数也改变带标签覆盖，不能把曲线直接解释为纯标签噪声。

FIXED λ=0.01；NORMALIZED λ=0.01·n/196。两条都报告，不宣称后者最优。五次排列先在同一目标块内平均，再按模型种子平均；196的排列／正则完全等价，共用一次物理解，不当作十次独立证据。6000个逻辑配置对应4920个物理解，首4目标块的3000条逻辑单步对应2460次实际独立更新。

| 任务 | 正则约定 | n16方向余弦 | n196方向余弦 | 余弦提升 | 相对梯度误差降低 | 同向种子 | 门槛 |
|---|---|---:|---:|---:|---:|---:|---|
{query}

query判据：196相对16，平均余弦提升≥0.03、相对误差降低≥10%、至少3/5种子两项同向、有效覆盖≥80%；至少2/3任务在两种正则约定均满足才支持监督数量是主要瓶颈。达到196仍差不证明总体数据规模无关。子样本波动、对全196估计的向量离散度、类计数、W与核谱、校准状态和单步CE公开。

D7-B合成多类别PSD案例及三个真实固定签名的复制恒等式通过={summary['duplicate_invariance_passed']}。固定λ复制两份等价于原λ/2；复制λ变2λ后按原图合并列恢复原W，误差阈值1e-8。该项用单列高精度工程配置，不改变正式曲线超参数。D7-C新增训练复用审计未执行。

![query数量曲线](figures/query_curves.png)
![标签与预测单步](figures/label_controls.png)

## 完整性与范围

- 所有逻辑记录数：{summary['counts']}。旧H FULL和n196最大指标差{summary['FULL_and_n196_replay_max_error']:.3e}；标签分解最大加法误差{summary['max_label_additivity_error']:.3e}。
- query校准未收敛逻辑配置{summary['query_nonconverged_logical']}，全部保留。真实final标签读取0、新训练0，临时参数不作为后续起点。
- 已观察的开发集作探索性诊断，三次null、五次query排列、快照／块均不当独立模型种子。
- 原方案不覆盖；新增标签判据和H-only D7修订见execution_protocol.md；原始日志、readout seals、样本ID、TensorBoard、代码及SHA均归档。
- 执行代码{manifest['git_commit']}；原方案SHA {manifest['plan_sha256']}；补充协议SHA {manifest['execution_protocol_sha256']}。
- 复现入口label_query_experiment.py prepare→d0→pilot→run，然后label_query_summarize.py。具体参数见reproduction_commands.sh。
'''


def plots(s,tasks,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for i,task in enumerate(['0_to_1','0_to_2','0_to_3']):
        for conv in ['FIXED','NORMALIZED']:
            rows=[r for r in curves if r['task']==task and r['convention']==conv]
            axes[0,i].plot([r['n'] for r in rows],[r['c_A'] for r in rows],marker='o',label=conv)
            axes[1,i].plot([r['n'] for r in rows],[r['e_A'] for r in rows],marker='o',label=conv)
        axes[0,i].set_title(task.replace('_to_','→'));axes[0,i].axhline(0,color='k',linewidth=.6);axes[1,i].set_xlabel('Source query labels')
    axes[0,0].set_ylabel('Cosine with A gradient');axes[1,0].set_ylabel('Relative gradient error');axes[0,-1].legend()
    fig.tight_layout();fig.savefig(s/'figures/query_curves.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3)
    for i,(key,label) in enumerate([('step_FULL_extra','Full vs source'),('step_BASE_extra','Base vs source'),('step_BASE_M_extra','Matched base vs source')]):
        ax.bar(x+(i-1)*.24,[r[key]*100 for r in tasks],width=.24,label=label)
    ax.set_xticks(x,['0→1','0→2','0→3']);ax.axhline(0,color='k',linewidth=.6);ax.set_ylabel('Extra relative B CE reduction (%)');ax.legend()
    fig.tight_layout();fig.savefig(s/'figures/label_controls.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();summarize(a.stage)
