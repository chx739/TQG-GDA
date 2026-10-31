"""Seed-level D8 and residual decomposition summaries with frozen gates."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import time

from p0_summarize import mean,deviation,balanced_snapshot_mean,export_csv,save,load_jsonl


def avg(rows,key):return mean([r.get(key) for r in rows])
def snapavg(rows,key):return balanced_snapshot_mean(rows,key)


def summarize(stage):
    stage=Path(stage);manifest=json.loads((stage/'experiment_manifest.json').read_text());settings=manifest['settings']
    files=['linear_gradient_blocks','graph_gradient_blocks','error_decomposition','synthetic_step_blocks',
        'real_gradient_blocks','real_step_blocks','real_residual_components','synthetic_baseline']
    allrows={name:[] for name in files}
    for seed in range(3405,3410):
        out=stage/'logs'/str(seed);done=json.loads((out/'done.json').read_text())
        assert done['status']=='completed' and done['git_commit']==manifest['git_commit']
        for name in files:allrows[name]+=load_jsonl(out/(name+'.jsonl'))
    expected=dict(linear_gradient_blocks=3960,graph_gradient_blocks=15840,error_decomposition=1800,
        synthetic_step_blocks=2880,real_gradient_blocks=6480,real_step_blocks=4680,real_residual_components=720,synthetic_baseline=60)
    for name,rows in allrows.items():
        assert len(rows)==expected[name],(name,len(rows),expected[name]);export_csv(stage/(name+'.csv'),rows)
    d8blocks=[]
    for layer in ['linear','graph']:
        grouped=defaultdict(dict)
        for r in allrows[layer+'_gradient_blocks']:grouped[(r['seed'],r['scenario'],r['snapshot_step'],r['block_id'],r['method'])][r['arm']]=r
        for (seed,scenario,snapshot,block,method),arms in grouped.items():
            true=arms['TRUE'];p=mean([arms[f'PERM-{i}']['c_A'] for i in [1,2,3]])
            rand=mean([arms[f'RAND-{i}']['c_A'] for i in [1,2,3]])
            valid=true['c_A'] is not None and all(arms[f'{kind}-{i}']['c_A'] is not None for kind in ['PERM','RAND'] for i in [1,2,3])
            d8blocks.append(dict(layer=layer,seed=seed,scenario=scenario,snapshot_step=snapshot,block_id=block,method=method,
                c_TRUE=true['c_A'],c_PERM=p,c_RAND=rand,c_SOURCE_CONDITIONAL=arms['SOURCE_CONDITIONAL']['c_A'],
                advantage_PERM=true['c_A']-p if valid else None,advantage_RAND=true['c_A']-rand if valid else None,
                e_TRUE=true['e_A'],e_SOURCE_CONDITIONAL=arms['SOURCE_CONDITIONAL']['e_A'],
                c_B_TRUE=true['c_B'],conditional_vs_empirical_cosine=true['conditional_vs_empirical_cosine'],
                coverage=float(valid),H_ST_norm=true['H_ST_norm'],K_ST_mean=true['K_ST_mean'],
                gradient_norm=true['gradient_norm'],conditional_gradient_norm=true['target_conditional_gradient_norm'],
                calibration_converged=float(true['solver']['converged']),exact_zero_feedback=float(true['gradient_norm']==0)))
    export_csv(stage/'synthetic_paired_blocks.csv',d8blocks)
    keys=['c_TRUE','c_PERM','c_RAND','c_SOURCE_CONDITIONAL','advantage_PERM','advantage_RAND','e_TRUE',
        'e_SOURCE_CONDITIONAL','c_B_TRUE','conditional_vs_empirical_cosine','coverage','H_ST_norm','K_ST_mean','gradient_norm',
        'conditional_gradient_norm','calibration_converged','exact_zero_feedback']
    synthseed=[];synthsnap=[]
    for layer in ['linear','graph']:
        for scenario in (['LINEAR'] if layer=='linear' else ['S1','S2','S3','S4']):
            for method in ['HA_JOINT','HA_MEAN','H_MEAN']:
                for seed in range(3405,3410):
                    rows=[r for r in d8blocks if (r['layer'],r['scenario'],r['method'],r['seed'])==(layer,scenario,method,seed)]
                    synthseed.append(dict(layer=layer,scenario=scenario,method=method,seed=seed,**{k:snapavg(rows,k) for k in keys}))
                    for snapshot in [0,1,2] if layer=='linear' else [0,50,200]:
                        sub=[r for r in rows if r['snapshot_step']==snapshot]
                        synthsnap.append(dict(layer=layer,scenario=scenario,method=method,seed=seed,snapshot_step=snapshot,**{k:avg(sub,k) for k in keys}))
    export_csv(stage/'synthetic_seed_summary.csv',synthseed);export_csv(stage/'synthetic_snapshot_seed_summary.csv',synthsnap)
    d8=[]
    for scenario in ['LINEAR','S1','S2','S3','S4']:
        for method in ['HA_JOINT','HA_MEAN','H_MEAN']:
            rows=[r for r in synthseed if r['scenario']==scenario and r['method']==method]
            result=dict(scenario=scenario,method=method,**{k:avg(rows,k) for k in keys},c_TRUE_SD=deviation([r['c_TRUE'] for r in rows]))
            qualified=sum(r['c_TRUE'] is not None and r['c_TRUE']>.2 and r['advantage_PERM'] is not None and r['advantage_PERM']>=.05
                and r['advantage_RAND'] is not None and r['advantage_RAND']>=.05 and r['coverage']>=.8 for r in rows)
            result['qualified_seeds']=qualified
            result['S1_passed']=scenario=='S1' and result['c_TRUE'] is not None and result['c_TRUE']>.2 and result['advantage_PERM'] is not None and result['advantage_PERM']>=.05 and result['advantage_RAND'] is not None and result['advantage_RAND']>=.05 and qualified>=3 and result['coverage']>=.8
            d8.append(result)
    # Real residual direction and incremental step contrasts, paired within a fixed W.
    grouped=defaultdict(dict)
    for r in allrows['real_gradient_blocks']:grouped[(r['task'],r['seed'],r['snapshot_step'],r['block_id'],r['method'])][r['arm']]=r
    realblocks=[]
    for (task,seed,snapshot,block,method),arms in grouped.items():
        c=arms['CENTERED'];p=mean([arms[f'PC-{i}']['c_A'] for i in [1,2,3]]);rand=mean([arms[f'RC-{i}']['c_A'] for i in [1,2,3]])
        valid=c['c_A'] is not None and all(arms[f'{kind}-{i}']['c_A'] is not None for kind in ['PC','RC'] for i in [1,2,3])
        realblocks.append(dict(task=task,seed=seed,snapshot_step=snapshot,block_id=block,method=method,
            c_FULL=arms['FULL']['c_A'],c_GLOBAL=arms['GLOBAL']['c_A'],c_CENTERED=c['c_A'],
            c_PERM_CENTERED=p,c_RAND_CENTERED=rand,c_B_FULL=arms['FULL']['c_B'],c_B_GLOBAL=arms['GLOBAL']['c_B'],c_B_CENTERED=c['c_B'],
            advantage_PERM=c['c_A']-p if valid else None,advantage_RAND=c['c_A']-rand if valid else None,coverage=float(valid)))
    rgkeys=[k for k in realblocks[0] if k not in ['task','seed','snapshot_step','block_id','method']]
    realseed=[]
    for method in ['H_JOINT','HA_JOINT']:
        for task in ['0_to_1','0_to_2','0_to_3']:
            for seed in range(3405,3410):
                rows=[r for r in realblocks if (r['method'],r['task'],r['seed'])==(method,task,seed)]
                realseed.append(dict(method=method,task=task,seed=seed,**{k:snapavg(rows,k) for k in rgkeys}))
    export_csv(stage/'real_direction_seed_summary.csv',realseed);export_csv(stage/'real_direction_paired_blocks.csv',realblocks)
    stepgroups=defaultdict(dict)
    for r in allrows['real_step_blocks']:stepgroups[(r['task'],r['seed'],r['snapshot_step'],r['block_id'],r['method'])][r['arm']]=r
    realsteps=[]
    for (task,seed,snapshot,block,method),arms in stepgroups.items():
        base=arms['S']['base_CE_B'];ce=lambda name:arms[name]['CE_B']
        valid=all(arms[f'S+GLOBAL+{kind}-{i}']['valid_comparison'] for kind in ['PC','RC'] for i in [1,2,3])
        pc=mean([ce(f'S+GLOBAL+PC-{i}') for i in [1,2,3]]);rc=mean([ce(f'S+GLOBAL+RC-{i}') for i in [1,2,3]])
        realsteps.append(dict(task=task,seed=seed,snapshot_step=snapshot,block_id=block,method=method,
            full_extra=(ce('S')-ce('S+FULL'))/base,global_extra=(ce('S')-ce('S+GLOBAL'))/base,
            centered_extra=(ce('S')-ce('S+CENTERED'))/base,
            centered_increment=(ce('S+GLOBAL')-ce('S+FULL'))/base,
            centered_advantage_PERM=(pc-ce('S+FULL'))/base if valid else None,
            centered_advantage_RAND=(rc-ce('S+FULL'))/base if valid else None,
            global_matched_extra=(ce('S')-ce('S+GLOBAL-M'))/base if arms['S+GLOBAL-M']['valid_comparison'] else None,
            centered_matched_extra=(ce('S')-ce('S+CENTERED-M'))/base if arms['S+CENTERED-M']['valid_comparison'] else None,
            oracle_extra=(ce('S')-ce('S+ORACLE-M'))/base if arms['S+ORACLE-M']['valid_comparison'] else None,
            coverage=float(valid)))
    skeys=[k for k in realsteps[0] if k not in ['task','seed','snapshot_step','block_id','method']]
    stepseed=[];stepsnap=[]
    for method in ['H_JOINT','HA_JOINT']:
        for task in ['0_to_1','0_to_2','0_to_3']:
            for seed in range(3405,3410):
                rows=[r for r in realsteps if (r['method'],r['task'],r['seed'])==(method,task,seed)]
                stepseed.append(dict(method=method,task=task,seed=seed,**{k:snapavg(rows,k) for k in skeys}))
                for snapshot in [0,50,200]:
                    sub=[r for r in rows if r['snapshot_step']==snapshot]
                    stepsnap.append(dict(method=method,task=task,seed=seed,snapshot_step=snapshot,**{k:avg(sub,k) for k in skeys}))
    export_csv(stage/'real_step_seed_summary.csv',stepseed);export_csv(stage/'real_step_snapshot_seed_summary.csv',stepsnap)
    export_csv(stage/'real_step_paired_blocks.csv',realsteps)
    real=[]
    for method in ['H_JOINT','HA_JOINT']:
        for task in ['0_to_1','0_to_2','0_to_3']:
            g=[r for r in realseed if r['method']==method and r['task']==task];s=[r for r in stepseed if r['method']==method and r['task']==task]
            r=dict(method=method,task=task,**{k:avg(g,k) for k in rgkeys},**{k:avg(s,k) for k in skeys if k!='coverage'},
                step_coverage=avg(s,'coverage'),increment_seed_wins=sum(x['centered_increment']>0 for x in s),
                step_PERM_seed_wins=sum(x['centered_advantage_PERM'] is not None and x['centered_advantage_PERM']>0 for x in s),
                step_RAND_seed_wins=sum(x['centered_advantage_RAND'] is not None and x['centered_advantage_RAND']>0 for x in s),
                direction_PERM_seed_wins=sum(x['advantage_PERM'] is not None and x['advantage_PERM']>0 for x in g),
                direction_RAND_seed_wins=sum(x['advantage_RAND'] is not None and x['advantage_RAND']>0 for x in g))
            r['direction_passed']=r['c_CENTERED'] is not None and r['c_CENTERED']>0 and r['advantage_PERM'] is not None and r['advantage_PERM']>=.05 and r['advantage_RAND'] is not None and r['advantage_RAND']>=.05 and r['direction_PERM_seed_wins']>=3 and r['direction_RAND_seed_wins']>=3 and r['coverage']>=.8
            r['increment_passed']=r['centered_increment']>=1e-4 and r['increment_seed_wins']>=3 and r['centered_advantage_PERM'] is not None and r['centered_advantage_PERM']>0 and r['centered_advantage_RAND'] is not None and r['centered_advantage_RAND']>0 and r['step_coverage']>=.8
            r['joint_passed']=r['direction_passed'] and r['increment_passed'];real.append(r)
    # Independent exact replay of FULL gradients against prior H/P0 raw observations.
    errors=[]
    for method,root in [('H_JOINT',settings['h_stage']),('HA_JOINT',settings['p0_stage'])]:
        prior={}
        for seed in range(3405,3410):
            for r in load_jsonl(Path(root)/'logs'/str(seed)/'gradient_blocks.jsonl'):
                if r['arm']=='TRUE':prior[(r['task'],r['seed'],r['snapshot_step'],r['block_id'])]=r
        for r in allrows['real_gradient_blocks']:
            if r['method']==method and r['arm']=='FULL':
                ref=prior[(r['task'],r['seed'],r['snapshot_step'],r['block_id'])]
                errors.extend(abs(r[k]-ref[k]) for k in ['c_A','c_B','e_A','gradient_norm'])
    assert max(errors)<1e-10,max(errors)
    # Identical target inputs must leave all unlabeled readouts identical in S1/S3.
    pair=defaultdict(dict)
    for r in allrows['graph_gradient_blocks']:
        if r['arm']=='TRUE' and r['scenario'] in ['S1','S3']:
            pair[(r['seed'],r['snapshot_step'],r['block_id'],r['method'])][r['scenario']]=r['W_hash']
    assert len(pair)==360 and all(v['S1']==v['S3'] for v in pair.values())
    report=dict(status='completed',D8=d8,real_decomposition=real,counts=expected,
        D8_S1_supported_methods=[r['method'] for r in d8 if r['S1_passed']],
        real_centered_supported_candidates=[m for m in ['H_JOINT','HA_JOINT'] if sum(r['joint_passed'] for r in real if r['method']==m)>=2],
        full_replay_max_metric_error=max(errors),S1_S3_identical_readout_pairs=360,
        synthetic_nonconverged=sum(not r['solver']['converged'] for name in ['linear_gradient_blocks','graph_gradient_blocks'] for r in allrows[name] if r['arm']=='TRUE'),
        max_residual_additivity_error=max(r['additivity_absolute_error'] for r in allrows['real_residual_components']),
        max_error_decomposition_identity_error=max(r['identity_absolute_error'] for r in allrows['error_decomposition']),
        real_final_labels_read=0,real_continuous_adaptation_jobs=0,new_synthetic_source_models=5)
    save(stage/'results_summary.json',report)
    results=[]
    for experiment,rows,key in [('D8',synthseed,'c_TRUE'),('RESIDUAL_DIRECTION',realseed,'c_CENTERED'),('RESIDUAL_STEP',stepseed,'centered_increment')]:
        for r in rows:results.append(dict(experiment_id=experiment,git_commit=manifest['git_commit'],config_name=settings['config_name'],
            hardware=manifest['hardware'],gpu_hours=0,event_flags='clean',metric_name=key,metric_value=r[key],**r))
    export_csv(stage/'results.csv',results)
    plots(stage,d8,real)
    text=render(report,manifest)
    for name in ['结果解读.md','run_report.md','decision.md']:(stage/name).write_text(text)
    (stage/'hand_off.md').write_text('# 机制实验交接\n\n'+f"D8 S1获支持方法：{report['D8_S1_supported_methods']}；真实逐图增量获支持候选：{report['real_centered_supported_candidates']}。无真实完整训练、无final标签。详见run_report.md。\n")
    ledger=json.loads((stage/'compute_ledger.json').read_text())
    save(stage.parent/'stage_3_done',dict(stage=3,scope='D8 and residual decomposition',started_at=ledger['started_at'],finished_at=time.time(),gpu_hours_consumed_so_far=0,status='complete'))
    print(json.dumps(report,ensure_ascii=False,indent=2))


def render(report,manifest):
    rows='\n'.join(f"| {r['scenario']} | {r['method']} | {r['c_TRUE'] if r['c_TRUE'] is not None else 'NA'} | {r['c_PERM'] if r['c_PERM'] is not None else 'NA'} | {r['c_RAND'] if r['c_RAND'] is not None else 'NA'} | {r['c_SOURCE_CONDITIONAL'] if r['c_SOURCE_CONDITIONAL'] is not None else 'NA'} | {r['qualified_seeds']}/5 | {'通过' if r['S1_passed'] else ('未通过' if r['scenario']=='S1' else '边界描述')} |" for r in report['D8'] if r['scenario']!='LINEAR')
    real='\n'.join(f"| {r['method']} | {r['task'].replace('_to_','→')} | {r['full_extra']*100:.6f} | {r['global_extra']*100:.6f} | {r['centered_increment']*100:.6f} | {r['centered_advantage_PERM']*100:.6f} | {r['centered_advantage_RAND']*100:.6f} | {r['step_PERM_seed_wins']}/5 | {'通过' if r['joint_passed'] else '未通过'} |" for r in report['real_decomposition'])
    return f'''# D8与残差分解：结果解读

D8共享规律S1获支持方法：**{report['D8_S1_supported_methods']}**；真实数据逐图残差联合门槛获支持候选：**{report['real_centered_supported_candidates']}**。这两组是机制诊断，不是完整适配分类结果。

## 先看实验在问什么

D8自己规定每张图属于类别1的概率η，因此可以计算没有随机标签噪声的条件梯度。真实源标签是按η抽取的一次0/1结果，方法只读这些抽样源标签，不读η。额外给读取器真实源η只作Oracle，区分源标签波动与函数近似／覆盖／跨域失配。

真实数据把源残差写成r_i=平均残差+(r_i−平均残差)。同一个V和W分别读取，完整反馈=平均反馈+逐图反馈。平均项对固定query池的标签置乱不变；源fit预训练仍使用过图与标签对应，不能把这个不变性外推成整个模型没用标签。

## D8设置和结果

- 五个配对生成／模型种子3405—3409；输入生成seed7300+(seed−3405)，角色偏移11/12/13/14/20/21，标签另加1000。
- cycle图含可选三角形；节点(1,u)，源u近似类别符号，η=sigmoid(4(2t−1))。源fit512／query196／validation128／audit256；目标构造1024、独立评价1024。
- 两层GINE hidden32 mean，固定200个Adam更新、batch32、lr0.003；保存0/50/200，共五个源模型复用于四情形。没有按目标结果选择快照。
- S1共享规则且部分重叠；S2特征捷径反转但标签仍由三角形决定；S3输入与S1相同、η翻转；S4目标u加10，不预先假定签名空间一定分离。
- 480个图块，每块目标16；三个原定读取器：HA-JOINT λ0.01、HA-MEAN λ0.005、H-MEAN λ0.005。各做真实／三次乱标签／三次同空间等范数随机、条件Oracle、经验Oracle、源条件读出。
- 首要比较当前A块条件梯度；独立目标池只用于辅助条件CE。前4块附加η0.0003、β0.1独立临时步，不迭代轨迹。
- 另有二维高斯线性组120块，θ=(0,0)/(1,1)/(0.5,−0.5)，真实w=(1,−1)。该组目标全池1024包含A，只作解析与噪声诊断，不宣称独立B泛化。

| 情形 | 方法 | 真实标签余弦 | 乱标签余弦 | 随机余弦 | 换成真实源概率的余弦（Oracle） | 满足S1阈值种子 | S1判断 |
|---|---|---:|---:|---:|---:|---:|---|
{rows}

S1判据在执行前固定：平均c>0.2，对两类null优势各≥0.05，至少3/5种子自身同时满足这三个阈值，可比覆盖≥80%。三个方法分别报告，不事后选择某个快照或生成器。其余情形只检验边界，不用S1门槛冒充成功率。

源标签噪声项(g_hat−g_read_source_cond)与合并失配项(g_read_source_cond−g_cond)相加为总误差。平方误差必须加交叉项；原始、归一化两项和交叉项全部在error_decomposition.csv，不把失配项单称为域偏移。S4同时公开K_ST、H_ST、E、反馈范数和精确数值零率；RBF下溢为0与数学支持严格为0不同。

## 真实Mutagenicity：平均项与逐图项

沿用0→1/2/3、五种子、0/50/200快照、源query196、fit16、A16、B271。H-JOINT和HA-JOINT各360个方向块；前4块共180×2×13=4680次临时步。β0.1、η0.0003，FP64。

完整、平均、逐图原幅度均公开；另将平均／逐图匹配完整反馈范数。关键受控比较：保持同一个源梯度和平均反馈，追加真实逐图反馈或同幅乱标签逐图／同空间随机逐图反馈。乱标签保留每图p，不把整个残差直接置乱。

下表都是**相对基准CE的额外降低百分比，不是准确率百分点**。完整／平均列相对源SGD；逐图增量=完整相对平均；最后两项为真实逐图相对两类等幅逐图null。

| 方法 | 任务 | 完整反馈 | 平均反馈 | 逐图增量 | 对乱标签优势 | 对随机优势 | 对乱标签胜种子 | 联合门槛 |
|---|---|---:|---:|---:|---:|---:|---:|---|
{real}

逐图方向门槛：c_A>0、对乱标签逐图／随机逐图各余弦优势≥0.05、各≥3/5种子为正、覆盖≥80%。增量单步门槛：追加逐图相对平均反馈收益≥1e-4、≥3/5种子为正、相对两类等幅逐图null均值优势均>0、覆盖≥80%。一个候选至少2/3任务两项均过才支持“稳定逐图监督增量”；不是显著性检验。

向量可相互抵消，平均／逐图反馈的范数比例不是贡献百分比；CE也有曲率，不能将两次单独更新收益无条件相加。分种子和快照CSV完整公开，不根据总体均值隐藏反转。

![D8条件梯度](figures/d8_conditions.png)
![真实残差分解](figures/residual_steps.png)

## 完整性、范围和复现

- 正式记录数：{report['counts']}。
- FULL与旧H/P0最大梯度指标误差{report['full_replay_max_metric_error']:.3e}；S1/S3的360组读出哈希相同。
- 最大残差加法误差{report['max_residual_additivity_error']:.3e}；误差平方分解恒等式误差{report['max_error_decomposition_identity_error']:.3e}。
- 合成校准非收敛配置{report['synthetic_nonconverged']}，全部保留状态与近似解。目标真相访问先封存W后显式记录；真实final读取0，不保存Oracle轨迹为模型。
- 旧原方案不修改；本轮未规定的块种子、附加条件CE步与逐图门槛等显式登记在execution_protocol.md。D8合成不能替代真实任务效果，开发集反复观察后的真实分解属于探索性诊断。
- git_commit={manifest['git_commit']}；plan_sha256={manifest['plan_sha256']}；execution_protocol_sha256={manifest['execution_protocol_sha256']}。
- 入口mechanism_experiment.py，prepare→pilot→run；汇总mechanism_summarize.py。详见复现命令及原始日志。数据／checkpoint／代码与角色种子按manifest保存。
'''


def plots(stage,d8,real):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    (stage/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,method in zip(axes,['HA_JOINT','HA_MEAN','H_MEAN']):
        rows=[r for r in d8 if r['scenario'] in ['S1','S2','S3','S4'] and r['method']==method];x=np.arange(4)
        for i,(key,label) in enumerate([('c_TRUE','TRUE'),('c_PERM','PERM'),('c_SOURCE_CONDITIONAL','Source conditional oracle')]):
            ax.bar(x+(i-1)*.24,[r[key] or 0 for r in rows],width=.24,label=label)
        ax.axhline(0,color='k',linewidth=.7);ax.set_xticks(x,['S1','S2','S3','S4']);ax.set_title(method);ax.set_ylim(-1,1)
    axes[0].set_ylabel('Cosine with known conditional gradient');axes[-1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(stage/'figures/d8_conditions.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,method in zip(axes,['H_JOINT','HA_JOINT']):
        rows=[r for r in real if r['method']==method];x=np.arange(3)
        for i,(key,label) in enumerate([('full_extra','Full vs source'),('global_extra','Global vs source'),('centered_increment','Centered increment')]):
            ax.bar(x+(i-1)*.24,[r[key]*100 for r in rows],width=.24,label=label)
        ax.set_xticks(x,['0→1','0→2','0→3']);ax.set_title(method);ax.axhline(0,color='k',linewidth=.7)
    axes[0].set_ylabel('Extra relative CE reduction (%)');axes[-1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(stage/'figures/residual_steps.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();summarize(a.stage)
