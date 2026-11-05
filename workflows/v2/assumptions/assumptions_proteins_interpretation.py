"""Read-only cross-dataset comparison for the completed PROTEINS assumption study."""
import argparse
import json
import csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def main(stage,repo):
    stage=Path(stage);repo=Path(repo)
    current=json.loads((stage/'results_summary.json').read_text())
    prior=json.loads((repo/'experiments/tqg_assumptions_20261103/results_summary.json').read_text())
    comparison=[]
    for name,r in current['overall'].items():
        m=prior['overall'][name]
        comparison.append(dict(candidate=name,proteins_source_tasks=r['same_source_tasks'],mutagenicity_source_tasks=m['same_source_tasks'],
            proteins_target_tasks=r['same_target_tasks'],mutagenicity_target_tasks=m['same_target_tasks'],
            proteins_gap_tasks=r['transfer_gap_tasks'],mutagenicity_gap_tasks=m['transfer_gap_tasks'],
            proteins_support_tasks=r['support_candidate_tasks'],mutagenicity_support_tasks=m['support_candidate_tasks'],
            proteins_local_compatibility_tasks=r['local_compatibility_tasks'],mutagenicity_local_compatibility_tasks=m['local_compatibility_tasks']))
    (stage/'cross_dataset_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
    projection=[]
    for domain in [1,2,3]:
        values=[json.loads((stage/f'logs/{seed}/{domain}_B_projection.json').read_text())['B_space_outside_energy_fraction'] for seed in range(3405,3410)]
        projection.append(dict(task=f'0_to_{domain}',mean_outside_A16_gradient_energy=float(np.mean(values)),per_seed=values))
    (stage/'B_projection_summary.json').write_text(json.dumps(projection,indent=2)+'\n')

    text='''## 先看本轮结论

**PROTEINS没有复现Mutagenicity上“源内／目标内稳定可学、同时有跨域读出缺口”的完整证据链。六个签名×探针组合的源内与目标内主门槛均为0/3。** 这是有限监督下的稳定增量未获支持，不是已证明签名无信息或四项数学条件全部不成立。

- 源query只有51张，42／9两类。H线性探针源内NLL约0.549，简单训练类比例基线约0.515，未稳定胜出；源内三个任务共用同一源数据／模型，不当三份独立源证据。
- 0→2的H-MLP目标内平均优于PRIOR约0.048 NLL，也优于乱标签，但只有3/5种子两项同向，未满足4/5。这是局部正信号，不应改写成完全没有信息。
- 虽然0→2／3存在较大的源→目标与目标→目标分数差，但目标内可学前提未过，不能按原判据宣称已定位条件失稳。0→1源训练LR／MLP平均还优于目标训练，不能只挑0→3讲故事。
- H／HA各任务支持分组的最小图数仅2～7，全部低于30。这里主要是分组证据不足，不能宣布“距离没有诊断价值”。共同近邻子集有局部信号，但相容区间无候选通过。
- 类别构成差异明显：源query的类别1比例9/51≈17.6%，目标B分别23/70、33/70、52/70≈32.9%／47.1%／74.3%。目标内参照也可能主要靠类比例获得较好NLL；均值类别差本身不等于同签名条件标签规律不同。
- 数学／执行验收通过：13项单测，Mutagenicity六组旧预测回归最大误差<1e−8，15份新几何的显式Jacobian／生产反馈／参数损失核验通过。真实PROTEINS的内层LR未收敛0条，MLP固定200步不声称优化收敛。工程通过不能代替研究门槛。

当前不依据PROTEINS单独推动局部bank或宣称跨域条件失稳。建议将其视为小样本／类别不平衡下的边界证据；先在Mutagenicity上验证既有局部支持与核读取候选。若继续解释PROTEINS，另登记分层划分敏感性、更简单／更强正则探针，或更多独立源监督；本轮不改阈值、不按标签重分折，也不补入骨干见过的fit图。

## 如何与Mutagenicity一起理解

这是同一套诊断的第二数据集复验，不是相同训练条件下仅替换数据集：PROTEINS的源query只有51张、探针每折38/39张；Mutagenicity为query196、每折147张。骨干GIN/GINE和源预训练预算也不同。因此，跨数据集差异不能单独归因于数据太少，门槛未过也不证明签名没有信息。

下表是满足各自原门槛的任务数，每列分母都是3。H源内重复三个任务共享同一源模型与数据，不当三份独立源证据。

| 候选 | PROTEINS源内／目标内可学 | Mutagenicity源内／目标内可学 | PROTEINS迁移缺口 | Mutagenicity迁移缺口 | PROTEINS支持候选 | Mutagenicity支持候选 |
|---|---|---|---|---|---|---|
'''
    for r in comparison:
        text+=f"| {r['candidate']} | {r['proteins_source_tasks']}/3，{r['proteins_target_tasks']}/3 | {r['mutagenicity_source_tasks']}/3，{r['mutagenicity_target_tasks']}/3 | {r['proteins_gap_tasks']}/3 | {r['mutagenicity_gap_tasks']}/3 | {r['proteins_support_tasks']}/3 | {r['mutagenicity_support_tasks']}/3 |\n"
    text+='\n当前A16空间外的B经验梯度能量平均约12.63%／7.04%／3.18%。这是梯度能量比例，不是未覆盖图的比例；也不能直接解释为源监督支持不足。\n'
    text+='\n支持候选要求每折近／远组均≥30张。PROTEINS的B只有70张，如果没有候选，先看分组数量是否充分，不能直接声称距离无用。目标内未通过可学门槛时，即使NLL有正差距，也不能跳过前置条件宣布迁移缺口候选成功。\n\n'
    path=stage/'结果解读.md';old=path.read_text();path.write_text(old.replace('## 同监督量的三角比较\n',text+'## 同监督量的三角比较\n'))
    geo=list(csv.DictReader((stage/'geometry_summary.csv').open()));correlations=[]
    for sig in ['H','HA']:
        selected=[r for r in geo if r['signature']==sig]
        for proxy in ['nearest_A_distance_mean','mean_error','max_error','weight_norm']:
            x=np.array([float(r[proxy]) for r in selected]);y=np.array([float(r['e_A']) for r in selected])
            corr=float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else None
            correlations.append(dict(signature=sig,proxy=proxy,spearman_with_empirical_A_error=corr,n=15,descriptive_only=True))
    (stage/'support_gradient_correlations.json').write_text(json.dumps(correlations,indent=2,allow_nan=False)+'\n')
    with (stage/'run_report.md').open('a') as f:
        f.write('\n## Cross-dataset descriptive comparison\n\nSee cross_dataset_comparison.json. Preserve the preregistered within-domain learnability prerequisites and 30-per-stratum support cutoff. PROTEINS uses fewer labeled query examples and a different backbone/pretraining budget; no size-only causal claim. Correlations over15 shared-model task references are descriptive, not new evidence of conditional-gradient correctness. No candidate was refitted or selected using B results.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--repo',required=True);a=p.parse_args();main(a.stage,a.repo)
