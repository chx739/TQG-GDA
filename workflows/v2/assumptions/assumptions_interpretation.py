"""Descriptive postprocessing of completed assumption diagnostics; no refits."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def main(stage):
    stage=Path(stage);result=json.loads((stage/'results_summary.json').read_text())
    geo=list(csv.DictReader((stage/'geometry_summary.csv').open()))
    correlations=[]
    for signature in ['H','HA']:
        selected=[r for r in geo if r['signature']==signature]
        for proxy in ['nearest_A_distance_mean','mean_error','max_error','weight_norm']:
            x=np.array([float(r[proxy]) for r in selected]);y=np.array([float(r['e_A']) for r in selected])
            rho=float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else None
            correlations.append(dict(signature=signature,proxy=proxy,n=len(x),spearman_with_empirical_A_relative_error=rho,
                                     scope='descriptive; 15 paired task/model geometries with shared source checkpoints'))
    (stage/'support_gradient_correlations.json').write_text(json.dumps(correlations,indent=2,allow_nan=False)+'\n')
    path=stage/'结果解读.md';text=path.read_text()
    intro='''## 先看结论和下一步

**当前最值得优先处理的是“局部源经验覆盖＋核读取方式”，还不能说已经证明η_S与η_T不同。**

1. **签名里确实有任务信息。** H与HA的线性／小MLP探针，在源内和目标内都过了可学门槛；同样147张训练图，各目标任务的目标内探针都优于类比例和乱标签对照。不能再把失败笼统解释成“当前签名完全不含标签信息”。但这不证明保留了完整图的全部必要信息。
2. **源规律迁过去存在稳定缺口。** H和HA的线性／MLP在三个目标任务都通过缺口筛选。目标探针是在使用目标诊断标签，故是定位工具，不是可部署的UDA效果。缺口可能混合条件关系、覆盖、类比例和有限样本的拟合差异，不能单凭它唯一归因。
3. **H距离能定位更困难的目标区域。** 三种探针的三个任务均出现“较远区域迁移缺口更大”，通过预设支持筛选。HA只在0→3过支持筛选；0→1/2的最小分组仅27/28张，未满足30张门槛，不能解读成HA支持诊断完全无效。距离仍只是代理，不是已证明数学支持。
4. **原HA核读取值得单独改。** HA残差KRR连源内可学门槛也未通过，换同签名的LR／MLP后能通过。它支持“核／残差读出受限”的诊断候选，不能证明某个RKHS绝不包含真实函数。HA的线性探针S→T NLL在三个任务反而都优于H线性探针，不能说响应A完全没有信息；“可预测信息”也不等于“原梯度读取已有额外收益”。
5. **只在局部有相容线索。** H线性探针在0→2共同近邻114张图上的Gap约0.00010，90%描述性区间[−0.01484,0.01537]落入预设±0.02；其他任务／探针未获跨任务相容支持。这不是普遍条件稳定性的证明。

建议下一批先登记A2-b：n64／128下H近邻bank与同数量均匀bank比较，并保留全196、两种正则、乱标签／等幅反馈对照；再做A4-a的核带宽／正则留出曲线。暂不优先扩骨干或加长适配；若局部读取改善预测，还必须经过A6方向／独立单步验收。

这些是完成实验后的路线建议，**本轮未执行A2-b或A3—A6**。

'''
    marker='## 同监督量的三角比较\n'
    assert marker in text
    text=text.replace(marker,intro+marker)
    text=text.replace('## 工程、范围与限制\n','''## 数值与范围补充

- H源内NLL在三个任务相同，因为它们共享同一源模型、query池、划分和H签名；重复的任务引用不是三份独立源数据证据。HA的V随目标任务变化，源端签名也随之变化。
- 残差KRR恢复的原始概率可能越过[0,1]。S→T越界比例H约4.24%／6.00%／14.76%，HA约6.73%／6.85%／12.08%；NLL按预设裁到[1e−6,1−1e−6]，同时保留原始残差MSE。尤其0→3的高KRR NLL可能部分受过强纠错与裁剪影响，不能全解释成标签关系反转。
- 内层LR有1条数值未收敛：3406、0→1、外折0、H、目标训练、第二个乱标签、L2=10，L-BFGS-B第10次迭代ABNORMAL。没有删除这条对照或改变网格。主真实标签拟合没有该标志。
- 代理指标与A经验梯度误差的描述性相关表另见support_gradient_correlations.json；只有15个共享模型的任务引用，不能当独立因果检验。B投影外能量复用历史同快照标量，不重新读B标签算梯度。

## 工程、范围与限制
''')
    path.write_text(text)
    with (stage/'run_report.md').open('a') as f:
        f.write('''\n## Post-result interpretation (no refits)

Same-signature LR/MLP learn within each domain, but every task has a source-to-target gap. H support strata locate larger far-region gaps on all tasks. This motivates separately registered local source-bank and kernel-complexity studies; it does not uniquely identify conditional shift, prove task sufficiency, or validate gradient feedback. HA+LR can improve predictive NLL even though the original HA residual KRR and prior gradient controls are weak. One H/LR task passes only the local descriptive equivalence margin. See 结果解读.md for exact qualifications.

One inner LR null fit has a finite ABNORMAL termination, retained. Original raw KRR probabilities and clipping rates are reported; high clipped NLL is not all assigned to transfer mismatch. All post-result correlation and interpretation code is archived separately from frozen execution code, with no model fitting or B-based selection.
''')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();main(a.stage)
