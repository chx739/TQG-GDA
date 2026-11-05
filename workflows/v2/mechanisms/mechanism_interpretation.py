"""Descriptive interpretation of completed frozen mechanism experiments."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def extend(stage):
    s=Path(stage);d=json.loads((s/'results_summary.json').read_text())
    errors=list(csv.DictReader((s/'error_decomposition.csv').open()))
    medians=[]
    for layer,scenario in [('linear','LINEAR'),('graph','S1'),('graph','S2'),('graph','S3')]:
        for method in ['HA_JOINT','HA_MEAN','H_MEAN']:
            rs=[r for r in errors if r['layer']==layer and r['scenario']==scenario and r['method']==method]
            medians.append(dict(layer=layer,scenario=scenario,method=method,
                **{k:statistics.median(float(r[k]) for r in rs if r[k]!='') for k in ['normalized_noise_squared','normalized_mismatch_squared','normalized_cross_term']}))
    (s/'error_decomposition_descriptive.json').write_text(json.dumps(dict(scope='Descriptive medians; medians of terms are not additive',rows=medians),indent=2)+'\n')
    friendly=[r for r in d['D8'] if r['scenario']=='S1']
    table='\n'.join(f"| {r['method']} | {r['c_TRUE']:.4f} ± {r['c_TRUE_SD']:.4f} | {r['c_PERM']:.4f} | {r['c_SOURCE_CONDITIONAL']:.4f} | {r['e_TRUE']:.3f} | {r['e_SOURCE_CONDITIONAL']:.4f} | {r['qualified_seeds']}/5 |" for r in friendly)
    real='\n'.join(f"| {r['task'].replace('_to_','→')} | {100*r['full_extra']:.6f} | {100*r['global_extra']:.6f} | {100*r['centered_increment']:.6f} | {r['step_PERM_seed_wins']}/5 |" for r in d['real_decomposition'] if r['method']=='H_JOINT')
    s4='\n'.join(f"| {r['method']} | {r['coverage']:.1%} | {r['exact_zero_feedback']:.1%} | {r['H_ST_norm']:.4f} | {r['gradient_norm']:.4f} |" for r in d['D8'] if r['scenario']=='S4')
    intro=f'''## 这轮最重要的结论

**在规定的友好合成图上，反馈方向可以恢复；真实数据的部分收益主要来自逐图残差，而不是平均残差。但“稳定利用逐图标签对应关系”仍未过门槛。**

### 1．友好条件下，基本机制能成立；幅度仍明显受源标签噪声影响

S1共享标签规律、特征捷径未反转、源目标部分重叠，三个预定读取器都通过了S1门槛。±是五模型种子标准差；source conditional只把源Y换成生成器给出的真实概率η_S，属于不可用于真实UDA的Oracle。

| 方法 | 抽样源标签方向余弦 | 乱标签余弦 | 真实源概率Oracle余弦 | 抽样标签相对梯度误差 | 源概率Oracle相对误差 | 满足完整种子阈值 |
|---|---:|---:|---:|---:|---:|---:|
{table}

方向通过并不等于恢复了准确幅度：抽样标签相对误差仍大于1，源概率Oracle的误差则很小。D8中的H版本按原方案为H-MEAN；真实数据分解的H版本为H-JOINT，不能将其称为完全相同配置的跨数据复现。

线性高斯组更明显：HA-JOINT抽样源标签平均余弦−0.1241，换真实源概率后为0.9988。其归一化源标签噪声平方项中位数7.179，合并失配项0.04359。说明在这组已知规律下，查询源标签噪声经读出后会被明显放大；不能据此直接量化真实Mutagenicity的噪声，那里没有η真值。中位数只描述尺度，不把中位数项硬加为总误差。

### 2．关系失配时，给真实源概率仍不能修复

S2反转特征捷径后，三个方法的抽样源标签平均方向均为负，换真实源概率后的平均方向也仍为负。因此在本轮固定源样本和读取器下，仅去掉源标签抽样噪声仍不能修复；当前签名下存在源目标关系或覆盖／近似失配。增大源查询量是否改变该结果尚未检验，不能据此断言更多样本永远无效。

S3只翻转目标标签规律，输入和S1逐个相同，360对读出完全一致，平均方向转负。这是方法所需迁移假设的边界；无标签输入本身不能辨别两个标签规律。

### 3．支持分离不等于已经学会可靠退让

S4的可比覆盖不足80%。主表的余弦只对可定义部分平均，特别HA-JOINT的约0.9997不能代表整体成功。精确零反馈为浮点计算结果，不当成RBF数学上严格无支持。

| 方法 | 成对有效覆盖 | 精确数值零反馈 | 平均H_ST范数 | 平均反馈范数 |
|---|---:|---:|---:|---:|
{s4}

H_ST通常减小，但H-MEAN的平均反馈范数反而高于S1；没有证明当前算法能自动、稳定地在弱支持时停止反馈。还需看每块支持与幅度，而不能只看原始特征加10或一个较高的有效子集余弦。

### 4．真实H-only的正收益并非主要来自平均残差

以下是完整独立单步相对CE百分比，非BA百分点。逐图增量严格用“完整更新减去只加平均项的更新”评价，而不是将两次单独CE收益强行相加。

| 任务 | 完整反馈相对源SGD | 平均反馈相对源SGD | 追加逐图项的增量 | 逐图项胜过等幅乱标签的种子 |
|---|---:|---:|---:|---:|
{real}

这修正了此前“收益可能主要来自平均残差”的猜测：本轮H-only在0→2、0→3的收益大部分保留在逐图项中。HA仍弱，去掉平均项并未恢复它。

不过，“逐图残差有用”仍不等于“逐图标签对应关系稳定有用”：中心化残差=(Y_i−平均Y)−(p_i−平均p)，仍包含各图预测差异；打乱Y不会删除这些预测差异。逐图方向门槛六个任务／候选组合均未过，H的两个正任务仅通过单步增量门槛。0→2对逐图乱标签仅2/5种子胜出，0→3为3/5。

### 下一步建议

- 保留H简化路线，暂不进入真实完整适配。S1方向成立是机制正对照，不是Mutagenicity效果证明。
- 优先研究查询源监督的方差：更充分查询、独立源留出验证、避免将小样本噪声读成强反馈；不能直接用未知η替代真实标签。
- 把中心化标签项与中心化预测项再分开做匹配对照，定位真实增益是否依赖源图与标签对应。
- 捷径反转、支持分离需要关系可靠性／支持门控证据；单纯加源样本不能保证修复S2/S3。

以上是后续建议，本轮未自动执行这些新实验。

'''
    for name in ['结果解读.md','run_report.md','decision.md']:
        p=s/name;t=p.read_text();assert '## 这轮最重要的结论' not in t
        anchor='## 先看实验在问什么';assert anchor in t;t=t.replace(anchor,intro+anchor)
        # Keep S4 low coverage visible alongside the full condition table.
        t=t.replace('满足S1阈值种子 | S1判断','满足S1阈值种子 | S1判断')
        p.write_text(t)
    p=s/'run_report.md';p.write_text(p.read_text()+'''\n## 执行与后处理补充\n\n正式五种子均一次完成，无训练错误、无种子删除。3个未达校准容差的逻辑配置原样保留：S2/50/块3/种子3405；S1与复用输入的S3/50/块4/种子3409；三者均为HA-JOINT，涉及2次独立物理解。未因非收敛改正则或迭代上限。\n\n正式执行代码为code_snapshot，提交81d0703。汇总代码analysis_code/mechanism_summarize.py仅将缺失余弦的柱改成图中空缺，所有聚合与门槛保持冻结版本；源码SHA在analysis_code/provenance.json。mechanism_interpretation.py只根据既有表格补充解释和误差中位数。\n\n原driver的试跑时间估计把总480图块误写为96，低估约5倍；compute_estimate.json保留原估计并另列修正估计。8小时硬上限、实验矩阵、正式计时与所有超参数未改。主仓库对后续运行修正了此处预估公式；冻结执行代码仍保留实际运行版本。\n''')
    # Add coverage labels to avoid interpreting S4 available-only cosine as full support.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,method in zip(axes,['HA_JOINT','HA_MEAN','H_MEAN']):
        rs=[r for r in d['D8'] if r['scenario'] in ['S1','S2','S3','S4'] and r['method']==method];x=np.arange(4)
        for i,(key,label) in enumerate([('c_TRUE','TRUE'),('c_PERM','PERM'),('c_SOURCE_CONDITIONAL','Source conditional oracle')]):
            ax.bar(x+(i-1)*.24,[r[key] if r[key] is not None else np.nan for r in rs],width=.24,label=label)
        ax.axhline(0,color='k',linewidth=.7);ax.set_xticks(x,[r['scenario']+f"\n{r['coverage']:.0%} valid" for r in rs]);ax.set_title(method);ax.set_ylim(-1,1)
    axes[0].set_ylabel('Available conditional-gradient cosine');axes[-1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(s/'figures/d8_conditions.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();extend(a.stage)
