"""Descriptive post-run explanation and input-only source-kernel spectra."""
import argparse
import json
import hashlib
from pathlib import Path
import statistics
import numpy as np
import torch


def extend(stage):
    s=Path(stage);d=json.loads((s/'results_summary.json').read_text());m=json.loads((s/'experiment_manifest.json').read_text())
    root=Path(m['settings']['h_stage'])
    spectra=[]
    for seed in range(3405,3410):
        with np.load(root/f'geometry_cache/{seed}/1_{seed}_0_0.npz',allow_pickle=False) as data:h=torch.from_numpy(data['source_h'].copy())
        ids=json.loads((root/f'logs/{seed}/1_{seed}_0_0_indices.json').read_text())['query']
        scale=torch.quantile(torch.pdist(h).square(),.5).clamp_min(1e-16)
        kernel=torch.exp(-torch.cdist(h,h,compute_mode='donot_use_mm_for_euclid_dist').square()/(2*scale))
        seen=set()
        for rep in range(5):
            order=sorted(ids,key=lambda i:hashlib.sha256(f'D7|{seed}|{rep}|{i}'.encode()).hexdigest())
            for n in [16,32,64,128,196]:
                selected=tuple(sorted(order[:n]))
                if selected in seen:continue
                seen.add(selected);indices=[ids.index(i) for i in selected]
                eigen=torch.linalg.eigvalsh(kernel[indices][:,indices]);positive=eigen.clamp_min(0)
                tol=n*torch.finfo(eigen.dtype).eps*max(float(eigen.max()),0.)
                weights=positive/positive.sum();valid=weights>0
                spectra.append(dict(seed=seed,n=n,example_rep=rep,selected_query_ids=list(selected),bandwidth_squared=float(scale),
                    eigenvalues=eigen.tolist(),numerical_rank=int((eigen>tol).sum()),
                    effective_rank=float(torch.exp(-(weights[valid]*weights[valid].log()).sum()))))
    assert len(spectra)==105
    (s/'derived_query_kernel_spectra.json').write_text(json.dumps(dict(scope='Input-only post-run derived spectra; no labels, no new solves, shared across target blocks and conventions',cases=spectra),indent=2)+'\n')
    components=[json.loads(line) for seed in range(3405,3410) for line in (s/f'logs/{seed}/label_components.jsonl').read_text().splitlines()]
    description=[]
    for task in ['0_to_1','0_to_2','0_to_3']:
        rs=[r for r in components if r['task']==task]
        description.append(dict(task=task,median_cancellation_ratio=statistics.median(r['cancellation_ratio'] for r in rs),
            median_LABEL_PRED_cosine=statistics.median(r['label_prediction_cosine'] for r in rs),
            mean_LABEL_norm=statistics.mean(r['LABEL_norm'] for r in rs),mean_PRED_norm=statistics.mean(r['PRED_norm'] for r in rs)))
    (s/'label_component_descriptive.json').write_text(json.dumps(dict(scope='Descriptive pooled medians only; efficacy aggregated by model seed separately',tasks=description),indent=2)+'\n')
    label='\n'.join(f"| {r['task'].replace('_to_','→')} | {100*r['step_FULL_extra']:.5f} | {100*r['step_BASE_extra']:.5f} | {100*r['step_BASE_M_extra']:.5f} | {r['label_increment_wins']}/5 |" for r in d['label_attribution'])
    text=f'''## 核心结论：两组总体门槛都未通过，但不是“所有因素都没作用”

### 1．逐图标签对应关系仍未稳定贡献增量

三个任务的标签归因方向门槛和新的严格单步门槛均未通过。完整H反馈的两个正收益任务仍在，但“追加标签项胜过预测基线”仅各2/5种子为正。不能只用跨种子平均增益称稳定标签迁移。

下表是相对源SGD的额外B池CE降低百分比，非BA百分点。BASE=平均项＋预测差异，仍保留query标签比例和源fit训练；BASE-M将该反馈匹配完整FULL范数。

| 任务 | FULL | BASE | BASE-M | FULL胜BASE的种子 |
|---|---:|---:|---:|---:|
{label}

特别0→3，BASE的方向余弦0.4485高于FULL的0.1885；匹配范数后的BASE-M单步平均收益约0.06626%，高于FULL的0.04469%。该比较有4/5种子支持，但仍是开发集上的描述性结果，未把BASE-M选成新主方法，也没有测试其长期效果。0→1没有由此得到修复，不能称预测项普遍解释全部收益。

B胜A也可能来自不同幅度或标签／预测项的抵消，因此同时保留单独项、复合null和完整等幅null，不凭某一较差基线宣布贡献。BASE依赖模型预训练，不能外推为整个系统完全不需要源标签。

### 2．query数有局部作用，但不支持“主要只是源标签数量不够”

- NORMALIZED约定下，0→1和0→3同时通过方向／误差门槛；FIXED约定下0/3通过，所以两种约定共同支持的总体门槛未过。
- 0→3从16到196的平均方向余弦明显提高：FIXED为0.0059→0.1813，NORMALIZED为−0.0070→0.1813。但FIXED相对梯度误差反而增加约1.1%；NORMALIZED误差降低约29.3%。不能只看角度，也不能把两种正则曲线混为同一证据。
- 0→2的平均方向随查询增加反而变差：FIXED为0.0423→−0.0536，NORMALIZED为0.0493→−0.0536。更多标签并不保证当前读取器更接近目标任务需要的方向。
- n196已经读取整个既有池。重复排列得到同一估计，抽样离散度为数值零是没有剩余子集选择的结果，**不意味着这个估计没有统计噪声或已经正确**。
- 196之外更多独立监督是否有帮助没有测试；源标签量还会改变带标签覆盖，本轮也没有测到纯粹的标签噪声强度。

### 当前决定

暂不进入完整适配，不再将“逐图源标签迁移已成立”或“失败主要因为query太少”作为结论。若继续研究，应明确选择新假设：把预测校准作为独立候选，或针对更可靠的跨域条件关系设计新读取器；它们都需要新登记的对照，不能用本轮开发结果直接宣称成功。本轮不自动启动这些后续实验。

'''
    for name in ['结果解读.md','run_report.md','decision.md']:
        p=s/name;t=p.read_text();anchor='## 标签项与预测项：实验如何拆开';assert '## 核心结论：' not in t;t=t.replace(anchor,text+anchor);p.write_text(t)
    p=s/'run_report.md';p.write_text(p.read_text()+'''\n## 后处理与执行记录\n\n正式五种子一次完成，耗时约117.05秒，CPU五worker、GPU时0；无运行错误，无校准非收敛配置。初始模型D7与所有快照标签归因分开聚合。input-only derived_query_kernel_spectra.json补充105个唯一源子核谱，不读取标签或改变已完成实验。主结果、门槛和冻结执行代码未更改，后处理源码SHA单列。\n''')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();torch.set_num_threads(1);extend(a.stage)
