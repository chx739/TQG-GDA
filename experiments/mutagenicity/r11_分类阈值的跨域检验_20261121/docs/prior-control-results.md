# R11 补充｜无标签类别先验对照结果

2026-11-21 完成 180/180 块，覆盖原 R11 的 3 个设置、12 个方向和 5 个种子。**MLLS 在 0/3 个设置通过门槛。** 用本域无标签 A 估计的先验在三个设置都胜过交换 A 的先验均值，但低于两个 Mutagenicity 设置的 B0，在 PROTEINS 上与 B0 几乎持平。[补充协议](prior-control.md) · [R11 主报告](../report.md)

## 同一批留出图的 Accuracy

SourceV 用源验证标签选阈值。SourceCal 用源验证标签拟合正温度和截距，保留校准后的零阈值。MLLS 用本域无标签 A 分数估计先验。SwapMean 平均同一源模型两个异域 A 的结果。PriorA 用本域 A 的真实正类比例替换 MLLS 估计。MajorityA 只预测该 A 子集的多数类。OwnA 沿用主实验中本域带标签 A 直接选出的阈值。

| 设置 | B0 | SourceV | SourceCal | MLLS | SwapMean | PriorA | MajorityA | OwnA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 69.04% | 68.89% | 69.23% | 67.05% | 65.50% | 70.00% | 55.98% | 71.19% |
| Mutagenicity 节点数 | 70.87% | 71.46% | 71.09% | 65.62% | 63.57% | 71.58% | 57.17% | 72.80% |
| PROTEINS 边数 | 55.46% | 55.48% | 54.99% | 55.46% | 41.09% | 65.09% | 72.09% | 73.92% |

以下是同图配对差值，单位为百分点。每个方向对五种子等权，设置内对十二方向等权。95% 区间使用 R7 原有的 2,000 次 H 图重采样，固定目标域内真类别数。每次抽样对同一目标域的各种子、源域及方法共享图权重。区间条件于固定模型、A 子集以及已拟合的参数。

| 设置 | MLLS−B0 [95% 区间] | MLLS−SourceV [95% 区间] | MLLS−SourceCal [95% 区间] | MLLS−SwapMean [95% 区间] | 后两项均为正的方向 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | −1.99 [−2.67, −1.30] | −1.83 [−2.46, −1.24] | −2.18 [−2.87, −1.48] | +1.55 [+1.05, +2.03] | 1/12 |
| Mutagenicity 节点数 | −5.25 [−6.24, −4.20] | −5.84 [−6.80, −4.82] | −5.47 [−6.46, −4.40] | +2.05 [+1.48, +2.64] | 3/12 |
| PROTEINS 边数 | −0.003 [−1.94, +1.90] | −0.02 [−1.53, +1.54] | +0.47 [−0.90, +1.82] | +14.38 [+13.80, +14.89] | 6/12 |

协议要求至少两个设置同时超过 B0、SourceV、SourceCal、SwapMean，其中相对后两项各至少 +1 pp，且至少 7/12 个方向同时为正。三个设置均未满足。PROTEINS 的交换 A 对照降到 41.09%，所以显著胜过交换对照与相对源模型近零收益可以同时成立。

## 类别比例与逐图纠错

| 设置 | MLLS 估计减 A 正类比例 | 绝对误差均值 | PriorA−MLLS [95% 区间] | PriorA−SourceCal [95% 区间] | PriorA−OwnA [95% 区间] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | +8.71 pp | 14.06 pp | +2.95 [+2.40, +3.51] | +0.77 [+0.26, +1.30] | −1.19 [−1.72, −0.65] |
| Mutagenicity 节点数 | +3.05 pp | 20.84 pp | +5.96 [+5.15, +6.73] | +0.48 [−0.14, +1.10] | −1.22 [−1.74, −0.71] |
| PROTEINS 边数 | +35.57 pp | 40.10 pp | +9.63 [+8.48, +10.80] | +10.10 [+8.78, +11.40] | −8.83 [−10.14, −7.51] |

替换成 A 的样本比例后，三种设置均优于 MLLS。这说明在既有校正规则内，比例估计误差确实影响了 Accuracy。三种设置仍低于直接用 A 标签选阈值的 OwnA。PROTEINS 上，PriorA 还比 A 多数类低 7.00 pp [−8.65, −5.38]，所以仅改进 MLLS 的比例估计不足以恢复 R11 的主要监督收益。A 的比例是同一子集上的估计，现有比较无法单独归因于源校准或类条件分数漂移。

| 设置 | MLLS BA | MLLS 负类／正类召回 | MLLS 常数预测块 | PriorA 常数预测块 | MLLS−MajorityA Accuracy [95% 区间] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 66.11% | 60.47%／71.74% | 3/60 | 0/60 | +11.07 [+9.80, +12.33] |
| Mutagenicity 节点数 | 62.79% | 63.94%／61.64% | 10/60 | 0/60 | +8.46 [+7.39, +9.55] |
| PROTEINS 边数 | 52.73% | 24.77%／80.68% | 43/60 | 18/60 | −16.63 [−17.51, −15.83] |

PROTEINS 的 SourceCal 负类／正类召回为 63.87%／51.21%，BA 为 57.54%。MLLS 的 Accuracy 近乎持平掩盖了两个类别之间的错误迁移。两个 Mutagenicity 设置中 MLLS 仍胜过 A 多数类，但各自低于已有 B0 和 SourceCal。逐块净纠错以及其他各臂的 BA、召回和常数预测计数见下方表格。

## 求解器、封存与复核

60 个源验证校准器全部可用。180 次本域先验拟合的 EM 与独立有界根解均返回结果。按边数 Mutagenicity、节点数 Mutagenicity、PROTEINS 排列，EM 触边分别为 1、5、31 块，根解为 1、5、33 块。两者估计最大绝对差为 6.28 × 10⁻⁸，本域及异域共 540 个预测数组逐图一致。EM 保留为登记的主臂。

使用提交 `9a07330` 执行。无标签阶段先封存 V 校准、MLLS、交换对照和留出图预测。第二阶段读取 21 个去重后的已登记 A 子集，只使用类别计数。第三阶段在 OBS 后读取 12 个留出图队列。独立归档核验重放 180 块预测、对齐主实验 540 个 B0／SourceV／OwnA 端点，并校对 4,860 条配对净纠错。24 项 R11/R7 工程回归通过。无新 GNN，使用此前已打开的开发设置。

[实际配置](../results/attempt-002-prior/executed-config.json) · [无标签封存](../results/attempt-002-prior/UNLABELED.json) · [监督后封存](../results/attempt-002-prior/OBS.json) · [评分封存](../results/attempt-002-prior/FINAL.json) · [独立归档核验](../results/attempt-002-prior/verification.json) · [运行日志](../results/attempt-002-prior/run.log) · [工程测试](../results/attempt-002-prior/engineering-tests.log)

[逐块 Accuracy／BA／召回](../results/attempt-002-prior/endpoints.csv) · [分设置的指标与常数预测](../results/attempt-002-prior/endpoint-summary.csv) · [逐块纠正与损害](../results/attempt-002-prior/paired-blocks.csv) · [配对汇总及区间](../results/attempt-002-prior/paired-summary.csv) · [先验误差](../results/attempt-002-prior/prior-error-summary.csv) · [EM 与独立根解](../results/attempt-002-prior/solver-parity.csv) · [登记门槛](../results/attempt-002-prior/gate.json)
