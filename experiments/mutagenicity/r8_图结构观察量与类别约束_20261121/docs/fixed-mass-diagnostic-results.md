# R8 固定 A 均值后的斜率诊断

2026-11-21 完成登记的 10/10 个开发块。只复用已有 SourceCal、源融合斜率和 JointExact 斜率，在本方向无标签 A 上匹配 SourceCal 的平均正类概率，再把同一截距和斜率应用于 H。全部输出使用普通 argmax。目标标签仅在新 OBS 封存后读取 H 一次用于评分。[登记协议](fixed-mass-diagnostic.md)

## 工作点校正解释了 M0→M1 的主要损失

| 方法 | M0→M1 Accuracy | M0→M3 Accuracy | 两方向等权 |
| --- | ---: | ---: | ---: |
| SourceCal | 73.68% | 65.98% | 69.83% |
| 固定均值的源斜率 | 74.01% | 66.98% | 70.49% |
| 固定均值的目标斜率 | 75.11% | 67.99% | 71.55% |
| 固定均值的交换斜率 | 75.24% | 66.57% | 70.91% |
| 原源融合 | 74.40% | 66.57% | 70.48% |
| 原 JointExact | 68.73% | 66.57% | 67.65% |

同一目标斜率重新定截距后，M0→M1 的 Accuracy 较原 JointExact 恢复 **6.38 pp**，NLL 从 0.95258 降至 0.54087，AUROC 保持 81.593%。H 的平均预测正类比例从 4.82% 升至 40.65%。M0→M3 的 Accuracy 相对原 JointExact 提高 1.42 pp，NLL 从 0.72269 降至 0.64684，AUROC 保持 74.524%。

在两个方向内，四个固定均值臂的 A 平均正类概率分别为 42.25% 和 48.78%。同均值下，阈值后的正类比例仍会不同。M0→M3 的目标斜率在 H 上预测 49.23% 为正类，源斜率预测 49.29%。

## 固定 A 均值后，目标方向的专属斜率尚未跨方向占优

比较为固定均值目标斜率减指定对照，单位 pp。括号为条件于冻结模型的 95% 配对图区间，沿用同一目标域及类别分层的 2000 次抽样和种子间共享图抽样。

| 对照 | M0→M1 | M0→M3 | 两方向等权 |
| --- | ---: | ---: | ---: |
| SourceCal | +1.43 [+0.46, +2.54] | +2.01 [−0.83, +5.03] | +1.72 [+0.20, +3.27] |
| 固定均值的源斜率 | +1.11 [+0.26, +2.02] | +1.01 [−1.30, +3.43] | +1.06 [−0.18, +2.35] |
| 固定均值的交换斜率 | −0.13 [−1.24, +1.11] | +1.42 [−0.53, +3.49] | +0.64 [−0.50, +1.87] |
| 原源融合 | +0.72 [−0.20, +1.69] | +1.42 [−0.83, +3.79] | +1.07 [−0.17, +2.35] |
| 原 JointExact | +6.38 [+2.74, +9.97] | +1.42 [−1.24, +4.14] | +3.90 [+1.78, +6.02] |

相对固定均值源斜率，M0→M1 的 5 个种子共纠正 34 次、损害 17 次，M0→M3 共纠正 111 次、损害 94 次。同一方向复用 H 图，次数可以重计同一张图。相对交换斜率，M0→M1 为 35 次纠正、37 次损害，M0→M3 为 122 次纠正、98 次损害。原方案要求两个方向的差值都为正，M0→M1 相对交换斜率的净纠错为 −2，未满足。交换只用于诊断，使用了另一个目标域的既有斜率。原 R8 Accuracy 门槛按原试点结果结案。

两方向等权 NLL 为 SourceCal 0.60264、固定均值源斜率 0.59487、固定均值目标斜率 0.59385、固定均值交换斜率 0.60168、原源融合 0.59347。点估计提示原工作点压低了目标斜率的分类成绩。目标斜率相对源斜率的两方向 Accuracy 差区间跨零，且交换对照结果随方向翻转，当前试点不足以支持目标域专属估计的重复优势。

## 复现与完整性

独立重做 A 的一维求根、H 的配对纠错和区间后，120 个 A/H 分类动作数组均逐值一致。原始区间文件继承 R7 的候选风险描述，本次诊断使用冻结分类分数；修订版明确标记候选最大值未重算，并逐块给出 Accuracy、BA 和 NLL 的配对图抽样区间。从同一 OBS 重放，50 个方向×种子×对照的三种区间共 150 行，与独立计算在小数点后七位吻合。原来的 45 行汇总和所有净纠错保持一致。M3 相对交换斜率的 24 个净纠正中，种子 3406 贡献 30 个，其余四个合计净损失 6 个。[逐块区间与独立核验](../results/fixed-mass-review/seed-interval-replay.json)

执行代码为 `a7f8ff9`。原 OBS、原 FINAL 和新诊断的 OBS、FINAL 成员哈希均已重检。A 平均概率误差最大为 $4.44\times10^{-16}$，20 个 H 同斜率对的秩与并列逐块一致，纯截距偏差最大为 $2.66\times10^{-15}$，30 个历史端点逐值复现。43 项 R7/R8 相关工程测试通过。逐种子指标及配对净数在原始表，方向和整体区间在原始区间文件。逐种子区间由后续只读评分补充归档。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.scripts.fixed_mass \
  --config experiments/mutagenicity/r8_图结构观察量与类别约束_20261121/configs/fixed-mass-diagnostic.json \
  --output /tmp/r8-fixed-mass-diagnostic
```

[实际执行配置](../results/fixed-mass-diagnostic/executed-config.json) · [截距与斜率](../results/fixed-mass-diagnostic/intercepts.json) · [逐种子指标](../results/fixed-mass-diagnostic/endpoints.csv) · [纠正与损害](../results/fixed-mass-diagnostic/paired-seeds.csv) · [方向及整体配对区间](../results/fixed-mass-diagnostic/paired-intervals.json) · [OBS](../results/fixed-mass-diagnostic/OBS.json) · [FINAL](../results/fixed-mass-diagnostic/FINAL.json) · [复核记录](../results/fixed-mass-diagnostic/verification.json) · [测试日志](../results/fixed-mass-diagnostic/engineering-tests.log)
