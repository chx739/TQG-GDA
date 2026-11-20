# R17｜同图外围读出一致性在 24 个源内方向的结果

**TargetPair 在三个设置均低于固定 C=1 的源 WL 和等预算 SourcePair，未达到真实四域开发门槛。** PROTEINS 在八方向中只有一个方向改变了分类，两次改判都使原正确分类变错。无标签动作分歧在评分前已排除 PROTEINS 满足「至少三个正增益源组」的可能。[登记协议](docs/paired-readout.md) · [冻结动作上界](results/attempt-001-paired-readout/action-bounds.json)

## 主结果

种子 3405，复用 R9 的原 D/C/U/E 角色。每个设置内八方向等权。TargetPair 相对固定 C=1 WL 的结果如下。

| 设置 | WL → TargetPair Accuracy | ΔAccuracy pp | ΔMacro-F1 pp | ΔNLL | 纠正／误改（净纠错） | 正／平／负方向 | 正增益源组 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 70.801% → 69.553% | −1.248 | −1.872 | +0.0138 | 14／25（−11） | 2／0／6 | 1／4 |
| Mutagenicity 节点数 | 67.705% → 66.558% | −1.147 | −1.834 | +0.0048 | 13／23（−10） | 2／1／5 | 0／4 |
| PROTEINS 边数 | 75.343% → 74.417% | −0.926 | −2.505 | +0.0042 | 0／2（−2） | 0／7／1 | 0／4 |

纠正和误改是同一个 E 图在本方向上的配对计数，各方向 Accuracy 差再取等权。PROTEINS 的两次改判集中在 M2 大→小的 27 张 E 图，单方向下降 7.407 pp。其余七方向逐图判决相同。封存后复核了全部 192 个端点和 168 个逐方向比较。[各方向完整指标](results/attempt-001-paired-readout/endpoints.csv) · [逐图配对汇总](results/attempt-001-paired-readout/paired-blocks.csv)

## 目标配对、源配对与错配

表内差值为 TargetPair 相对各对照的八方向等权 Accuracy，单位为百分点。

| 设置 | SourcePair | Shuffled | 等阳性数 | 源选模 WL | Size | 联合 Base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | −1.046 | +11.014 | −1.144 | −6.670 | +9.618 | −4.509 |
| Mutagenicity 节点数 | −0.238 | +7.711 | +0.203 | −8.179 | +8.858 | −8.889 |
| PROTEINS 边数 | −0.926 | −1.788 | −0.926 | +2.386 | +5.737 | +2.824 |

两种 Mutagenicity 划分中，同图配对优于错配；同样预算下用源图构成正则，表现均高于目标配对。两设置相对等阳性数对照分别纠正／误改 **9／19** 和 **19／17**，额外排序收益随划分变化。PROTEINS 的目标配对低于三个主要机制对照。Shuffled 同时改变差矩阵的方向和归一化标量，其差距衡量真实配对的整体作用。[各源组与设置配对汇总](results/attempt-001-paired-readout/paired-summary.csv)

下表每格依次为 Accuracy %、Macro-F1 %、NLL。等阳性数对照定义硬判决。

| 方法 | Mutagenicity 边数 | Mutagenicity 节点数 | PROTEINS 边数 |
| --- | ---: | ---: | ---: |
| 固定 C=1 WL | 70.801／67.406／0.5774 | 67.705／63.379／0.6009 | 75.343／48.188／0.5619 |
| TargetPair | 69.553／65.535／0.5912 | 66.558／61.545／0.6057 | 74.417／45.683／0.5661 |
| SourcePair | 70.600／66.501／0.5901 | 66.796／62.051／0.6048 | 75.343／48.188／0.5632 |
| Shuffled | 58.539／46.075／0.6512 | 58.847／45.521／0.6547 | 76.205／46.763／0.5870 |
| 等阳性数 | 70.697／66.751／— | 66.355／61.444／— | 75.343／47.349／— |
| 源选模 WL | 76.223／74.071／0.5046 | 74.737／72.428／0.5623 | 72.031／51.389／0.5993 |
| Size | 59.935／47.632／0.7939 | 57.700／47.812／0.6713 | 68.680／41.627／0.6144 |
| 联合 Base | 74.062／70.211／0.6579 | 75.447／71.294／0.5986 | 71.593／48.248／0.6337 |

源组相对固定 WL 的 Accuracy 差按组内两方向等权。边数 Mutagenicity 的 M0–M3 分别为 −1.385、+0.921、−1.835、−2.692 pp。节点数为 −2.710、0.000、−1.393、−0.485 pp。PROTEINS 为 0.000、0.000、−3.704、0.000 pp。[全部指标及源组汇总](results/attempt-001-paired-readout/summary.csv) · [按整数正确数计算的门槛](results/attempt-001-paired-readout/gate.json)

## 无标签上界与参考支持

| 设置 | TargetPair 对固定 WL 的最大可能 Accuracy 增量 | 最多改善方向 | 最多改善源组 |
| --- | ---: | ---: | ---: |
| Mutagenicity 边数 | +4.452 pp | 8／8 | 4／4 |
| Mutagenicity 节点数 | +4.110 pp | 8／8 | 4／4 |
| PROTEINS 边数 | +0.926 pp | 1／8 | 1／4 |

上界只依赖两种预测的分歧，取各方向分歧率的等权均值。PROTEINS 的均值上界仍允许超过 0.5 pp，但一个可能改善的源组低于所需三个。结合已评分的历史固定 WL，分子设置即使将分歧全改对，也未达到已知源选模 WL 的 Accuracy。[上界与强起点对比](docs/source-screen-sensitivity.md)

| 设置 | 每侧参考 m | TargetPair 非零行 | TargetPair 的 T | 单行最大平方范数占比 | TargetPair／SourcePair 分歧图次 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 396–427 | 396–427 | 306.082–417.694 | 0.437% | 25 |
| Mutagenicity 节点数 | 386–419 | 386–419 | 309.338–410.108 | 0.440% | 12 |
| PROTEINS 边数 | 107–112 | 1–32 | 0.00384–0.740 | 100% | 2 |

PROTEINS 选择了约百张图，但有些方向经 D 词表投影后，整个配对惩罚只由一张参考图贡献。三个设置的 72 个增广拟合均使用非零惩罚，归一化迹为 1。逐方向保留 T、支持行数、最大单行占比和源目标配对分歧。[完整几何与求解诊断](results/attempt-001-paired-readout/geometry.csv)

## 证据范围与复核

按登记结束这项固定外围读出与损失的变体。源内大小差、训练规模和真实域分布均可能不同。已发布的同种子参照中，PROTEINS 的源选模 WL 与 Size 在大小伪方向上的优劣顺序与真实迁移相反。本轮可以否决该固定方案的投入；真实跨域分类价值仍需同图验证。[源内与真实任务的历史比较](docs/source-screen-sensitivity.md) · [独立的自然包含诊断](docs/natural-containment-results.md)

执行与评分均使用 `2169350`。只解析各方向 D 的拟合标签，C 的原作用由冻结基线保留；E 标签仅在 24 方向全部封存、重载和盲算复核后按原角色打开。24 个方向的 1,955 个 E 图次、八方法共 **15,640 项判决**与独立实现逐字节一致。比较了 24 组参照选择、72 个目标函数，最大目标差为 9.33×10⁻¹⁵，最大一阶残差为 2.45×10⁻⁸。四项历史参照的 96 行既有端点按各自可用指标复现。8 项专项回归通过。

[实际配置](results/attempt-001-paired-readout/executed-config.json) · [OBS](results/attempt-001-paired-readout/OBS.json) · [FINAL](results/attempt-001-paired-readout/FINAL.json) · [D 读口](results/attempt-001-paired-readout/fit-label-access.json) · [E 读口](results/attempt-001-paired-readout/e-label-access.json) · [模型与参考 ID](results/attempt-001-paired-readout/models) · [独立盲算](results/attempt-001-paired-readout/blind-review/FROZEN.json) · [盲算比对](results/attempt-001-paired-readout/blind-verification.json) · [评分后核验](results/attempt-001-paired-readout/final-verification.json) · [测试日志](results/attempt-001-paired-readout/engineering-tests.log) · [拟合日志](results/attempt-001-paired-readout/freeze.log) · [评分日志](results/attempt-001-paired-readout/score.log)
