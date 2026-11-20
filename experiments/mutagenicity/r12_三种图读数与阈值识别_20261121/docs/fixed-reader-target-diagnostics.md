# R12｜给出目标正类比例后，原赔率校正能改对多少图？

固定源端读数的无监督 SizeRange 在 Mutagenicity 边数与 PROTEINS 边数的 Accuracy 分别为 70.10% 和 65.09%，相对各自强源起点变化 −1.16 和 +0.01 个百分点。即使无标签比例估计有误，投入更好的估计器也需要一个前提：把更准的比例交给同一分类规则，能在留出图上多分对几张。下面用两种不同的目标标签信息检验这个前提。[主试验实测](fixed-reader-transfer-results.md) · [开分前登记](fixed-reader-identification.md)

## 比例与阈值的配对比较

两个设置各含十二个开发方向，种子 3405。Anchor 是各自在源域训练后冻结的分类器，分子用 WL，蛋白质用节点数和边数。每个方向都按冻结的源 GNN 分数将图分为四档。把每个物理目标域的留出图 H 固定分成两半，来自不同源模型的同一目标图始终同折。在参考半的每个档读取标签，随后在另一半的同档预测，交换两半并汇总。两种监督诊断共用源模型、分档和回退规则。常规无监督方法只用独立适配池 A 的无标签读数产生更新。[实际折号](../results/fixed-reader-target-diagnostics/physical-folds.csv) · [逐档回退](../results/fixed-reader-target-diagnostics/fallback.csv)

| 规则及所用目标信息 | Mutagenicity Accuracy | 相对 Anchor 的纠正／误改 | PROTEINS Accuracy | 相对 Anchor 的纠正／误改 |
|---|---:|---:|---:|---:|
| 强源 Anchor | 71.254% | — | 65.081% | — |
| 无标签 SizeRange，正式部署规则 | 70.097% | 80／125 | 65.090% | 13／12 |
| 参考半给每档正类比例，按原赔率公式调分数 | 68.963% | 335／425 | 64.129% | 36／46 |
| 参考半给单图标签，直接选每档分类阈值 | 71.234% | 205／207 | 67.761% | 156／127 |

Accuracy 为方向等权均值，纠正和误改按图次求和。若只知道另一半在每档有多少正例，原赔率修正相对 Anchor 分别下降 2.292 和 0.952 个百分点。若还能知道参考半的正例排在源分类器分数的什么位置，就可以直接在这一半选阈值。后一诊断在两个设置都只赢了 4/12 个方向。分半后的档内比例和阈值都有抽样误差。[完整汇总](../results/fixed-reader-target-diagnostics/summary.csv) · [逐方向结果](../results/fixed-reader-target-diagnostics/directions.csv) · [逐折逐档结果](../results/fixed-reader-target-diagnostics/cells.csv)

## 两个看过主结果后的复核

1. **消除参考半比例的抽样差异。** 将每个 H 档内的实际正类比例直接代入原公式，保留原来的赔率裁剪和回退。这个诊断使用了被评分图自己的类别，只描述这批图。Mutagenicity 的 Accuracy 为 69.651%，低于 Anchor **1.603** 个百分点。PROTEINS 为 65.674%，高于 Anchor **0.593** 个百分点，3 胜、8 平、1 负。分半估计影响了蛋白质上的效应符号，这一数据也保留了目标比例在部分方向有用的可能。[事后诊断汇总](../results/fixed-reader-target-diagnostics/sensitivity/summary.csv)
2. **给直接选阈值的办法加上相同的移动范围。** 在参考半直接找准确率最高的阈值，但阈值只能在任意合法比例经过原赔率裁剪后可实现的范围内移动。这样仍用另一半计分。Mutagenicity 相对 Anchor 为 −0.021 个百分点，PROTEINS 为 +2.938 个百分点。PROTEINS 的 M1→M3 单独纠正 61 张、误改 21 张，共 83 张，给十二方向均值贡献了 +4.016 个百分点。去掉这一方向，另外十一个方向平均为 −1.176 个百分点。放开范围时同一方向也净纠正 40 张，其余方向为 −1.457 个百分点。该方向的改善能在原范围内实现，仍须解释部署时怎样识别它，以及怎样从无标签信息选中有益的移动。[单方向贡献](../results/fixed-reader-target-diagnostics/sensitivity/proteins-M1-M3.csv) · [逐折选中的阈值](../results/fixed-reader-target-diagnostics/sensitivity/cells.csv)

## 对下一轮投入的影响

分子数据给出的反例最清楚。在当前强源分数、分档、赔率映射和同批图上，精确知道各档有多少正例，平均仍会多分错图。继续提高同一比例估计器的精度，缺少值得扩大实验的分类证据。蛋白质上的正结果则有明确边界。某个源与目标的组合确实存在可迁移的阈值，但一个方向就足以使总平均由负转正。更值得验证的具体假设是，无标签信息能否指出**哪些原判决会错**，并在未用于设计的图上比沿原排序匹配预测阳性数多纠正图。两个设置使用单一种子和已参与开发的留出图。监督诊断只用于定位问题，沿用主试验事前登记的扩种子判定。

独立复核按原授权读取八个 H 角色，重建正式计分的十一张表，核对 2,102 行。分半诊断另以逐阈值枚举和累计计数对照，并验证更改查询半标签不会改变由参考半产生的动作。[正式结果的独立复算](../results/fixed-reader-bounds/independent-final-verification.json) · [开分前动作复算](../results/fixed-reader-bounds/independent-obs-verification.json) · [监督诊断核验](../results/fixed-reader-target-diagnostics/audit.json) · [参考折隔离检查](../results/fixed-reader-target-diagnostics/reference-check.json) · [事后检查及数据使用](../results/fixed-reader-target-diagnostics/sensitivity/audit.json)
