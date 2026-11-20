# 看到一处图结构，能预测同图远处另一处吗？

在 R9 的三个设置、各八个来源内大小方向上，用一半未标记伪目标 U 图学习局部结构的配对，在另一半 U 上预测。每个局部词是一个中心节点的离散类型及其一跳邻居的类型、边类型多重集。中心间距超过两跳的有序词对提供观测，避免两个一跳邻域直接重叠。在 D 按出现的图数固定十六个常见词，余词合并。每张有效图的全部词对总权重相同。档案列出无有效词对的图数与每折分母，两种分子划分复用实体。[词表、源锚及分折](../results/source-context-checks/source-anchors-and-fold-ids.json) · [逐折记录](../results/source-context-checks/anchor-folds.csv)

| 目标图上预测第二词的交叉熵，nat，越小越好 | 固定源两锚 | 本域两锚 | 本域第二词边际 | 本域完整条件表 |
|---|---:|---:|---:|---:|
| Mutagenicity 边数 | 2.0150 | 2.0009 | 1.9480 | 1.9409 |
| Mutagenicity 节点数 | 2.0296 | 2.0109 | 1.9637 | 1.9568 |
| PROTEINS 边数 | 2.5204 | 2.4752 | 2.4607 | 2.4232 |

完整条件表在 24/24 个方向都优于直接用本域词频。有些差值很小，蛋白 M3 大→小仅约 0.000049 nat，该方向 111 张 U 仅 50 张有有效词对。固定两个源锚点后，限制其他条件分布落在两个锚词条件分布之间，本域版本胜过本域词频的方向只有 0/8、0/8、2/8，48 个折的训练半均观察到两锚。在这里，域内远处结构的统计关系能跨图重复，而这两个固定锚点解释不了大部分配对优势。任意置换图的类别都不会改变上表，分类价值需要单独检验。锚点思想借鉴可分主题模型，本次两个锚的具体选取及拟合属于探索性诊断。[逐方向比较](../results/source-context-checks/anchor-directions.csv) · [精确汇总](../results/source-context-checks/summary.csv) · [Arora 等原文](https://proceedings.mlr.press/v28/arora13.html)

还检验了一种具体的图扰动解释。如果比例 q 的图在图内均匀打乱节点类型，其余图保留原样，那么在固定供体分布下，一跳和两跳节点对的同类型率减去随机放置期望后的两个超额量，都应缩放为源均值的 1−q 倍。分子边数 M0 小→大的一跳超额从 +0.036 变成 −0.033，节点数 M0 同向从 +0.020 变成 −0.040。正负号不同，任何合法强度都无法同时复现这两批观测间的一跳经验均值。此项只针对同一供体分布及随机置换机制，源图与目标图本来还可以有不同的结构分布。[两个矩及样本量](../results/source-context-checks/attenuation-directions.csv) · [输入哈希](../results/source-context-checks/input-hashes.json) · [原始复算脚本](../results/source-context-checks/original-checks.zip)

## 按真类建表后，能否多分对整张图？

另用每方向 D 的类别将局部词对分成两个类条件表。在独立源验证图 C 上，用该图归一词对频率作为权重，对条件表的逐对 log 似然比取平均。对照用相同的权重、词典、平滑及源类别先验，仅将条件概率换成第二词的类内边际。无有效词对的 22 张图均回退源先验。两个分类器的全部 1,955 个 C 图次分数先封存，再按各方向 C 身份评分。这个实验直接检验从共现过渡到分类的那一步。[固定公式与角色](../results/source-context-checks/classification/plan.md) · [预测封存](../results/source-context-checks/classification/freeze.json)

| 来源内验证 | 边际 → 条件 Accuracy | 纠正／误改 | 边际 → 条件 NLL |
|---|---:|---:|---:|
| Mutagenicity 边数 | 63.866% → 63.509% | 4／7 | 0.62636 → 0.63148 |
| Mutagenicity 节点数 | 64.369% → 63.763% | 9／14 | 0.62033 → 0.62271 |
| PROTEINS 边数 | 70.937% → 70.937% | 2／2 | 0.55038 → 0.56123 |

三个设置的条件读数均未提高平均 Accuracy，且 NLL 均变差。24/24 的结构预测改善和这个分类结果可以同时成立：两类都共有的配对规律能帮助预测另一个词，在正负类的概率比里贡献却很小。这一现象与该解释相容；本次真正验证的是上述固定特征与读出，仍可有少量不同的配对包含类别信息。当前停止扩大这项直接读出。[逐方向结果](../results/source-context-checks/classification/directions.csv) · [设置汇总](../results/source-context-checks/classification/summary.csv) · [原始脚本、预测、建表及访问记录](../results/source-context-checks/classification/original-checks.zip)
