# 用已公布的正负图数，能否判断条件秩规则在 PROTEINS 上的上限？

种子 3405 的 36 个方向已在目标类别盲态下独立生成条件秩动作。本次计算使用这些动作、图号和 R12 先前公布的八个票型各有多少正负图。它回答这批固定动作在那批图上最多可能取得多少净纠错。这里的区间是给定已有计数的可行范围。无标签算法的实际分类成绩需要另行评分。[新规则的冻结协议](conditional-rank.md) · [R12 已公开的逐格正负人数](../results/attempt-002-conditional-votes/h-cells.csv)

PROTEINS 的 P2→P3 可以直接手算。只看大小的源分类器把 83 张 H 图全判正，已知其中有 62 张为正。新规则将其中 57 张改判为负。即使这 57 张包含全部 21 张负图，仍至少误改 36 张正图，净损失最少 15 张。无论那 57 张图内部的排序有多好，这个方向的 Accuracy 相对大小基线都至少低 15/83，即 18.07 个百分点。这里失败的充分原因是判负数量过多。

对每一格同样能算更紧的上下界。给定原八票型中的一格含 \(n\) 张图、\(P\) 张正类，给每图记 \(d_i\) 为 OwnRank 二值预测减对照的二值预测。原来正确总数到新正确总数的差是 \(2\sum_{i:Y_i=1}d_i-\sum_i d_i\)。分别将 \(P\) 个正类分配给最大和最小的 \(d_i\)，即可得到这一格的精确最大与最小净纠错。先允许每个方向分别取最优时，PROTEINS 的十二方向平均差至多为 −0.693 pp，最多三个方向胜过 Size。

同一物理目标图在另外两个源方向也有同一类别。加上这一条件后，将每个目标域中三源票型和动作差均相同的图合为一组。以组中正类图数为整数变量，要求各源八格人数与历史表相符，可得下表。每一项均单独求最优和最差，各对照的极值可能需要不同的类别分配。两个分子设置复用了同一批图，本表在每个设置内独立求界。

| 设置 | OwnRank − 原起点的均值可行范围 | 最多胜原起点 | OwnRank − 只用源刻度的均值可行范围 | 最多胜源刻度 |
|---|---:|---:|---:|---:|
| Mutagenicity 边数 | [−5.607, +2.435] pp | 10/12 | [−4.619, +5.217] pp | 12/12 |
| Mutagenicity 节点数 | [−7.117, +1.093] pp | 10/12 | [−4.272, +3.174] pp | 11/12 |
| PROTEINS 边数 | [−2.585, −0.894] pp | 3/12 | [−2.439, +0.042] pp | 1/12 |

PROTEINS 的 16 个源域与票型组合中有 7 格源验证集为空，6 格选出恒定判决，仅 3 格能随目标参考池的分数分布改变阈值。对于此次特定规则，即使把逐图未知的真类安排在最有利的位置，这批目标图也达不到扩种子的 Accuracy 门槛。目标图上的连续分数仍可能排对同票型的正负图；已登记的排序指标和异半标签选阈值对照分别检验这一点及其分类价值。

[逐方向八格界](../results/conditional-rank-bounds/direction-bounds.csv) · [设置内联合最优和最差](../results/conditional-rank-bounds/shared-setting-bounds.csv) · [按物理目标分域的结果](../results/conditional-rank-bounds/shared-target-bounds.csv) · [输入摘要](../results/conditional-rank-bounds/inputs.json)

原始独立动作和复核程序收在[复算压缩包](../results/conditional-rank-bounds/recompute.zip)。解压到单独目录，运行 `python bounds.py --self-test` 可穷举检验三个微型例子。用 `python bounds.py --workdir . --actions actions.csv --h-cells /绝对路径/h-cells.csv --output bounds-output` 复算真实界限，其中 `h-cells.csv` 用本页首段所链接的原表。另一个独立评分器在动作核验后用原角色 H 重放，覆盖 36 个方向的 26,280 个分半交叉预测，全部与正式评分一致。[独立评分核验](../results/conditional-rank-bounds/independent-score.json)
