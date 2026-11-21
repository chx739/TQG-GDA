# R9 固定词频 dropout 的补充图复验

2026-11-21。固定原 3405 的 24 个来源内分类器块，只换评价图。**PROTEINS 的初测 +2.652 pp 在补充图上降到 +0.005 pp，纠正与误改各 12 张，也低于两种节点数基线。本轮只保留局部效果记录，不进入真实域试验。** 两种分子划分的 Accuracy 差仍为负。[登记协议](feature-dropout-transfer-check.md) · [初测与原始快照](feature-dropout-screen.md)

## 初测与补充的同口径比较

各设置八个方向等权。纠正／误改是和静态岭在同一批图上的成对改判。分母是方向图次。初测 E 与补充图在各设置内实体无交集。相同数据集的两种分子划分以及正反方向的模型角色存在实体复用。

| 设置 | 原 E 图次 | 初测 Target−Static Accuracy | 初测纠正／误改 | 补充图次 | 补充 Target−Static Accuracy | 补充 Macro-F1 差 | 补充纠正／误改 | 胜／平／负 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mutagenicity 边数 | 869 | −6.240 pp | 31／85 | 2,599 | −3.890 pp | −4.816 pp | 80／182 | 0／0／8 |
| Mutagenicity 节点数 | 865 | −2.757 pp | 43／67 | 2,607 | −0.891 pp | −0.790 pp | 132／160 | 3／0／5 |
| PROTEINS 边数 | 221 | +2.652 pp | 9／3 | 671 | +0.005 pp | −0.587 pp | 12／12 | 2／4／2 |

PROTEINS 的图次净值为零，+0.005 pp 来自方向样本数不同后的等权平均。它低于 +0.5 pp 投入门槛。两个分子设置的负方向与初测一致，负差幅度有所缩小。[初测汇总](../results/feature-dropout-transfer-001/score/initial-summary.csv) · [逐方向同图配对](../results/feature-dropout-transfer-001/score/paired-blocks.csv)

## 补充池分类端点

每格为 Accuracy／Macro-F1，单位为百分数。Count 沿静态分数降序、图号升序，选取与 Target 相同数量的阳性。Curve 只用该方向 D 的图总节点数和源标签拟合；CurvePool 用同一曲线在原 U 参照子集求平均后对补充池给统一分类。

| 方法 | Mutagenicity 边数 | Mutagenicity 节点数 | PROTEINS 边数 |
|---|---:|---:|---:|
| 目标词频 dropout（Target） | 73.293／70.578 | 74.348／72.816 | 73.147／50.887 |
| 静态岭（Static） | 77.183／75.394 | 75.238／73.606 | 73.141／51.474 |
| 等预算源参照（Source） | 75.335／73.441 | 74.525／72.881 | 71.949／49.770 |
| 分组置换（Perm） | 76.190／74.625 | 73.390／71.788 | 71.086／48.755 |
| 静态同阳性数（Count） | 75.998／73.553 | 75.278／73.656 | 73.136／50.528 |
| 节点数曲线（Curve） | — | — | 74.191／43.705 |
| 节点数池多数（CurvePool） | — | — | 73.922／42.249 |

在 PROTEINS，Target 相对同阳性数对照的差为 +0.011 pp，纠正与误改各 9 张，只有 1 个方向改善、1 个方向退化。初测的同阳性数差为 +0.862 pp，净纠正 2 张。补充池中 Target 优于等预算源参照 +1.197 pp、优于置换 +2.061 pp，分别低于两种节点数基线 1.044 和 0.775 pp。Target 的 Macro-F1 高于两种节点数基线，低于静态岭。协议以普通 Accuracy 决定投入。[绝对端点](../results/feature-dropout-transfer-001/score/summary.csv) · [所有对照的配对汇总](../results/feature-dropout-transfer-001/score/paired-summary.csv)

## 去掉一个源组

每次等权平均其余六个方向。下表为 Target−Static 的 Accuracy 百分点。

| 设置 | 去 M0 | 去 M1 | 去 M2 | 去 M3 |
|---|---:|---:|---:|---:|
| Mutagenicity 边数 | −4.623 | −4.819 | −4.319 | −1.800 |
| Mutagenicity 节点数 | −0.495 | −2.154 | −0.483 | −0.431 |
| PROTEINS 边数 | −0.194 | −0.196 | +0.201 | +0.210 |

PROTEINS 的小幅正差在去 M0 或 M1 后转负，且节省的错误未超过节点数基线。门槛回执为 `local_effect_only`。补充评价属于有既往初测信息的来源内开发验证，真实跨域数据没有进入本轮。[去源计数及分母](../results/feature-dropout-transfer-001/score/leave-source.csv) · [门槛回执](../results/feature-dropout-transfer-001/score/gate.json)

## 模型重建和封存

- 执行提交 `e62c8e7722989ab2459aed00aadf89c77dbd3fa9`，51 项 R9 工程回归通过。用 24 个原 D/C 角色逐方向重建四个已登记的岭分类器。24 块正则选值、词表大小和零保留词数一致，原 E 共 7,820 项连续分数绝对最大误差为 0，硬判和同阳性数结果均完全一致。[配置](../configs/feature-dropout-transfer-check.json) · [重建记录](../results/feature-dropout-transfer-001/model-replay.csv) · [工程日志](../results/feature-dropout-transfer-001/logs/r9-dropout-transfer-tests.log)
- OBS SHA-256 为 `b1bb43490443d5af0ed3d060dbf9844eeece7e5de564be18d8f189d849441862`，53 个成员。封存原角色、D/C 读取回执、参照子集、模型系数及补充图分数与判决。随后才打开 5,877 个补充方向图次的标签。没有重新读取原 E 标签。[OBS](../results/feature-dropout-transfer-001/OBS.json) · [图号与角色](../results/feature-dropout-transfer-001/roles.json) · [评分读取回执](../results/feature-dropout-transfer-001/score/label-access.json)
- FINAL SHA-256 为 `22c0a83993c717a7d238183bde433955d639a4e12efe26c1365ddfc150822e0e`，10 个成员。按混淆矩阵和同图净纠错复核 136 个端点与 112 个配对，所有封存成员摘要均匹配。[FINAL](../results/feature-dropout-transfer-001/score/FINAL.json) · [归档核验](../results/feature-dropout-transfer-001/archive-verification.json) · [执行日志](../results/feature-dropout-transfer-001/logs)
