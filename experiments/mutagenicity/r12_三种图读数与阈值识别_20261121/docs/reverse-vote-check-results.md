# R12 图级反向验证结果

2026-11-21，种子 3405。三套设置各 12 个方向，全部完成。**Reverse 在三套设置均低于强源起点及 Direct，未达到补种子门槛。** 相比破坏逐图对应的 RowPerm，Reverse 在三套设置都更准；这一增量仍不足以支持用反向验证挑选目标域判决表。[登记协议](reverse-vote-check.md) · [RowPerm 对照](reverse-vote-identification.md)

## 分类端点

每设置按 12 个方向等权，表内指标均为百分数。Mutagenicity 的 Anchor 为 WL，PROTEINS 的 Anchor 为 Size。CountMatched 用精确期望计分，跨票型的数学同分共享同一个选择比例。其他两域的平均值先在同一 H 上计算，再跨方向平均。

| 方法 | Mutagenicity 边数 Accuracy | Macro-F1 | Mutagenicity 节点数 Accuracy | Macro-F1 | PROTEINS 边数 Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | 71.254 | 70.599 | 72.776 | 72.100 | 65.081 | 42.413 |
| **Reverse** | **68.527** | **67.605** | **72.596** | **71.931** | **53.744** | **46.137** |
| Direct | 71.254 | 70.599 | 72.776 | 72.100 | 63.786 | 49.209 |
| PermMean | 56.828 | 55.693 | 48.972 | 48.091 | 50.760 | 44.278 |
| RowPerm | 57.595 | 55.467 | 55.585 | 54.176 | 51.085 | 41.543 |
| Other1 | 67.231 | 66.177 | 72.324 | 71.566 | 53.817 | 42.300 |
| Other2 | 68.961 | 68.000 | 71.024 | 69.827 | 64.408 | 44.401 |
| OtherMean | 68.096 | 67.088 | 71.674 | 70.697 | 59.113 | 43.351 |
| CountMatched | 70.833 | 69.929 | 72.437 | 71.779 | 53.047 | 45.334 |
| B0 | 69.775 | 69.267 | 70.927 | 70.089 | 57.682 | 50.962 |
| WL | 71.254 | 70.599 | 72.776 | 72.100 | 53.583 | 45.233 |
| Size | 60.437 | 55.482 | 58.526 | 46.352 | 65.081 | 42.413 |
| B | 70.838 | 70.304 | 71.226 | 70.137 | 58.041 | 49.476 |
| B₂ | 71.493 | 70.885 | 71.341 | 70.350 | 57.808 | 47.673 |
| PoolMajority | 51.993 | 34.015 | 55.577 | 35.572 | 66.948 | 39.365 |
| DirectP | 69.263 | 68.543 | 70.659 | 69.701 | 65.990 | 56.357 |

下表为 Reverse 减对照的方向等权 Accuracy 百分点。括号内为由同图聚合格计数得出的纠正／误改图次。

| 对照 | Mutagenicity 边数 | Mutagenicity 节点数 | PROTEINS 边数 |
|---|---:|---:|---:|
| Anchor | −2.727（167／275） | −0.181（12／19） | −11.337（164／276） |
| Direct | −2.727（167／275） | −0.181（12／19） | −10.041（119／218） |
| PermMean | +11.699（1,031／577） | +23.624（1,462／555） | +2.984（114／82） |
| RowPerm | +10.932（923／495） | +17.011（1,166／508） | +2.659（60／32） |
| Other1 | +1.296（286／239） | +0.272（34／23） | −0.073（149／144） |
| Other2 | −0.434（228／247） | +1.571（132／71） | −10.664（116／221） |
| OtherMean | +0.431（257／243） | +0.922（83／47） | −5.368（132.5／182.5） |
| CountMatched | −2.306 | +0.158 | +0.697 |

相对 Anchor 的胜／平／负分别为 **1／7／4、1／9／2、4／1／7**。三个设置均未达到 +0.5 pp 和至少 7 个方向改善，且均输给各自最佳固定参照 B₂、WL、PoolMajority。PROTEINS 的 Macro-F1 相对 Anchor 上升 3.724 pp，Accuracy 下降 11.337 pp。Reverse 胜过 RowPerm 的方向数依次为 10、9、4。两者差异对应保留 V 上逐图标签和反向预测的配对，在这批开发图上有可量化贡献；相比有源标签的 Direct，对目标表的选择仍退化。[逐方向端点及召回](../results/attempt-004-reverse-vote/endpoints.csv) · [配对方向表](../results/attempt-004-reverse-vote/paired-blocks.csv) · [配对汇总](../results/attempt-004-reverse-vote/paired-summary.csv) · [门槛](../results/attempt-004-reverse-vote/gate.json)

## 目标域和判决空间

Reverse 相对 Anchor 的 Accuracy 差，单位为百分点。逐目标平均三个源方向。删去一个目标后，等权平均余下九个方向。

| 设置 | M0 | M1 | M2 | M3 | 去 M0 | 去 M1 | 去 M2 | 去 M3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mutagenicity 边数 | −4.747 | +0.760 | −2.484 | −4.438 | −2.054 | −3.890 | −2.808 | −2.157 |
| Mutagenicity 节点数 | +0.104 | −0.531 | 0.000 | −0.297 | −0.276 | −0.064 | −0.241 | −0.142 |
| PROTEINS 边数 | −18.900 | −19.718 | +9.738 | −16.466 | −8.815 | −8.543 | −18.362 | −9.627 |

评分前只有 5、3、11 个方向的 Reverse 预测与 Anchor 有过改判，分别涉及 442／3,870、31／3,870、440／1,020 图次。两种分子设置即使所有改判都正确，胜出方向数仍不足 7。PROTEINS 有 9/12 个方向的最优 Reverse 候选在源 V 上与次优候选精确同分，分子边数和节点数分别有 1/12、5/12 同分。[分组汇总](../results/attempt-004-reverse-vote/summary.csv) · [评分前动作上界](../results/attempt-004-reverse-vote/action-bounds-summary.csv) · [全部候选损失](../results/attempt-004-reverse-vote/candidate-losses.csv)

| 设置 | 逐方向事后最优 Accuracy | 相对 Anchor | 同源三目标共用表 Accuracy | 相对 Anchor | 逐方向最优减 Reverse |
|---|---:|---:|---:|---:|---:|
| Mutagenicity 边数 | 73.501% | +2.247 pp | 73.063% | +1.808 pp | +4.974 pp |
| Mutagenicity 节点数 | 73.278% | +0.501 pp | 72.776% | +0.000 pp | +0.682 pp |
| PROTEINS 边数 | 73.734% | +8.653 pp | 67.191% | +2.110 pp | +19.990 pp |

PROTEINS 的四格动作空间仍有可利用的事后余量。逐方向选表比每源共用一张好 6.543 pp，当前反向规则也远低于这两个参考。Mutagenicity 节点数的四格空间本身只比强源起点多 0.501 pp。这些参考都用旧 H 的聚合真类数选表，刻画的是这批图上四格规则的表达空间。[逐方向与同源事后最优表](../results/attempt-004-reverse-vote/oracle-headroom.csv)

## 复现与封存

- 执行代码为 `94cb0436cf1711000031cdc5673741ed3cedb2a6`，R12 相关 42 项工程测试通过。冻结只读取原 12 个源 V 的标签，共 1,955 条记录和 20,223 个 A 方向图次。H 端复用 8,760 个方向图次的原硬判。无新训练。[配置](../configs/reverse-vote-check.json) · [执行记录](../results/attempt-004-reverse-vote/executed-config.json) · [测试日志](../results/attempt-004-reverse-vote/logs/r12-reverse-vote-tests.log)
- OBS SHA-256 为 `f3891107308cf44dc6a03479925006461eb6121c7d905a5962afabd0de123a80`，43 个成员。评分前独立核对 36 个方向、2,304 项精确候选损失、23,460 项 V→A 票型权重和 144 项按原 H 顺序的动作摘要，全部一致。[OBS](../results/attempt-004-reverse-vote/OBS.json) · [复核回执](../results/attempt-004-reverse-vote/independent-obs-verification.json) · [冻结日志](../results/attempt-004-reverse-vote/logs/r12-reverse-vote-freeze.log)
- FINAL SHA-256 为 `cc3f3c7496a8ca7af0234f1dd582b26b1a92dab25f9e52c94dffa6ddcc531c07`，9 个成员。评分仅打开原已发布的 `h-cells.csv` 和 `endpoints.csv`，没有读目标逐图类别。独立重算核对了 648 个端点、540 个方向配对、36 个方向的事后表达上限、486 行分组汇总、45 行配对汇总和门槛，全部一致。结果来自既有开发 H、单一个种子。[FINAL](../results/attempt-004-reverse-vote/FINAL.json) · [独立评分核对](../results/attempt-004-reverse-vote/independent-final-verification.json) · [聚合评分回执](../results/attempt-004-reverse-vote/historical-aggregate-access.json) · [评分日志](../results/attempt-004-reverse-vote/logs/r12-reverse-vote-score.log)
