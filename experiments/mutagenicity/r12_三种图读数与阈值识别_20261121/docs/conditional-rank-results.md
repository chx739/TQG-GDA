# R12 条件秩迁移结果

2026-11-21，种子 3405。三种划分各完成 12 个方向。**OwnRank 在三个设置的 Accuracy 均低于主起点、SignMix 和 R7 Centered，未达到扩种子门槛。** H 上的连续 B0 分数在同票型内部保留了可测的排序信息。本次用无标签 A 搬运源端切点，在三种划分上都没有将这项信息转成分类收益。[登记协议](conditional-rank.md) · [源端证据](conditional-rank-source.md)

## 正式无监督比较

Accuracy 为主要终点。每图一个真实类且只预测一个类时，按所有类别计算的 Micro-F1 等于 Accuracy。表中为 12 个方向等权平均，指标以百分数呈现。分子主起点是 WL，蛋白质主起点是 Size。R7 Centered 使用对应强起点的历史逐图预测。[GDABench 附录 E 表 15 与表 16](https://proceedings.neurips.cc/paper_files/paper/2024/file/a2c548b2ed3141400d86a235eaa6fad0-Paper-Datasets_and_Benchmarks_Track.pdf)

| 方法 | Mutagenicity 边数 Accuracy | Macro-F1 | Mutagenicity 节点数 Accuracy | Macro-F1 | PROTEINS 边数 Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Anchor | 71.254 | 70.599 | 72.776 | 72.100 | 65.081 | 42.413 |
| **OwnRank** | **70.057** | **69.719** | **69.280** | **67.851** | **63.614** | **49.666** |
| SourceRank | 69.726 | 69.175 | 71.035 | 69.733 | 64.718 | 50.593 |
| SignMix | 70.278 | 69.817 | 70.237 | 69.128 | 63.929 | 50.185 |
| Other1 | 70.062 | 69.565 | 68.280 | 66.486 | 64.144 | 49.611 |
| Other2 | 70.747 | 70.141 | 69.261 | 67.854 | 64.625 | 50.360 |
| OtherMean | 70.405 | 69.853 | 68.771 | 67.170 | 64.385 | 49.985 |
| CountMatched | 70.994 | 70.653 | 69.935 | 68.564 | 62.724 | 48.327 |
| B0 | 69.775 | 69.267 | 70.927 | 70.089 | 57.682 | 50.962 |
| WL | 71.254 | 70.599 | 72.776 | 72.100 | 53.583 | 45.233 |
| Size | 60.437 | 55.482 | 58.526 | 46.352 | 65.081 | 42.413 |
| B | 70.838 | 70.304 | 71.226 | 70.137 | 58.041 | 49.476 |
| B₂ | 71.493 | 70.885 | 71.341 | 70.350 | 57.808 | 47.673 |
| PoolMajority | 51.993 | 34.015 | 55.577 | 35.572 | 66.948 | 39.365 |
| DirectP | 69.263 | 68.543 | 70.659 | 69.701 | 65.990 | 56.357 |
| Static | 69.921 | 69.430 | 73.032 | 72.156 | 61.840 | 46.190 |
| R7 Centered | 71.932 | 71.241 | 72.925 | 72.200 | 65.620 | 43.176 |

下表为 OwnRank 减对照的 Accuracy 百分点，括号内为同图纠正／误改数。OtherMean 为两个供体对照的平均。CountMatched 按 B₂ 连续分数取与 OwnRank 相同的阳性数，截止并列时按均匀选择计算期望混淆矩阵，固定阳性数也使 Macro-F1 分母固定。

| 对照 | Mutagenicity 边数 | Mutagenicity 节点数 | PROTEINS 边数 |
|---|---:|---:|---:|
| Anchor | −1.197（173／217） | −3.496（140／274） | −1.467（56／70） |
| SourceRank | +0.331（135／122） | −1.755（79／146） | −1.104（11／22） |
| SignMix | −0.221（71／79） | −0.957（25／62） | −0.316（5／8） |
| OtherMean | −0.348（107.5／120） | +0.510（79.5／60） | −0.771（14／21.5） |
| CountMatched | −0.938（136／172） | −0.654（167／193） | +0.890（94／84） |
| R7 Centered | −1.875（192／262） | −3.644（165／305） | −2.007（69／88） |

相对主起点的胜／平／负分别为 **5／1／6、3／1／8、3／6／3**。以 R7 Centered 作敏感性对照，则为 **3／0／9、4／0／8、4／3／5**。两种口径均达不到 +0.5 pp 且至少七胜。开分前相对主起点实际存在动作分歧的方向分别为 12、12、7 个，涉及 390、414、126 个图次。PROTEINS 中 OwnRank 的 Macro-F1 比 Size 高 7.252 pp，且比等阳性数 B₂ 准 0.890 pp，仍低于 SourceRank、SignMix、两异域平均及多项固定参照。两个分子设置源于同一数据集。[逐方向端点](../results/attempt-005-conditional-rank/score/endpoints.csv) · [配对汇总](../results/attempt-005-conditional-rank/score/paired-summary.csv) · [开分前动作上界](../results/attempt-005-conditional-rank/action-bounds-summary.csv) · [门槛](../results/attempt-005-conditional-rank/score/gate.json)

## 错误来自阳性数还是格内选图

按四个 WL/Size 格分别计算。P 为真实正类数，K 为预测正类数，错误精确分为阳性数偏差 |P−K| 与等数目内可由更好排序避免的错误 2·min(FP,FN)。表内为各格和方向的图次总和。Anchor 每格恒定，第二项因此为零。

| 设置 | Anchor 总错误 | OwnRank 总错误 | OwnRank 阳性数偏差 | OwnRank 格内选图错误 |
|---|---:|---:|---:|---:|
| Mutagenicity 边数 | 1,115 | 1,159 | 801 | 358 |
| Mutagenicity 节点数 | 1,057 | 1,191 | 915 | 276 |
| PROTEINS 边数 | 357 | 371 | 361 | 10 |

两种分子划分里，OwnRank 缩小了阳性数差距，但格内选图错误超出节省的图次。PROTEINS 中 OwnRank 的 371 次错误有 361 次来自阳性数差距，当前改变切点时更大的问题是预测正类数偏离实际。[逐格计数](../results/attempt-005-conditional-rank/score/cell-errors.csv)

## 封存后监督诊断

同一 WL/Size 格内，对 H 正负图对逐方向计算一致率，连续分数相等计半。再等权平均有效方向。限制到三张硬票都相同后，B0 硬票自身必然为 50%。覆盖指 H 中落入有两类的格的图次比例，分别报告负类与正类。

| 设置 | 同 WL/Size 连续／硬票 | 正负图对数 | 三票同值连续／硬票 | 正负图对数 | 三票同值负／正类覆盖 |
|---|---:|---:|---:|---:|---:|
| Mutagenicity 边数 | 67.61%／57.87% | 86,116 | 64.19%／50.00% | 56,732 | 99.76%／99.15% |
| Mutagenicity 节点数 | 67.92%／58.04% | 100,089 | 64.64%／50.00% | 66,559 | 99.15%／100.00% |
| PROTEINS 边数 | 67.00%／56.57% | 11,337 | 64.04%／50.00% | 8,898 | 96.57%／95.34% |

三套设置、两种粒度各有 12/12 个方向可定义。即使所有硬票相同，B0 连续分数也保留了同格正负图对的排序能力。[逐方向排序](../results/attempt-005-conditional-rank/score/ranking-directions.csv) · [排序和覆盖汇总](../results/attempt-005-conditional-rank/score/ranking-summary.csv)

分半诊断按同目标图固定折号，两半分别选规则并互作评价集。HalfRank 在参考半用真实类别选格内比例。HalfFour 与 HalfEight 在相同参考半按四格／八格多数投票。表内为 Accuracy 与 Macro-F1。末列列出 HalfRank 相对主起点有分歧与胜出的方向数。

| 设置 | HalfRank Accuracy／Macro-F1 | HalfFour Accuracy／Macro-F1 | HalfEight Accuracy／Macro-F1 | HalfRank 相对 Anchor | 有分歧／胜出 |
|---|---:|---:|---:|---:|---:|
| Mutagenicity 边数 | 73.507%／72.391% | 73.191%／72.342% | 73.609%／72.695% | +2.252 pp（307／222） | 12／8 |
| Mutagenicity 节点数 | 72.328%／71.121% | 73.252%／70.364% | 72.623%／71.797% | −0.448 pp（237／255） | 12／3 |
| PROTEINS 边数 | 71.391%／53.740% | 72.090%／46.575% | 71.710%／49.867% | +6.310 pp（145／77） | 11／7 |

HalfRank 相对同折 HalfFour 的 Accuracy 差为 +0.315、−0.923、−0.699 pp，相对 HalfEight 为 −0.103、−0.295、−0.319 pp。PROTEINS 有可见的监督工作点余量，四格多数已经获得高于 HalfRank 的 Accuracy，HalfRank 则给出更高的 Macro-F1。这次分半诊断未建立连续切分超越同折八格多数的 Accuracy 优势。单次哈希分半的结果受每格样本数限制。[分半预测](../results/attempt-005-conditional-rank/score/half-diagnostic) · [诊断同图配对](../results/attempt-005-conditional-rank/score/paired-summary.csv)

## 复现与封存

- 执行代码为 `bc1eec4fd1242dda370ee5acb499aa43c730587f`，R12 相关 51 项工程回归通过。冻结使用 12 个源 V 的 1,955 条标签记录，复用 R11 等预算 A 的 20,223 个方向图次及 R7/R16 的已封存打分。无模型训练更新。正式评分按 12 个 H 角色打开 2,920 条图标签，应用于 8,760 个方向图次。A 标签未读取。[配置](../configs/conditional-rank.json) · [执行记录](../results/attempt-005-conditional-rank/executed-config.json) · [H 读取回执](../results/attempt-005-conditional-rank/score/h-label-access.json) · [测试日志](../results/attempt-005-conditional-rank/logs/r12-conditional-rank-tests.log)
- OBS SHA-256 为 `8ddd9ac95477c6759cac6643a38006354a7e2c2c5a606cd7e051d90aa5a1ffb3`，54 个成员。评分前独立核对 43,800 项硬判、48 个源格（含 7 个空格）与 1,989 个候选切分，全部一致。[OBS](../results/attempt-005-conditional-rank/OBS.json) · [盲算比对](../results/attempt-005-conditional-rank/independent-obs-verification.json) · [V 候选表](../results/attempt-005-conditional-rank/source-candidates.csv)
- FINAL SHA-256 为 `5c9fe3b463723fac5e6f4f8da0eacf70842461e8aec57b8c5b6556157ddf2d54`，47 个成员。归档核对了 720 个端点、756 个方向配对、1,152 项逐格错误和 144 项排序。独立评分复算还核对了 8,760 个分半分配及 26,280 项监督诊断逐图判决，全部一致。结果来自既有开发 H 和单一种子。[FINAL](../results/attempt-005-conditional-rank/score/FINAL.json) · [归档核对](../results/attempt-005-conditional-rank/archive-verification.json) · [独立评分复算](../results/attempt-005-conditional-rank/independent-final-verification.json) · [冻结日志](../results/attempt-005-conditional-rank/logs/r12-conditional-rank-freeze.log) · [评分日志](../results/attempt-005-conditional-rank/logs/r12-conditional-rank-score.log)
