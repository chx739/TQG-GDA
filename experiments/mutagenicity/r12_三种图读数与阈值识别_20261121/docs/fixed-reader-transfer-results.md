# R12 固定源端读数试验结果

2026-11-21，种子 3405。Mutagenicity 边数和 PROTEINS 边数各完成 12 个开发方向。**两个设置均未达到扩种子门槛。** SizeRange 在两套数据都比 Point 和 HashRange 准确，但相对指定强源起点，Mutagenicity 下降 1.157 个百分点，PROTEINS 仅增加 0.009 个百分点。[预先登记的协议](fixed-reader-transfer.md) · [配置](../configs/fixed-reader-transfer.json)

## 分类效果

Accuracy 和 Macro-F1 均以百分数呈现。先计算每个方向的指标，再对设置内的 12 个方向等权平均。Anchor 在 Mutagenicity 为 WL，在 PROTEINS 为 Size。

| 方法 | Mutagenicity Accuracy | Mutagenicity Macro-F1 | PROTEINS Accuracy | PROTEINS Macro-F1 |
|---|---:|---:|---:|---:|
| Anchor | 71.254 | 70.599 | 65.081 | 42.413 |
| **SizeRange** | **70.097** | **69.348** | **65.090** | **42.568** |
| Point | 62.462 | 58.718 | 52.618 | 39.672 |
| HashRange | 65.503 | 62.668 | 63.956 | 42.716 |
| EqualCount | 70.571 | 69.833 | 64.528 | 41.923 |
| Other1 | 68.744 | 67.243 | 63.128 | 42.873 |
| Other2 | 67.394 | 66.795 | 65.103 | 42.338 |
| OtherMean | 68.069 | 67.019 | 64.116 | 42.605 |
| B0 | 69.775 | 69.267 | 57.682 | 50.962 |
| WL | 71.254 | 70.599 | 53.583 | 45.233 |
| Size | 60.437 | 55.482 | 65.081 | 42.413 |
| B | 70.838 | 70.304 | 58.041 | 49.476 |
| B₂ | 71.493 | 70.885 | 57.808 | 47.673 |
| PoolMajority | 51.993 | 34.015 | 66.948 | 39.365 |
| DirectP | 69.263 | 68.543 | 65.990 | 56.357 |

下表每格为 SizeRange 减指定对照的 Accuracy 百分点；括号内为同图纠正/误改的图次。OtherMean 的图次是两个异域对照的平均。

| 对照 | Mutagenicity | PROTEINS |
|---|---:|---:|
| Anchor | −1.157（80 / 125） | +0.009（13 / 12） |
| Point | +7.636（657 / 364） | +12.472（188 / 62） |
| HashRange | +4.594（285 / 109） | +1.134（14 / 3） |
| EqualCount | −0.474（87 / 105） | +0.562（18 / 12） |
| Other1 | +1.353（208 / 155） | +1.961（25 / 8） |
| Other2 | +2.703（224 / 122） | −0.014（6 / 6） |
| OtherMean | +2.028（216 / 138.5） | +0.974（15.5 / 7） |

相对 Anchor，Mutagenicity 胜/平/负为 1/5/6，PROTEINS 为 1/9/2，均低于预定的 +0.5 个百分点且至少 7 个方向改善的门槛。Mutagenicity 同时低于 EqualCount、B 和 B₂。PROTEINS 同时低于 PoolMajority 和 DirectP。本域更新超过两异域更新的平均值。按登记结束当前固定规则，不补种子。[设置汇总](../results/attempt-003-fixed-reader-transfer/summary.csv) · [逐方向端点](../results/attempt-003-fixed-reader-transfer/endpoints.csv) · [逐方向配对](../results/attempt-003-fixed-reader-transfer/paired-blocks.csv) · [配对汇总](../results/attempt-003-fixed-reader-transfer/paired-summary.csv) · [门槛明细](../results/attempt-003-fixed-reader-transfer/gate.json)

## 覆盖、比例和改判

| 设置 | 方法 | H 类均值覆盖 | 类均值区间平均宽度 | H 实际比例覆盖 | 按 H 图数加权的比例覆盖 | 非空可行区间平均宽度 |
|---|---|---:|---:|---:|---:|---:|
| Mutagenicity | SizeRange | 38 / 89 | 0.1051 | 20 / 45 | 51.72% | 0.6003 |
| Mutagenicity | HashRange | 31 / 89 | 0.0913 | 11 / 45 | 26.21% | 0.4397 |
| PROTEINS | SizeRange | 33 / 69 | 0.3191 | 27 / 37 | 81.73% | 0.5827 |
| PROTEINS | HashRange | 31 / 69 | 0.2514 | 25 / 37 | 69.71% | 0.5414 |

H 类均值的分母是 H 中该类有图、源 V 中该档两类都有图且读数对比非零的组合。两种分组分别在 Mutagenicity 的 5 个及 PROTEINS 的 10 个已观测组合因源组缺类而使用全读数范围。每种方法另有 7 个和 27 个组合不可比较，分别由 H 缺类 1/19 和源 V 缺类 6/8 构成。比例覆盖只计 A/H 均有图且源读数可识别的方向档，空区间记作未覆盖。两种分组的人数相同，真实大小组得到的区间也更宽。[逐类覆盖](../results/attempt-003-fixed-reader-transfer/class-coverage.csv) · [逐档比例覆盖](../results/attempt-003-fixed-reader-transfer/prior-coverage.csv) · [覆盖汇总](../results/attempt-003-fixed-reader-transfer/coverage-summary.csv)

| 比例来源 | Mutagenicity 绝对误差 | PROTEINS 绝对误差 |
|---|---:|---:|
| 源比例 p₀ | 13.46 pp | 24.98 pp |
| **SizeRange** | **14.59 pp** | **25.27 pp** |
| Point | 31.69 pp | 35.23 pp |
| HashRange | 19.49 pp | 27.42 pp |
| Other1 | 17.28 pp | 24.16 pp |
| Other2 | 17.67 pp | 23.85 pp |

先在各方向按 H 档人数加权绝对误差，再对方向等权。所有有 H 图的回退档计入。两个设置的 SizeRange 都比 Point 和 HashRange 误差小，但比直接保留源比例误差大。真实大小分组扩大了经验覆盖，仍未恢复足够准确的部署比例；本轮也无法把分类差异单独归因于大小分组，因为两者的区间宽度不同。[估计误差汇总](../results/attempt-003-fixed-reader-transfer/proportion-summary.csv) · [逐方向误差](../results/attempt-003-fixed-reader-transfer/proportion-by-direction.csv) · [逐档估计与回退](../results/attempt-003-fixed-reader-transfer/proportion-errors.csv)

在可同时分解两项的档上，按 H 图数加权的绝对矩偏差，Mutagenicity 的 H 类条件漂移为 0.0423，A−H 均值差为 0.0163；PROTEINS 对应 0.0956 和 0.0548。分母分别是 45 档的 3,755 图次和 37 档的 1,007 图次。H 只出现一个类别时，缺失类别贡献按零计算。当前偏差支持优先检查类条件矩的跨域稳定性。在同一批 H 图上，SizeRange 实际仅改判 Mutagenicity 的 205/3,870 和 PROTEINS 的 25/1,020 图次。后者的分歧上界表明，按本次固定规则即便所有改判都正确，最多也只有 3 个方向改善。[逐档矩分解](../results/attempt-003-fixed-reader-transfer/moment-drift.csv) · [评分前动作上界](../results/attempt-003-fixed-reader-transfer/action-bounds.json)

## 复现与封存

- 执行提交 `eb74cb35eb01b736201b4cff910b69aee4d23c5b`，R12 相关工程测试 31 项通过。复用八个源的冻结探针和切点，无训练更新。源 V 的 1,090 个读数复现原记录，冻结阶段读取源 V 标签建表。目标 A 标签未读取。评分阶段读取原八个 H 角色，共 1,630 张图，按 24 个方向应用各自的固定预测，共 4,890 个方向图次。[执行配置](../results/attempt-003-fixed-reader-transfer/executed-config.json) · [测试日志](../results/attempt-003-fixed-reader-transfer/logs/r12-fixed-reader-engineering.log) · [H 读取回执](../results/attempt-003-fixed-reader-transfer/h-label-access.json)
- 封存 OBS 为 `2deb9b577e080920638d766cdd199c036fa34b3ce5d1cba573edeb7c203e6db1`，85 个成员。H 开分前，独立盲算核对了两套 V 分组与逐类计数、288 个比例，16,092 行 A/H 读数和 68,460 个判决。判决完全一致，六组更新前后 margin 的最大差异为 1.78e−15。[OBS](../results/attempt-003-fixed-reader-transfer/OBS.json) · [独立复核回执](../results/attempt-003-fixed-reader-transfer/review-receipt.json)
- FINAL 为 `bf355dec5a5cf2f753cf77f4980497b0dd0ce059f6954e7a8ce8f5061dce4b2d`，15 个成员。重载后用混淆矩阵和精确配对计数核对 336 行端点、336 行方向比较及 28 行等权配对汇总。这些结果来自既有开发 H、单一种子。[FINAL](../results/attempt-003-fixed-reader-transfer/FINAL.json) · [冻结日志](../results/attempt-003-fixed-reader-transfer/logs/r12-fixed-reader-freeze.log) · [评分日志](../results/attempt-003-fixed-reader-transfer/logs/r12-fixed-reader-score.log)
