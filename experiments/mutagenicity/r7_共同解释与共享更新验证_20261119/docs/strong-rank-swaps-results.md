# R7 强基座上的等量重排归因

2026-11-21。E 在 Mutagenicity 节点数设置相对同阳性数 B0 排序净纠正 116 图次，其中 91 图次对应的净收益已反映在 WL 动作中。只交换 WL 尚未采用的 E 动作并严格保持 WL 阳性数，期望净纠正降到 6.626 图次，即 **+0.032 个 Accuracy 百分点**。同预算下本域优于异域均值 **+0.128 pp**，但相对 WL 的小幅收益在删去一个源域后最低为 −0.075 pp。分子边数及 PROTEINS 的指定基座均为负收益。[事后诊断协议](strong-rank-swaps.md)

## E 的排序收益落在哪些图上

(C) 是按 B0 排名、和该 E 头预测相同阳性数的冻结动作。对每张 E 与 C 判决不同的图，强基座 B 必须在两者中选一个标签。B=C 的图可用于本次交换。B=E 的图已经采用了 E 的判决。表内每格为 E 相对 C 的纠正／误改（净）。它们是图次，不是去重的图数。

| 本域 E 与主基座 | B=C，0→1 | B=C，1→0 | B=E，0→1 | B=E，1→0 | E−C 合计净 |
|---|---:|---:|---:|---:|---:|
| 分子边数，WL | 85／141（−56） | 113／96（+17） | 150／85（+65） | 172／80（+92） | +118 |
| 分子节点数，WL | 107／123（−16） | 144／103（+41） | 127／93（+34） | 130／73（+57） | +116 |
| PROTEINS 边数，Size | 32／34（−2） | 4／18（−14） | 23／16（+7） | 49／34（+15） | +6 |

分子节点数的 WL 已采用组净值为 +91，待交换两组分别为 −16 和 +41，合计 +25。这里待交换组的两侧人数不同，必须再按每块各取同样多张计算严格等量的收益。另以 SourceV 作为基座，三个设置已采用组净值为 +56、+80、+4，待交换组净值为 +62、+36、+2。完整的正向和反向拆分、两个异域参考及种子 3405 的附加基座均在表中。[分组汇总](../results/iteration-12/strong-rank-swaps-001/score/attribution-summary.csv) · [逐块分组](../results/iteration-12/strong-rank-swaps-001/score/attribution-blocks.csv)

## 严格保阳性数的效果

自身预算 (m) 取可换入与可换出人数的较小值。每块对两侧各交换 (m) 张，保持原强基座的预测阳性数。表中准确率与 Macro-F1 按方向内先平均种子，再对十二方向等权。纠正、误改与净值是跨块求和的精确期望，展示时四舍五入。

| 设置 | 原基座 Accuracy／Macro-F1 | 本域交换后 Accuracy／Macro-F1 | Accuracy 差 | 期望纠正／误改 | (m>0) 块／全部块 | 共交换对数 |
|---|---:|---:|---:|---:|---:|---:|
| 分子边数，WL | 71.485%／70.886% | 71.404%／70.805% | −0.081 pp | 140.990／157.010 | 48／60 | 149 |
| 分子节点数，WL | 70.994%／70.102% | 71.027%／70.133% | +0.032 pp | 161.313／154.687 | 50／60 | 158 |
| PROTEINS 边数，Size | 65.601%／42.985% | 65.566%／42.914% | −0.034 pp | 1／3 | 2／57 | 2 |

两个分子设置各有 19,350 H 方向种子图次，PROTEINS 有 4,843 图次。完整组保留原交叉试验的 60／60／57 块，PROTEINS 的 M3 源、种子 3407 的三个目标块仅保留在逐块证据。每个设置仍含全部十二方向。分子节点数的 +0.032 pp 在 12 方向中六胜六负。分子边数虽有八个方向改善，等权净效果为负。[各参考头绝对端点](../results/iteration-12/strong-rank-swaps-001/score/swap-summary.csv) · [逐块期望矩阵与交换组正类率](../results/iteration-12/strong-rank-swaps-001/score/swap-blocks.csv)

## 等量异域对照和去域检验

共同预算 (L) 取同一 H 块三参考头的最小 (m)，三头各交换 (L) 对。(L=0) 的块照常计入，三个头均沿用基座。异域均值在同块先平均，后续对方向等权。去域每次删掉一个源或目标，余下九方向各自等权；下表列所有删法中的最小 Accuracy 差。

| 设置 | 各自 (m) 本域−异域 Accuracy | 同 (L) 本域−异域 Accuracy／Macro-F1 | (L>0) 块／全部块 | 同 (L) 最小去源／去目标 | 自身 (m) 对基座最小去源／去目标 |
|---|---:|---:|---:|---:|---:|
| 分子边数，WL | +0.138 pp | +0.062／+0.065 pp | 35／60 | +0.034／+0.034 pp | −0.134／−0.129 pp |
| 分子节点数，WL | +0.255 pp | +0.128／+0.130 pp | 37／60 | +0.070／+0.080 pp | −0.075／−0.008 pp |
| PROTEINS 边数，Size | −0.017 pp | 0.000／0.000 pp | 0／57 | 0.000／0.000 pp | −0.046／−0.046 pp |

分子节点数在同预算上有本域优势，删任一源域或目标域后方向均值仍为正。但对原 WL 的增益很小且去域后可转负。分子边数也有同预算本域优势，实际交换仍不如 WL，按协议结束该设置的直接移植路线。PROTEINS 的 Size 在全部 57 块均无共同预算。以 SourceV 为基座时，分子边数自身 (m) 增加 +0.262 pp，共同 (L) 的本域优势为 −0.023 pp。分子节点数对应 +0.182 与 +0.013 pp，后者删去一个源域时最低 −0.004 pp。[对照汇总](../results/iteration-12/strong-rank-swaps-001/score/contrast-summary.csv) · [逐方向](../results/iteration-12/strong-rank-swaps-001/score/direction-contrasts.csv) · [全部去域结果](../results/iteration-12/strong-rank-swaps-001/score/leave-domain.csv)

种子 3405 每个设置另有十二方向。下表每格依次为自身 (m) 比基座的 Accuracy 差、同 (L) 本域比异域均值的 Accuracy 差，单位为 pp。共同预算为零的补充基座只支持零差结论。

| 补充基座 | 分子边数 | 分子节点数 | PROTEINS 边数 |
|---|---:|---:|---:|
| Static | +0.004／−0.062 | +0.212／+0.073 | +0.015／+0.004 |
| R7 Centered | −0.297／+0.046 | −0.026／+0.092 | 0.000／0.000 |
| B₂ | −0.196／+0.125 | +0.022／+0.066 | 0.000／0.000 |
| DirectP | −0.035／+0.113 | +0.348／0.000 | +0.057／0.000 |
| PoolMajority | 0.000／0.000 | 0.000／0.000 | 0.000／0.000 |

这些结果基于整批旧 H 排序和事后评分。分子节点数的正值只支持本组开发图上的必要条件，独立图和逐图可部署的推断规则仍未验证。

## 封存和回放

- 执行提交 `f131a0233f35acf98fdef1013522f422a48bc5c3`，56 项 R7 回归通过。没有重新训练或读取 A 类别。使用 12 个原 H 角色的 2,920 条标签记录。[配置](../configs/strong-rank-swaps.json) · [执行记录](../results/iteration-12/strong-rank-swaps-001/executed-config.json) · [标签回执](../results/iteration-12/strong-rank-swaps-001/score/h-label-access.json) · [测试日志](../results/iteration-12/strong-rank-swaps-001/logs/r7-strong-rank-swaps-tests.log)
- OBS SHA-256 为 `eafaee853e712abdf11e4a534b7102528ad1b1051a1c5bdbad2c4123945bcc64`。评分前的 1,620 个块／参考／基座记录、17,406 个分歧图次、34,812 个精确入选概率与独立盲算完全一致。原脚本与摘要一同归档，可在同一历史数据路径重建独立包。[OBS](../results/iteration-12/strong-rank-swaps-001/OBS.json) · [比对回执](../results/iteration-12/strong-rank-swaps-001/blind-verification.json) · [独立脚本及摘要](../results/iteration-12/strong-rank-swaps-001/independent)
- FINAL SHA-256 为 `4e40754573b76bdffc613dd022975dd6e119637944dc0551b7fc39d63c1fd33f`。归档复核保数约束、3,240 行等量交换端点与 6,480 行归因分组，并确认四组求和与原交叉试验 180 块的 E 对 C 配对完全相同。独立复算的 3,240 行端点与 81 行对照汇总均一致。[FINAL](../results/iteration-12/strong-rank-swaps-001/score/FINAL.json) · [归档复核](../results/iteration-12/strong-rank-swaps-001/archive-verification.json) · [执行日志](../results/iteration-12/strong-rank-swaps-001/logs)
