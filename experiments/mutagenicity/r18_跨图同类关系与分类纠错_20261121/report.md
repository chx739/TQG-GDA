# R18｜学习图对同类关系能否纠正强单图模型？

**24 个来源内方向已完成，三个设置均未达到真实四域验证门槛。** Full 的图对 Brier 在三个设置都低于未校准乘积，但均高于校准乘积。TargetFull 的分类动作与同阳性数的原强模型排序在 1,955 个 E 图次上逐项相同，观测到的判决变化均可由预测阳性数的改变复现。[登记协议](docs/source-screen.md) · [同类关系与有向票的机制](docs/what-a-peer-vote-can-correct.md)

## 整图分类

种子 3405，每个设置内八个方向等权。Strong 是每方向按 C 选择并冻结的 WL、Size 或联合 Base。下表以绝对 Accuracy 显示，净纠错来自与 Strong 同图配对。

| 设置 | Strong | TargetFull | EqualCount | RelationOnly | TargetFull − Strong | 纠正／误改（净纠错） | 正增益源组 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 74.794% | 74.452% | 74.452% | 74.372% | −0.342 pp | 2／5（−3） | 0／4 |
| Mutagenicity 节点数 | 75.227% | 74.765% | 74.765% | 74.052% | −0.461 pp | 0／4（−4） | 0／4 |
| PROTEINS 边数 | 71.568% | 72.031% | 72.031% | 74.094% | +0.463 pp | 1／0（+1） | 1／4 |

TargetFull 相对 Strong 的 Macro-F1 差依次为 −0.263、−0.449、+2.617 pp。两个分子设置相对 Strong 各只有三个方向改变了判决，PROTEINS 只有 M0 小→大方向的一张图获得纠正。其 +0.463 pp 已达到预先计算的动作分歧上界，仍低于 +0.5 pp 门槛，并且最多改善一个源组。节点数划分的动作上界也是 +0.461 pp。两个分子设置复用部分图实体，本轮为单种子既有开发图结果。[逐方向端点](results/attempt-001-source-screen/endpoints.csv) · [逐方向纠正、误改和分歧](results/attempt-001-source-screen/paired-blocks.csv) · [标签前冻结的动作上界](results/attempt-001-source-screen/action-bounds.json)

## 同伴与分类对照

表内差值为 TargetFull 减所列对照的等权 Accuracy，单位为百分点。U/C 分歧是更换同伴池后的同图原始次数。

| 设置 | TargetUnary | SourceFull | PermMean | CalAnchor | EqualCount | U／C 分歧图次 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | −0.454 | −0.215 | −0.244 | −0.562 | 0.000 | 6 |
| Mutagenicity 节点数 | −0.236 | −0.124 | −0.234 | +0.357 | 0.000 | 5 |
| PROTEINS 边数 | +0.567 | +2.025 | +0.521 | −2.392 | 0.000 | 4 |

PROTEINS 上 TargetFull 优于等预算源校准池的 SourceFull，换池分歧为四次纠正、零次误改；相对不使用任何同伴的 Strong 只净纠正一张。CalAnchor 的 Accuracy 为 74.423%，固定 C=1 WL 为 75.343%，均高于 TargetFull。纯关系票 RelationOnly 在 PROTEINS 上为 74.094%，仍低于这两项单图参照。两个分子设置的 U 同伴表现均低于等预算 C 同伴。原三种源选模模型和固定 WL 的 Accuracy、Macro-F1 均保留在同一张汇总表。[所有模型绝对值](results/attempt-001-source-screen/summary.csv) · [源组与设置配对](results/attempt-001-source-screen/paired-summary.csv)

## 留出图对的同类概率

每方向先平均 E 内所有无序异图对，再对八方向等权。Brier 越低越好。图对共享端点，总图对数描述计算规模。

| 设置 | E 图对 | Product | CalProduct | Unary | Full | Full − CalProduct | Full − Unary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 47,000 | 0.236637 | 0.234540 | 0.234810 | 0.234866 | +0.000325 | +0.000055 |
| Mutagenicity 节点数 | 46,367 | 0.239131 | 0.235497 | 0.235500 | 0.235887 | +0.000391 | +0.000387 |
| PROTEINS 边数 | 2,966 | 0.241320 | 0.231466 | 0.235324 | 0.234010 | +0.002544 | −0.001314 |

在两个分子设置，Full 相对 Unary 也没有降低 Brier。PROTEINS 的结构特征相对 Unary 带来一些关系估计增量，但只校准单图分数后计算乘积，Brier 仍更低。三设置均未满足联合门槛，按登记结束该固定关系头和投票规则。[图对 Brier 逐方向](results/attempt-001-source-screen/pair-brier-blocks.csv) · [等权 Brier](results/attempt-001-source-screen/pair-brier-summary.csv) · [精确晋级判定](results/attempt-001-source-screen/gate.json)

## 封存与核对

执行提交 `b27add7`。历史三模型的 72 份原 E margin 逐图复现，四项基线的 96 个分类端点与 R9 一致。四折校准及关系头拟合只使用方向授权的 D/C 类别。E 标签在全部 24 个方向的预测和图对概率封存并复核后才读取。无真实四域评价。

独立复核确认 96 项四折选择、同伴 ID 和 1,955 图次的 25,415 项判决一致。这些复核判决包含与 Strong 严格同一的 Product 动作。96,333 个无序图对的未校准乘积逐值相等，其他图对概率的最大绝对差为 8.12×10⁻⁸。54 个 OBS 成员和 11 个 FINAL 成员均通过哈希校验，复算了 288 个分类端点、264 项分类配对、96 个方向图对 Brier 行和 60 个 Brier 汇总。8 项 R18 专项回归和 53 项 R17/R9 回归通过。[动作与图对盲算核验](results/attempt-001-source-screen/blind-verification.json) · [评分后核验](results/attempt-001-source-screen/final-verification.json) · [读口和拟合诊断](results/attempt-001-source-screen/diagnostics.csv)

[实际配置](results/attempt-001-source-screen/executed-config.json) · [OBS](results/attempt-001-source-screen/OBS.json) · [FINAL](results/attempt-001-source-screen/FINAL.json) · [D/C 读口](results/attempt-001-source-screen/fit-label-access.json) · [E 读口](results/attempt-001-source-screen/e-label-access.json) · [模型与四折状态](results/attempt-001-source-screen/models) · [逐图动作及图对概率](results/attempt-001-source-screen/predictions) · [独立盲算回执](results/attempt-001-source-screen/blind-review/receipt.json) · [测试日志](results/attempt-001-source-screen/engineering-tests.log) · [封存日志](results/attempt-001-source-screen/freeze.log) · [评分日志](results/attempt-001-source-screen/score.log)
