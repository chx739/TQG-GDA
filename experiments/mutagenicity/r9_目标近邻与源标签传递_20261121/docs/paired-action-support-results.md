# R9｜一处节点类型差异的跨图参照有多少改判空间？

**固定的二点保序操作在 PROTEINS 上的改判上界为 0.425 个百分点，低于登记的 0.5 门槛。** Mutagenicity 的边数和节点数划分分别为 1.254 和 0.969 个百分点。联合推进条件未达到，本项在宽松筛查结束。[登记协议](paired-action-support.md)

## 36 个方向的结构和判决条件

每一列统计至少找到一张同域 R 图满足条件的 Q 图次数。同一张 Q 在各方向只计一次。括号内为先除以本方向的全部 Q 数、再对 12 个方向等权的比例。

| 设置 | Q 图次数 | 有结构候选 | 候选 B0 类别相反 | 固定操作可能改判 |
| --- | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 4,575 | 417（9.184%） | 103（2.271%） | 57（1.254%） |
| Mutagenicity 节点数 | 4,572 | 408（8.995%） | 78（1.717%） | 44（0.969%） |
| PROTEINS 边数 | 1,164 | 27（2.281%） | 6（0.510%） | 5（0.425%） |

PROTEINS 有改判空间的 5 次均在 M3 作目标的方向。M0→M3 有 2 次，M2→M3 有 3 次，其余 10 个方向为零。两种 Mutagenicity 划分各有 9 个方向存在可能改判的图。结构候选能否对应一次真实的单节点类型差异，以及改判后是否正确，都需要进一步证据。当前上界允许每张 Q 独立挑选最有利的实际 R，并允许未知相对顺序取最有利的一侧。宽松拓扑条件还能容纳非同构图，严格配对只会让候选变少。

[逐方向计数](../results/paired-action-support-001/directions.csv) · [等权汇总](../results/paired-action-support-001/settings.csv) · [门槛判定](../results/paired-action-support-001/gate.json)

## 输入范围与复算

执行提交为 `e2ecc20`。复用 R13 种子 3405 的同域 Q/R 和 R7 的 A 池 B0，全部 10,311 次 Q 动作与 R13 的 B0 核对一致。结构条件保留原始有向边记录、边类型、自环、平行边及孤立节点。复算只使用 Q/R 图和相应分数。运行期间没有训练、读取图类别或使用 H 输入。归档只提供聚合的候选和改判计数。

15 项 R9 工程回归通过，其中 5 项覆盖本次筛查。使用独立的计数多重集实现并穷举 Q×R，复算了 36 个方向的 108 个计数、三个等权汇总和门槛。

[实际配置](../results/paired-action-support-001/executed-config.json) · [读取范围与身份哈希](../results/paired-action-support-001/access.json) · [FINAL](../results/paired-action-support-001/FINAL.json) · [复算](../results/paired-action-support-001/verification.json) · [运行日志](../results/paired-action-support-001/run.log) · [工程测试](../results/paired-action-support-001/engineering-tests.log)
