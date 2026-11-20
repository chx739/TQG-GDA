# R9 实测｜伪目标图的结构共变能否提高源监督分类？

**24 个源内大小伪迁移已完成，三个设置均未达到真实四域先导门槛。** TargetFull 相对联合模型 Base 的 Accuracy 分别下降 1.097、0.919 和 0.906 个百分点。完整共变相对等预算源参照、目标词方差两项对照的优势随设置改变。按登记结束此固定表示的跨设置推进。[登记协议](reference-geometry.md) · [机制与双解反例](reference-geometry-mechanism.md)

## 主结果

三个设置各有四个历史源组，每组包含小图到大图和大图到小图两个方向。以下每设置八方向等权。差值的左边均为 TargetFull。

| 开发设置 | TargetFull Accuracy | 相对 Base pp | 相对 SourceFull pp | 相对 TargetDiag pp | 相对 WL pp | 相对 Size pp | 胜 Base 的源组 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 72.965% | −1.097 | −0.006 | −0.451 | −3.257 | +13.030 | 1／4 |
| Mutagenicity 节点数 | 74.528% | −0.919 | −0.119 | +0.223 | −0.209 | +16.828 | 1／4 |
| PROTEINS 边数 | 70.687% | −0.906 | +0.495 | −0.058 | −1.344 | +2.008 | 1／4 |

每个设置的晋级要求包括相对 Base 至少 +0.5 pp、严格胜过 SourceFull、TargetDiag、WL 和 Size，并至少在三个源组净增益为正。进入真实四域验证要求 PROTEINS 和至少一个 Mutagenicity 设置通过。该判定按整数正确数及有理数汇总。三个设置同时落后于 Base 和单独的 WL。

PROTEINS 上换成本域完整参照比源参照高 0.495 pp，但目标对角对照仍高 0.058 pp。Mutagenicity 节点数划分上，完整参照高于目标对角 0.223 pp，低于同预算源参照 0.119 pp。本轮没有哪个设置在 Accuracy 上同时胜过这两项对照。这些结果来自既有 E 开发池、单种子及有复用的源组。[判定与精确差值](../results/reference-geometry-001/gate.json) · [逐方向原始指标](../results/reference-geometry-001/endpoints.csv)

## 分类指标与误改

下表依序列 Accuracy %、Macro-F1 %、NLL。等阳性数对照沿用 Base 的 E 排序、匹配 TargetFull 的阳性数，按图 ID 打破平局。该对照仅定义硬分类决策。

| 方法 | Mutagenicity 边数 | Mutagenicity 节点数 | PROTEINS 边数 |
| --- | ---: | ---: | ---: |
| WL | 76.223／74.071／0.5046 | 74.737／72.428／0.5623 | 72.031／51.389／0.5993 |
| Size | 59.935／47.632／0.7939 | 57.700／47.812／0.6713 | 68.680／41.627／0.6144 |
| Base | 74.062／70.211／0.6579 | 75.447／71.294／0.5986 | 71.593／48.248／0.6337 |
| TargetFull | 72.965／69.045／0.6563 | 74.528／70.003／0.5861 | 70.687／51.270／0.6394 |
| SourceFull | 72.971／68.891／0.6495 | 74.646／70.113／0.5850 | 70.192／48.710／0.6169 |
| TargetDiag | 73.416／69.338／0.6544 | 74.304／69.699／0.5865 | 70.745／49.142／0.6384 |
| Base 等阳性数 | 73.805／69.898／— | 74.528／70.003／— | 71.521／52.572／— |

TargetFull 相对 Base 的纠正／误改分别为 **5／15、6／14、2／4**，对应方向等权分歧率 **2.210%、2.299%、2.694%**。计算采用图次，同一图进入不同设置时重复计数。将阳性数固定后，TargetFull 相对 Base 排序的 Accuracy 差为 **−0.840、0.000、−0.833 pp**。节点数划分虽有 16 项分歧，但恰好纠正 8 项、误改 8 项，等权差也为零。PROTEINS 相对 Base 的 Macro-F1 提高 3.022 pp，等阳性数 Base 的 Macro-F1 再高 1.302 pp。[设置与源组指标](../results/reference-geometry-001/endpoint-summary.csv) · [逐方向纠正、误改与分歧](../results/reference-geometry-001/paired-blocks.csv) · [源组合计与等权分歧](../results/reference-geometry-001/paired-summary.csv)

## 方差、封存与复核

24 个方向的参照预算为每池 24–122 张。TargetFull、SourceFull、TargetDiag 的尺度因子范围分别为 5.531–10.546、4.769–8.733、9.507–16.887。72 个增广拟合均完成方差匹配，没有触发退回 Base。24 个方向中，Base 选择 C=100、10、1 的次数分别为 12、9、3。每个增广臂沿用对应方向的该值。[尺度与所选 C](../results/reference-geometry-001/geometry.csv) · [拟合后模型状态](../results/reference-geometry-001/models)

执行代码为 `052b026`。先用 24 个方向各自的 D 标签拟合、C 标签校准，再封存 24 个模型文件和 24 个预测文件。盲算核对在评分前完成。正式与独立核 IRLS 的 1,955 个 E 图次、七方法共 **13,685 项分类动作**逐项一致，72 项模型选择及每个参照 ID 一致。选模 NLL 最大绝对差为 8.6×10⁻⁸，尺度最大绝对差为 1.1×10⁻¹⁴，margin 最大差为 1.1×10⁻⁵。合成回归与相关 R9 测试共 45 项通过。评分按方向只读取 E，随后复核了 168 个混淆端点、144 个配对以及 195 行等权汇总。

[实际配置](../results/reference-geometry-001/executed-config.json) · [OBS](../results/reference-geometry-001/OBS.json) · [FINAL](../results/reference-geometry-001/FINAL.json) · [D/C 读口](../results/reference-geometry-001/fit-label-access.json) · [E 读口](../results/reference-geometry-001/e-label-access.json) · [盲算动作](../results/reference-geometry-001/blind-review/actions.tsv) · [盲算核验](../results/reference-geometry-001/blind-verification.json) · [评分后核验](../results/reference-geometry-001/final-verification.json) · [测试](../results/reference-geometry-001/engineering-tests.log) · [拟合日志](../results/reference-geometry-001/freeze.log) · [评分日志](../results/reference-geometry-001/score.log)

评分后独立审阅复现了一个分阶段入口的重试问题。首次提供错误 OBS 哈希时，旧入口在验证前创建评分执行记录，阻碍随后用正确哈希重试。修复将独占评分记录放到全部预测核验之后，并保留首次失败日志。修复版在 24 个合成方向分别跑通正常全流程和“两次错误哈希、一次正确哈希”；错误尝试解析 E 标签 0 次，正确尝试 24 次。正式实验第一次就使用正确哈希，原封存输出仍可用事前独立预测逐图复核。[修复提交](https://github.com/chx739/TQG-GDA/commit/4ff4a71)
