# R10 实测｜PROTEINS 节点属性中位数能多分对哪些图？

**加属性后，八方向等权 Accuracy 比 WL＋Size 高 0.666 个百分点。四个源组只有两个改善，未达到登记的继续条件。** 17 项改判中纠正 9 项、误改 8 项。按协议结束这组固定摘要的筛查。[登记协议](source-attribute-screen.md) · [原始节点字段核对](size-support.md)

## 八方向等权结果

单种子 3405，四个历史源域内各做小图→大图和大图→小图，共八方向、221 个 E 图次。WL、Size 和 Base 的模型选择、D/C/E 角色和 E 预测已按原归档核对。新臂只在 Base 后拼接三个按节点类型计算的属性中位数。所有尺度来自本方向 D，四个候选正则参数均用 C 选择。

| 方法 | Accuracy | Macro-F1 | NLL |
| --- | ---: | ---: | ---: |
| WL | 72.031% | 51.389% | 0.5993 |
| Size | 68.680% | 41.627% | 0.6144 |
| Base（WL＋Size） | 71.593% | 48.248% | 0.6337 |
| Attribute（Base＋属性中位数） | 72.259% | 50.640% | 0.6080 |

| Attribute 相对 | Accuracy 差 pp | Macro-F1 差 pp | NLL 差 | 纠正／误改 | 分歧图次 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WL | +0.228 | −0.749 | +0.0087 | 16／14 | 30 |
| Size | +3.579 | +9.013 | −0.0064 | 15／7 | 22 |
| Base | +0.666 | +2.392 | −0.0257 | 9／8 | 17 |

Accuracy 差先在各方向按其 E 图数计算，再平均八个方向。纠正和误改是原始次数。新臂相对 Base 等权分歧率为 8.321%，221 个图次中实际增加 1 次正确分类。[逐方向指标](../results/attempt-005-source-attributes/endpoints.csv) · [与三个参照的同图配对](../results/attempt-005-source-attributes/paired-blocks.csv) · [等权汇总](../results/attempt-005-source-attributes/summary.csv)

## 收益集中在哪些方向

下表为 Attribute 相对 Base。源组两方向等权，C 为新臂选定的 L2 正则参数。

| 源组 | 大小方向 | E 图数 | C | 纠正／误改 | Accuracy 差 pp | Macro-F1 差 pp | NLL 差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | 小→大 | 27 | 1 | 0／0 | 0.000 | 0.000 | +0.0125 |
| M0 | 大→小 | 29 | 100 | 1／0 | +3.448 | +15.153 | +0.1302 |
| M1 | 小→大 | 24 | 0.1 | 7／5 | +8.333 | +5.830 | −0.4971 |
| M1 | 大→小 | 30 | 0.1 | 1／1 | 0.000 | 0.000 | +0.0738 |
| M2 | 小→大 | 29 | 10 | 0／0 | 0.000 | 0.000 | +0.0040 |
| M2 | 大→小 | 27 | 10 | 0／0 | 0.000 | 0.000 | +0.0072 |
| M3 | 小→大 | 24 | 10 | 0／0 | 0.000 | 0.000 | −0.0191 |
| M3 | 大→小 | 31 | 10 | 0／2 | −6.452 | −1.844 | +0.0829 |

源组 M0、M1、M2、M3 的 Accuracy 差依次为 **+1.724、+4.167、0.000、−3.226 pp**。总体超过 +0.5 pp、严格超过独立 WL、严格超过 Size 三项成立。正增益源组为 **2／4**，低于要求的 3／4。M1 小→大贡献了大部分等权 Accuracy 和 NLL 改善。新臂使用独立 C 选模，结果度量的是完整源监督流程在既有 E 开发图上的增量。[逐源组配对](../results/attempt-005-source-attributes/paired-summary.csv) · [有理数判定](../results/attempt-005-source-attributes/gate.json)

## 输入、封存与复核

按 R5 SHA-256 核对图归属、节点类型和属性三个原始 TU 文件，并在 1,113 张图的 43,471 个节点上验证行归属及三类 one-hot 顺序。含负值的属性照原值处理，同图同类型内反转属性没有改变任一图的中位数。不同方向只以自己的 D/C 标签拟合和选模。全部八方向的四臂预测与八个新模型文件先封存，再验证 OBS 并按原角色读取 E 标签。原 U 未用于本轮，真实四域的 A/H 未进入运行。

正式执行提交为 `25e086b`。独立核 IRLS 的 884 个动作和 32 项 C 选择逐项一致，新臂 margin 的最大绝对差为 9.8×10⁻⁷。评分后核验了全部 32 个端点、24 项配对、35 行等权汇总和晋级判定。原执行版本通过 30 项 R10 工程回归。审查还发现实际运行的代码清单漏列了两个 R8 依赖，已补核两文件与拟合及评分所用 Git 提交的字节一致，并由 `618448b` 修复后续运行的清单，修复后 31 项回归通过。

[实际配置](../results/attempt-005-source-attributes/executed-config.json) · [原始输入核对](../results/attempt-005-source-attributes/input-verification.json) · [D/C 读口](../results/attempt-005-source-attributes/fit-label-access.json) · [E 读口](../results/attempt-005-source-attributes/e-label-access.json) · [OBS](../results/attempt-005-source-attributes/OBS.json) · [FINAL](../results/attempt-005-source-attributes/FINAL.json) · [属性尺度](../results/attempt-005-source-attributes/attribute-scaling.json) · [独立盲算核验](../results/attempt-005-source-attributes/blind-verification.json) · [盲算动作清单](../results/attempt-005-source-attributes/blind-review/actions.tsv) · [评分后复核](../results/attempt-005-source-attributes/final-verification.json) · [依赖补核](../results/attempt-005-source-attributes/code-audit.json) · [测试](../results/attempt-005-source-attributes/engineering-tests.log) · [修复后回归](../results/attempt-005-source-attributes/reviewed-tests.log) · [拟合日志](../results/attempt-005-source-attributes/freeze.log) · [评分日志](../results/attempt-005-source-attributes/score.log)
