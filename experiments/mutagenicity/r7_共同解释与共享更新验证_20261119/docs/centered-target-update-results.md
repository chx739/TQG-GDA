# R7 实测｜只应用两个冻结教师的权重差

**固定单位系数的教师差分在三个开发设置都略高于指定强源模型，三个设置均未通过扩种子门槛。** Mutagenicity 边数和 PROTEINS 的等权 Accuracy 增量超过 0.5 个百分点，但改善方向不足 7/12。Mutagenicity 节点数有 7/12 个改善方向，等权增量只有 0.148 个百分点。[评分前登记](centered-target-update.md) · [原强源起点试验](strong-anchor-refit-results.md)

## 指定强源起点

种子 3405，每设置十二方向等权。Centered 使用原强源 margin 加上 $u^\top(d_{\mathrm{Full}}-d_{\mathrm{Cal}})$。同一 H 上分别计分两个异域更新，再取平均。同阳性数对照按原强源 margin 排序，匹配本域 Centered 的预测阳性数。

| 设置／起点 | Accuracy 原值 → Centered | ΔAccuracy pp | ΔMacro-F1 pp | ΔNLL | 纠正／误改 | 正／平／负方向 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数／WL | 71.254% → 71.932% | +0.678 | +0.643 | +0.0008 | 89／63 | 6／2／4 |
| Mutagenicity 节点数／WL | 72.776% → 72.925% | +0.148 | +0.100 | +0.0059 | 47／41 | 7／1／4 |
| PROTEINS 边数／Size | 65.081% → 65.620% | +0.539 | +0.763 | −0.0016 | 20／15 | 2／9／1 |

同图配对的纠正和误改是跨方向图次，主指标从各方向的百分比等权得出。PROTEINS 的两个正方向是 M0→M1 和 M3→M2，分别提高 5.634 和 2.247 个百分点。M3→M1 下降 1.408 个百分点，另外九个方向持平，其中八个方向没有改判。[逐方向端点](../results/iteration-12/centered-target-update-001/endpoints.csv) · [全部逐方向配对](../results/iteration-12/centered-target-update-001/paired-blocks.csv)

## 方向性与同阳性数对照

表内是 Centered 相对所列对照的 Accuracy 差，单位为百分点。

| 设置／起点 | 原 WL | 原 Size | 反号 | 异域 1 | 异域 2 | 异域均值 | 同阳性数 | 原 Full | 原 CalTeacher |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数／WL | +0.678 | +11.495 | +1.461 | +0.397 | +0.334 | +0.366 | +0.683 | +1.716 | +2.067 |
| Mutagenicity 节点数／WL | +0.148 | +14.399 | +0.662 | +0.074 | +0.235 | +0.155 | +0.153 | +0.504 | +0.731 |
| PROTEINS 边数／Size | +12.038 | +0.539 | +1.734 | +1.742 | +1.048 | +1.395 | +0.781 | +4.968 | +5.845 |

在三个指定起点上，Centered 的平均 Accuracy 均高于两个单模型参照、反号更新、异域参考平均和等阳性数对照。在既有 H 上，增量包含排序变化的贡献。PROTEINS 的同阳性数比较为 20 次纠正与 12 次误改。边数 Mutagenicity 的异域相对差为 +0.366 pp，节点数为 +0.155 pp。正式晋级仍要求 PROTEINS 和至少一个 Mutagenicity 设置同时通过平均增益、方向覆盖及对照条件。按协议，以这批已有开发图结束单位系数规则。[精确判定](../results/iteration-12/centered-target-update-001/gate.json) · [所有对照的 Accuracy、Macro-F1 与 NLL](../results/iteration-12/centered-target-update-001/paired-summary.csv)

## 逐方向纠正与误改

表格每格为相对指定原起点的纠正／误改次数，后附 Accuracy 差值。M0–M3 在三个设置内各自指对应的历史域。

| 方向 | Mutagenicity 边数／WL | Mutagenicity 节点数／WL | PROTEINS 边数／Size |
| --- | ---: | ---: | ---: |
| M0→M1 | 5／5 (0.000 pp) | 1／6 (-1.592 pp) | 5／1 (+5.634 pp) |
| M0→M2 | 8／9 (-0.311 pp) | 1／1 (0.000 pp) | 0／0 (0.000 pp) |
| M0→M3 | 2／4 (-0.592 pp) | 2／0 (+0.593 pp) | 0／0 (0.000 pp) |
| M1→M0 | 8／3 (+1.548 pp) | 5／2 (+0.935 pp) | 3／3 (0.000 pp) |
| M1→M2 | 2／4 (-0.621 pp) | 3／2 (+0.314 pp) | 0／0 (0.000 pp) |
| M1→M3 | 7／1 (+1.775 pp) | 4／2 (+0.593 pp) | 0／0 (0.000 pp) |
| M2→M0 | 5／6 (-0.310 pp) | 9／3 (+1.869 pp) | 0／0 (0.000 pp) |
| M2→M1 | 8／8 (0.000 pp) | 3／7 (-1.274 pp) | 0／0 (0.000 pp) |
| M2→M3 | 6／4 (+0.592 pp) | 1／2 (-0.297 pp) | 0／0 (0.000 pp) |
| M3→M0 | 13／12 (+0.310 pp) | 3／2 (+0.312 pp) | 0／0 (0.000 pp) |
| M3→M1 | 16／6 (+3.257 pp) | 12／9 (+0.955 pp) | 0／1 (-1.408 pp) |
| M3→M2 | 9／1 (+2.484 pp) | 3／5 (-0.629 pp) | 12／10 (+2.247 pp) |

## 另外三个起点组合

表内三项依次为 Accuracy %、Macro-F1 %、NLL。完整两起点全部方法在 CSV 中。

| 设置／起点 | 原起点 | Centered | 相对原起点 Accuracy pp |
| --- | ---: | ---: | ---: |
| Mutagenicity 边数／Size | 60.437／55.482／0.6793 | 62.341／58.623／0.6764 | +1.905 |
| Mutagenicity 节点数／Size | 58.526／46.352／0.7179 | 58.826／49.180／0.7114 | +0.300 |
| PROTEINS 边数／WL | 53.583／45.233／0.8002 | 54.210／46.425／0.8070 | +0.627 |

三个组合的增量均未使其超过该设置的另一个原源模型。两个 Mutagenicity 设置复用分子实体，所有结果来自一个种子和已参与开发的 H。[两起点完整绝对指标](../results/iteration-12/centered-target-update-001/endpoint-summary.csv)

## 运行与复核

执行代码为 `ab0d312`。复用原 Full 与 CalTeacher 的冻结更新，重新拟合次数为零。先封存全部 36 个方向与两种起点的动作并逐项重放，再按原 12 个 H 角色评分。39 个 OBS 成员逐项校验。17,520 个方向、起点和图的组合共 **140,160 项判决**与事前独立盲算逐字节一致，margin 最大绝对差为 6.01×10⁻⁸。独立清单 SHA-256 为 `006fcf44f2b1264008bceb5f3f383c6dd474206c4d320478d0dc692500c51817`。封存后复核 792 行端点、720 行配对、72 个同阳性数和 60 项等权配对汇总。原 WL、Size、Full、CalTeacher 和 Anchor 的 360 行端点复现。52 项 R7 回归通过。

[实际配置](../results/iteration-12/centered-target-update-001/executed-config.json) · [OBS](../results/iteration-12/centered-target-update-001/OBS.json) · [FINAL](../results/iteration-12/centered-target-update-001/FINAL.json) · [权重差](../results/iteration-12/centered-target-update-001/weight-differences.json) · [H 读口](../results/iteration-12/centered-target-update-001/h-label-access.json) · [盲算比对](../results/iteration-12/centered-target-update-001/blind-verification.json) · [评分后复核](../results/iteration-12/centered-target-update-001/final-verification.json) · [测试日志](../results/iteration-12/centered-target-update-001/engineering-tests.log) · [封存日志](../results/iteration-12/centered-target-update-001/freeze.log) · [评分日志](../results/iteration-12/centered-target-update-001/score.log)
