# R7 实测｜在强源起点重新求解原解释的更新

**三个开发设置均未达到扩种子门槛。** 从预先指定的强源分类器起点直接重新求解，Full 的 Accuracy 相对原模型分别下降 1.038、0.355 和 4.429 个百分点。Full 略胜匹配正类质量和 GNN 分数一阶矩的 CalTeacher，说明两参数校准未复现全部判决。这个增量仍不足以修复完整更新对原强源模型的损失。[主协议](strong-anchor-refit.md) · [教师对照协议](strong-anchor-score-teacher.md)

## 指定强源起点

种子 3405，三个设置各十二方向等权。Full、教师对照和 ScoreOnly 在各自完整的无标签 A 上求解。同源两个异域 Full 使用同一 H 分别计分后等权。原 E 软修正将旧 E 相对 GNN 的 margin 差以单位系数加到强源 margin。

| 设定／指定起点 | 原起点 → Full Accuracy | Full − 起点 Accuracy pp | Full − 起点 Macro-F1 pp | Full − 起点 NLL | 纠正／误改 | 正／平／负方向 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数／WL | 71.254% → 70.216% | −1.038 | −1.106 | +0.0129 | 158／197 | 5／0／7 |
| Mutagenicity 节点数／WL | 72.776% → 72.421% | −0.355 | −0.797 | −0.0021 | 141／155 | 2／1／9 |
| PROTEINS 边数／Size | 65.081% → 60.652% | −4.429 | +8.688 | +0.1283 | 172／216 | 6／0／6 |

所有设置相对原起点的 Accuracy 均低于登记要求的 +0.5 pp，方向数也均低于 7/12。PROTEINS 的 Macro-F1 上升，同时 Accuracy 与 NLL 恶化。纠正与误改按图次计，跨源方向可重复计入同一 H 图。晋级使用普通 Accuracy 的精确有理数和原预先指定的起点。[逐方向绝对值](../results/iteration-12/strong-anchor-refit-001/endpoints.csv) · [逐方向配对](../results/iteration-12/strong-anchor-refit-001/paired-blocks.csv) · [门槛](../results/iteration-12/strong-anchor-refit-001/gate.json)

## 机制对照

[起点导数与单位步 NLL 的分解](strong-anchor-directional-risk.md)显示，三个设置的 Full 方向在原强源起点均给出负的平均 NLL 导数；分子边数和 PROTEINS 在单位步的平均 NLL 上升。该诊断区分了局部概率变化与分类改判。

表内均为 Full 相对对照的等权 Accuracy 差，单位为百分点。同阳性数对照按原强源 margin 排序，以 Full 的阳性数截取，因此属于整批 H 上的排序诊断。

| 设置／指定起点 | B0Teacher | CalTeacher | ScoreOnly | 原 E 软修正 | 异域 Full 平均 | 同阳性数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数／WL | −0.865 | +0.351 | −0.737 | +12.594 | +0.002 | −0.898 |
| Mutagenicity 节点数／WL | +0.494 | +0.228 | +0.053 | +4.788 | +2.256 | −0.946 |
| PROTEINS 边数／Size | +0.734 | +0.877 | +0.311 | +1.876 | +0.426 | +5.582 |

- **CalTeacher。** Full 相对其纠正／误改分别为 79／65、53／44、16／7，Macro-F1 差 +0.368、+0.300、+1.777 pp，NLL 差 −0.0045、−0.0037、−0.0099。此结果限于当前的二参数教师族与既有开发图。
- **同阳性数。** 两个 Mutagenicity 设置都能用原强源排序达到更高 Accuracy。PROTEINS 在相同的预测阳性数下，Full 的排序优于原 Size 排序 5.582 pp。Full 将方向等权预测阳性率从 32.022% 移到 38.404%，最终仍比原 Size 低 4.429 pp。
- **原 E 软修正。** 重新求解在三个设置均改善直接相加的结果，尤其是边数 Mutagenicity。相对零更新强源起点的主结果仍为负。[全部对照的 Accuracy、Macro-F1 和 NLL](../results/iteration-12/strong-anchor-refit-001/paired-summary.csv)

## 两种起点的完整范围

表中三项依次为 Accuracy %、Macro-F1 %、NLL。两个分子划分复用图实体，每个设置只包含一个训练种子。

| 设置 | 起点 | 原起点 | Full | 相对起点 Accuracy pp |
| --- | --- | ---: | ---: | ---: |
| Mutagenicity 边数 | WL | 71.254／70.599／0.6045 | 70.216／69.493／0.6174 | −1.038 |
| Mutagenicity 边数 | Size | 60.437／55.482／0.6793 | 69.940／68.877／0.5969 | +9.503 |
| Mutagenicity 节点数 | WL | 72.776／72.100／0.5980 | 72.421／71.303／0.5959 | −0.355 |
| Mutagenicity 节点数 | Size | 58.526／46.352／0.7179 | 71.479／69.918／0.5842 | +12.953 |
| PROTEINS 边数 | WL | 53.583／45.233／0.8002 | 59.831／52.826／0.8028 | +6.249 |
| PROTEINS 边数 | Size | 65.081／42.413／0.6945 | 60.652／51.101／0.8227 | −4.429 |

从另一较低 Accuracy 的起点重求解，三个设置均有增量。其更新后的 Accuracy 仍分别比原 WL、原 WL 和原 Size 低 1.314、1.297、5.250 pp。按登记停止这条完整更新规则，保持其余四个种子未运行。[两起点全部方法的绝对指标](../results/iteration-12/strong-anchor-refit-001/endpoint-summary.csv)

## 封存、数值与独立核验

使用正式版 `e4bc894`。独立审查在 freeze 前修复了极大 margin 的相对目标消减误差，以及大共同截距下的校准矩失真。完整 A 的图数范围分别为 747–777、748–770、181–207。两起点合计 288 个增广或对照求解均获得原坐标或规定子空间的 KKT 证书，最大驻点残差 9.85×10⁻¹¹。CalTeacher 最大矩残差 4.10×10⁻¹⁵。重求原 E 的 A/H 硬判决与历史逐项一致，H margin 最大绝对差为 5.54×10⁻⁵。历史 R10 的 72 个强源端点复现了 Accuracy、Macro-F1 和 NLL。

三份评分前独立盲算表各覆盖 17,520 个「方向、起点、图」组合，合计 **157,680 项硬判决逐项一致**。76 个 OBS 成员先核验，随后按原 12 个 H 角色读标签。评分后复核 1,008 行端点、792 行配对、66 行汇总和三项门槛。48 项 R7 回归通过。[独立审查及恢复记录](strong-anchor-review.md) · [盲算比对](../results/iteration-12/strong-anchor-refit-001/blind-verification.json) · [最终复核](../results/iteration-12/strong-anchor-refit-001/final-verification.json)

[实际配置](../results/iteration-12/strong-anchor-refit-001/executed-config.json) · [OBS](../results/iteration-12/strong-anchor-refit-001/OBS.json) · [FINAL](../results/iteration-12/strong-anchor-refit-001/FINAL.json) · [求解及每个 A 图池大小](../results/iteration-12/strong-anchor-refit-001/fit-notes.json) · [更新权重与 A 分数](../results/iteration-12/strong-anchor-refit-001/fits) · [H 读口](../results/iteration-12/strong-anchor-refit-001/h-label-access.json) · [回归日志](../results/iteration-12/strong-anchor-refit-001/engineering-tests.log) · [封存日志](../results/iteration-12/strong-anchor-refit-001/freeze.log) · [评分日志](../results/iteration-12/strong-anchor-refit-001/score.log)
