# R9｜知道真实比例以后，按比例定阈能否超过原 WL？

**PROTEINS 上，将 U 的比例换成 E 的真实比例提高了 2.288 个 Accuracy 百分点。再改用 E 的排序定阈，Accuracy 下降 3.356 个百分点。** 最终仍比原 WL 低 **4.583 pp**，Macro-F1 则高 **10.710 pp**。Mutagenicity 节点数划分中，E 比例和 E 排序的同一规则比原 WL 高 **2.093 Accuracy pp**。结论随设置和指标变化。[事前登记与可行范围](true-rate-and-accuracy.md)

## 实验范围与端点

复用种子 3405 的全部 24 个源内大小段方向。保持 D 上的 WL/Size 输出、U/E 图 ID 与层次不变。三组规则先根据已发布的分层正例数和图数定阈，再在同一 E 评分。A 按 U 的真实比例在 U 定阈。B 换用 E 的真实比例，仍在 U 定阈。C 同时使用 E 的真实比例和 E 排序定阈。三个规则都保留同分、同距和空 U 层回退。B 和 C 使用 E 的监督汇总，C 还用到整个 E 层的排序，均作为评分后诊断。

下表为八方向等权的 **Accuracy／Macro-F1**，单位为 %。

| 设置 | 原 WL | A，真 U 比例／U 阈值 | B，真 E 比例／U 阈值 | C，真 E 比例／E 阈值 |
| --- | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 70.801／67.406 | 69.033／66.978 | 69.969／67.770 | 70.191／68.530 |
| Mutagenicity 节点数 | 67.705／63.379 | 69.440／66.649 | 69.341／65.951 | 69.798／67.067 |
| PROTEINS 边数 | 75.343／48.188 | 71.827／61.612 | 74.116／63.631 | 70.760／58.898 |

A 的 Accuracy 在三个设置均落入事前推导的可行范围。原 WL 的 24 个混淆矩阵全部复现。[逐方向混淆矩阵](../results/attempt-003-true-rate-replay/endpoints.csv) · [等权端点](../results/attempt-003-true-rate-replay/endpoint-summary.csv) · [A 区间复核](../results/attempt-003-true-rate-replay/a-range-checks.json)

## 相邻规则的配对差

每格为 **Accuracy 差／Macro-F1 差**，单位为百分点。末尾括号为逐图纠正／误改次数。差值对方向等权，纠正和误改跨方向求和。

| 设置 | B − A，换比例 | C − B，换定阈池 | C − 原 WL |
| --- | ---: | ---: | ---: |
| Mutagenicity 边数 | +0.936／+0.792（24／16） | +0.222／+0.760（24／22） | −0.610／+1.124（57／62） |
| Mutagenicity 节点数 | −0.099／−0.697（23／24） | +0.457／+1.115（24／20） | +2.093／+3.688（83／65） |
| PROTEINS 边数 | +2.288／+2.019（13／8） | −3.356／−4.733（6／14） | −4.583／+10.710（17／27） |

在 PROTEINS，三个真实比例规则中 B 的 Accuracy 最高，为 74.116%，比原 WL 低 1.227 pp。把比例改为 E 实测值能追回部分损失。使用 E 的真实比例并直接在 E 排序上定阈，仍出现总体 Accuracy 代价。这表明按比例给出阳性动作本身可以成为瓶颈。同时，三个规则的 Macro-F1 均高于原 WL，符合两项指标的不同侧重。

节点数划分则提供了正例，C 的两项指标均高于原 WL。边数划分的 C 提升 Macro-F1，同时 Accuracy 略低。这些是在锁定模型、单种子历史源图和既定阈值策略下的净变化。两个大小段方向复用同一源组的原始图，两个 Mutagenicity 划分来自同一批分子。原 R9 的晋级判定继续有效。

[逐方向相邻及对 WL 配对](../results/attempt-003-true-rate-replay/paired-blocks.csv) · [等权配对汇总](../results/attempt-003-true-rate-replay/paired-summary.csv)

## 封存及复核

执行提交为 `fb79066`。诊断从原 OBS 的冻结预测和 FINAL 的已发布计数生成 72 个动作向量。完整封存后，评分只按 24 个已登记 E 角色解码标签，期间未加载图容器或运行训练。E 标签权威与原 R9 OBS 记录的 R5 INPUT、OBS 及偏移登记核对一致。

策略审查的评分前盲算与正式 OBS 的 **72/72** 个 E 动作向量逐一吻合。盲算清单 SHA-256 为 `949535423fb0fee81dff42de30daab74f44b3413118b768d2f65e5e76a5e9ffb`。另用独立枚举复算三个规则，核对 96 个端点、120 条配对和原 WL 混淆矩阵。149 项相关工程回归通过。

[实际配置](../results/attempt-003-true-rate-replay/executed-config.json) · [比例和阈值](../results/attempt-003-true-rate-replay/rates.json) · [OBS](../results/attempt-003-true-rate-replay/OBS.json) · [FINAL](../results/attempt-003-true-rate-replay/FINAL.json) · [E 读取记录](../results/attempt-003-true-rate-replay/e-label-access.json) · [盲算比对](../results/attempt-003-true-rate-replay/blind-review-match.json) · [复核](../results/attempt-003-true-rate-replay/verification.json) · [运行日志](../results/attempt-003-true-rate-replay/run.log)
