# R9｜按图大小段检验加权源残差

**24 个源内伪方向全部完成，三个设置均未通过登记门槛。** 配平缩小了每个有效层的 WL 均值距离。相对直接使用 D 模型的 WL 概率均值，加权残差在三个设置均增加 U 和 E 的分层比例误差，并降低 E 的 Accuracy。这次实验在源内筛查收束。[登记协议](weighted-source-residuals.md)

## 范围和主要指标

固定种子 3405。三个设置各四个历史源，每个源按原定义的节点数或边数划分大小段，配对执行小→大和大→小。只在各方向的 D 重训 WL/Size，C 计算源残差。U 参与配平、比例估计及定阈，E 只用于应用已定阈值及评分。两个方向共享同一批原始 S/V 图，不能视为独立重复。

| 设置 | 完成方向 | D 图数范围 | C 图数范围 | U 图数范围 | E 图数范围 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 8/8 | 396–470 | 95–122 | 396–470 | 95–122 |
| Mutagenicity 节点数 | 8/8 | 386–482 | 103–114 | 386–482 | 103–114 |
| PROTEINS 边数 | 8/8 | 107–115 | 24–31 | 107–115 | 24–31 |

下表为每方向先按层大小加权的比例绝对误差，再对每设置八方向等权。单元格为 **U／E**，单位为百分点。

| 比例估计 | Mutagenicity 边数 | Mutagenicity 节点数 | PROTEINS 边数 |
| --- | ---: | ---: | ---: |
| **加权残差** | **6.675／9.373** | **9.176／13.139** | **12.567／19.750** |
| WL 概率均值 | 6.344／8.486 | 8.073／10.359 | 11.986／13.306 |
| 均匀源残差 | 7.440／10.170 | 9.488／13.168 | 12.298／20.733 |
| 同权直接标签 | 7.829／10.310 | 10.464／13.724 | 12.440／19.953 |
| R9 八近邻 | 10.037／12.473 | 10.281／13.637 | 13.265／21.717 |
| Size 概率均值 | 14.599／14.105 | 12.336／14.139 | 13.177／16.301 |
| C 正例比例 | 14.309／16.359 | 15.933／18.260 | 13.495／21.806 |

加权残差相对均匀残差，在两个 Mutagenicity 设置的 U 分层误差分别低 0.765 和 0.312 个百分点，均低于登记的 1 个百分点改进门槛。PROTEINS 的 U 误差反而高 0.269 个百分点。主方法的 E 全池有符号误差／绝对误差，三个设置依次为 **+1.718／9.079**、**+0.411／12.348** 和 **−2.788／19.750** 个百分点。

[逐方向与角色大小](../results/attempt-002-weighted-residuals/directions.csv) · [逐层比例误差](../results/attempt-002-weighted-residuals/proportion-strata.csv) · [逐方向全池误差](../results/attempt-002-weighted-residuals/proportion-blocks.csv) · [等权比例及全池误差汇总](../results/attempt-002-weighted-residuals/proportion-summary.csv)

## E 分类及逐图配对

单元格为 E 的 **Accuracy／Macro-F1**，单位为 %，每设置八方向等权。七个比例估计都在同一 U 层内按预测阳性率定阈，再到 E 分类。原 WL 和原 Size 使用各自的零 margin 阈值。

| 方法 | Mutagenicity 边数 | Mutagenicity 节点数 | PROTEINS 边数 |
| --- | ---: | ---: | ---: |
| **加权残差** | **68.474／66.974** | **65.524／63.352** | **67.836／58.145** |
| WL 概率均值 | 68.817／67.340 | 66.320／63.972 | 70.969／60.167 |
| 均匀源残差 | 68.123／66.587 | 65.852／63.730 | 68.299／58.387 |
| 同权直接标签 | 68.840／67.352 | 65.273／63.132 | 67.836／58.145 |
| R9 八近邻 | 68.893／67.495 | 66.792／64.344 | 67.012／55.567 |
| Size 概率均值 | 69.810／66.694 | 66.822／63.935 | 68.649／59.016 |
| C 正例比例 | 67.662／66.222 | 62.275／60.087 | 67.257／56.748 |
| 原 WL | 70.801／67.406 | 67.705／63.379 | 75.343／48.188 |
| 原 Size | 59.922／47.908 | 58.523／49.247 | 65.555／39.299 |

| 设置 | 主方法 − WL 均值定阈（pp） | 纠正／误改 | 动作分歧率 | 主方法 − 原 WL（pp） | 纠正／误改 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | −0.343 | 8／11 | 2.192% | −2.327 | 45／65 |
| Mutagenicity 节点数 | −0.796 | 23／30 | 6.154% | −2.181 | 45／64 |
| PROTEINS 边数 | −3.134 | 4／11 | 6.617% | −7.507 | 18／35 |

纠正与误改为逐图计数之和，Accuracy 差和分歧率按方向等权。PROTEINS 上主方法与同权直接标签的 E 动作完全相同。主方法相对原 WL 的 Macro-F1 更高，Accuracy 更低，同一阈值协议下的 WL 均值对照在两项指标上均更高。

[逐方向分类](../results/attempt-002-weighted-residuals/endpoints.csv) · [分类汇总](../results/attempt-002-weighted-residuals/endpoint-summary.csv) · [逐图配对计数](../results/attempt-002-weighted-residuals/paired-blocks.csv) · [等权纠正、误改及分歧率](../results/attempt-002-weighted-residuals/paired-summary.csv) · [登记门槛](../results/attempt-002-weighted-residuals/gate.json)

## 配平是否成功，以及它解释了什么

38 个非空 U 层均使用该方向的全部 C 图完成配平。配平前后平均 WL 均值距离在两种 Mutagenicity 划分分别从 **0.264 降到 0.107**、从 **0.233 降到 0.092**，在 PROTEINS 从 **0.238 降到 0.196**。权重有效样本数均值为 65.9、68.9 和 23.0。10 个空 U 层在对应 E 中也为空。没有估计触发 [0,1] 截断。

实测最大一阶最优性间隙为 **9.96×10⁻¹¹**，纳入了目标独有 WL 坐标。独立 SLSQP 对照的 70 个小型凸问题中，两个求解器的最大目标值差为 **5.26×10⁻¹⁴**。正式分层结果的残差分解恒等式逐项通过。本次数据说明缩小 WL 均值差可以实现，但这个变化未使加权源残差优于不校正的 WL 均值。结果范围是原源域内的大小段，尚无法据此判断所有跨域残差估计。

[原始估计、阈值和配平诊断](../results/attempt-002-weighted-residuals/estimates.json) · [独立优化核对](../results/attempt-002-weighted-residuals/independent-qp-check.json)

## 审查与封存

正式执行提交为 `b634670`。独立审查后固定角色和配置，24 个方向在拟合和评分阶段各完成 48 次角色读取。OBS 封存全部预测后才开启 U/E 标签。历史跨域 A/H 文件和 R5 整域类计数均未参与。封存后复核了 12 个切点、336 个比例端点、216 个分类端点和 360 项配对计数。

事后数值回归发现，完全均匀的权重或 C/U 层完全相同的 WL 概率可使理论相同的规则在浮点平局处分开。修复提交 `ea1b4f2` 统一了这些解析等价的阈值。使用已封存模型和无标签特征检查全部 38 个有效层，三个条件在实测均未触发。48 个 D 模型的 U/E margin 回放一致，保留原 OBS 和 FINAL，无需重跑。

[正式配置](../results/attempt-002-weighted-residuals/executed-config.json) · [角色](../results/attempt-002-weighted-residuals/roles.json) · [拟合读口](../results/attempt-002-weighted-residuals/fit-label-access.json) · [评分读口](../results/attempt-002-weighted-residuals/score-label-access.json) · [OBS](../results/attempt-002-weighted-residuals/OBS.json) · [FINAL](../results/attempt-002-weighted-residuals/FINAL.json) · [复核](../results/attempt-002-weighted-residuals/verification.json) · [均匀权重影响](../results/attempt-002-weighted-residuals/uniform-equivalence-impact.json) · [相同 WL 概率影响](../results/attempt-002-weighted-residuals/shared-WL-equivalence-impact.json) · [日志](../results/attempt-002-weighted-residuals/run.log) · [事后回归](../results/attempt-002-weighted-residuals/post-run-numerical-tests.log)
