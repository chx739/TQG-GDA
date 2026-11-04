# TQG v2：当前研究入口

v2让无标签目标图定义当前参数梯度查询，依据图表示／Jacobian响应的签名核，构造矩阵W读取源标签残差，再形成参数更新反馈。

| 阅读顺序 | 入口 |
|---|---|
| 1．精读方法 | [v2设计稿](method/design.md) |
| 2．核对公式和代码 | [实现说明](implementation/method.md) · [代码导航](../code/README.md) |
| 3．看哪些实验做完了 | [执行状态](results/status.md) · [完整结果目录](results/README.md) |
| 4．设计下一步 | [实验设计目录](plans/README.md) · [§3.8四条件验证](plans/assumptions_20261103/README.md) |
| 5．追溯日志、指标与模型 | [v2原始归档目录](../../experiments/v2/README.md) |

截至2026-11-03：友好合成数据的方向恢复通过；真实任务逐图源标签稳定增量尚未得到支持，增加query的收益也依赖正则约定。当前不直接扩完整适配。§3.8第一批A0／A1／A2-a已完成：同域可学、跨域存在缺口，H支持代理可定位困难区域；[结果](results/tqg_assumptions_20261103/结果解读.md)。该历史第一批之后，A2-b、A4-a和A6已完成，见下方最新结果；A3/A5未在本轮范围内执行。

配置见[配置导航](../../configs/README.md)。旧v1使用有限扰动与odd/even；不要把那些公式和实验结论直接套到本版。

[PROTEINS §3.8复验（2026-11-04）](results/tqg_assumptions_proteins_20261104/结果解读.md)已完成。源内／目标内主门槛未过，支持分层数量不足；与Mutagenicity分开解释，不能仅归因于数据量。

最新：[2026-11-04优化研究结论](results/optimization_20261104/README.md)。源内读取改善未形成稳定目标反馈；LOCAL/KERNEL均0/3通过完整A6门槛，不触发连续适配。
