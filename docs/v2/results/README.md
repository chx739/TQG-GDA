# v2实验结果与进度

最新先读[新SEGA式v1/v2最小验证结果，11-07](../../results/tqg_sega_minimal_20261107/README.md)：三个新源模型、48个版本块完成，v2为0/2任务通过，M2未触发。旧骨干结论见[优化研究](optimization_20261104/README.md)和[执行状态总表](status.md)。下面明确区分说明与原始证据。

| 阶段 | 结果解读 | 完整报告 | 原始归档 | 核心结论 |
|---|---|---|---|---|
| 新SEGA式M0/M1，11-07 | [详细结果](../../results/tqg_sega_minimal_20261107/README.md) | [原门槛与方案](../../research_plans/tqg_sega_minimal_20261106/README.md) | [归档](../../../experiments/tqg_sega_minimal_20261107/README.md) | 数值正常，v1/v2均0/2研究门槛通过，M2不触发 |
| 目标研究A6，11-04 | [解读](tqg_optimization_feedback_20261104/结果解读.md) | [报告](tqg_optimization_feedback_20261104/run_report.md) | [归档](../../../experiments/tqg_optimization_feedback_20261104/README.md) | 两候选均0/3通过，教师及标签归因未支持TQG稳定增量 |
| 目标研究A2-b，11-04 | [解读](tqg_optimization_local_20261104/结果解读.md) | [报告](tqg_optimization_local_20261104/run_report.md) | [归档](../../../experiments/tqg_optimization_local_20261104/README.md) | 没有局部配置通过至少两个任务；公开聚合口径敏感性 |
| 目标研究A4-a，11-04 | [解读](tqg_optimization_a4_20261104/结果解读.md) | [报告](tqg_optimization_a4_20261104/run_report.md) | [归档](../../../experiments/tqg_optimization_a4_20261104/README.md) | 核学习曲线与审计完成；跨域读取仍弱，A6验收见本轮新结果 |
| Mutagenicity完整训练，10-26 | [解读](mutagenicity_v2_20261026/结果解读.md) | [报告](mutagenicity_v2_20261026/run_report.md) | [归档](../../../experiments/mutagenicity_v2_20261026/README.md) | 未获跨任务稳定收益 |
| 目标batch16／32／64，10-28 | [解读](mutagenicity_target_batch_20261028/结果解读.md) | [报告](mutagenicity_target_batch_20261028/run_report.md) | [归档](../../../experiments/mutagenicity_target_batch_20261028/README.md) | 幅度更稳定不等于分类更好 |
| 目标batch196，10-28 | [解读](mutagenicity_target_batch196_20261028/结果解读.md) | [报告](mutagenicity_target_batch196_20261028/run_report.md) | [归档](../../../experiments/mutagenicity_target_batch196_20261028/README.md) | 没有稳定解决问题 |
| P0：D0／D1／D2，10-31 | [解读](tqg_v2_p0_20261031/结果解读.md) | [报告](tqg_v2_p0_20261031/run_report.md) | [归档](../../../experiments/tqg_v2_p0_20261031/README.md) | 工程通过，方向及单步门槛未过 |
| D3校准／D4签名，10-31 | [解读](tqg_d3_d4_20261031/结果解读.md) | [报告](tqg_d3_d4_20261031/run_report.md) | [归档](../../../experiments/tqg_d3_d4_20261031/README.md) | JOINT和加入响应距离无稳定增量 |
| H-only完整对照，10-31 | [解读](tqg_h_only_controls_20261031/结果解读.md) | [报告](tqg_h_only_controls_20261031/run_report.md) | [归档](../../../experiments/tqg_h_only_controls_20261031/README.md) | 方向门槛未过，两个任务单步有收益 |
| D8／残差分解，10-31 | [解读](tqg_d8_residual_20261031/结果解读.md) | [报告](tqg_d8_residual_20261031/run_report.md) | [归档](../../../experiments/tqg_d8_residual_20261031/README.md) | 友好合成成立，真实逐图标签贡献仍需拆分 |
| 标签／预测归因与D7-H，11-02 | [解读](tqg_label_query_20261102/结果解读.md) | [报告](tqg_label_query_20261102/run_report.md) | [归档](../../../experiments/tqg_label_query_20261102/README.md) | 稳定标签增量及跨正则数量瓶颈门槛未过 |
| §3.8 A0／A1／A2-a，11-03 | [解读](tqg_assumptions_20261103/结果解读.md) | [报告](tqg_assumptions_20261103/run_report.md) | [归档](../../../experiments/tqg_assumptions_20261103/README.md) | 同域有信息，跨域有缺口，H支持代理有诊断价值 |
| PROTEINS §3.8复验，11-04 | [解读](tqg_assumptions_proteins_20261104/结果解读.md) | [报告](tqg_assumptions_proteins_20261104/run_report.md) | [归档](../../../experiments/tqg_assumptions_proteins_20261104/README.md) | 同域稳定可学门槛未过；支持分层图数不足，未复现Mutagenicity证据链 |

[完整训练数值表与统一设置](overview.md)。P0及其后各类是冻结快照／独立临时单步诊断，不是新的连续适配准确率。研究门槛未过与程序报错不同。目标final标签未用于这些诊断。
