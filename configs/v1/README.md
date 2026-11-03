# v1配置索引

这些文件仍位于原路径。本目录只做导航，不复制配置。

| 配置 | 用途 |
|---|---|
| [tqg_development.json](../tqg_development.json) | 基础v1 query16示例 |
| [full_query/](../full_query/) | PROTEINS n51／n250完整query对照 |
| [mutagenicity/sega_full_query.json](../mutagenicity/sega_full_query.json) | Mutagenicity SEGA骨干／源训练及v1实验矩阵 |
| [revision_v1/](../revision_v1/) | v1改进与机制验证的历史配置 |
| [diagnostics/](../diagnostics/) · [pilot/](../pilot/) | 早期方向、泛化和pilot诊断 |
| [odd_only](../odd_only.json) · [linear_jvp](../linear_jvp.json) | 响应消融；同目录另有8cols变体 |
| [source_only_same_fit](../source_only_same_fit.json) · [source_only_all_labels](../source_only_all_labels.json) | 源监督基线 |
| [random_probes](../random_probes.json) · [source_probes](../source_probes.json) | 方向来源对照 |
| [mean_residual_q16](../mean_residual_q16.json) · [intercept_q16](../intercept_q16.json) | 残差基线；同目录另有q32变体 |

配置本身不代表实验已执行。历史报告、种子、优化器与数据SHA须同时核对。[v1结果](../../docs/v1/results/README.md)。
