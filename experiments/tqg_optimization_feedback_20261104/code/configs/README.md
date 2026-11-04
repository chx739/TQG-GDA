# 配置导航

配置文件保留既有执行路径，避免破坏历史命令、数据清单和代码快照的引用。

| 用途 | 入口 |
|---|---|
| v1原方法、改进、完整query对照 | [v1配置索引](v1/README.md) |
| v2完整训练与基线 | [v2配置索引](v2/README.md) |
| v2 P0诊断 | [p0_diagnostics.json](p0_diagnostics.json) |
| v2 D3／D4 | [d34_diagnostics.json](d34_diagnostics.json) |
| v2 H-only完整对照 | [h_only_diagnostics.json](h_only_diagnostics.json) |
| v2 D8／残差分解 | [mechanism_diagnostics.json](mechanism_diagnostics.json) |
| v2 标签／预测与D7-H | [label_query_diagnostics.json](label_query_diagnostics.json) |

§3.8第一批A0／A1／A2-a已实现并完成，配置见[assumptions_diagnostics.json](assumptions_diagnostics.json)，[结果](../docs/v2/results/tqg_assumptions_20261103/结果解读.md)。后续A2-b及A3—A6尚未实现／执行，不把本配置当全部方案入口。

PROTEINS同方案复验：[assumptions_proteins.json](assumptions_proteins.json)，[结果](../docs/v2/results/tqg_assumptions_proteins_20261104/结果解读.md)。配置的query51、38/39折大小、GIN与旧源起点应整体匹配，不直接改数据路径冒充原Mutagenicity配置。
