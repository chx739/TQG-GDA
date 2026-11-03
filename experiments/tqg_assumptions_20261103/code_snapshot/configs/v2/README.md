# v2训练配置

| 配置 | 用途 |
|---|---|
| [tqg_v2.json](tqg_v2.json) | HA核、联合校准主方法 |
| [mean_only_krr.json](mean_only_krr.json) | 去掉最大方向项的核回归对照 |
| [representation_only.json](representation_only.json) | H-only核，保留目标梯度查询 |
| [permuted_labels.json](permuted_labels.json) | query标签置乱 |
| [source_only_sgd.json](source_only_sgd.json) | 继续源fit SGD |
| [source_all_sgd.json](source_all_sgd.json) | 全源监督SGD |
| [pseudo_labels_sgd.json](pseudo_labels_sgd.json) | 置信度伪标签SGD |

[实现与命令](../../docs/v2/implementation/method.md) · [诊断配置](../README.md) · [结果](../../docs/v2/results/README.md)
