# 实验原始归档

| 版本 | 归档入口 | 可读结果 |
|---|---|---|
| 新SEGA式 v1/v2共同验证，11-07 | [完整原始归档](tqg_sega_minimal_20261107/README.md) | [M0/M1结果、M2不触发原因](../docs/results/tqg_sega_minimal_20261107/README.md) |
| v1：有限扰动、odd/even | [v1归档索引](v1/README.md) | [结果与完整表格](../docs/v1/results/README.md) |
| v2：Jacobian、核查询、梯度反馈 | [v2归档索引](v2/README.md) | [结果及最新状态](../docs/v2/results/README.md) |

归档目录保持原路径，避免改写代码快照、manifest、checkpoint引用及历史SHA。这里的v1／v2目录是版本导航，不复制大文件。每份归档的README说明范围；`manifest.json`、评价索引、代码提交与日志提供追溯依据。

完整训练归档通常含history.jsonl、checkpoint、配置及模型评价索引；机制诊断可能只引用已有模型，不含新增训练模型。不可将复用模型计成新训练。

核验完整训练归档（不需要PyTorch）：

```bash
python scripts/verify_experiment_archive.py experiments/proteins_full_query_20261026
```

诊断归档的manifest结构和评价类型不同，按其README及artifact_audit核查。模型文件不使用Git LFS；真实数据、凭据和未完成运行不进入归档。新增实验完成后，必须加入相应版本索引与结果目录。
