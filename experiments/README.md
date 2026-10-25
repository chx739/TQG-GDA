# 实验文件目录

| 实验 | 任务 | 内容 | 结果说明 |
|---|---|---|---|
| [PROTEINS 全量查询，2026-10-26](proteins_full_query_20261026/README.md) | 0→1、0→2、0→3，各5个模型种子 | 原始训练日志、源模型、适配checkpoint、配置、来源清单与SHA256 | [完整报告](../docs/experiments/latest/run_report.md) |
| [Mutagenicity SEGA配置，2026-10-26](mutagenicity_sega_20261026/README.md) | 0→1、0→2、0→3，各5个模型种子 | 115份训练日志、755份checkpoint、配置、恢复记录与SHA256 | [完整报告](mutagenicity_sega_20261026/results/run_report.md) |

每个数据集的归档均覆盖105次适配及10次共享源预训练。PROTEINS包含20次复用基线；Mutagenicity全部为本轮适配。归档不包含所有历史消融或诊断文件。早期的设计与验证摘要见[文档目录](../docs/README.md)与[历史说明](../docs/history/README.md)。

归档中的路径按任务、模型种子、方法组织；统计结论请以配套报告为准。模型文件不需要 Git LFS，单个文件均小于1 MiB。
