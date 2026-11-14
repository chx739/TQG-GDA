# r2_20261111 证据入口

状态：已完成。具体结论与适用范围以[报告](report.md)为准。

- 计划配置：[r2-plan.json](r2-plan.json)；[实验方案](r2-protocol.md)。
- 实际执行计划：[executed-plan.json](executed-plan.json)。
- 实际配置快照及各文件 SHA256：[executed-configs.json](executed-configs.json)。
- 来源、计划差异与代码提交：[evidence.json](evidence.json)。

这些执行记录从既有档案逐字节提取并核对成员哈希，并非根据当前计划重新填写。配置快照不包含所有选中 checkpoint 和逐任务动态状态；这些仍由报告和原始档案提供。历史路径和旧提交号保留原貌；清理后执行提交号见本轮 `evidence.json`。

后续重跑必须创建新 attempt，不覆盖这些文件。命令、数据角色及评价权限见方案与报告；目录整理不授权重跑或使用目标标签。

计划与实际配置字段一致。

## 本轮代码

| 文件 | 用途 |
|---|---|
| [diagnostics.py](diagnostics.py) | 本轮专用实现，由运行或分析入口调用 |
| [interpret.py](interpret.py) | 命令行入口：`python -m experiments.mutagenicity.r2_20261111.interpret --help` |
| [publish.py](publish.py) | 命令行入口：`python -m experiments.mutagenicity.r2_20261111.publish --help` |
| [run.py](run.py) | 命令行入口：`python -m experiments.mutagenicity.r2_20261111.run --help` |
| [run_resources.py](run_resources.py) | 命令行入口：`python -m experiments.mutagenicity.r2_20261111.run_resources --help` |

共用实现见 [experiments/common](../../common/README.md)。在仓库根目录运行；实际训练与评价需遵循本轮协议。
