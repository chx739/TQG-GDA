# r4_20261112 证据入口

状态：已完成。具体结论与适用范围以[报告](report.md)为准。

- 计划配置：[r4-plan.json](r4-plan.json)；[实验方案](r4-protocol.md)。
- 实际执行计划：[executed-plan.json](executed-plan.json)。
- 实际配置快照及各文件 SHA256：[executed-configs.json](executed-configs.json)。
- 来源、计划差异与代码提交：[evidence.json](evidence.json)。

配置来源文件从既有档案提取并核对成员哈希，R4 的展示记录合并了主运行与 A0，并非根据当前计划重新填写。配置快照不包含所有选中 checkpoint 和逐任务动态状态；这些仍由报告和原始档案提供。历史路径和旧提交号保留原貌；清理后执行提交号见本轮 `evidence.json`。

后续重跑必须创建新 attempt，不覆盖这些文件。命令、数据角色及评价权限见方案与报告；目录整理不授权重跑或使用目标标签。

R4 主运行和 A0 补跑统一记录在本目录、同一份报告及执行配置中。`executed-plan.json` 的 `main` 和 `A0_extension` 保留两个实际运行配置；它是整合记录，不是可直接传给运行器的单一计划。补跑登记、原始配置哈希见 `executed-configs.json`，来源见 `evidence.json` 和原存档索引。

计划与实际配置字段一致。

## 本轮代码

| 文件 | 用途 |
|---|---|
| [evaluate.py](evaluate.py) | 本轮专用实现，由运行或分析入口调用 |
| [protocol.py](protocol.py) | 本轮专用实现，由运行或分析入口调用 |
| [report.py](report.py) | 本轮专用实现，由运行或分析入口调用 |
| [run.py](run.py) | 命令行入口：`python -m experiments.mutagenicity.r4_20261112.run --help` |
| [supplement.py](supplement.py) | 命令行入口：`python -m experiments.mutagenicity.r4_20261112.supplement --help` |
| [training.py](training.py) | 本轮专用实现，由运行或分析入口调用 |
| [trajectory.py](trajectory.py) | 本轮专用实现，由运行或分析入口调用 |

共用实现见 [experiments/common](../../common/README.md)。在仓库根目录运行；实际训练与评价需遵循本轮协议。
