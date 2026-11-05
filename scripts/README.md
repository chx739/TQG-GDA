# 通用命令行与维护工具

从仓库根目录运行 `python scripts/<文件>.py ...`。按实验阶段编排的代码集中在 [`workflows/`](../workflows/README.md)。

| 用途 | 工具 |
|---|---|
| 准备源划分 | [prepare_source_split.py](prepare_source_split.py) |
| 基础训练／批量调度 | [run_experiment.py](run_experiment.py)、[run_matrix.py](run_matrix.py) |
| 显式评价 | [evaluate.py](evaluate.py) |
| 早期诊断／汇总／后续验证 | [run_diagnostic.py](run_diagnostic.py)、[summarize_diagnostic.py](summarize_diagnostic.py)、[run_followup.py](run_followup.py) |
| 合成数据端到端检查 | [smoke_test.py](smoke_test.py)、[smoke_v2.py](smoke_v2.py) |
| 完整训练归档 | [archive_full_query.py](archive_full_query.py)、[archive_v2.py](archive_v2.py)、[archive_target_batch.py](archive_target_batch.py) |
| P0、D3/D4、H-only 归档 | [archive_p0.py](archive_p0.py)、[archive_d34.py](archive_d34.py)、[archive_h_only.py](archive_h_only.py) |
| 残差分解、标签归因归档 | [archive_mechanisms.py](archive_mechanisms.py)、[archive_label_query.py](archive_label_query.py) |
| 条件验证、优化实验归档 | [archive_assumptions.py](archive_assumptions.py)、[archive_optimization_phase.py](archive_optimization_phase.py) |
| 核验完整训练归档 | [verify_experiment_archive.py](verify_experiment_archive.py) |
| 汇报生成器 | [build_assumptions_brief.py](build_assumptions_brief.py) |

归档工具只用于已完成、已审计的实验。不同实验的 manifest 和模型关联规则有区别，使用对应归档器。部分早期工具依赖特定运行目录，参数与限制以脚本及原实验记录为准。

汇报生成器是历史模板；它的默认内容包含 11 月 4 日记录，会覆盖本地汇报。当前本地汇报已手动改成仅含 11 月 3 日数据，请先核对模板再使用。该汇报文件继续由 `.gitignore` 排除，不随仓库更新发布。
