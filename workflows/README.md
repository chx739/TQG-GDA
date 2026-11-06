# 实验脚本导航

这里放实验编排、汇总和报告生成脚本；可复用的算法实现位于 [`tqg_gda/`](../tqg_gda/)。已完成实验的原始产物和当时的代码位于 [`experiments/`](../experiments/README.md)。

## 按研究阶段找入口

| 阶段 | 目录 | 执行入口 | 汇总／报告 |
|---|---|---|---|
| 当前SEGA式TQG v1/v2 | [tqg/](tqg/) | [run_task.py](tqg/run_task.py)、[run_all.py](tqg/run_all.py) | 每个任务保存完整配置、历史、模型及产物SHA；[说明](../docs/code/sega_style_tqg.md) |
| v1 早期验证 | [v1/](v1/) | [experiment.py](v1/experiment.py)、[diagnostics_pilot.py](v1/diagnostics_pilot.py) | [plot.py](v1/plot.py) |
| v1 改进验证 | [v1/](v1/) | [run_revision_pipeline.py](v1/run_revision_pipeline.py)、[experiment_revision.py](v1/experiment_revision.py) | [summarize_revision.py](v1/summarize_revision.py) |
| v1 完整 query 对照 | [v1/](v1/) | [full_query_experiment.py](v1/full_query_experiment.py)、[mutagenicity_experiment.py](v1/mutagenicity_experiment.py) | [mutagenicity_evaluate.py](v1/mutagenicity_evaluate.py) |
| v2 P0：梯度与单步诊断 | [v2/p0/](v2/p0/) | [p0_experiment.py](v2/p0/p0_experiment.py) | [p0_summarize.py](v2/p0/p0_summarize.py) |
| v2 D3／D4：校准与签名 | [v2/d34/](v2/d34/) | [d34_experiment.py](v2/d34/d34_experiment.py) | [d34_summarize.py](v2/d34/d34_summarize.py)、[d34_report.py](v2/d34/d34_report.py) |
| v2 H-only | [v2/h_only/](v2/h_only/) | [h_only_experiment.py](v2/h_only/h_only_experiment.py) | [h_only_report.py](v2/h_only/h_only_report.py) |
| v2 D8／残差分解 | [v2/mechanisms/](v2/mechanisms/) | [mechanism_experiment.py](v2/mechanisms/mechanism_experiment.py) | [mechanism_summarize.py](v2/mechanisms/mechanism_summarize.py) |
| v2 标签归因／query 数量 | [v2/label_query/](v2/label_query/) | [label_query_experiment.py](v2/label_query/label_query_experiment.py) | [label_query_summarize.py](v2/label_query/label_query_summarize.py) |
| v2 §3.8 条件验证 | [v2/assumptions/](v2/assumptions/) | [assumptions_experiment.py](v2/assumptions/assumptions_experiment.py)，含 PROTEINS backend | [assumptions_summarize.py](v2/assumptions/assumptions_summarize.py) |
| v2 LOCAL／核读取优化 | [v2/optimization/](v2/optimization/) | [optimization_a4.py](v2/optimization/optimization_a4.py)、[optimization_feedback.py](v2/optimization/optimization_feedback.py)、[optimization_candidates.py](v2/optimization/optimization_candidates.py) | [optimization_a4_report.py](v2/optimization/optimization_a4_report.py)、[optimization_feedback_report.py](v2/optimization/optimization_feedback_report.py) |

同阶段的 `*_interpretation.py` 生成结果说明，`*_audit.py` 校验结果，`*_plot.py` 绘图。v1 的 `revision_common.py`、`training_revision.py` 等是该阶段共用代码。

## 当前代码怎么运行

从仓库根目录使用模块命令；例如查看参数：

```bash
python -m workflows.v2.assumptions.assumptions_experiment --help
python -m workflows.v2.optimization.optimization_a4 --help
python -m workflows.v1.full_query_experiment --help
```

目录层级变成模块名中的点，省略 `.py`。模块间使用完整包名导入，支持子进程和冻结代码副本。基础训练 CLI 仍是 `python scripts/run_experiment.py ...`，见[使用指南](../docs/usage.md)。

准备真实实验时仍需匹配对应方案的缓存、配置、源／目标角色、旧模型和独立输出目录。部分 v1 脚本依赖原 run 的同级目录；移动脚本没有取消这些依赖。冻结新实验代码时，应包含整个 `workflows/`、`tqg_gda/` 及该实验需要的配置、脚本和索引。

## 查旧路径

2026-11-05 从根目录迁移了 52 个 Python 文件，文件名保留。[完整旧路径→新路径映射](../docs/workflow_migration.json)可用于查找旧文档中的入口。

旧归档中的命令对应当时的代码快照，应在那份快照中运行；当前主分支使用上面的 `python -m` 形式。历史日志、模型、协议、代码快照及其 SHA 未随本次整理改写。
