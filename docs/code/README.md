# 代码阅读与执行入口

本次整理调整文档和导航；算法模块、脚本导入名、配置路径及训练行为保持兼容。所有命令从仓库根目录执行。根目录的历史编排脚本仍保留原名，避免破坏模块互相导入和冻结复现命令。

## 先读共用部分

| 文件 | 内容 |
|---|---|
| [data.py](../../tqg_gda/data.py) | 图数据、域任务及数据角色 |
| [models.py](../../tqg_gda/models.py) | GIN／GINE骨干与图级读出 |
| [training.py](../../tqg_gda/training.py) | 源预训练、适配、模型恢复和更新 |
| [tests/](../../tests/) | 数学、状态、训练与标签边界检查 |

## v1

| 文件／入口 | 内容 |
|---|---|
| [feedback.py](../../tqg_gda/feedback.py) | 探测、有限响应、残差组合和输出纠错 |
| [revision.py](../../tqg_gda/revision.py) | 改进实验反馈分支 |
| [scripts/run_experiment.py](../../scripts/run_experiment.py) | 基础训练命令 |
| [full_query_experiment.py](../../full_query_experiment.py) | PROTEINS完整query矩阵 |
| [mutagenicity_experiment.py](../../mutagenicity_experiment.py) | Mutagenicity早期完整对照 |
| [run_revision_pipeline.py](../../run_revision_pipeline.py) | v1改进实验历史编排 |

[方法与实验设计](../v1/README.md) · [配置](../../configs/v1/README.md)。带原运行目录依赖的编排脚本不等于开箱即用复现入口。

## v2：建议按这个顺序精读

| 顺序 | 文件 | 对应内容 |
|---|---|---|
| 1 | [feedback_v2.py](../../tqg_gda/feedback_v2.py) | 固定快照、目标Jacobian空间、响应、签名核、残差读取与反馈 |
| 2 | [calibration_v2.py](../../tqg_gda/calibration_v2.py) | W的强凸校准求解、Ω及停止准则 |
| 3 | [training.py](../../tqg_gda/training.py) | 源CE与参数反馈合并、停止梯度和SGD |
| 4 | [p0_diagnostics.py](../../tqg_gda/p0_diagnostics.py) | 几何封存、目标标签Oracle与梯度／单步诊断 |
| 5 | [d34_diagnostics.py](../../tqg_gda/d34_diagnostics.py) | 核与校准对照 |
| 6 | [mechanism_diagnostics.py](../../tqg_gda/mechanism_diagnostics.py) · [query_label_diagnostics.py](../../tqg_gda/query_label_diagnostics.py) | 已知规律、残差拆分及标签／预测归因 |

### v2实验编排及实现说明

| 阶段 | 主入口 | 解释 |
|---|---|---|
| 完整训练 | [scripts/run_experiment.py](../../scripts/run_experiment.py)，配合v2配置 | [方法实现](../v2/implementation/method.md) |
| P0 | [p0_experiment.py](../../p0_experiment.py) | [P0实现](../v2/implementation/p0.md) |
| D3／D4 | [d34_experiment.py](../../d34_experiment.py) | [D3／D4实现](../v2/implementation/d3_d4.md) |
| H-only | [h_only_experiment.py](../../h_only_experiment.py) | [H-only实现](../v2/implementation/h_only.md) |
| D8／残差 | [mechanism_experiment.py](../../mechanism_experiment.py) | [D8实现](../v2/implementation/d8_residual.md) |
| 标签／query | [label_query_experiment.py](../../label_query_experiment.py) | [标签／query实现](../v2/implementation/label_query.md) |

同名前缀的`*_summarize.py`、`*_report.py`、`*_interpretation.py`分别承担汇总、报告和后处理；不是新的训练方法。历史精确执行代码以各实验的code_snapshot与提交为准。

## 归档工具

[scripts/](../../scripts/)下的`archive_*.py`负责已完成产物的导出，不进行训练。[verify_experiment_archive.py](../../scripts/verify_experiment_archive.py)验证完整训练归档的字节、评价记录与模型链接；机制诊断归档还需核对对应artifact_audit和manifest。
