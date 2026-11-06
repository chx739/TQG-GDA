# 代码阅读与执行入口

**2026-11-06 起，当前主实现请先读 [SEGA式 TQG 流水线](sega_style_tqg.md)。** 阅读顺序：`pipeline.py` → `config.py`／`data.py` → `encoder.py`／`classifier.py` → `pretrain.py` → 对应版本的 `query.py`（v2 再读 `calibration.py`）→ `adapt.py` → `predict.py`。下面的实验编排和旧训练入口仍保留，用于已有研究记录。

算法模块位于`tqg_gda/`，阶段实验与报告脚本位于`workflows/`，通用CLI位于`scripts/`。所有命令从仓库根目录执行；阶段入口使用`python -m workflows...`。完整分类见[实验脚本导航](../../workflows/README.md)和[旧路径映射](../workflow_migration.json)。

## 先读共用部分

| 文件 | 内容 |
|---|---|
| [data.py](../../tqg_gda/data.py) | 图数据、域任务及数据角色 |
| [encoder.py](../../tqg_gda/encoder.py)／[classifier.py](../../tqg_gda/classifier.py) | 当前SEGA骨干与TQG线性头 |
| [pipeline.py](../../tqg_gda/pipeline.py)／[adapt.py](../../tqg_gda/adapt.py) | 当前完整流水线和逐步更新 |
| [models.py](../../tqg_gda/models.py)／[training.py](../../tqg_gda/training.py) | 历史骨干、checkpoint与训练兼容接口 |
| [tests/](../../tests/) | 数学、状态、训练与标签边界检查 |

## v1

| 文件／入口 | 内容 |
|---|---|
| [v1/query.py](../../tqg_gda/v1/query.py) | 探测、有限响应、残差组合和输出纠错；feedback.py保留导入兼容 |
| [revision.py](../../tqg_gda/revision.py) | 改进实验反馈分支 |
| [scripts/run_experiment.py](../../scripts/run_experiment.py) | 基础训练命令 |
| [full_query_experiment.py](../../workflows/v1/full_query_experiment.py) | PROTEINS完整query矩阵 |
| [mutagenicity_experiment.py](../../workflows/v1/mutagenicity_experiment.py) | Mutagenicity早期完整对照 |
| [run_revision_pipeline.py](../../workflows/v1/run_revision_pipeline.py) | v1改进实验历史编排 |

[方法与实验设计](../v1/README.md) · [配置](../../configs/v1/README.md)。带原运行目录依赖的编排脚本不等于开箱即用复现入口。

## v2：建议按这个顺序精读

| 顺序 | 文件 | 对应内容 |
|---|---|---|
| 1 | [v2/query.py](../../tqg_gda/v2/query.py) | 固定快照、目标Jacobian空间、响应、签名核、残差读取与反馈 |
| 2 | [v2/calibration.py](../../tqg_gda/v2/calibration.py) | W的强凸校准求解、Ω及停止准则 |
| 3 | [adapt.py](../../tqg_gda/adapt.py) | 源CE与参数反馈合并、停止梯度和SGD；旧training.py供历史流程使用 |
| 4 | [p0_diagnostics.py](../../tqg_gda/p0_diagnostics.py) | 几何封存、目标标签Oracle与梯度／单步诊断 |
| 5 | [d34_diagnostics.py](../../tqg_gda/d34_diagnostics.py) | 核与校准对照 |
| 6 | [mechanism_diagnostics.py](../../tqg_gda/mechanism_diagnostics.py) · [query_label_diagnostics.py](../../tqg_gda/query_label_diagnostics.py) | 已知规律、残差拆分及标签／预测归因 |

### v2实验编排及实现说明

| 阶段 | 主入口 | 解释 |
|---|---|---|
| 完整训练 | [scripts/run_experiment.py](../../scripts/run_experiment.py)，配合v2配置 | [方法实现](../v2/implementation/method.md) |
| P0 | [p0_experiment.py](../../workflows/v2/p0/p0_experiment.py) | [P0实现](../v2/implementation/p0.md) |
| D3／D4 | [d34_experiment.py](../../workflows/v2/d34/d34_experiment.py) | [D3／D4实现](../v2/implementation/d3_d4.md) |
| H-only | [h_only_experiment.py](../../workflows/v2/h_only/h_only_experiment.py) | [H-only实现](../v2/implementation/h_only.md) |
| D8／残差 | [mechanism_experiment.py](../../workflows/v2/mechanisms/mechanism_experiment.py) | [D8实现](../v2/implementation/d8_residual.md) |
| 标签／query | [label_query_experiment.py](../../workflows/v2/label_query/label_query_experiment.py) | [标签／query实现](../v2/implementation/label_query.md) |
| §3.8 A0／A1／A2-a | [assumptions_experiment.py](../../workflows/v2/assumptions/assumptions_experiment.py) | [条件诊断实现](../v2/implementation/assumptions.md) |
| PROTEINS §3.8复验 | [PROTEINS backend](../../workflows/v2/assumptions/proteins_assumptions_experiment.py)，由共同入口按配置选择 | [方案与适配](../v2/plans/assumptions_proteins_20261104/README.md) |

LOCAL／核读取优化的执行、审计和报告位于[workflows/v2/optimization/](../../workflows/v2/optimization/)，对应[11月4日优化方案](../v2/plans/optimization_execution_20261104/README.md)。

同名前缀的`*_summarize.py`、`*_report.py`、`*_interpretation.py`分别承担汇总、报告和后处理；不是新的训练方法。历史精确执行代码以各实验的code_snapshot与提交为准。

## 归档工具

[scripts/](../../scripts/)下的`archive_*.py`负责已完成产物的导出，不进行训练。[verify_experiment_archive.py](../../scripts/verify_experiment_archive.py)验证完整训练归档的字节、评价记录与模型链接；机制诊断归档还需核对对应artifact_audit和manifest。
