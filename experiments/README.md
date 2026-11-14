# TSBA 实验

实验按数据类型和真实数据集归档。先看各分类 README，再进入每轮报告、协议与证据。

| 分类 | 阅读入口 | 内容 |
|---|---|---|
| 合成数据 | [synthetic](synthetic/README.md) | E1/E2、控制实验、源后验、静态 score 与 P0 训练预算 |
| Mutagenicity | [mutagenicity](mutagenicity/README.md) | R1–R4 真实图实验；R4 含原始 TSBA 补充对照 |

## 最新结果

[Mutagenicity R4（含 A0 补跑）](mutagenicity/r4_all_direction_transfer_20261112/report.md)：原始 TSBA 的平均 Accuracy 为 70.001%，源模型为 69.658%；零请求对照为 70.020%，尚未观察到 score 的平均额外收益。M11 为 69.701%。这些是完整目标域上的开发评价，不是独立最终测试。

[P0 合成数据报告](synthetic/p0_20261110/report.md)：增加训练预算后，二维固定查询上的方向一致率达到 99.980%；这不是图分类 Accuracy，也不直接证明真实图适配有效。

## 实验状态索引

所有下列研究轮次均已完成；目录整理不新增研究结果。每个入口包含计划、实际配置、来源和报告。

| 研究问题 | 状态 | 结论与后续关系 |
|---|---|---|
| [符号、几何与适配能否按设计运行？](synthetic/e1_e2_20261109/README.md) | 已完成 | 存在高维反例；后续控制拆解原因 |
| [目标请求能否带来额外收益？](synthetic/e2_controls_20261109/README.md) | 已完成 | 正确请求的小增量不等于整体优于初始化 |
| [小批次噪声是否解释适配失败？](synthetic/e2_full_source_20261110/README.md) | 已完成 | 全源更新仍可出现目标下降 |
| [适配如何影响源泛化、目标置信度和边界？](synthetic/e2_replay_20261110/README.md) | 已完成 | 经验源 CE 改善不保证总体风险改善 |
| [源标签抽样噪声如何影响结论？](synthetic/e2_source_posterior_20261110/README.md) | 已完成 | 消除噪声后总体 score 收益仍小 |
| [样本量与带宽如何影响 score？](synthetic/e2_static_score_20261110/README.md) | 已完成；预算解释由 P0 更新 | 400 步结果不能解释为网络不可学 |
| [训练不足能否解释 score 方向误差？](synthetic/p0_20261110/README.md) | 已完成 | 二维长预算可学到方向；不等于真实图适配有效 |
| [冻结图表示是否保留类信息和边界支持？](mutagenicity/r1_representation_boundary_support_20261111/README.md) | 已完成 | 局部支持不足；不能据此确认低密度分隔 |
| [密度尺度与单步适配诊断揭示了什么？](mutagenicity/r1b_density_and_single_step_diagnostics_20261111/README.md) | 已完成 | 多项诊断与单步结果见报告，不能当作闭环有效性证明 |
| [训练预算与多轮适配如何影响 score？](mutagenicity/r2_score_budget_and_adaptation_20261111/README.md) | 已完成；旧数据口径 | 后续由 R3 检查约束与源拟合问题；不可跨数据池直接比分 |
| [源风险与置信度约束如何影响适配？](mutagenicity/r3_risk_and_confidence_constraints_20261112/README.md) | 已完成 | 固定配置扩展到 R4；开发调参不等于独立测试 |
| [固定配置在各迁移方向上是否带来收益？](mutagenicity/r4_all_direction_transfer_20261112/README.md) | 已完成 | score 未表现出平均额外收益；全部方法共用一份 R4 报告 |

## 实验项目结构

```text
experiments/
  common/                  # 纯通用工具
  synthetic/<experiment_id>/
  mutagenicity/<experiment_id>/
    README.md              # 问题、结论、运行方式、依赖
    src/                   # 该轮实现
    configs/               # 计划与实际配置
    docs/                  # 方案及补充说明
    results/               # 汇总、图表、日志、核验
    report.md              # 完整结论
scripts/<数据集>/<experiment_id>/   # 仓库根目录的启动脚本
```

新实验的目录名包含轮次编号、研究主题和日期，使用 `<round>_<topic>_<YYYYMMDD>`，并在索引中用研究问题作为显示名。方案、报告和目录结构见[新实验模板](templates/README.md)。已有合成实验沿用目录名，编号见各轮 README。

`common` 不按轮次命名，不保存训练或适配流程。R2/R3 等特定协议的代码归本轮 `src/`；后续实验沿用时明确列出依赖。核心算法仍位于 `tsba/`。

## 运行代码

从仓库根目录安装并调用模块入口：

```sh
python -m pip install -e '.[experiments]'
python -m scripts.mutagenicity.r4_all_direction_transfer_20261112.run --help
python -m scripts.synthetic.p0_20261110.run_p0 --help
```

各轮 README 列出具体入口。命令参数与标签使用规则仍以该轮协议为准；目录整理不授权重跑真实实验。研究入口面向仓库 checkout，使用 editable 安装；历史数据、封存配置和外部档案不会随 Python 包安装自动提供。
