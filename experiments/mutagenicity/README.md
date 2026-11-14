# Mutagenicity 实验

[返回实验总览](../README.md) · [合成数据](../synthetic/README.md)

## 最新结果：R4 全方向及原始 TSBA 对照

[R4 完整报告](all_direction_transfer_20261112/report.md)覆盖 12 个方向、5 个种子。原实验 300 条轨迹，A0 事后补跑 60 条，合计 360 条轨迹、10,800 次尝试。

| 方法 | 12 方向平均 Accuracy |
|---|---:|
| 原始源模型 B0 | 69.658% |
| M00 零请求 B1 | 70.020% |
| 原始 TSBA：M00＋score A0 | 70.001% |
| M01＋score A1 | 69.917% |
| M11＋score A2（登记主方案） | 69.701% |

原始 TSBA 小幅改善起点，但未超过零请求；新增约束降低 NLL，同时牺牲部分 Accuracy。完整统计、辅助指标和限制见报告，不能将目标密度请求的贡献与整个更新流程的收益混为一谈。

## 按研究问题阅读

| 研究问题 | 历史轮次 | 报告 | 协议 / 计划 | 存档 |
|---|---|---|---|---|
| 固定配置全方向；含 A0 补跑 | R4 | [结果](all_direction_transfer_20261112/report.md) | [协议](all_direction_transfer_20261112/docs/r4-protocol.md) / [计划](all_direction_transfer_20261112/configs/r4-plan.json) | [索引](all_direction_transfer_20261112/results/archive-index.json) |
| 源风险、置信度约束和参数敏感性 | R3 | [结果](risk_and_confidence_constraints_20261112/report.md) | [协议](risk_and_confidence_constraints_20261112/docs/r3-protocol.md) / [计划](risk_and_confidence_constraints_20261112/configs/r3-plan.json) | [索引](risk_and_confidence_constraints_20261112/results/archive-index.json) |
| score 训练预算、λ 与 30 次适配 | R2 | [结果](score_budget_and_adaptation_20261111/report.md) | [协议](score_budget_and_adaptation_20261111/docs/r2-protocol.md) / [计划](score_budget_and_adaptation_20261111/configs/r2-plan.json) | [索引](score_budget_and_adaptation_20261111/results/archive-index.json) |
| 参考头、密度尺度、静态 score 和单步更新 | R1b | [结果](density_and_single_step_diagnostics_20261111/report.md) | [协议](density_and_single_step_diagnostics_20261111/docs/r1b-protocol.md) / [计划](density_and_single_step_diagnostics_20261111/configs/next-round-plan.json) | [索引](density_and_single_step_diagnostics_20261111/results/archive-index.json) |
| 源模型、类别可读性和边界支持量 | R1 | [结果](representation_boundary_support_20261111/report.md) | [协议](representation_boundary_support_20261111/docs/mutagenicity-stage1-protocol.md) / [计划](representation_boundary_support_20261111/configs/stage1-plan.json) | [索引](representation_boundary_support_20261111/results/archive-index.json) |

## 数据与评价口径

- 沿用 SEGA 四域索引，数量为 1084 / 1084 / 1084 / 1085；与 CoCo 论文划分的对应关系尚未核实。
- R1–R2：源训练/验证/诊断为 779/109/196；目标 A/B 各 271 张，历史 final 标签未用于这些运行。R1b 探针使用 118 优化＋29 内部验证，B 仅评价。
- R3：重新训练源模型，源训练/验证为 868/216；封存后在两个完整目标域开发计分。
- R4：四个源域分别按类划分约 80%/20%，完整目标域计分；主终点为第 30 次尝试的 Accuracy。每个任务只允许其源标签参与训练，目标标签在封存后用于评价。
- 不直接把不同表示、数据池或选模规则下的跨轮数字解释为方法改进。工程测试通过与研究方法有效是不同结论。

## 执行与复现

实验实现位于各轮 `src/`，纯工具集中到 `experiments/common/`。从仓库根目录使用 `python -m scripts.mutagenicity.<experiment_id>.<入口>` 运行，各轮 README 列出入口。各轮精确命令、代码提交及模型来源见协议和报告。重新执行使用新目录，不覆盖历史 attempt；本 README 不授权重新训练或读取目标标签。

[输入来源与哈希](stage1-inputs.json) · [R1 报告模板](representation_boundary_support_20261111/docs/stage1-report-template.md) · [历史总设计](../validation-plan.md)

旧的源诊断角色及 A/B/final 划分只适用于相应历史实验，不是当前 TSBA 的固定算法要求。历史 JSON、日志和存档索引中的路径保留执行时原貌。
