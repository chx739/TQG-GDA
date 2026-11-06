# v1实验设计与失败分析

这些方案针对有限扰动、odd/even与输出残差反馈。方案提出的项目不代表都已完成；具体状态以当时报告和验证记录为准。

| 文档 | 用途 | 配套证据 |
|---|---|---|
| [新SEGA式v1/v2最小验证，2026-11-06](../../research_plans/tqg_sega_minimal_20261106/README.md) | 新起点、固定反馈/单步、条件性50步对照 | 方案与配置已登记，尚未真实训练 |
| [有效性验证路线](validation_roadmap.md) | 源监督、伪标签、方向和响应消融的整体比较 | [E1—E4历史记录](../../validation/README.md) |
| [E1首个机制实验](E1_mechanism.md) | 验证方向、响应和反馈的基本机制 | [E1记录](../../validation/e1/) |
| [失败原因与改进方案](failure_analysis.md) | 数学、实现及统计失效分析 | 不把分析当作已证明的因果结论 |
| [v1改进实验计划](improvement_plan.md) | query数量、响应读取、局部匹配、幅度和执行步骤 | [历史revision说明](../../history/README_revision.md) |
| [E4残差基线说明](E4_residual_baselines.md) | 平均残差和常数项等对照 | [E4记录](../../validation/e4/) |

最终完整训练结果另见[v1结果](../results/README.md)。新SEGA式v1/v2共用上方最小验证方案；既有v2研究计划另见[v2计划](../../v2/plans/README.md)。
