# v2实验设计

| 方案 | 作用 | 状态入口 |
|---|---|---|
| [新SEGA式v1/v2最小验证，2026-11-06](../../research_plans/tqg_sega_minimal_20261106/README.md) | 新骨干同起点的标签/随机/直接query-CE控制；按门槛进入50步训练 | 方案与配置已登记，尚未真实训练 |
| [下一轮优化与目标模式 prompt，2026-11-04](optimization_goal_20261104/README.md) | 局部源bank、核读取、简单教师与A6验收的有限研究闭环；[完整prompt](optimization_goal_20261104/goal_prompt.md) | 已采用，执行中；[冻结执行协议](optimization_execution_20261104/README.md) |
| [四项迁移条件验证，2026-11-03](assumptions_20261103/README.md) | 对齐跨域条件稳定性、签名充分性、函数复杂度和源监督支持；包含旧结果链接及A0—A6方案 | A0／A1／A2-a已完成；[结果](../results/tqg_assumptions_20261103/结果解读.md)；后续未执行 |
| [PROTEINS条件诊断复验，2026-11-04](assumptions_proteins_20261104/README.md) | 将A0/A1/A2-a迁到query51、匹配38/39训练图；原门槛不变 | [已完成结果](../results/tqg_assumptions_proteins_20261104/结果解读.md)；同域主门槛未过、支持分层证据不足 |
| [D0—D9详细实验总表，2026-10-31](../../research_plans/tqg_v2_20261031/README.md) | 梯度质量、独立单步、校准、签名、合成机制及条件改进 | [最新执行状态](../results/status.md) |
| [统一方法与数据协议](../../research_plans/tqg_v2_20261031/method.md) | 数据角色、符号及梯度诊断定义 | 属于冻结方案 |
| [预算与执行清单](../../research_plans/tqg_v2_20261031/execution.md) | 执行顺序、预算、产物和停止规则 | 属于冻结方案 |

2026-10-31方案保留原目录和字节，页面中的“未执行”是设计当时的状态。判断实际进度请读[执行状态](../results/status.md)，不能把冻结计划的状态当作今天的状态。

新方案不覆盖旧实验门槛；新候选先冻结实现与协议，再进行实验。
