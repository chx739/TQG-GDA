# 最小验证执行器：启动前验收

日期：2026-11-07。针对[预登记方案](../../research_plans/tqg_sega_minimal_20261106/README.md)补充[执行器](../../../workflows/tqg/minimal_validation/README.md)，不改变 canonical TQG 算法、SEGA 骨干、数据划分或研究门槛。

| 检查 | 结果 | 证据 |
|---|---|---|
| 新实验执行器 | 10项通过 | [测试](../../../tests/test_minimal_validation.py)、[原始日志](driver.tests.log) |
| 原生产流水线回归 | 13项通过 | [原始日志](canonical.tests.log) |
| 模块CLI | `--help` 正常 | `python -m workflows.tqg.minimal_validation.run --help` |

检查包含：TRUE与canonical梯度一致，v2置乱仅更改残差，MEAN-Y保留预测项，固定图ID标签置乱，RAND空间/范数，临时FP32步与FP64审计、原状态复位，query-CE准确联合梯度与同fit调度，PERM训练数据角色和批次调度，未封存/final访问拒绝，以及完整双版本诊断块封存、评价、复用流程。预设门槛的通过/缺失/失败种子路径均做合成检查。

主训练之前固定执行commit、代码SHA、预登记文件SHA、数据SHA和全部M1批次及三组置乱映射。所有新实验统计单独归档，不回写历史结果或预登记门槛。确定性重放只作工程检查，不增加独立实验块数。

这些是合成工程验收，未包含真实数据效果。真实执行结果另建结果页；M2仅由M1预设门槛触发。
