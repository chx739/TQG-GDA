# D8已知规律与真实残差分解

五个生成／模型配对种子；480个合成图块、120个线性块；真实Mutagenicity三个任务共360块×H/HA两读取器。新增五个合成源warmup模型，1000个Adam更新；无真实预训练或连续适配，无真实final标签。

**S1友好条件下三个预定读取器均通过方向门槛；真实H-only的局部单步收益主要来自逐图残差，但逐图方向门槛仍未过。** 线性组表明抽样源标签噪声可显著放大；S2即便给真实源条件概率仍失效，不能把所有问题归因于样本少。S4低覆盖不作为成功。

- [白话结论与完整表格](结果解读.md)
- [执行、收敛和后处理记录](run_report.md)
- [主汇总](results_summary.json)、[种子指标](results.csv)
- [合成分种子](synthetic_seed_summary.csv)、[合成分快照](synthetic_snapshot_seed_summary.csv)
- [真实逐图方向](real_direction_seed_summary.csv)、[真实逐图单步](real_step_seed_summary.csv)
- [误差分解描述](error_decomposition_descriptive.json)

[原始日志、模型和代码归档](../../../experiments/tqg_d8_residual_20261031/README.md)
