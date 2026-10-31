# H-only D1/D2：完整对照记录

Mutagenicity 0→1/2/3、五个模型种子、360个方向块、180个单步块及6480次独立临时更新。仅H核，JOINT λ=0.01，其余遵守原D1/D2冻结门槛。无新预训练／连续适配，无final标签。

**D1方向门槛0/3未通过；D2单步门槛2/3通过。** 0→2对乱标签仅2/5种子胜出，0→3为3/5，不能把总体平均收益直接称为稳定监督迁移。暂不进入完整训练。

- [白话结果与稳定性](结果解读.md)
- [完整执行报告](run_report.md)
- [主汇总](results_summary.json)和[种子级指标](results.csv)
- [方向种子表](gradient_seed_summary.csv)、[单步种子表](one_step_seed_summary.csv)

[原始记录、代码及哈希归档](../../../experiments/tqg_h_only_controls_20261031/README.md)
