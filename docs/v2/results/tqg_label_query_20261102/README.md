# H-only标签／预测归因与D7-H查询数量验证

三个Mutagenicity迁移任务、五模型种子。标签归因沿用0/50/200快照共360块、3420独立更新；D7-H只用初始模型120块，6000逻辑配置／4920物理解，3000逻辑单步／2460实际更新。无新训练、无final标签。

**标签稳定增量与“数量是主要瓶颈”两项总体门槛均未通过。** 0→3的BASE-M不使用query逐图标签对应，平均单步效果高于FULL，有4/5种子支持，但并非新盲测或长期训练结果。query增加在NORMALIZED正则下改善两任务，FIXED下没有任务同时通过方向／误差门槛。

- [白话结论及详细表格](结果解读.md)
- [完整执行报告](run_report.md)
- [汇总JSON](results_summary.json)／[模型种子指标](results.csv)
- [标签方向种子表](label_direction_seed_summary.csv)／[标签单步种子表](label_step_seed_summary.csv)
- [query数量种子表](query_size_seed_summary.csv)／[两正则曲线](query_curve_summary.csv)

[原始记录、代码与校验](../../../../experiments/tqg_label_query_20261102/README.md)
