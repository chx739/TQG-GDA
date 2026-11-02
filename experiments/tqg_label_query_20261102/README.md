# H-only标签／预测归因与D7-H查询数量验证

三个Mutagenicity迁移任务、五模型种子。标签归因沿用0/50/200快照共360块、3420独立更新；D7-H只用初始模型120块，6000逻辑配置／4920物理解，3000逻辑单步／2460实际更新。无新训练、无final标签。

**标签稳定增量与“数量是主要瓶颈”两项总体门槛均未通过。** 0→3的BASE-M不使用query逐图标签对应，平均单步效果高于FULL，有4/5种子支持，但并非新盲测或长期训练结果。query增加在NORMALIZED正则下改善两任务，FIXED下没有任务同时通过方向／误差门槛。

- [白话结论及详细表格](结果解读.md)
- [完整执行报告](run_report.md)
- [汇总JSON](results_summary.json)／[模型种子指标](results.csv)
- [标签方向种子表](label_direction_seed_summary.csv)／[标签单步种子表](label_step_seed_summary.csv)
- [query数量种子表](query_size_seed_summary.csv)／[两正则曲线](query_curve_summary.csv)

logs/包含逐种子JSONL、TensorBoard、全部样本ID和readout seals／Oracle访问记录；D0/含23单测、50项代数／权限检查及8个复制恒等式案例。code_snapshot/和execution_history.bundle保留冻结执行代码（0ff5627）；analysis_code/包含只读后处理与源核谱计算，publication_code/为独立归档审计。

protocol/为2026-10-31旧方案的原样副本；execution_protocol.md登记新标签归因判据和H-only的D7变体。原HA版D7、D7-C训练复用审计均未执行。源query角色保持196池，不混入fit。n196多逻辑配置是同一解，不代表新增监督。

模型与高维几何缓存沿用旧H/P0归档，只提供SHA和本机引用；不重复上传checkpoint或真实数据。无新训练模型。manifest.json提供文件校验和；artifact_audit.json保存角色、时间顺序、嵌套选集、原始均值与回放审计。
