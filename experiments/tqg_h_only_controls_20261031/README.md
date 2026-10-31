# H-only D1/D2：完整对照记录

Mutagenicity 0→1/2/3、五个模型种子、360个方向块、180个单步块及6480次独立临时更新。仅H核，JOINT λ=0.01，其余遵守原D1/D2冻结门槛。无新预训练／连续适配，无final标签。

**D1方向门槛0/3未通过；D2单步门槛2/3通过。** 0→2对乱标签仅2/5种子胜出，0→3为3/5，不能把总体平均收益直接称为稳定监督迁移。暂不进入完整训练。

- [白话结果与稳定性](结果解读.md)
- [完整执行报告](run_report.md)
- [主汇总](results_summary.json)和[种子级指标](results.csv)
- [方向种子表](gradient_seed_summary.csv)、[单步种子表](one_step_seed_summary.csv)

`logs/`保存每种子原始JSONL、TensorBoard、样本ID及标签访问记录；`code_snapshot/`与`execution_history.bundle`保存冻结执行代码；`protocol/`为原方案字节副本，`candidate_protocol.md`为候选修订；`analysis_code/`为后处理稳定性解释；`publication_code/`为归档审计代码。`manifest.json`列出发布文件SHA，`artifact_audit.json`保存独立数值与角色审计。

模型checkpoint沿用已有归档，不重复上传。数据及高维geometry_cache不上传；`input_cache_manifest.json`／`local_geometry_manifest.json`提供本机引用及SHA。H权重及梯度指标逐块与D4 H臂一致。
