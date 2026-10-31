# D8已知规律与真实残差分解

五个生成／模型配对种子；480个合成图块、120个线性块；真实Mutagenicity三个任务共360块×H/HA两读取器。新增五个合成源warmup模型，1000个Adam更新；无真实预训练或连续适配，无真实final标签。

**S1友好条件下三个预定读取器均通过方向门槛；真实H-only的局部单步收益主要来自逐图残差，但逐图方向门槛仍未过。** 线性组表明抽样源标签噪声可显著放大；S2即便给真实源条件概率仍失效，不能把所有问题归因于样本少。S4低覆盖不作为成功。

- [白话结论与完整表格](结果解读.md)
- [执行、收敛和后处理记录](run_report.md)
- [主汇总](results_summary.json)、[种子指标](results.csv)
- [合成分种子](synthetic_seed_summary.csv)、[合成分快照](synthetic_snapshot_seed_summary.csv)
- [真实逐图方向](real_direction_seed_summary.csv)、[真实逐图单步](real_step_seed_summary.csv)
- [误差分解描述](error_decomposition_descriptive.json)

原始JSONL、TensorBoard、生成器隐变量和真相访问日志在logs/；15份正式合成checkpoint在checkpoints/3405—3409，pilot_3405仅为未训练0步快照，不计新增完整模型。synthetic_checkpoint_manifest.json提供快照SHA与评价链接。真实checkpoint沿用旧归档，不重复上传；真实数据和高维几何缓存不上传。

冻结执行代码在code_snapshot/，执行提交81d0703；execution_history.bundle包含其版本历史。analysis_code/保留真实使用的汇总和后处理代码及SHA；publication_code/为独立归档审计。protocol/为旧方案原样副本，execution_protocol.md为执行前细节。manifest.json列出所有发布文件SHA。

原始后处理审计错误与重试日志均保留。图中S4余弦仅在可定义部分平均，覆盖率显示在图上；不能把有效子集当全体。条件梯度Oracle不是可部署UDA方法，所有临时步均从原θ独立开始。
