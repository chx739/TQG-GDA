# TQG v2 P0完整诊断归档

[结果白话解读](结果解读.md) · [完整执行报告](run_report.md) · [汇总JSON](results_summary.json) · [源快照与哈希](snapshot_manifest.json) · [原始D1](gradient_blocks.csv) · [原始D2](one_step_blocks.csv)

本归档只包含D0/D1/D2，不是新增的连续适配实验。最终标签未读取。protocol保存不可变原方案；code_snapshot为执行代码，execution_history.bundle可恢复实际执行Git历史。

源模型引用主仓库已有experiments/mutagenicity_target_batch196_20261028/checkpoints；未重复上传35个输入模型。geometry_cache只在原始本地run中保存，local_geometry_manifest.json给出文件SHA和大小，CSV中的geometry_cache路径相对原始source_run。公开数据包括完整指标与代码，不包含原始数据集或高维梯度缓存。
