# Mutagenicity v2：目标batch16／32／64

**结果：反馈幅度波动减小、目标覆盖增加，但分类性能没有跨任务稳定改善，计算开销明显增加。** 见[结果解读](结果解读.md)、[完整执行报告](run_report.md)和[逐种子结果](results.csv)。

只改变目标batch。源fit每步16、query全部196、相同五个fit源模型、200步、SGD lr0.0003、反馈权重0.1、lambda_W0.01和FP64均保持不变。源预训练沿用100 epoch、batch32、2层hidden32 GINE。源域1084张＝779 fit＋196 query＋109源验证，目标域1/2/3为1084/1084/1085张；四域是SEGA提供的边数划分，不是Mutagenicity官方域。

batch16与继续源SGD各15个作业复用自上一轮；batch32和64各新跑15个作业。复用前验证配置、源模型哈希、数据角色及14个数值代码文件。五个种子为3405–3409，三个迁移方向均保留，使用同一源采样序列和目标随机排列前缀。

- [原始指标CSV](results.csv)、[均值/样本标准差及配对结果](results_summary.json)。
- [逐作业反馈诊断](training_diagnostics.json)、[逐任务诊断汇总](diagnostics_summary.csv)。CV只描述幅度波动，未测量梯度方向夹角。
- [独立审计](independent_audit.json)：配置差异、3000组配对步骤、模型来源、汇总复算、历史指标复现和标签访问边界。
- [冻结计划](experiment_plan.yaml)、[完成记录](completion_audit.json)、[交接说明](hand_off.md)。
- [准确率图](figures/target_batch_accuracy.png)、[反馈范数CV图](figures/feedback_norm_cv.png)、[每步计时图](figures/step_seconds_median.png)，均另有PDF版本。
- [完整日志、模型与执行快照](../../../experiments/mutagenicity_target_batch_20261028/README.md)。

主结果固定200步，次结果按源验证CE选点。所有新训练完成并封存后才评价开发集，final标签未读。开发集曾被历史实验观察，本轮为探索性敏感性分析；固定步数并非等计算预算，也不能把数量、有效方向空间与核的共同变化单独解释为采样噪声变化。

执行快照commit为`587072a4371e16da22cf5b6e9fc4db8a02dd90ab`，属于原运行独立Git仓库，完整受版本控制文件在归档`code_snapshot/`。原始运行目录为`/home/hxchen/runs/2026-10-28-tqg-v2-target-batch-ablation/`。`results.csv`为复用记录保留原训练commit，并另列本轮`evaluation_commit`。
