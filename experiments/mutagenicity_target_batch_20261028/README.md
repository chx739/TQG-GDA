# Mutagenicity：目标batch对照归档，2026-10-28

三个迁移方向0→1/2/3、五个种子3405–3409，比较TQG v2目标batch16/32/64。**反馈幅度波动降低，但分类收益很小、没有跨任务稳定改善。** 见[结果解读](results/结果解读.md)和[完整报告](results/run_report.md)。

源fit每步16、query全部196、相同fit预训练模型、200步、SGD lr0.0003、反馈权重0.1、lambda_W0.01、FP64不变。预训练沿用100 epoch和batch32。batch32/64新增30次适配；batch16与源SGD各15次复用自[上一轮v2](../mutagenicity_v2_20261026/README.md)，另复用5次源fit预训练。不重新训练源模型。

| 原始文件 | 数量 |
|---|---:|
| 新适配日志，200步/份 | 30份 |
| 复用batch16与源SGD日志 | 30份 |
| 复用源预训练日志 | 5份 |
| 适配checkpoint，step50/100/150/200、last、selected | 360份 |
| 源预训练checkpoint，best、last | 10份 |
| 冻结、末步和源验证选点评价 | 135条 |

清单中的原始文件合计88,894,752字节，约84.8 MiB，不含归档说明和索引。模型单文件小于1 MiB，不需要Git LFS。图数据缓存和TensorBoard事件文件未上传，逐步指标完整保存在JSONL中；TensorBoard原件仍在本机运行目录。

- [训练索引](job_index.csv)：65条来源记录，包含真实训练commit与复用标记。
- [评价索引](evaluation_index.csv)：每条指标对应的checkpoint、SHA256和训练日志。
- [日志](logs/)、[模型](checkpoints/)、[逐作业配置及环境](metadata/jobs/)。
- [冻结协议和样本ID](protocol/)、[训练封存和标签访问审计](metadata/stage/)。
- [执行代码快照](code_snapshot/)：独立执行仓库commit`587072a4371e16da22cf5b6e9fc4db8a02dd90ab`的全部受版本控制文件。该提交不属于主仓库历史。
- [独立审计和描述性分析脚本](metadata/run/)；[诊断汇总](results/diagnostics_summary.csv)；[图表](results/figures/)。
- [文件清单](manifest.json)和[SHA256SUMS](SHA256SUMS)。

复用B16原作业名为V2_joint，归档使用B16便于比较；原completion和RUN_INFO保留原名、路径与提交，不把复用记录记为新训练。不同档的源fit/query序列逐步一致，目标图取同一随机排列的不同长度前缀。目标标签只在开发评价阶段读取，final标签未读。

源域为779 fit＋196 query＋109源验证；各目标域开发集542张，完整无标签目标输入池分别1084/1084/1085张。域索引沿用SEGA按边数构造的划分，不是官方域定义。固定200步时，目标使用次数为3200/6400/12800，因此此比较不等计算预算。旧batch16耗时是历史参考，不能排除机器负载影响。

原绝对路径只作为来源信息，评价索引中的`archived_checkpoint`提供仓库内路径。重跑需要提供同哈希数据缓存并配置计划路径，不能声称代码快照无需环境配置即可直接重跑。

在仓库根目录核验，不需要PyTorch：

```bash
python scripts/verify_experiment_archive.py experiments/mutagenicity_target_batch_20261028
```

导出脚本为`scripts/archive_target_batch.py`，只归档已完成运行，逐字节校验并检查指标与模型链接；导出过程不训练模型、不读取图标签。
