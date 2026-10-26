# Mutagenicity v2 实验归档

2026-10-26完成的三个迁移方向（0→1/2/3）、五个模型种子（3405–3409）、七种适配方法，共105次适配。固定200步和源验证选点分开报告，15个冻结源模型评价引用共享源模型。**v2当前未获得跨任务稳定收益。** 见[完整报告](results/run_report.md)和[白话解读](results/结果解读.md)。

源域1,084张＝975训练＋109验证；975＝779 fit＋196 query。每步使用全部196张独立query。目标开发集每域542张；final标签未读取。骨干、源模型和域索引沿用[上一轮Mutagenicity实验](../mutagenicity_sega_20261026/README.md)。本轮改用无动量SGD，并重新运行所有适配对照，不能把旧Adam表格当作相同训练条件的对照。

| 归档内容 | 数量 |
|---|---:|
| 适配训练日志，每份200步 | 105份 |
| 复用源预训练日志 | 10份 |
| 适配模型，step50/100/150/200、last、selected | 630份 |
| 复用源模型，best、last | 20份 |
| 开发集评价记录 | 225条 |

原始文件合计146,516,218字节（约140 MiB，不含归档清单和索引）。单个模型小于1 MiB，无需Git LFS。TensorBoard事件文件及图数据缓存未收录；逐步指标完整保留于`history.jsonl`。本轮没有新增源预训练，归档操作没有训练或重新评价模型。

- [训练索引](job_index.csv)：定位任务、种子、方法、原训练提交、日志和模型。
- [评价索引](evaluation_index.csv)：从每条指标定位实际checkpoint和历史日志，含SHA256。
- [逐种子结果](results/results.csv)、[汇总](results/results_summary.json)、[图表](results/figures/balanced_accuracy.png)。
- [日志](logs/)、[模型](checkpoints/)、[配置及审计记录](metadata/)、[冻结协议及样本ID](protocol/)。
- [执行代码快照](code_snapshot/)：原运行独立仓库的提交`7280cdd48ed1c3ba306c9661162f3411d53f03ba`中的全部受版本控制文件。该提交不属于主仓库历史；[执行清单](metadata/stage/experiment_manifest.json)记录数值代码哈希及原路径。
- [归档清单](manifest.json)、[SHA256SUMS](SHA256SUMS)：每份原始文件的来源、大小与哈希。

源预训练记录来自上一轮归档，并保留各自真实训练提交。原始绝对路径作为来源信息保留，`archived_checkpoint`等字段给出仓库内路径。数据缓存需自行提供；执行控制器保留原运行目录结构，因此代码快照不等于无需配置即可直接重跑的入口。主仓库的[v2实现说明](../../docs/TQG_V2_IMPLEMENTATION.md)提供可用的训练命令。

下载后在仓库根目录校验，无需PyTorch：

```bash
python scripts/verify_experiment_archive.py experiments/mutagenicity_v2_20261026
```

导出脚本为`scripts/archive_v2.py`，要求原运行已完成、代码快照干净、训练封存哈希一致；目标目录必须不存在。独立实验审计与代码测试见[结果文档](../../docs/experiments/mutagenicity_v2_20261026/README.md)。
