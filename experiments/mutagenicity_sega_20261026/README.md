# Mutagenicity：SEGA配置下的TQG实验归档

完整结果与执行偏差见[实验报告](results/run_report.md)，[逐种子结果](results/results.csv)及[汇总](results/results_summary.json)。

采用SEGA原边数域0→1/2/3、五个模型种子。骨干为GINE，14维节点标签、3维边标签，32维、2层、mean readout、float64。源预训练按SEGA通用配置100 epoch；TQG适配200 step。SEGA包没有Mutagenicity专用训练/验证划分，源划分为本轮固定协议。

源域1084张＝975训练＋109验证；训练部分为779 fit＋196 query。比较全部196张query/全部975张源训练图的odd和odd＋even，以及冻结源模型、继续fit监督、全源监督和置信度伪标签四类基线。源预训练按源验证CE选择起点；本轮不沿用PROTEINS的700次更新预算。

## 文件导航

- [训练索引](job_index.csv)：105次适配及10次共享源预训练，共115份history.jsonl。
- [评价索引](evaluation_index.csv)：225条开发集评价，映射到对应日志与checkpoint；final标签未评价。
- [日志](logs/)、[755份checkpoint](checkpoints/)、[配置与来源记录](metadata/)。未压缩归档合计约260.1 MiB。
- [静态结果图](results/figures/balanced_accuracy.png)。

原始文件逐字节复制并校验SHA256，未重新保存模型。每作业保留配置、数据角色、环境、原代码提交；恢复的两次全源odd作业在event_flags中标为restart。源预训练模型存在68bb171/b59cbd7两个提交来源，详见报告执行记录。

## 校验

从仓库根目录运行：

```bash
python scripts/verify_experiment_archive.py experiments/mutagenicity_sega_20261026
```

图数据缓存和TensorBoard事件文件未随此归档上传；history.jsonl保留逐步标量记录。内部绝对路径保留来源含义，evaluation_index提供仓库内路径。完整编排仍依赖原运行目录结构和匹配的数据缓存，不能声称clone即可直接重放全部实验。
