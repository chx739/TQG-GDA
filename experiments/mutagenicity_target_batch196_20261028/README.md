# Mutagenicity：目标batch196追加对照

**结论：196相对64未带来分类收益，反馈幅度CV更小、计算更慢；相对16未满足预设支持条件。**

新增15次B196训练；60次适配对照及5次源预训练模型复用，保留各自原训练提交。0→1/2/3各五种子，源fit每步16、query全部196、相同预训练起点、适配200步。

[结果解读](results/结果解读.md) · [完整报告](results/run_report.md) · [评价与模型对应](evaluation_index.csv) · [作业索引](job_index.csv)

本归档包含80份JSONL日志、460份checkpoint、165条评价记录；来源文件共110,010,230字节（约104.9 MiB）。

`manifest.json`与`SHA256SUMS`记录原始字节、来源路径及哈希。复用结果同时核对训练代码、配置、样本角色、模型起点与旧指标。没有在归档时训练或读取标签。

目标域图数为1084/1084/1085，196是每步目标抽样数，不是全量目标图。域来自SEGA图规模划分。目标标签只在全部训练封存后的开发评价阶段读取；final标签未读。数据缓存与TensorBoard二进制事件留在本机，JSONL及执行代码快照保留。

校验：`python scripts/verify_experiment_archive.py experiments/mutagenicity_target_batch196_20261028`。模型文件均小于1 MiB，不需要Git LFS。
