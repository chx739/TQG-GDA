# 本次归档的范围

本目录归档2026-10-26完成的 PROTEINS 全量源查询实验：三个迁移方向、五个模型种子、四种基线与四种 TQG 设置。建仓期间没有启动新的真实数据训练。

- `results.csv`、`results_summary.json`、审计 JSON 和图表来自原始已完成实验。
- `run_report.md` 与 `结果解读.md` 仅将指向本地归档的 Markdown 链接改为仓库内相对链接。
- `artifact_hashes.json` 保留原始路径和原始文件哈希，描述运行时归档，而非改写链接后的文件。
- `repository_provenance.json` 同时记录源文件与仓库副本的 SHA256。
- `configs/full_query/0_to_2_3.json` 与 `configs/full_query/0_to_1.json` 分别保留两个执行阶段的原始配置。
- 原始执行代码分别对应提交 `8923cc9` 与 `3ea6b72`，均保留在仓库历史中。

## 可以独立运行的部分

核心 `tqg_gda/`、普通命令行 `scripts/run_experiment.py`、合成测试和 smoke test 可在安装依赖后使用。真实数据训练需要自行提供 TU/PyG 缓存，并指定与其内容哈希匹配的源划分清单。

`data/splits/PROTEINS_official_cache_source.json` 对应本轮新处理的官方原始数据缓存。这里的 official 指原始数据来源，不表示四个域是 PROTEINS 官方提供的；域索引沿用用户上传的 SEGA 代码包。默认的 `PROTEINS_idx_0_source.json` 是更早缓存的清单，不能与新缓存混用。

## 不能仅靠仓库直接重放的部分

根目录的 `full_query_experiment.py`、`experiment_revision.py` 等属于原运行目录的编排快照，依赖相邻阶段的角色清单、执行锁、源模型 checkpoint、计算账本和绝对路径。这些运行资产、数据缓存和模型权重没有上传，因而不能声称 clone 后即可逐位重现整轮实验。

`revision_common.py` 增加了归档方案目录的回退位置，使合成训练测试能读取仓库内方案；`full_query_experiment.py` 的配置读取位置随归档目录调整。这两处路径调整不补齐真实实验编排的外部运行资产，也不改变原实验的数值计算。归档方案 YAML 中的状态和授权字段是当时的历史记录；不代表当前进度，也不授权自动开始其中尚未执行的实验。

普通命令行示例用于运行基础 TQG。它与本轮全查询编排的批次调度、探测随机数和复用起点不完全相同，不能直接把新输出当成表格的逐位复现结果。
