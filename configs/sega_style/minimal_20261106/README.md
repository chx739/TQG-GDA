# 最小验证配置（尚未执行）

[实验列表与继续门槛](../../../docs/research_plans/tqg_sega_minimal_20261106/README.md) · [执行清单](../../../docs/research_plans/tqg_sega_minimal_20261106/execution.md)

15个核心任务：v1 TRUE 6个、v2 TRUE 6个、FIT-CE 3个。每个配置适配50步；所有种子采用相同源划分，v1/v2共享源初始化和批次调度。FIT-CE只训练一次，由后续开发评价器分别评价0→1/3。

`core_tasks.json`是现有`Config`可解析的15配置列表，但它不包含PERM/QUERY-CE控制、阶段门槛或开发标签评价。**不是完整实验runner，不能在M0/M1前直接批量开跑。** 控制任务及评价器按执行清单补齐后才能启动正式矩阵。

数据缓存使用占位路径`/path/to/Mutagenicity/processed/data.pt`；执行时替换为SHA匹配的路径。所有配置共用`runs/tqg_sega_minimal_20261106`下的内容身份缓存。当前生成配置不代表已进行预训练或适配。
