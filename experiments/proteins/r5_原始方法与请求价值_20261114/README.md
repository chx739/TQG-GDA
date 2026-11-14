# R5 原始方法与请求价值

状态：计划中。研究问题：原始 TSBA 在 PROTEINS 边数划分的全部迁移方向上，是否具有整体收益及定向 score 的额外价值？

[完整跨数据集协议](../../r5_原始方法与请求价值_20261114.md) · [计划配置](configs/plan.json) · [文档入口](docs/README.md) · [返回 PROTEINS](../README.md)

## 范围

使用 `idx_PROTEINS_0.npy` 至 `idx_PROTEINS_3.npy` 的既有域成员和顺序，四域 278/278/278/279 张图；运行 12 个方向和种子 3405–3409。

新训练 20 个源模型、60 个 score；B0 固定基线，B1/A0 各 60 条、每条 30 次尝试。A0 在第 1、30 次尝试前生成同状态的正常、零、反向候选。score 固定 7200 次更新，对应每 epoch 三次更新、2400 epochs，不能沿用 Mutagenicity 的 800 epochs。

采用 3 维节点类别 one-hot，不使用连续节点属性；无边标签时 `edge_attr=None`。核心编码器使用 GIN。完整目标域只在观察封存后开发计分，不能作为独立最终测试或与不同输入协议的论文分数直接比较。

## 状态与运行接口

当前只有协议和计划配置，未实现运行器，未开始训练。未来入口为 `python -m experiments.proteins.r5_原始方法与请求价值_20261114.scripts.run`，模板见完整协议第 10 节；该命令目前不可执行。

实现时增加本轮 `src/`、`scripts/`、`results/`，显式调用 [Mutagenicity R5](../../mutagenicity/r5_原始方法与请求价值_20261114/README.md) 的跨设置实现，不复制算法内核。真实运行后才写实际配置、来源、结果及报告。
