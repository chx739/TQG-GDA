# R5 原始方法与请求价值

状态：计划中。研究问题：原始 TSBA 在 Mutagenicity 两种划分及 PROTEINS 上，是否具有整体分类收益及定向 score 的额外价值？

[完整协议](docs/r5-protocol.md) · [Mutagenicity 计划](configs/mutagenicity-plan.json) · [PROTEINS 计划](configs/proteins-plan.json) · [返回 Mutagenicity](../README.md)

## 范围

本轮所有设置统一归档于本目录，PROTEINS 作为同轮跨数据集验证设置。

| 设置 | 安排 |
|---|---|
| mutagenicity_edge | 复用 R4 的 B0/B1、A0 及 A0 第 1、30 次尝试前的同状态旁路 |
| mutagenicity_node | 新训练 20 个源模型、60 个 score；运行 B1/A0 各 60 条轨迹及 A0 旁路 |
| proteins_edge | 新训练 20 个源模型、60 个 score；运行 B1/A0 各 60 条轨迹及 A0 旁路 |

每个设置均为 12 个有向迁移任务、种子 3405–3409、每轨迹 30 次尝试。完整目标域作为传导式开发评价池，历史结果与新结果分别标记。

PROTEINS 四域为 278/278/278/279 张图，使用 3 维节点类别 one-hot，不使用连续节点属性，无边标签时 `edge_attr=None`。score 固定 7200 次更新，Mutagenicity 对应 800 epochs，PROTEINS 对应 2400 epochs。完整参数、标签规则和评价方法见协议。

## 状态与运行接口

当前只有协议和计划配置，未实现运行器，未开始训练。实际配置、执行来源、结果和报告仅在真实运行后生成。

统一入口拟为 `python -m experiments.mutagenicity.r5_原始方法与请求价值_20261114.scripts.run`，通过 `--plan` 和 `--setting` 选择设置。参数与调用模板见协议第 10 节；该命令目前不可执行。实现时在本目录增加 `src/`、`scripts/` 和 `results/`。

## 实现依赖

复用核心 `tsba/`，以及 [R3](../r3_源风险与置信度约束_20261112/README.md) 的源角色、同状态旁路与数值诊断、[R4](../r4_全方向迁移对照_20261112/README.md) 的训练与证据组织。本轮实现负责三个设置的输入适配与运行，不改写旧 R3/R4 证据。
