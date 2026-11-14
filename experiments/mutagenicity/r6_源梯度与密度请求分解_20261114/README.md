# R6 源梯度与密度请求分解

状态：计划中。研究问题：R5 的泛化代价来自源梯度推动、密度请求推动，还是两者在有限步长下的交互？

[实验协议](docs/r6-protocol.md) · [计划配置](configs/r6-plan.json) · [R5 结果](../r5_原始方法与请求价值_20261114/report.md) · [返回 Mutagenicity](../README.md)

本轮统一包含 Mutagenicity 边数、Mutagenicity 节点数和 PROTEINS 边数三个设置，全部 12 个方向、种子 3405–3409。复用 R5 A0 第 1、30 次尝试前的状态，构造“源梯度开／关 × 密度请求正常／零／反向”六组离线候选；保留同一矩阵、边界和批次。

不重新训练源模型或 score，不提交适配更新。主分析为初始状态、R5 旁路原步长；末轮前状态与半步长、四分之一步长为预先固定的辅助分析。目标标签和目标梯度仅用于候选封存后的评价。该实验是基于已观察 R5 结果的事后机制诊断，不是新的独立最终测试。

当前仅完成协议与计划配置，运行器尚未实现，尚未启动实验。未来实现放本轮 `src/`，入口放 `scripts/`；实际运行后再生成执行配置、`results/` 与 `report.md`。

拟议入口如下，**尚不可执行**，实现后须核验并更新：

```sh
python -m experiments.mutagenicity.r6_源梯度与密度请求分解_20261114.scripts.run --plan experiments/mutagenicity/r6_源梯度与密度请求分解_20261114/configs/r6-plan.json --stage prepare
```

后续阶段拟为 `candidates`、`replay`、`evaluate`、`report`。依赖核心 `tsba/` 与 R5 的输入恢复、状态索引、functional 诊断和评价逻辑；不修改 R3–R5 的历史证据，不将本轮流程移入 `experiments/common/`。
