# R6 源梯度与密度请求分解

状态：实现与独立代码审查已完成，等待执行启动记录。研究问题：R5 的泛化代价来自源梯度推动、密度请求推动，还是两者在有限步长下的交互？

[实验协议](docs/r6-protocol.md) · [计划配置](configs/r6-plan.json) · [R5 结果](../r5_原始方法与请求价值_20261114/report.md) · [返回 Mutagenicity](../README.md)

本轮统一包含 Mutagenicity 边数、Mutagenicity 节点数和 PROTEINS 边数三个设置，全部 12 个方向、种子 3405–3409。复用 R5 A0 第 1、30 次尝试前的状态，构造“源梯度开／关 × 密度请求正常／零／反向”六组离线候选；保留同一矩阵、边界和批次。

不重新训练源模型或 score，不提交适配更新。主分析为初始状态、R5 旁路原步长；末轮前状态与半步长、四分之一步长为预先固定的辅助分析。目标标签和目标梯度仅用于候选封存后的评价。该实验是基于已观察 R5 结果的事后机制诊断，不是新的独立最终测试。

实现位于本轮 `src/`，入口位于 `scripts/run.py`。计划配置保留原始登记内容；实际配置、代码版本、输入哈希和命令随运行另存，不将计划预算写成已完成数量。

从仓库根目录启动全部三个设置：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r6_源梯度与密度请求分解_20261114.scripts.run --plan experiments/mutagenicity/r6_源梯度与密度请求分解_20261114/configs/r6-plan.json --stage all
```

要求先提交已经 review 的代码，运行器会检查干净工作树。`--setting mutagenicity_edge|mutagenicity_node|proteins_edge` 可单独执行一个完整设置。默认输出使用计划指定的 `r6/<setting>/attempt-001`；`--archive-root` 可指定新 attempt 的父目录。恢复同一运行使用原命令，代码、输入或配置不匹配会拒绝恢复，不覆盖旧证据。

阶段依次为 `prepare`、`candidates`、`replay`、`evaluate`、`report`，`all` 顺序执行。先准备全部设置，再按设置执行，最多四个 worker，每个线程数一。候选生成只读取源训练标签；重载核验后封存 R6_OBS，评价阶段才读取源验证与目标标签。报告和完整 CSV 在各设置档案的 `evaluation/` 中，完成时建立 R6_FINAL。

核心核验：旧预测和线性系统分解、六组固定候选、功能调用不改变状态、有限越界保留、候选封存门控、方向导数与有限差分、分组贡献守恒，以及固定方向/种子分母。合成工程测试位于 `tests/tsba/test_r6.py`，不能替代真实数据结果。

[独立审查与工程检查记录](results/engineering/review.md)。原协议中的“计划中”描述保留为登记历史；运行状态以本页及实际执行记录为准。
