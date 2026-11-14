# R5 原始方法与请求价值

状态：已完成。三设置均通过 R5_INPUT / R5_OBS / R5_FINAL 封存及独立统计复核。研究问题：原始 TSBA 在 Mutagenicity 两种划分及 PROTEINS 上，是否具有整体分类收益及定向 score 的额外价值？

[完整报告](report.md) · [档案索引](results/archive-index.json) · [启动记录](results/startup.md) · [完整协议](docs/r5-protocol.md) · [Mutagenicity 计划](configs/mutagenicity-plan.json) · [PROTEINS 计划](configs/proteins-plan.json) · [返回 Mutagenicity](../README.md)

## 结果摘要

| 设置 | A0−B0 Accuracy（百分点） | A0−B1 Accuracy（百分点） |
|---|---:|---:|
| Mutagenicity 边数（历史复用） | +0.34283 | −0.01844 |
| Mutagenicity 节点数（新增） | +0.22594 | −0.00154 |
| PROTEINS 边数（新增） | −0.16739 | +0.01199 |

原始 TSBA 未建立跨设置稳定收益；学习 score 没有稳定超越零请求。三设置 A0 的目标 NLL 均比 B0/B1 更差。详见[完整报告](report.md)中的配对 SD、五种子结果与旁路证据。

360 条轨迹、10,800 次尝试、11,160 个状态预测全部完成并核验；其中 240 条轨迹为新增，120 条为历史复用。旁路共 348/360 块可用，PROTEINS 的 12 块无有效边界保留缺失和分母，没有删除任务或补延长预算。

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

运行器已实现。两份计划 JSON 保留原登记信息；它们的 `planned/not_implemented/training_started=false` 是计划登记时的状态，不作为实际执行记录。运行时生成 `protocol/requested-plan.json`、解析配置 `protocol/plan.json`、Git/环境/命令来源、输入封存、checkpoint 与最终评价；实际进度另记在 `results/`。

已验证的 Python 环境为 `/home/hxchen/anaconda3/envs/bfn4jet/bin/python`。从仓库根目录执行：

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m experiments.mutagenicity.r5_原始方法与请求价值_20261114.scripts.run --plan experiments/mutagenicity/r5_原始方法与请求价值_20261114/configs/mutagenicity-plan.json --setting mutagenicity_edge --root /home/hxchen/research/GDA/TSBA/experiments/r5/mutagenicity_edge/attempt-001
python -m experiments.mutagenicity.r5_原始方法与请求价值_20261114.scripts.run --plan experiments/mutagenicity/r5_原始方法与请求价值_20261114/configs/mutagenicity-plan.json --setting mutagenicity_node --root /home/hxchen/research/GDA/TSBA/experiments/r5/mutagenicity_node/attempt-001
python -m experiments.mutagenicity.r5_原始方法与请求价值_20261114.scripts.run --plan experiments/mutagenicity/r5_原始方法与请求价值_20261114/configs/proteins-plan.json --setting proteins_edge --root /home/hxchen/research/GDA/TSBA/experiments/r5/proteins_edge/attempt-001
```

将 `python` 指向上述环境。`--stage prepare` 只登记输入，`run` 运行到评价，`replay` 只重载核验并封存，`evaluate` 必须已有本设置的 `R5_OBS`。`--help` 不读真实类别。正式运行要求已提交的干净工作区；每个设置最多 4 个 worker、每 worker 1 线程。工程中断可原目录续跑，数值异常留下 terminal failure 并阻止重入，修复后必须另开 attempt。

## 结果字段

每设置 `evaluation/` 输出主轨迹、旁路、源/score 曲线、图表与 `report.md`。`paired_seeds.csv` 是五个 seed 的原始等权均值；`side_seeds.csv` / `side_summary.csv` 同时列覆盖与缺失。缺失方向不冒充完整 12 方向均值，`available_mean` 仅为注明分母的可用子集描述。Accuracy/BA/F1 原始列为 0–1，`*_pp` 为百分点。旁路 `nll_contribution` 是该固定子集对全目标 ΔNLL 的加权贡献，正确/错误两组可加和。

历史 edge 的原文件保持不变，新目录保存独立字节副本、来源与 SHA256；补算的请求分解/几何字段注明历史状态新增分析。

## 实现依赖

复用核心 `tsba/`，以及 [R3](../r3_源风险与置信度约束_20261112/README.md) 的源角色、同状态旁路与数值诊断、[R4](../r4_全方向迁移对照_20261112/README.md) 的训练与证据组织。本轮实现负责三个设置的输入适配与运行，不改写旧 R3/R4 证据。

## 完成后的核验与发布

所有设置最终封存后可独立复核并复制原字节结果：

```sh
python -m experiments.mutagenicity.r5_原始方法与请求价值_20261114.scripts.archive --archive-root /home/hxchen/research/GDA/TSBA/experiments/r5
```

该入口只读取封存结果，复核哈希、方向等权配对、精确平局、纠错计数和 NLL 分组守恒，不训练或调整模型。结论正文保存在 `results/interpretation.md`，生成总报告时一并引用。实际配置见 [executed-configs.json](configs/executed-configs.json)，最终日志与封存清单见 `results/final-logs/`、`results/final-seals/`。启动记录是保留的历史快照，不代表当前仍在运行。
