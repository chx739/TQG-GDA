# TSBA 实验

实验按数据类型和真实数据集归档。先看各分类 README，再进入每轮报告、协议与证据。

| 分类 | 阅读入口 | 内容 |
|---|---|---|
| 合成数据 | [合成实验](synthetic_data/README.md) | E1/E2、控制实验、源后验、静态 score 与 P0 训练预算 |
| Mutagenicity | [mutagenicity](mutagenicity/README.md) | R1–R4 结果与 R5 计划；R5 包含 PROTEINS 验证设置 |

## 当前计划

[R5｜原始 TSBA 在不同数据设置上是否有整体收益与请求价值？](mutagenicity/r5_原始方法与请求价值_20261114/README.md)：复用 Mutagenicity 边数划分的 R4，新增 Mutagenicity 节点数划分和 PROTEINS 边数划分；各设置 12 方向、5 种子。统一 B0/B1/A0 与 A0 首尾同状态正常/零/反向旁路。状态为计划中，当前仅有协议和计划配置，尚未实现运行器或启动训练。

## 最新结果

[Mutagenicity R4（含 A0 补跑）](mutagenicity/r4_全方向迁移对照_20261112/report.md)：原始 TSBA 的平均 Accuracy 为 70.001%，源模型为 69.658%；零请求对照为 70.020%，尚未观察到 score 的平均额外收益。M11 为 69.701%。这些是完整目标域上的开发评价，不是独立最终测试。

[P0 合成数据报告](synthetic_data/p0_分数训练预算_20261110/report.md)：增加训练预算后，二维固定查询上的方向一致率达到 99.980%；这不是图分类 Accuracy，也不直接证明真实图适配有效。

## 实验状态索引

下列历史研究轮次均已完成；R5 计划见上节。历史入口包含计划、实际配置、来源和报告，计划中实验不生成虚构的执行记录。

| 编号与研究问题 | 状态 | 结论与后续关系 |
|---|---|---|
| [E1/E2｜符号、几何与适配能否按设计运行？](synthetic_data/e1_e2_符号几何与适配_20261109/README.md) | 已完成 | 存在高维反例；后续控制拆解原因 |
| [E2｜目标请求能否带来额外收益？](synthetic_data/e2_目标请求对照_20261109/README.md) | 已完成 | 正确请求的小增量不等于整体优于初始化 |
| [E2｜小批次噪声是否解释适配失败？](synthetic_data/e2_全源批次对照_20261110/README.md) | 已完成 | 全源更新仍可出现目标下降 |
| [E2｜适配如何影响源泛化、目标置信度和边界？](synthetic_data/e2_适配轨迹回放_20261110/README.md) | 已完成 | 经验源 CE 改善不保证总体风险改善 |
| [E2｜源标签抽样噪声如何影响结论？](synthetic_data/e2_解析源后验对照_20261110/README.md) | 已完成 | 消除噪声后总体 score 收益仍小 |
| [E2｜样本量与带宽如何影响 score？](synthetic_data/e2_静态分数估计_20261110/README.md) | 已完成；预算解释由 P0 更新 | 400 步结果不能解释为网络不可学 |
| [P0｜训练不足能否解释 score 方向误差？](synthetic_data/p0_分数训练预算_20261110/README.md) | 已完成 | 二维长预算可学到方向；不等于真实图适配有效 |
| [R1｜冻结图表示是否保留类信息和边界支持？](mutagenicity/r1_图表示与边界支持_20261111/README.md) | 已完成 | 局部支持不足；不能据此确认低密度分隔 |
| [R1b｜密度尺度与单步适配诊断揭示了什么？](mutagenicity/r1b_密度与单步适配诊断_20261111/README.md) | 已完成 | 多项诊断与单步结果见报告，不能当作闭环有效性证明 |
| [R2｜训练预算与多轮适配如何影响 score？](mutagenicity/r2_分数训练预算与多轮适配_20261111/README.md) | 已完成；旧数据口径 | 后续由 R3 检查约束与源拟合问题；不可跨数据池直接比分 |
| [R3｜源风险与置信度约束如何影响适配？](mutagenicity/r3_源风险与置信度约束_20261112/README.md) | 已完成 | 固定配置扩展到 R4；开发调参不等于独立测试 |
| [R4｜固定配置在各迁移方向上是否带来收益？](mutagenicity/r4_全方向迁移对照_20261112/README.md) | 已完成 | score 未表现出平均额外收益；全部方法共用一份 R4 报告 |

## 实验项目结构

```text
experiments/
  common/                  # 纯通用工具
  synthetic_data/<experiment_id>/
  mutagenicity/<experiment_id>/
    README.md              # 问题、结论、运行方式、依赖
    src/                   # 该轮实现
    scripts/               # 该轮启动与分析入口
    configs/               # 计划与实际配置
    docs/                  # 方案及补充说明
    results/               # 汇总、图表、日志、核验
    report.md              # 完整结论
scripts/                   # 通用 TSBA 入口与仓库工具
```

实验目录采用「编号_中文主题_日期」，索引采用「编号｜研究问题」。新实验可参考[模板](templates/README.md)。

新增文件前先阅读本页、所属数据集 README 和模板，确定所属轮次，再按以下规则落位：

- **方案属于轮次目录。** 协议与补充说明放在该轮 `docs/`，计划与实际配置放在 `configs/`。即使尚未实现代码、仅有实验计划，也使用这一结构；不要在 `experiments/` 或数据集根目录单独新建轮次协议。
- **跨数据集仍可属于同一轮。** 沿用已确定的主数据集目录，将同轮所有数据集的协议、配置和结果统一归档，并在轮次 README 中列明范围。不要仅因增加验证数据集而另建并列轮次或把共享协议提到根目录。
- **历史总设计不是新文件的例外。** 根目录中已有的跨轮历史总设计可以保留；这一安排不适用于新轮次协议，也不因协议涉及多个数据集而扩大适用范围。
- **先归档，后更新导航。** 总索引链接数据集或轮次入口，各轮 README 链接自己的方案与配置。检查时同时核验目录归属与链接有效性，不能修改目录规则来迁就错误位置。

例如，R5 的 Mutagenicity 两种划分与 PROTEINS 验证统一放在 Mutagenicity 目录：

```text
experiments/mutagenicity/r5_原始方法与请求价值_20261114/
  README.md
  docs/r5-protocol.md
  configs/mutagenicity-plan.json
  configs/proteins-plan.json
```

该示例为计划阶段；实现与运行后再增加本轮 `src/`、`scripts/`、`results/` 和 `report.md`，不提前生成虚构的执行记录。

`common` 不按轮次命名，不保存训练或适配流程。R2/R3 等特定协议的代码归本轮 `src/`；后续实验沿用时明确列出依赖。核心算法仍位于 `tsba/`。

## 运行代码

从仓库根目录安装并调用模块入口：

```sh
python -m pip install -e '.[experiments]'
python -m experiments.mutagenicity.r4_全方向迁移对照_20261112.scripts.run --help
python -m experiments.synthetic_data.p0_分数训练预算_20261110.scripts.run_p0 --help
```

各轮 README 列出具体入口。命令参数与标签使用规则仍以该轮协议为准；目录整理不授权重跑真实实验。研究入口面向仓库 checkout，使用 editable 安装；历史数据、封存配置和外部档案不会随 Python 包安装自动提供。
