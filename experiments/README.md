# TSBA 实验

实验按数据类型和真实数据集归档。先看各分类 README，再进入每轮报告、协议与证据。

| 分类 | 阅读入口 | 内容 |
|---|---|---|
| 合成数据 | [synthetic](synthetic/README.md) | E1/E2、控制实验、源后验、静态 score 与 P0 训练预算 |
| Mutagenicity | [mutagenicity](mutagenicity/README.md) | R1–R4 真实图实验；R4 含原始 TSBA 补充对照 |

## 最新结果

[Mutagenicity R4（含 A0 补跑）](mutagenicity/r4-20261112/report.md)：原始 TSBA 的平均 Accuracy 为 70.001%，源模型为 69.658%；零请求对照为 70.020%，尚未观察到 score 的平均额外收益。M11 为 69.701%。这些是完整目标域上的开发评价，不是独立最终测试。

[P0 合成数据报告](synthetic/p0-20261110/report.md)：增加训练预算后，二维固定查询上的方向一致率达到 99.980%；这不是图分类 Accuracy，也不直接证明真实图适配有效。

## 实验状态索引

所有下列研究轮次均已完成；目录整理不新增研究结果。每个入口包含计划、实际配置、来源和报告。

| 实验 | 状态 | 研究问题 | 结论与后续关系 |
|---|---|---|---|
| [E1/E2](synthetic/e1-e2-20261109/README.md) | 已完成 | score 与边界适配能否工作 | 存在高维反例；后续控制拆解原因 |
| [控制实验](synthetic/e2-controls-20261109/README.md) | 已完成 | 目标请求的额外贡献 | 正确请求的小增量不等于整体优于初始化 |
| [全源 CE](synthetic/e2-full-source-20261110/README.md) | 已完成 | 小批次噪声是否主因 | 全源更新仍可出现目标下降 |
| [回放](synthetic/e2-replay-20261110/README.md) | 已完成 | 源泛化、置信度与边界变化 | 经验源 CE 改善不保证总体风险改善 |
| [源后验](synthetic/e2-source-posterior-20261110/README.md) | 已完成 | 标签抽样噪声的作用 | 消除噪声后总体 score 收益仍小 |
| [静态 score](synthetic/e2-static-score-20261110/README.md) | 已完成；预算解释由 P0 更新 | 样本量与带宽 | 400 步结果不能解释为网络不可学 |
| [P0](synthetic/p0-20261110/README.md) | 已完成 | 是否训练不足 | 二维长预算可学到方向；不等于真实图适配有效 |
| [R1](mutagenicity/r1-20261111/README.md) | 已完成 | 表示与边界支持量 | 局部支持不足；不能据此确认低密度分隔 |
| [R1b](mutagenicity/r1b-20261111/README.md) | 已完成 | 参考头、密度尺度、静态 score | 多项诊断与单步结果见报告，不能当作闭环有效性证明 |
| [R2](mutagenicity/r2-20261111/README.md) | 已完成；旧数据口径 | 训练预算、λ、30 次适配 | 后续由 R3 检查约束与源拟合问题；不可跨数据池直接比分 |
| [R3](mutagenicity/r3-20261112/README.md) | 已完成 | 约束与参数敏感性 | 固定配置扩展到 R4；开发调参不等于独立测试 |
| [R4（含 A0）](mutagenicity/r4-20261112/README.md) | 已完成 | 固定配置全方向比较 | score 未表现出平均额外收益；全部方法共用一份 R4 报告 |

## 目录约定

```text
experiments/
  README.md
  validation-plan.md        # 跨轮历史总设计
  templates/                # 新实验方案与报告模板
  synthetic/<轮次-日期>/   # 每轮证据包
  mutagenicity/
    README.md
    stage1-inputs.json       # 共享输入来源
    <轮次-日期>/           # 方案、计划、实际配置、报告、汇总与档案索引
```

新增真实数据集使用数据集名称目录。方法与使用说明见 [docs](../docs/README.md)，实验规范见[记录规范](../docs/experiment-records.md)。算法和运行器继续放在 `tsba/` 与 `workflows/tsba/`。

## 存档与迁移

2026-11-14 将 `docs/tsba/experiments/` 移到仓库根目录 `experiments/`，各轮协议和计划与报告归拢；共享输入来源仍在数据集目录。运行器和测试路径同步修改。历史 JSON、CSV、日志和图表保持原始字节；其中旧路径和提交号属于执行时来源，不作改写。新增加的实际配置副本与来源记录用于补全证据入口。

大模型、数组、完整日志留在仓库外。旧 TQG 在独立分支归档，见[历史说明](../docs/archive/README.md)。目录整理不重新训练、不改变评价权限或研究结论。
