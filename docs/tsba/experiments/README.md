# TSBA 实验

实验按数据类型和真实数据集归档。先看各分类 README，再进入每轮报告、协议与证据。

| 分类 | 阅读入口 | 内容 |
|---|---|---|
| 合成数据 | [synthetic](synthetic/README.md) | E1/E2、控制实验、源后验、静态 score 与 P0 训练预算 |
| Mutagenicity | [mutagenicity](mutagenicity/README.md) | R1–R4 真实图实验；R4 含原始 TSBA 补充对照 |

## 最新结果

[Mutagenicity R4（含 A0 补跑）](mutagenicity/r4-20261112/report.md)：原始 TSBA 的平均 Accuracy 为 70.001%，源模型为 69.658%；零请求对照为 70.020%，尚未观察到 score 的平均额外收益。M11 为 69.701%。这些是完整目标域上的开发评价，不是独立最终测试。

[P0 合成数据报告](synthetic/p0-20261110/report.md)：增加训练预算后，二维固定查询上的方向一致率达到 99.980%；这不是图分类 Accuracy，也不直接证明真实图适配有效。

## 目录约定

```text
experiments/
├── README.md
├── validation-plan.md       # 跨合成/真实数据的历史总设计
├── synthetic/              # 合成实验的计划、报告、图表和证据
└── mutagenicity/            # 以真实数据集命名
    ├── README.md
    ├── *-protocol.md       # 执行协议
    ├── *-plan.json         # 机器计划
    └── r*-日期/            # 每轮报告、指标和存档索引
```

新增真实数据集使用独立的数据集名称目录；同一轮的图表与指标留在该轮目录中。

[历史总设计](validation-plan.md)覆盖早期 E0–E5；实际执行范围以各轮协议为准。[方法设计](../TSBA_Graph_Level_GDA_Design.md) · [实现说明](../implementation.md) · [仓库首页](../../../README.md)

## 存档与迁移

本次整理将原根目录合成实验移入 `synthetic/`，将 `real-data/` 改为 `mutagenicity/`。报告链接、运行器和测试入口同步更新；历史 JSON、CSV、日志、输入哈希和存档索引保持原始字节，其中记载的旧路径属于执行时来源，不作改写。旧版页面可从 Git 历史查看。

Git 保存文档、代码与紧凑证据；大体积模型及原始输入保存在报告注明的本地档案。目录整理不重新训练、不改变实验结论或数据权限。
