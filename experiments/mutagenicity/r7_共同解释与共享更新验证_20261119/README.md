# R7 共同解释与共享更新验证

第十一轮控制候选标签后验的置信度，两个真实方向的平均 BA 都转正，M0→M1 达到该方向门槛。M0→M3 的 NLL 仍缺充分改善，P 未通过。第十二轮固定一个更低的后验强度继续验证。研究问题是，源标签和无标签目标图约束出的共同监督统计，能否为固定分类器提供有用的更新，以及保留图之间的共同约束是否比逐图区间更有价值。

[完整实验协议](docs/r7-protocol.md) · [计划配置](configs/r7-plan.json) · [方法实现](../../../docs/method/shared-update.md) · [返回总览](../../README.md)

| 阶段 | 登记范围 | 决策 |
|---|---|---|
| S 合成 | 已知后验的四种世界，含只能用特征相关修正恢复的分类方向变化 | 检查有效修正、无迁移退化与适用范围 |
| P 小规模真实图 | Mutagenicity 边数划分 M0→M1、M0→M3，各 5 seed | 用留出图判断分类收益，并分开检验类别比例、校准和共同约束 |
| C 全方向开发评价 | Mutagenicity 两种划分和 PROTEINS 边数划分，各 12 方向、5 seed | 通过效果门槛后按固定设置扩展，报告配对区间和同口径 TSBA 对照 |
| D 外部数据集确认 | NCI1 边数划分，M0→M1/M2/M3，各 5 seed | 冻结方法后独立确认，已核对域索引，原始数据待准备 |

三个开发设置、NCI1 外部确认和合成机制对照属于同一轮，统一归档于此。方法依赖 `tsba.shared_update`，开发设置复用 [R5 源模型和冻结表示档案](../r5_原始方法与请求价值_20261114/results/archive-index.json)。初次运行封存了 20 个合成块。各次迭代重复原合成口径，主条件通过后才运行真实图 P。初次结果和各轮开发结果见下方。

[初次 S 实测报告](report.md) · [门槛输出](results/gate.json) · [S 阶段补充计划](docs/s-separation-followup.md)

[独立分离审计与第一轮迭代](docs/iteration-01.md) · [第二轮实测](docs/iteration-02.md) · [第三轮投影实测](docs/iteration-03.md) · [第四轮源类条件锚定](docs/iteration-04.md) · [第五轮源混合模板](docs/iteration-05.md) · [第六轮判别后验锚定](docs/iteration-06.md) · [第七轮类别质量约束](docs/iteration-07.md) · [第八轮混合模板与类别质量](docs/iteration-08.md) · [第九轮角向解释](docs/iteration-09.md) · [第十轮图读出成对对照](docs/iteration-10.md) · [第十一轮候选后验置信度](docs/iteration-11.md) · [第十二轮较低后验强度](docs/iteration-12.md)

## 合成阶段执行

在仓库根目录用已安装项目依赖的 Python 执行。`check` 只运行小型工程夹具。以下示例重放第六轮 S，使用原执行提交 `b69aa5b` 的干净工作区。初次 S 使用其原执行提交 `bd259d3`。每次执行指定一个新运行目录。

```bash
python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.run check
python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.run S \
  --plan experiments/mutagenicity/r7_共同解释与共享更新验证_20261119/configs/r7-iteration06-posterior.json \
  --output /tmp/r7-iteration06-replay/S --workers 4
```

S 使用固定生成参数的解析源 logits。真实目标条件概率与抽样标签只在对应封存后进入 oracle 或评价。主宽度的门槛决定是否继续 P，`gate.json` 保存逐世界、逐条件的决策。工程回归固定到本方法及本轮入口的相关测试。

`S-audit` 使用初次 S 的封存档案，做固定 W 独立分离诊断。`--plan` 可用于登记后的迭代版本。P 入口需要优化版本通过的同配置 S 封存目录，入口会校验合成阶段的拟合实现哈希。

`C` 入口需要同配置通过的 `--p-output`。它沿用 R5 M00 源模型缓存和 R3 的候选更新验收，按登记的实际迭代数重训 A 上的 score，并和 J 共用 A/H 折。已完成小规模工程联调。

登记 `source_refit` 的分支用该轮新训练的冻结源模型及其匹配 TSBA 坐标。P 和 C 将源模型与冻结表示纳入输入封存。C 复用 P 的同一源检查点，其他源检查点按场景和种子各训练一次。

`D` 入口需要同配置通过的 `--c-output` 及本地原始文件认证记录。记录含 `zip_sha256` 和以原始文件名为键的 `raw_sha256`。入口核对 C 判定与代码，封存执行锁，再加载 NCI1 并按方案选源模型。当前通过了生成图上的完整流程联调，真实运行等待 C 晋级。

```bash
python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.run D \
  --plan /path/to/executed-plan.json --c-output /path/to/passed-C \
  --acquisition /path/to/nci1-sha256.json --output /path/to/new-D --workers 4
```
