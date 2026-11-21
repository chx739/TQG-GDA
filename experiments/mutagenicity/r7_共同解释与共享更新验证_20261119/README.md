# R7 共同解释与共享更新验证

[原参考算法在真实图上的风险归因](docs/original-risk-attribution-results.md)复用第二轮两个方向、五个种子。留出图上逐图分配误差均大于比例项，多个解释减轻了单参考更新的损害，两方向的平均 Accuracy 仍低于源模型。[归因协议](docs/original-risk-attribution.md) · [逐块结果](results/iteration-02/original-risk-attribution-001/score/attribution.csv)

受控补充在原始分子和蛋白图上交换节点类型码表，分别检验能否用一池图恢复编码，以及恢复后能多分对哪些查询图。[角色、预检与分类对照](docs/finite-codebook-control.md)

[第十二轮已有更新头的跨目标域交叉验证](docs/reference-head-crossover-results.md)已完成 180 个评价块。E 在三种开发设置均通过目标匹配门槛，PROTEINS 的匹配增益几乎全由同阳性数对照解释。[预登记协议](docs/reference-head-crossover.md)

[将 E 改判移植到强源模型](docs/transfer-of-flips-results.md)已完成 180 块。三个设置均未达到相对指定强源模型的收益门槛。[预登记协议](docs/transfer-of-flips.md)

[E 排序收益与强基座等量交换](docs/strong-rank-swaps-results.md)复用原交叉结果。分子节点数设定的 WL 已吸收多数 E 净收益，严格保阳性数后的增量为 +0.032 pp，去域时可转负。[事后诊断协议](docs/strong-rank-swaps.md)

[强源起点重求解](docs/strong-anchor-refit-results.md)已完成单种子 36 个既有开发方向。Full 在三个设置均低于指定强源起点，未达扩种子门槛。[登记协议](docs/strong-anchor-refit.md)
使用相同完整更新空间的[教师校准对照](docs/strong-anchor-score-teacher.md)检验先验和 GNN 分数斜率能否解释这项更新。
[八样本算例和 36 个未标记适配池的矩核验](docs/strong-anchor-mechanism.md)说明重新求解何时能改变判决，以及该改变的正确性还取决于什么。
[独立核验记录](docs/strong-anchor-review.md)涵盖两个数值修复、角色和判据的核对及中断评分恢复。
[两教师权重差的回放](docs/centered-target-update-results.md)已完成 36 个方向。指定强源起点均有小幅 Accuracy 增量，三个设置未同时达到均值与方向覆盖条件。[登记协议](docs/centered-target-update.md)

第十二轮按原门槛首次通过真实 P。全方向 C 已封存，Mutagenicity 边数划分的平均 BA 提高 0.53 个百分点，节点数划分接近零，PROTEINS 有主比较缺失。C 整体尚未通过。第十三轮合成扩展未过门槛。第十四轮用配对消融定位到类别比例锚点过强，固定伪计数后的正确设定诊断为 15/15。独立种子的正式 S 已通过，同配置真实 P 未通过。[真实试点](docs/iteration-12.md#同配置重跑通过-p) · [全方向结果](docs/iteration-12.md#全方向-c-的封存结果) · [均衡分类验证](docs/iteration-13.md) · [固定类别比例锚点](docs/iteration-14.md)

研究问题是，源标签和无标签目标图约束出的共同监督统计，能否为固定分类器提供有用的更新，以及保留图之间的共同约束是否比逐图区间更有价值。

[完整实验协议](docs/r7-protocol.md) · [计划配置](configs/r7-plan.json) · [方法实现](../../../docs/method/shared-update.md) · [返回总览](../../README.md)

| 阶段 | 登记范围 | 决策 |
|---|---|---|
| S 合成 | 已知后验的四种世界，含只能用特征相关修正恢复的分类方向变化 | 检查有效修正、无迁移退化与适用范围 |
| P 小规模真实图 | Mutagenicity 边数划分 M0→M1、M0→M3，各 5 seed | 用留出图判断分类收益，并分开检验类别比例、校准和共同约束 |
| C 全方向开发评价 | Mutagenicity 两种划分和 PROTEINS 边数划分，各 12 方向、5 seed | 通过效果门槛后按固定设置扩展，报告配对区间和同口径 TSBA 对照 |
| D 外部数据集确认 | NCI1 边数划分，M0→M1/M2/M3，各 5 seed | 冻结方法后独立确认，已核对域索引，原始数据待准备 |

三个开发设置、NCI1 外部确认和合成机制对照属于同一轮，统一归档于此。方法依赖 `tsba.shared_update`，开发设置复用 [R5 源模型和冻结表示档案](../r5_原始方法与请求价值_20261114/results/archive-index.json)。初次运行封存了 20 个合成块。各次迭代重复原合成口径，主条件通过后才运行真实图 P。初次结果和各轮开发结果见下方。

[初次 S 实测报告](report.md) · [门槛输出](results/gate.json) · [S 阶段补充计划](docs/s-separation-followup.md)

[独立分离审计与第一轮迭代](docs/iteration-01.md) · [第二轮实测](docs/iteration-02.md) · [第三轮投影实测](docs/iteration-03.md) · [第四轮源类条件锚定](docs/iteration-04.md) · [第五轮源混合模板](docs/iteration-05.md) · [第六轮判别后验锚定](docs/iteration-06.md) · [第七轮类别质量约束](docs/iteration-07.md) · [第八轮混合模板与类别质量](docs/iteration-08.md) · [第九轮角向解释](docs/iteration-09.md) · [第十轮图读出成对对照](docs/iteration-10.md) · [第十一轮候选后验置信度](docs/iteration-11.md) · [第十二轮较低后验强度](docs/iteration-12.md) · [第十三轮均衡分类和合成扩展](docs/iteration-13.md) · [第十四轮固定类别比例锚点](docs/iteration-14.md)

[第十四轮 P 五种子损失与排序复盘](docs/iteration-14-p-risk-replay.md) · [十块源校准与 MLLS 对照及复现入口](docs/accuracy-and-prior-control-results.md)

## 复现原参考算法的误差归因

从仓库根目录运行。先在新目录冻结预测；第二条命令的 `obs_sha` 取第一条命令打印的 OBS SHA-256。原档案路径、环境版本和已执行命令见[实际配置](results/iteration-02/original-risk-attribution-001/executed-config.json)。

```bash
r7=experiments/mutagenicity/r7_共同解释与共享更新验证_20261119
replay_dir=/tmp/r7-original-risk-replay
python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.original_risk_attribution --config "$r7/configs/original-risk-attribution.json" --output "$replay_dir" --phase freeze
python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.original_risk_attribution --config "$r7/configs/original-risk-attribution.json" --output "$replay_dir" --phase score --obs-sha256 "$obs_sha"
```

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

登记 `matched_tsba_reuse` 的 C 分支需要额外指定 `--matched-from` 指向已封存的基线运行。入口核对指定的 INPUT 和 OBS、模型与折 ID、预算及代码，再重新核验复制出的四个池预测。

`D` 入口需要同配置通过的 `--c-output` 及本地原始文件认证记录。记录含 `zip_sha256` 和以原始文件名为键的 `raw_sha256`。入口核对 C 判定与代码，封存执行锁，再加载 NCI1 并按方案选源模型。当前通过了生成图上的完整流程联调，真实运行等待 C 晋级。

```bash
python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.run D \
  --plan /path/to/executed-plan.json --c-output /path/to/passed-C \
  --acquisition /path/to/nci1-sha256.json --output /path/to/new-D --workers 4
```

## 冻结更新头交叉推断

复用第十二轮修复版 C 的原有 E/J、源坐标和 H。使用指定的新输出目录，全部交叉预测封存后才按原角色评分。

```sh
round=experiments/mutagenicity/r7_共同解释与共享更新验证_20261119
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.reference_head_crossover --config "$round/configs/reference-head-crossover.json" --output /path/to/new-head-crossover
```

## 冻结改判移植

复用上节已封存的交叉预测，在相同 H 上先组合并封存 E、WL 与 Size 的新动作，再按原角色评分。

```sh
round=experiments/mutagenicity/r7_共同解释与共享更新验证_20261119
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.transfer_flips --config "$round/configs/transfer-of-flips.json" --output /path/to/new-flip-transfer
```

### 强基座等量交换归因

从仓库根目录选新的输出路径。第一次运行输出 OBS 摘要。打开 H 类别前必须先核对独立分组和动作概率。原独立包也可在同一数据路径用归档的两个脚本重建。

```sh
round=experiments/mutagenicity/r7_共同解释与共享更新验证_20261119
out=/path/to/new-strong-rank-swaps
entry=experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.strong_rank_swaps
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m "$entry" --config "$round/configs/strong-rank-swaps.json" --output "$out" --phase freeze
python -m "$entry" --config "$round/configs/strong-rank-swaps.json" --output "$out" --phase verify --obs-sha256 "$obs_sha" --blind /tmp/r7-swap-specific-obs
python -m "$entry" --config "$round/configs/strong-rank-swaps.json" --output "$out" --phase score --obs-sha256 "$obs_sha"
```

复核已有 OBS 时，应在空目录重建独立回执，以保留原运行的回执字节。

```sh
original="$round/results/iteration-12/strong-rank-swaps-001"
audit=$(mktemp -d)
cp "$original/OBS.json" "$original/executed-config.json" "$original/blocks.csv" "$original/graphs.csv" "$audit/"
obs_sha=$(sha256sum "$audit/OBS.json" | cut -d' ' -f1)
python -m "$entry" --config "$round/configs/strong-rank-swaps.json" --output "$audit" --phase verify --obs-sha256 "$obs_sha" --blind /tmp/r7-swap-specific-obs
cmp "$audit/blind-verification.json" "$original/blind-verification.json"
```

## 复现强源起点重求解

先在完整 A 上拟合、封存 H 预测并核对 OBS。第二个命令使用第一步打印的 OBS SHA-256，只读取原 H 角色标签。

```sh
round=experiments/mutagenicity/r7_共同解释与共享更新验证_20261119
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.strong_anchor_refit --config "$round/configs/strong-anchor-refit.json" --output /path/to/new-anchor --phase freeze
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.strong_anchor_refit --config "$round/configs/strong-anchor-refit.json" --output /path/to/new-anchor --phase score --obs-sha256 OBS_SHA256
```

## 复现冻结教师差分

先封存同一个 H 上的新判决。使用输出的 OBS SHA-256 在同一目录评分。

```sh
round=experiments/mutagenicity/r7_共同解释与共享更新验证_20261119
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.centered_target_update --config "$round/configs/centered-target-update.json" --output /path/to/new-centered --phase freeze
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.centered_target_update --config "$round/configs/centered-target-update.json" --output /path/to/new-centered --phase score --obs-sha256 OBS_SHA256
```
