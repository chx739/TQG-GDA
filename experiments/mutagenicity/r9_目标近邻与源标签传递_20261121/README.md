# R9｜未标记邻居能否帮助一张新图找到更可靠的源标签？

首轮只使用 Mutagenicity 边数划分 M0 内已有的源训练图。在同一来源内构造小图到大图、大图到小图两个伪迁移，检验通过中间图改变源样本的投票权是否能提高图分类 Accuracy。两个方向共十块已完成并封存，未达到真实开发比较的登记条件。

[来源内试验与停止条件](docs/source-bridge.md) · [图级强基线的同协议接入准备](docs/native-comparators.md) · [实验索引](../../README.md)

[筛查实测报告与全部证据](report.md)

已在三个真实开发设置的无标签输入上检查局部模式能否跨图复现。半径 2 的本域命中率比等预算异域平均高 1.87 个百分点。这一差值衡量结构覆盖，未使用图类别。[协议、实测与复核](docs/local-vocabulary.md)

[一处节点类型差异的跨图参照](docs/paired-action-support-results.md)已完成三个设置 36 个方向。PROTEINS 的可改判上界为 0.425%，低于 0.5% 门槛。[登记协议](docs/paired-action-support.md)

[源校准残差的结构加权协议](docs/weighted-source-residuals.md)使用每个源域内的自然大小段，检验局部结构能否帮助估准目标组的正例比例并改善分类。

[按大小段借用源分类残差](docs/weighted-source-residuals-results.md)完成了三个设置共 24 个源内伪方向。加权残差在三个设置均输给未经残差校正的 WL 概率均值，未进入真实跨域验证。[登记协议](docs/weighted-source-residuals.md)

[真实 U/E 比例定阈诊断](docs/true-rate-replay-results.md)复用了全部 24 个源内方向。PROTEINS 上将 E 真实比例和 E 排序同时用于定阈，Accuracy 仍比原 WL 低 4.583 个百分点。

[伪目标结构共变筛查](docs/reference-geometry-results.md)完成了全部 24 个源内方向。TargetFull 在三个设置均低于联合 Base，未进入真实四域验证。

从仓库根目录执行。输出使用一个新的存档目录。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.scripts.run   --config experiments/mutagenicity/r9_目标近邻与源标签传递_20261121/configs/source-bridge.json   --output /tmp/r9-source-screen
```

运行器复用 R8 的类型校验和源 logistic 拟合，以及 R7 的配对图重采样。R5 输入和 R8 来源角色按封存哈希认领。两个方向完成 OBS 后才应用动作分歧前置条件并评分。

从仓库根目录复算已封存 Q/R 的改判上界。

```sh
round=experiments/mutagenicity/r9_目标近邻与源标签传递_20261121
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.scripts.paired_action_support --config "$round/configs/paired-action-support.json" --output /path/to/new-paired-action-attempt
```

## 运行源内大小段残差试验

复用 R10 的历史源图角色与 R9 的 WL，使用 R11 的硬阳性率阈值。D/C 与 U/E 标签按方向分阶段读取。

```sh
round=experiments/mutagenicity/r9_目标近邻与源标签传递_20261121
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.scripts.weighted_source_residuals --config "$round/configs/weighted-source-residuals.json" --output /path/to/new-weighted-residual-attempt
```

## 复算已知比例的评分后诊断

使用原加权残差试验的封存预测和公开分层总数，封存三组动作后按原 E 角色评分。

```sh
round=experiments/mutagenicity/r9_目标近邻与源标签传递_20261121
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.scripts.true_rate_replay --config "$round/configs/true-rate-replay.json" --output /path/to/new-true-rate-replay
```

## 复现伪目标参照共变试验

复用 attempt-002 的 24 个源内方向。组合预测并封存 OBS 后才读取对应的 E 标签。

```sh
round=experiments/mutagenicity/r9_目标近邻与源标签传递_20261121
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.scripts.reference_geometry --config "$round/configs/reference-geometry.json" --output /path/to/new-reference-geometry
```

也可先加 `--phase freeze`，随后对同一目录使用 `--phase score --obs-sha256` 并传入第一阶段输出的 OBS 摘要。

[结构词 dropout 的源内实测](docs/feature-dropout-screen.md)在 24 个来源内方向上比较了以真实词频、源词频及打乱后的词频调整分类惩罚。蛋白质有单种子正结果，两种分子划分均落后于静态岭分类器。
