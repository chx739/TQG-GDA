# R9｜未标记邻居能否帮助一张新图找到更可靠的源标签？

首轮只使用 Mutagenicity 边数划分 M0 内已有的源训练图。在同一来源内构造小图到大图、大图到小图两个伪迁移，检验通过中间图改变源样本的投票权是否能提高图分类 Accuracy。两个方向共十块已完成并封存，未达到真实开发比较的登记条件。

[来源内试验与停止条件](docs/source-bridge.md) · [图级强基线的同协议接入准备](docs/native-comparators.md) · [实验索引](../../README.md)

[筛查实测报告与全部证据](report.md)

已在三个真实开发设置的无标签输入上检查局部模式能否跨图复现。半径 2 的本域命中率比等预算异域平均高 1.87 个百分点。这一差值衡量结构覆盖，未使用图类别。[协议、实测与复核](docs/local-vocabulary.md)

[一处节点类型差异的跨图参照](docs/paired-action-support-results.md)已完成三个设置 36 个方向。PROTEINS 的可改判上界为 0.425%，低于 0.5% 门槛。[登记协议](docs/paired-action-support.md)

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
