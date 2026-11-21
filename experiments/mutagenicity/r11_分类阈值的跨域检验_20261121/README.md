# R11｜固定源模型后，不同目标域是否需要不同的分类阈值？

用历史开发数据做监督诊断。冻结源分类分数，把从本域适配池和另外两个目标域适配池学出的阈值应用到同一批留出图，比较普通 Accuracy。覆盖 Mutagenicity 边数、节点数和 PROTEINS 边数三种划分，各 12 个方向、5 个种子。登记时已核对 180 块的身份和文件字段，尚无本轮计分结果。复用 R7 的分数与按角色读取标签的入口，无需重训图网络。[实施与判读](docs/threshold-transfer.md) · [研究复盘](../../../docs/guides/research-retrospective.md)

已完成 180/180 个块，三个设置达到带标签阈值诊断门槛。[实测报告](report.md) · [实现复核](docs/implementation-check.md)

[固定排序的收益上限与可识别性](docs/rank-identification.md)对照五种子的纯源结构参照，并给出相同 ROC 和无标签分数对应不同最优阈值的例子。

[图大小比例与 ROC 工作点迁移](docs/roc-rate-transfer-results.md)已完成三种设置的 36 个方向。三个设置未达到补种子条件。[登记协议](docs/roc-rate-transfer.md)

[蛋白质的图大小与目标多数类](docs/protein-size-exposure-results.md)完成 PROTEINS 边数的 60 块。新池多数 Accuracy 为 70.035%，新分位在 3406–3409 比它低 3.743 pp，未达晋级门槛。[登记协议](docs/protein-size-exposure.md)

从仓库根目录运行，选择一个新的结果目录：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.scripts.run --config experiments/mutagenicity/r11_分类阈值的跨域检验_20261121/configs/threshold-transfer.json --output /tmp/r11-threshold-transfer
```

复用 R7 的历史封存、R5 角色认证和配对重采样。执行要求可访问原 C 档案与认证标签文件。

## 事后补充｜无标签类别先验

已完成 180 块，MLLS 在三个设置均未达到登记门槛。[补充结果](docs/prior-control-results.md) · [登记协议](docs/prior-control.md) · [执行配置](configs/prior-control.json)。复用主实验所选的 A 图与冻结的 B0 分数。无标签决策、A 比例监督对照、H 评分分阶段封存。历史 EM 是主臂，独立有界根解单独归档。两者均复用 R7 的校准、先验界与同一凹似然。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.scripts.run_prior --config experiments/mutagenicity/r11_分类阈值的跨域检验_20261121/configs/prior-control.json --output /tmp/r11-prior-control
```

## 补充｜无标签 ROC 工作点迁移

复用历史 Size 模型、原 R11 的等数 A 和原 H，在新的档案目录执行。

```sh
round=experiments/mutagenicity/r11_分类阈值的跨域检验_20261121
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.scripts.roc_rate_transfer --config "$round/configs/roc-rate-transfer.json" --output /path/to/new-roc-attempt
```

## 运行 PROTEINS 图规模曲线

从仓库根目录选新的输出路径。冻结只读取源 S 标签。核验 60 块 OBS 后，将 `obs_sha` 设为当前冻结阶段打印的 SHA-256，再统一计分和复核源 V。

```sh
round=experiments/mutagenicity/r11_分类阈值的跨域检验_20261121
out=/path/to/new-protein-size-exposure-attempt
entry=experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.scripts.protein_size_exposure
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m "$entry" --config "$round/configs/protein-size-exposure.json" --output "$out" --phase freeze
python -m "$entry" --config "$round/configs/protein-size-exposure.json" --output "$out" --phase score --obs-sha256 "$obs_sha"
```
