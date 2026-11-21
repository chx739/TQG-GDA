# R10｜公开图级适配方法在同一批留出图上能多分对多少？

以 GDABench 发布的 graph-mode GRADE 及同架构 source-only 为参照，在两个既有 Mutagenicity 开发方向建立同图的 Accuracy 比较。15 个模型已完成，GRADE 相对同架构源模型的两方向等权 Accuracy 差 +0.074 pp，配对区间跨零。[实测报告](report.md) · [实施与判读](docs/matched-grade.md) · [实验索引](../../README.md)

[作者原文件与一步等价核验](docs/implementation-check.md) · [初始启动记录](results/startup/launch.json) · [生成回归记录](results/startup/r10-tests.log)

[冻结 GRADE 的交叉推断](docs/reference-crossover-results.md)已完成五种子和两个既有开发域。两域等权的本域匹配增量为 +0.166 pp，95% 配对区间跨零。[交叉协议](docs/reference-crossover.md)

[全方向源监督 WL 和大小参照](docs/structural-reference-results.md)已完成三个开发设置各十二方向五种子，共 180 块。源选模 WL 相对 B0 的 Accuracy 差依次为 +2.44、+0.12、−2.18 pp。[结构参照协议](docs/structural-reference.md)

在同一源的三个目标间改选 WL 或 B0，两种 Mutagenicity 划分的事后额外空间均小于 0.1 个百分点。PROTEINS 为 1.08 个百分点。[选择上界与分歧图分析](docs/classifier-choice.md)

在 PROTEINS 上，即使逐图知道哪个模型正确，B0 与 WL 二选一仍低于 Size 0.92 个百分点。[共同错误与三张票的改判空间](docs/vote-bounds.md)

图大小和 PROTEINS 原始属性的源端核对见[结构输入核查](docs/size-support.md)。PROTEINS 的八方向源内筛查已完成。加入按节点类型的属性中位数后，Accuracy 相对 WL＋Size 提升 0.666 pp，两个源组改善，未达继续条件。[登记协议](docs/source-attribute-screen.md) · [实测结果](docs/source-attribute-screen-results.md)

ATC 补充已完成 36 个方向。PROTEINS 同域池比异域池平均高 7.99 个百分点，但低于 R11 曲线池多数，两个 Mutagenicity 划分也低于固定 WL。[协议](docs/target-confidence-selection.md) · [结果](docs/target-confidence-selection-results.md)

从仓库根目录检查 60 个源组与 180 个开发块的角色与历史档案身份。源监督参照沿用 R7 C 的角色、R9 的 WL 表示与 R8 的 logistic 目标、R11 的配对重抽样以及 R13 验证过的历史参照读取方式。

```sh
round=experiments/mutagenicity/r10_原生图适配基线与增量_20261121
python -m experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.structural_reference --config "$round/configs/structural-reference.json" --preflight
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.structural_reference --config "$round/configs/structural-reference.json" --reviewed-commit REVIEWED_COMMIT --output /path/to/new-structural-attempt
```

结构参照先写入 FIT，封存每组的 S/V ID、模型与全部 V 候选指标，再逐图推断三个 H。OBS 绑定 FIT、完整预测网格和对应历史参照。评分程序验证 OBS 与全部 H 读口后打开标签，由 FINAL 记录结果与读口。

交叉推断直接复用原 R10 的模型和输入，先保存交叉预测，再按原 H 角色统一评分。

```sh
round=experiments/mutagenicity/r10_原生图适配基线与增量_20261121
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.reference_crossover --config "$round/configs/reference-crossover.json" --output /path/to/new-crossover-attempt
```

从仓库根目录选择新档案路径运行。中断后使用下面的独立验收入口续跑。

```sh
python -m experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.run prepare --config experiments/mutagenicity/r10_原生图适配基线与增量_20261121/configs/r10-grade.json --root /path/to/new-r10-attempt
python -m experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.run train --root /path/to/new-r10-attempt
python -m experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.run evaluate --root /path/to/new-r10-attempt
```

本次运行位于 `/home/hxchen/research/GDA/TSBA/experiments/r10/attempt-001`。五个种子的 source-only 各训练一次，推断同种子的两个 H。每个方向单独训练 GRADE。前置检查通过后共安排 15 个模型。H 只在模型和同批历史预测全部重载核验、封存后由评价步骤读取标签。

已有 INPUT 的运行如需续跑，执行下面的命令。已完成的模型须通过版本、程序身份、角色和完整成员校验；不完整模型保留为上一尝试，同配置从起点重做。

```sh
python -m experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.validate --root /path/to/r10-attempt --resume
```

## 复现 PROTEINS 节点属性筛查

从仓库根目录使用新输出位置。全部 E 预测封存后才进入评分。

```sh
round=experiments/mutagenicity/r10_原生图适配基线与增量_20261121
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.source_attribute_screen --config "$round/configs/source-attribute-screen.json" --output /path/to/new-source-attribute-attempt
```

如需在评分前人工核对，先加 `--phase freeze`，再用同一目录与 `--phase score --obs-sha256` 传入第一阶段输出的摘要。

## 复算 ATC 选型

冻结阶段复用 R10、R11、R12、R16 与 R7/R5 的认证模型、图和分数。计分阶段只读取已公布的聚合表，先核对独立盲算。

```sh
round=experiments/mutagenicity/r10_原生图适配基线与增量_20261121
run=/tmp/r10-atc-replay
entry=experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.target_confidence_selection
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m "$entry" \
  --config "$round/configs/target-confidence-selection.json" --output "$run" --phase freeze
obs=$(sha256sum "$run/OBS.json" | cut -d ' ' -f 1)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m "$entry" \
  --config "$round/configs/target-confidence-selection.json" --output "$run" --phase score --obs-sha256 "$obs" \
  --blind-script "$round/results/attempt-006-target-confidence/independent/tsba_peer_atc_unlabeled_20261121.py"
```
