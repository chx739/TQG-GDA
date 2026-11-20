# R10｜公开图级适配方法在同一批留出图上能多分对多少？

以 GDABench 发布的 graph-mode GRADE 及同架构 source-only 为参照，在两个既有 Mutagenicity 开发方向建立同图的 Accuracy 比较。登记时尚无该参照在这里的分类结果。[实施与判读](docs/matched-grade.md) · [实验索引](../../README.md)

[作者原文件与一步等价核验](docs/implementation-check.md) · [初始启动记录](results/startup/launch.json) · [生成回归记录](results/startup/r10-tests.log)

从仓库根目录选择新档案路径运行。已有 INPUT 的运行用同路径续跑，已封存的模型会先核验再跳过。

```sh
python -m experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.run prepare --config experiments/mutagenicity/r10_原生图适配基线与增量_20261121/configs/r10-grade.json --root /path/to/new-r10-attempt
python -m experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.run train --root /path/to/new-r10-attempt
python -m experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts.run evaluate --root /path/to/new-r10-attempt
```

本次运行位于 `/home/hxchen/research/GDA/TSBA/experiments/r10/attempt-001`。五个种子的 source-only 各训练一次，推断同种子的两个 H。每个方向单独训练 GRADE。前置检查通过后共安排 15 个模型。H 只在模型和同批历史预测全部重载核验、封存后由评价步骤读取标签。
