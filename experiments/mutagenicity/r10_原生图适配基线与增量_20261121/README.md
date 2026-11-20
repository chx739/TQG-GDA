# R10｜公开图级适配方法在同一批留出图上能多分对多少？

以 GDABench 发布的 graph-mode GRADE 及同架构 source-only 为参照，在两个既有 Mutagenicity 开发方向建立同图的 Accuracy 比较。15 个模型已完成，GRADE 相对同架构源模型的两方向等权 Accuracy 差 +0.074 pp，配对区间跨零。[实测报告](report.md) · [实施与判读](docs/matched-grade.md) · [实验索引](../../README.md)

[作者原文件与一步等价核验](docs/implementation-check.md) · [初始启动记录](results/startup/launch.json) · [生成回归记录](results/startup/r10-tests.log)

[全方向源监督 WL 和规模参照](docs/structural-reference.md)已登记，覆盖三个既有开发设置，各十二方向五种子。

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
