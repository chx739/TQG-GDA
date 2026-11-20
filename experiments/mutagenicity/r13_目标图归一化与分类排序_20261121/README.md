# R13｜无标签目标图的统计量能否改善冻结图分类器的排序和准确率？

R12 中依赖同图三项读数推断阈值的办法未通过源内筛查。本轮直接改变冻结编码器的 BatchNorm 统计量，测试已有图分类器能否利用本域无标签图改善分类。先跑三个已开发设置的所有方向、一个源种子。普通 AdaBN 是方法基线，每张图等权的变体需要证明额外贡献。种子 3405 的 36 块已完成，两条路线均未达到补种子门槛。见[先导报告](report.md)。

[设计、反例与投入标准](docs/pilot.md) · [实验总览](../../README.md)

## 运行

以下入口从仓库根目录运行。预检核验原 buffers 的 B0 逐图 logits、类别与 H 身份，使用冻结的 R10/R11 对照，不读取 H 标签。正式运行先固定全部 S/A 统计及 H 预测，然后读取 H 标签评分。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.scripts.run --config experiments/mutagenicity/r13_目标图归一化与分类排序_20261121/configs/pilot.json --preflight
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.scripts.run --config experiments/mutagenicity/r13_目标图归一化与分类排序_20261121/configs/pilot.json --output /tmp/r13-pilot
```

[配置](configs/pilot.json)规定数值复核容差。正式路径使用历史 batch size 128，另以 47 重做统计及前向，核对矩、logits 和类别。编码器使用原 checkpoint 的 float32，分类头使用历史 R7 的 float64 乘加。先导达到登记条件才使用 `--extend-from <先导目录>` 和独立输出目录补种子。
