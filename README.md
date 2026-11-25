# CELA v3

Connected Edge-Law Adaptation 用已标注的源图学习各类别如何逐步选择一条边，再用一批无标签目标图估计哪些结构选择在目标域变得更常见。目标阶段只更新可由组内比较识别的结构偏移，以及预先声明的可选类别偏置。任务是整图分类。

```bash
python -m pip install -e .
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python -m experiments.s1_流程验收_20261126.scripts.run --task synthetic --output smoke-runs/synthetic
```

运行命令时使用仓库根目录。合成 smoke 自建输入。两项真实 smoke 需要 TU 原始文本，已附样本清单和逐图索引。每次运行指定新的输出目录。源模型的正式默认预算是 100 epoch，smoke 使用 2 epoch。

| 内容 | 入口 |
|---|---|
| 方法流程、默认参数、概率口径和参考实现 | [方法说明](docs/method.md) |
| 数据协议、角色和原始图索引 | [数据说明](data/README.md) |
| 合成、Mutagenicity、PROTEINS 的复现命令和实测结果 | [实验目录](experiments/README.md) |
| 独立复核的发现和回归依据 | [审查记录](docs/review.md) |

核心模块在 `cela/`，统一命令行在 `scripts/`，自动化测试在 `tests/`。实验直接归档到 `experiments/<实验编号>_<主题>_<日期>/`。
