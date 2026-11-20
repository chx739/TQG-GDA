# R15｜其他图中的远处结构能否给一个新局部模式提供类别信息？

在已有 R9 来源内双向伪迁移上做一次低成本筛查。源模型无法为在训练中从未出现的两跳模式拟合专属权重。R15 检验能否从独立无标签图中为这种模式估计一个可迁移到整图分类的增量。输出角色、拟合、对照和判定依据见[来源内方案](docs/source-screen.md)。本次授权范围仅为既有的 868 张源图。[实验索引](../../README.md)

## 运行

从仓库根目录执行，输出使用新目录。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m experiments.mutagenicity.r15_新局部结构与图类别_20261121.scripts.run \
  --output experiments/mutagenicity/r15_新局部结构与图类别_20261121/results/attempt-001
```

[计划配置](configs/source-screen-plan.json) · [文档索引](docs/README.md)

输入沿用 R9 attempt-001 的 OBS 封存和源图档案，依赖 R9 的源图认证、角色划分与 WL 归一化，以及 R8 对图类型和边的校验。复用 R9 已拟合的源模型并验证原边际与分类动作。先封存两个方向的词典与 E 预测，再按方案判断能否读取两方向的 E 标签。

## 来源内结果

两方向共完成 433 张 E 图的计分。真实对应比 B 与交换对照的等权 Accuracy 分别高 0.227 和 0.455 个百分点，与等阳性数 B 的分类动作一致。反向 NLL 改善斜率为负，筛查未晋级。[报告](report.md) · [执行与封存证据](results/evidence.json)
