# R14｜恢复被遮住的节点类型，能否给整图分类提供有益的更新？

先在已用过的三个设置各跑十二个方向、种子 3405。测量节点类型与局部上下文的对应关系能否提供超出图内类型比例的分类更新，以每张留出图的普通 Accuracy 评价。用本域无标签图产生更新，同一源模型分别用异域图和源验证图产生对照。36 块已完成，三种设置均未达到补种子门槛。见[先导报告](report.md)。

[固定设计、对照和停止条件](docs/pilot.md) · [实验总览](../../README.md)

## 运行

从仓库根目录使用 [固定配置](configs/pilot.json)。预检仅核验历史 S/V/A 图、角色和源 checkpoint。正式执行先封存全部恢复头、源 BN 统计及各池更新方向，再前向 H 图。H 标签在全批预测封存后打开。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r14_节点类型与分类更新_20261121.scripts.run --config experiments/mutagenicity/r14_节点类型与分类更新_20261121/configs/pilot.json --preflight
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r14_节点类型与分类更新_20261121.scripts.run --config experiments/mutagenicity/r14_节点类型与分类更新_20261121/configs/pilot.json --output /tmp/r14-pilot
```

有限步使用原生前向精度与最后外 BN 的更新仿射参数。局部风险使用从原生 ReLU 基点出发的 float64 解析右导数，另以首个激活拐点以内的正向差分核验。该诊断的差分幅度与登记的有限移动长度独立。达到投入门槛后可用 `--extend-from <先导目录>` 启动独立的补种子运行。
