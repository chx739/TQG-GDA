# R17｜同一目标图的外围结构能否指出应当削弱的分类变化？

本轮在 Mutagenicity 边数、节点数和 PROTEINS 边数划分的历史源域内部模拟大小迁移。完整源图提供分类标签，无标签图提供成对读出。研究主张、对照、角色和停止条件见[来源内协议](docs/paired-readout.md)。[自然包含与标签一致性的独立诊断](docs/natural-containment-results.md)另行归档。

[主报告](report.md)覆盖全部 24 个方向。TargetPair 相对固定 WL 的 Accuracy 在三个设置均下降，未达到真实开发门槛。[源内筛查与真实方向的关系](docs/source-screen-sensitivity.md)

## 运行

从仓库根目录先封存全部 E 上的预测。完成无标签复核后，在同一运行目录使用第一步输出的 OBS SHA-256 评分。

```sh
round=experiments/mutagenicity/r17_目标图外围与分类一致性_20261121
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r17_目标图外围与分类一致性_20261121.scripts.paired_readout --config "$round/configs/paired-readout.json" --output /path/to/new-r17-attempt --phase freeze
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r17_目标图外围与分类一致性_20261121.scripts.paired_readout --config "$round/configs/paired-readout.json" --output /path/to/new-r17-attempt --phase score --obs-sha256 OBS_SHA256
```

角色、已拟合的固定 WL 和三项源选模参照复用 R9。WL 的逐节点颜色沿用 R15 已验证的同一递推。
