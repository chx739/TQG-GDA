# R8｜独立图结构分数能否为目标类别提供可迁移的约束？

真实图试点使用 Mutagenicity 边数划分 M0→M1、M0→M3，各五个源模型种子。研究问题是，加入一项从图中直接计算、由源标签训练的分数，能否提供超出源校准、先验平移和源监督融合的逐图纠错信息。登记的十个开发块已完成并封存，主臂全部为零更新，未通过 Accuracy 晋级条件。[试点实测报告](report.md)

[试点设计与停止规则](docs/pilot.md) · [实验总览](../../README.md)

## 执行

预检逐图比对 R5 INPUT 封存的离散属性与无标签拓扑，并核验 R7 的十块角色和预测身份。正式入口先使用源 fit/validation 与无标签 A 拟合，再推断 H，封存 A/H 预测后统一计分。沿用 R7 认证过的 SourceCal、MLLS、B0 和 J 预测身份，以及按目标域、类别分层的共享图重采样。历史预测与本次结论分别存档。

从仓库根目录调用，输出用新目录：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.scripts.run preflight --config experiments/mutagenicity/r8_图结构观察量与类别约束_20261121/configs/pilot-plan.json --output /tmp/r8-preflight
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.scripts.run run --config experiments/mutagenicity/r8_图结构观察量与类别约束_20261121/configs/pilot-plan.json --output /path/to/new-r8-attempt
```

[运行前复核与预检结果](docs/preflight-review.md) · [冻结配置](configs/pilot-plan.json)

[固定 A 均值的工作点诊断](docs/fixed-mass-diagnostic-results.md)

[分数档内漂移的历史分解](docs/conditional-drift.md)显示，M0→M1 的类 0 在相同主分类器分数档内仍有明显的结构分数差异，M0→M3 的类 1 则有两项相反的变化。
