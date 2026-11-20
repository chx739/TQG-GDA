# R18｜从源图学到的同类关系能否帮助纠正强分类器？

在 Mutagenicity 两种大小划分和 PROTEINS 边数划分的 24 个历史源内方向，分别测试留出图对的同类概率、无标签同伴的分类增量及同一查询换参考池后的表现。[固定方案](docs/source-screen.md)

[实测报告](report.md)覆盖全部 24 个方向。Full 的 Brier 在三个设置均高于校准乘积。TargetFull 的判决与等阳性数对照逐项相同，未达到真实四域门槛。[同伴投票的机制分析](docs/what-a-peer-vote-can-correct.md)

## 运行

从仓库根目录先封存全部动作和 E 无序图对概率。核验 OBS 后，用第一步打印的 SHA-256 进行评分。

```sh
round=experiments/mutagenicity/r18_跨图同类关系与分类纠错_20261121
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r18_跨图同类关系与分类纠错_20261121.scripts.source_screen --config "$round/configs/source-screen.json" --output /path/to/new-r18-attempt --phase freeze
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r18_跨图同类关系与分类纠错_20261121.scripts.source_screen --config "$round/configs/source-screen.json" --output /path/to/new-r18-attempt --phase score --obs-sha256 OBS_SHA256
```

角色、冻结基线与 typed WL 复用 R9，输入认证沿用 R17。逐折及关系模型保存在本轮的封存结果。
