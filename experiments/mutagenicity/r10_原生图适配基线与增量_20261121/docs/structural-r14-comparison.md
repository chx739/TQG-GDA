# R14 与源监督 WL 的同种子对照

比较共同种子 3405 的 **36/36 块**，三个设置各 12 个方向。两个封存独立完成拟合。逐块核验原 R7 任务与 manifest 哈希、H 的有序图 ID、共同 B0 和 SourceCal 的 logits 与类别。完整匹配后才读取 H 标签，沿用 2,000 次目标域×类别的共享重采样。[R10 结构参照协议](structural-reference.md) · [R14 原报告](../../r14_节点类型与分类更新_20261121/report.md)

## Accuracy

| 设置 | R14 Base | R14 Binding | 固定 C=1 的 WL | 源 V 选模 WL | Base−选模 WL [95% 区间] | Binding−选模 WL [95% 区间] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 70.24% | 69.89% | 63.43% | 71.25% | −1.02 [−2.31, +0.47] | −1.37 [−2.67, +0.15] |
| Mutagenicity 节点数 | 72.25% | 71.97% | 64.14% | 72.78% | −0.53 [−1.91, +0.86] | −0.81 [−2.19, +0.55] |
| PROTEINS 边数 | 59.21% | 59.15% | 51.13% | 53.58% | +5.62 [+2.96, +8.43] | +5.57 [+2.90, +8.41] |

差值单位为百分点。两种 Mutagenicity 划分中，源验证选模的 WL 点估计高于 R14 两臂，区间跨零。PROTEINS 上两条 R14 臂均高于该 WL 参照，且 Base 的点估计略高于 Binding。这与 R14 相对自身起点无增益的结论一致。选择 WL 正则强度的影响很大，固定 C 的对照会给出不同的相对排名。该表的比较条件于种子 3405 和已拟合模型。

## Macro-F1

| 设置 | R14 Base | R14 Binding | 固定 C=1 的 WL | 源 V 选模 WL |
| --- | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 69.67% | 69.26% | 61.39% | 70.60% |
| Mutagenicity 节点数 | 70.99% | 70.66% | 62.20% | 72.10% |
| PROTEINS 边数 | 54.47% | 54.61% | 38.82% | 45.23% |

本补充评分复用冻结预测，包含 144 个方法端点与 144 条配对。保留每方向准确率、Macro-F1、BA、AUROC、NLL 及纠正损害计数。[端点](../results/structural-r14-3405/endpoints.csv) · [逐块配对](../results/structural-r14-3405/paired-blocks.csv) · [汇总和区间](../results/structural-r14-3405/paired-summary.csv) · [封存](../results/structural-r14-3405/COMPARISON.json) · [可重放入口](../scripts/compare_structural_r14.py)
