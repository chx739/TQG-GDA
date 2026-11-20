# R13｜目标图归一化与分类排序的先导结果

2026-11-21 完成种子 3405 的全部 **36/36 块**，覆盖 Mutagenicity 边数、节点数和 PROTEINS 边数各 12 个方向。**图等权和节点等权路线均为 0/3 设置达到投入门槛。** 按登记规则，先导在这个种子结束。两个本域 A 校准臂在三个设置的普通 Accuracy 都低于原 B0。[登记协议](docs/pilot.md)

## 同一批 H 图的 Accuracy

每行对 12 个方向等权。S 和本域或异域 A 的统计使用同一个源 checkpoint、相同图数和相同选取规则。Other 列是两个异域臂的 Accuracy 等权平均。

| 设置 | B0 | SourceCal | S-node | S-graph | A-node | A-graph | Other-node | Other-graph | B0 等阳性数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 69.775% | 69.337% | 70.111% | 70.445% | 67.100% | 67.190% | 68.134% | 67.691% | 69.712% |
| Mutagenicity 节点数 | 70.927% | 71.521% | 72.222% | 72.278% | 67.566% | 67.213% | 69.142% | 68.946% | 70.042% |
| PROTEINS 边数 | 57.682% | 56.474% | 58.347% | 58.595% | 56.937% | 56.667% | 56.561% | 55.866% | 57.039% |

| 设置 | A-node−B0 | A-graph−B0 | A-graph−A-node | A-graph−B0 等阳性数 | 图等权交互 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | −2.68 [−3.86, −1.44] | −2.59 [−3.82, −1.30] | +0.09 [−0.37, +0.55] | −2.52 [−3.74, −1.32] | −0.24 [−0.80, +0.31] |
| Mutagenicity 节点数 | −3.36 [−4.60, −2.10] | −3.71 [−4.98, −2.43] | −0.35 [−0.74, +0.01] | −2.83 [−4.02, −1.59] | −0.41 [−0.92, +0.08] |
| PROTEINS 边数 | −0.74 [−3.04, +1.44] | −1.01 [−3.26, +1.16] | −0.27 [−1.28, +0.73] | −0.37 [−2.46, +1.63] | −0.52 [−1.97, +0.94] |

差值单位为百分点。交互是 (A-graph−A-node) 减 (S-graph−S-node)。95% 区间采用历史 2,000 次目标域×类别共享图重采样。同一目标域内不同源模型和方法使用相同抽样权重，区间条件于种子 3405、这批固定图和已经校准的统计。

图等权在 3、0、2 个方向同时超过 B0 和 A-node，节点等权在 2、0、6 个方向超过 B0。两条路线每个设置都达不到 7/12 方向门槛。两种 Mutagenicity 划分上，本域图等权也低于异域图等权。PROTEINS 本域两臂都超过各自异域均值，但低于原 B0 和对应的 S 重算统计。

## 排序与工作点

沿 B0 原排序选取与 A-graph 相同数量的正类，所得 Accuracy 在三个设置都高于 A-graph。相对 B0，A-graph 的 AUROC 分别降低 **3.26、3.22、1.53 个百分点**。两个 Mutagenicity 划分中，A-graph 的 BA 也下降。PROTEINS 的 A-graph BA 从 59.96% 升至 60.53%，同时 Accuracy 和 AUROC 均下降。

这批开发图没有显示本域 BN 统计能提升分类排序。源端精确重算统计在三个设置的 Accuracy 点估计均高于 B0。图等权相对节点等权的增量在目标端也没有超过源端。覆盖仍是一个种子，两个 Mutagenicity 设置是同一个数据集的两种划分。

## 历史同图对照

R10 的三个对照只覆盖 Mutagenicity 边数的 **2/36 块**。以下是配对 Accuracy 差，单位为百分点。R10 中的 B0、SourceCal 在这两块的 margin 与类别逐图对齐。

| 方向 | A-graph−SourceOnly | A-graph−SourceFusion | A-graph−GRADE |
| --- | ---: | ---: | ---: |
| M0→M1 | −9.45 | −5.86 | −8.79 |
| M0→M3 | −0.30 | −10.36 | +0.30 |

这两块上的比较不能当作全部 12 个方向的排名。每方向的两套异域、历史对照及净纠错保留在逐块表。

## 执行与核验

执行提交为 `6be5b31`。12 个源模型组共保存 96 套 BN 统计，每套六处 BN，按实际内1、外1、内2、外2、内3、外3 顺序估计。只替换 running_mean 和 running_var。从相同原 checkpoint 起步，统计分批为 128、47 张图，记录每处矩、图数、节点数和重分批误差。各组原始 logits 与 R7 逐图复算最大误差为 **0**，B0 重分批最大差为 **1.91×10⁻⁶**，校准臂重分批最大差为 **3.52×10⁻⁶**，所有离散类别一致。

36 块完整预测封存后，评分读口共解码了 12 个 H 队列。归档核验覆盖 402 个端点、882 条逐块配对。工程回归共 29 项通过，独立审查另对 48 组 GIN/GINE 合成校准进行了显式数值比对。

[执行配置](results/attempt-001/executed-config.json) · [BN 统计目录](results/attempt-001/stats) · [角色、预测来源与误差](results/attempt-001/roles.json) · [OBS](results/attempt-001/OBS.json) · [FINAL](results/attempt-001/FINAL.json) · [H 评分读口](results/attempt-001/label-access.json)

[全端点](results/attempt-001/endpoints.csv) · [各臂 Accuracy、BA、AUROC、NLL 和召回](results/attempt-001/endpoint-summary.csv) · [逐块纠正、损害及净差](results/attempt-001/paired-blocks.csv) · [配对汇总与区间](results/attempt-001/paired-summary.csv) · [门槛判定](results/attempt-001/gate.json)

[预检日志](results/attempt-001/preflight.log) · [运行日志](results/attempt-001/run.log) · [工程回归](results/attempt-001/engineering-tests.log) · [归档核验](results/attempt-001/verification.json)
