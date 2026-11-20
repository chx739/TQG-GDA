# R14｜节点类型恢复方向的整图分类先导

2026-11-21 完成种子 3405 的 **36/36 块**。三个设置均未通过投入门槛，按登记规则结束先导。遮点类型在三个设置的源验证集上都可预测。扣除同图类型比例后，Binding 在三个设置的留出 Accuracy 均低于共同起点 Base，局部 H NLL 的平均方向也都不利。[登记协议](docs/pilot.md)

## 留出图分类

下表为逐方向 Accuracy 在各设置 12 个方向上的等权平均。Base 用全部源训练图逐层重估节点等权 BN，全部更新臂均从 Base 出发。Other 均值在同一 H 上将两个异域臂等权平均。

| 设置 | B0 | SourceCal | Base | Binding | True | Hist | V-Binding | Other ① | Other ② | Other 均值 | Base 等阳性数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 69.775% | 69.337% | 70.238% | 69.886% | 70.399% | 70.312% | 70.034% | 69.930% | 69.984% | 69.957% | 70.034% |
| Mutagenicity 节点数 | 70.927% | 71.521% | 72.246% | 71.969% | 72.196% | 72.093% | 72.044% | 72.020% | 71.941% | 71.980% | 72.121% |
| PROTEINS 边数 | 57.682% | 56.474% | 59.207% | 59.153% | 58.762% | 58.544% | 59.021% | 59.263% | 59.325% | 59.294% | 58.952% |

| 设置 | Binding−Base [95% 区间] | Binding−True | Binding−Hist | Binding−V-Binding | Binding−Other 均值 | Binding−等阳性数 Base | 胜 Base 的方向 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | −0.35 [−0.69, −0.01] | −0.51 | −0.43 | −0.15 | −0.07 | −0.15 | 1/12 |
| Mutagenicity 节点数 | −0.28 [−0.56, +0.00] | −0.23 | −0.12 | −0.08 | −0.01 | −0.15 | 1/12 |
| PROTEINS 边数 | −0.05 [−0.54, +0.45] | +0.39 | +0.61 | +0.13 | −0.14 | +0.20 | 2/12 |

配对单位为百分点。区间使用历史 R11 目标域×类别分层、2,000 次共享图重采样，条件于种子 3405 和冻结模型与方向。节点数设置的区间上端约为 +0.0014 pp，跨零。三个设置均低于相对 Base 的 +0.5 pp 门槛，也低于 7/12 胜方向门槛。

Binding 相对原 B0 仍高 +0.11、+1.04、+1.47 pp。其共同起点 Base 相对 B0 已高 +0.46、+1.32、+1.53 pp。按种子与同图配对，增加节点类型更新后，三个设置各自的 Accuracy 都回落。

| 设置 | 评估图次数 | Base／Binding 阳性数 | Binding 纠正／损害 | Other 均值阳性数 | Other 均值纠正／损害 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 3,870 | 1,907／1,879 | 15／29 | 1,878 | 14.5／25.5 |
| Mutagenicity 节点数 | 3,870 | 1,716／1,697 | 10／21 | 1,697.5 | 9.5／20 |
| PROTEINS 边数 | 1,020 | 440／443 | 3／4 | 442.5 | 3.5／3 |

计数相对 Base，跨方向累计，同一 H 图可在多个源模型下重复计入。两个 Other 的逐端点及等权值、所有臂的 Accuracy、BA、两类召回、NLL、AUROC、阳性数与纠正损害见[设置计数与指标](results/attempt-001/summary/setting-counts.csv)和[异域三行对照](results/attempt-001/summary/other-binding-endpoints.csv)。

## 辅助任务与分类风险

| 设置 | V 恢复准确率／S 多数类对照 | A 内至少两种遮点类型的图次数 | A 未知类型遮点 | Binding 局部 NLL 改善 | 去除平移与温度方向后的读数 | Binding 有限步 NLL 改善 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | 87.95%／41.69% | 2,448／2,607 | 11／10,428 | −0.01116 | +0.00245 | −0.00516 |
| Mutagenicity 节点数 | 86.25%／44.00% | 2,436／2,595 | 8／10,380 | −0.01317 | +0.00152 | −0.00571 |
| PROTEINS 边数 | 68.90%／52.97% | 498／663 | 0／2,652 | −0.00428 | −0.00157 | −0.00170 |

正值表示 NLL 降低。局部读数单位为沿单位参数方向的 nats／图，有限步 NLL 改善为固定步长后的 nats／图。两种 Mutagenicity 划分的残差局部读数略正，但协议主方向的总局部读数均为负，且有限步 Accuracy 未达门槛。PROTEINS 的 Binding 比 True、Hist 和等阳性数控制略高，仍低于 Base 与异域均值。当前证据支持关闭这组遮点信号与最后 BN 仿射参数的固定更新分支。该结论条件于一个种子和这批开发图。

图等权的遮点恢复头在所有源模型上均可拟合。各本域 A 池 Binding 未归一化梯度范数，按上表设置依次为 0.827–1.163、0.844–1.142、0.040–0.114。本域三臂均未触发回退。Binding 的 AUROC 相对 Base 分别变化 −0.052、+0.017、−0.001 pp。

R13 的 S-node 和 S-graph 各覆盖 36 块。SourceFusion 和 GRADE 各覆盖 Mutagenicity 边数的两个方向，配对保留在逐块表。各历史参照均按封存 ID 对齐。

## 归档

执行提交 `439ee60`。12 个源模型的全 S BN 矩、遮点恢复头、掩码位置、144 个梯度方向和逐图贡献都在 FIT 中封存。随后用原始未遮 H 图计算 216 组有限步预测及解析右方向导数，正向差分最大绝对差为 **1.24×10⁻¹³**。全部 H 预测写入 OBS 后才打开 H 类别标签。归档核验复算 436 个端点、724 条配对以及所有方向。18 项相关工程回归通过。

[执行配置](results/attempt-001/executed-config.json) · [FIT](results/attempt-001/FIT.json) · [恢复头、统计和梯度](results/attempt-001/fit) · [OBS](results/attempt-001/OBS.json) · [FINAL](results/attempt-001/FINAL.json) · [验证记录](results/attempt-001/verification.json)

[逐块配对](results/attempt-001/paired-blocks.csv) · [方向配对与区间](results/attempt-001/paired-summary.csv) · [局部与有限步指标](results/attempt-001/local-summary.csv) · [计算门槛](results/attempt-001/gate.json) · [运行日志](results/attempt-001/run.log) · [工程回归](results/attempt-001/engineering-tests.log)
