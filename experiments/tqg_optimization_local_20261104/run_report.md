# A2-b：目标附近的源图是否更有用

Mutagenicity，0→1/2/3，五模型种子3405—3409；fit779/query196，源fit16/目标A16，B271。各快照都是原源模型的独立诊断；未重新预训练GNN，未进行连续TQG训练。±为五模型种子标准差，括号为正值种子数；不是五次独立数据集划分。开发A/B历史已观察，不能称盲测；final标签读取为0。

**结论：没有任何局部配置达到至少2/3任务通过的要求。** 不能从“距离与困难相关”直接推出“只保留近邻就能得到可靠反馈”。

## 实验方案

保持原JOINT读取器；H/HA读取核各比较64/128张、H近邻/HA近邻/五个均匀排列，保留全196。公共带宽固定196输入；FIXED使用λ=.01，NORMALIZED使用λ=.01×n/196。只用目标A输入选bank，不用B。120目标块、6960配置；前4块共3480配置进行独立一步。

## 全部局部配置与同数量均匀源图比较

五个均匀排列先在同一块平均，再做模型种子配对。误差降低存在原协议未明确的聚合口径：主列先逐块算比例，敏感性列先在种子内平均误差再算比例。两列完整公开，不按哪列更有利决定结论。

| 任务 | 配置 | 余弦提升 | 逐块误差降低% | 种子均值误差降低% | 主口径通过 | 敏感性通过 |
|---|---|---|---|---|---|---|
| 0_to_1 | HA_128_NEAR_HA_FIXED | 0.15904 ± 0.19143 (4/5) | -1.63068 ± 15.30564 (3/5) | 1.268 (3/5) | False | False |
| 0_to_1 | HA_128_NEAR_HA_NORMALIZED | 0.14733 ± 0.19941 (4/5) | -4.79130 ± 15.35842 (3/5) | -2.056 (3/5) | False | False |
| 0_to_1 | HA_128_NEAR_H_FIXED | -0.02660 ± 0.09466 (1/5) | -10.37756 ± 11.92506 (0/5) | -9.990 (2/5) | False | False |
| 0_to_1 | HA_128_NEAR_H_NORMALIZED | -0.03334 ± 0.08097 (1/5) | -12.32731 ± 13.31551 (1/5) | -11.926 (1/5) | False | False |
| 0_to_1 | HA_64_NEAR_HA_FIXED | -0.00089 ± 0.12402 (1/5) | -19.86618 ± 20.01127 (1/5) | -16.288 (1/5) | False | False |
| 0_to_1 | HA_64_NEAR_HA_NORMALIZED | -0.00809 ± 0.13574 (1/5) | -29.83297 ± 22.44224 (0/5) | -26.598 (0/5) | False | False |
| 0_to_1 | HA_64_NEAR_H_FIXED | -0.04588 ± 0.18380 (2/5) | -22.84311 ± 30.23250 (1/5) | -19.352 (1/5) | False | False |
| 0_to_1 | HA_64_NEAR_H_NORMALIZED | -0.03715 ± 0.22289 (2/5) | -27.78617 ± 31.21575 (1/5) | -24.360 (1/5) | False | False |
| 0_to_1 | H_128_NEAR_HA_FIXED | 0.02855 ± 0.16807 (2/5) | -27.84422 ± 20.46147 (1/5) | -18.929 (2/5) | False | False |
| 0_to_1 | H_128_NEAR_HA_NORMALIZED | 0.00768 ± 0.14534 (3/5) | -39.19802 ± 21.49884 (0/5) | -28.265 (0/5) | False | False |
| 0_to_1 | H_128_NEAR_H_FIXED | 0.14278 ± 0.09067 (5/5) | -25.21856 ± 46.68599 (2/5) | -29.975 (2/5) | False | False |
| 0_to_1 | H_128_NEAR_H_NORMALIZED | 0.13818 ± 0.08487 (5/5) | -27.53254 ± 49.05229 (2/5) | -31.041 (2/5) | False | False |
| 0_to_1 | H_64_NEAR_HA_FIXED | -0.01058 ± 0.09580 (2/5) | -95.72824 ± 120.66463 (0/5) | -80.008 (1/5) | False | False |
| 0_to_1 | H_64_NEAR_HA_NORMALIZED | 0.02668 ± 0.10690 (4/5) | -120.48240 ± 145.01346 (1/5) | -91.394 (1/5) | False | False |
| 0_to_1 | H_64_NEAR_H_FIXED | -0.14712 ± 0.09196 (0/5) | -44.58877 ± 37.02006 (1/5) | -30.262 (1/5) | False | False |
| 0_to_1 | H_64_NEAR_H_NORMALIZED | -0.18408 ± 0.14965 (1/5) | -47.62450 ± 37.39428 (1/5) | -41.561 (1/5) | False | False |
| 0_to_2 | HA_128_NEAR_HA_FIXED | 0.03563 ± 0.06078 (4/5) | -0.73040 ± 6.63018 (2/5) | 0.542 (3/5) | False | False |
| 0_to_2 | HA_128_NEAR_HA_NORMALIZED | 0.03802 ± 0.07368 (3/5) | -2.47420 ± 7.78128 (2/5) | -1.400 (3/5) | False | False |
| 0_to_2 | HA_128_NEAR_H_FIXED | -0.02368 ± 0.08766 (3/5) | -2.83404 ± 10.10753 (2/5) | -1.668 (2/5) | False | False |
| 0_to_2 | HA_128_NEAR_H_NORMALIZED | -0.02378 ± 0.08737 (3/5) | -4.14293 ± 10.49396 (2/5) | -3.421 (2/5) | False | False |
| 0_to_2 | HA_64_NEAR_HA_FIXED | 0.15132 ± 0.20486 (4/5) | -10.51040 ± 20.81267 (2/5) | -7.460 (1/5) | False | False |
| 0_to_2 | HA_64_NEAR_HA_NORMALIZED | 0.16250 ± 0.22275 (4/5) | -20.75786 ± 23.94863 (1/5) | -19.550 (1/5) | False | False |
| 0_to_2 | HA_64_NEAR_H_FIXED | 0.00465 ± 0.08184 (3/5) | 2.08420 ± 11.23402 (3/5) | 8.533 (3/5) | False | False |
| 0_to_2 | HA_64_NEAR_H_NORMALIZED | 0.00516 ± 0.09199 (3/5) | -0.98220 ± 13.69821 (2/5) | 6.353 (4/5) | False | False |
| 0_to_2 | H_128_NEAR_HA_FIXED | 0.02724 ± 0.25438 (2/5) | -31.75280 ± 40.79646 (1/5) | -21.701 (2/5) | False | False |
| 0_to_2 | H_128_NEAR_HA_NORMALIZED | 0.03147 ± 0.26543 (2/5) | -38.28486 ± 42.89076 (1/5) | -28.906 (2/5) | False | False |
| 0_to_2 | H_128_NEAR_H_FIXED | 0.17865 ± 0.18853 (4/5) | 4.89148 ± 14.12452 (3/5) | 10.295 (4/5) | False | True |
| 0_to_2 | H_128_NEAR_H_NORMALIZED | 0.18404 ± 0.20938 (4/5) | 0.01660 ± 13.23113 (2/5) | 8.120 (4/5) | False | False |
| 0_to_2 | H_64_NEAR_HA_FIXED | 0.21371 ± 0.35811 (3/5) | -56.38343 ± 51.06200 (0/5) | -53.907 (0/5) | False | False |
| 0_to_2 | H_64_NEAR_HA_NORMALIZED | 0.21140 ± 0.28701 (4/5) | -75.33102 ± 51.38507 (0/5) | -70.349 (0/5) | False | False |
| 0_to_2 | H_64_NEAR_H_FIXED | 0.02745 ± 0.20540 (2/5) | -23.34442 ± 34.82121 (2/5) | -9.504 (2/5) | False | False |
| 0_to_2 | H_64_NEAR_H_NORMALIZED | 0.06381 ± 0.13548 (4/5) | -26.94928 ± 37.53073 (1/5) | -8.187 (3/5) | False | False |
| 0_to_3 | HA_128_NEAR_HA_FIXED | -0.07378 ± 0.12113 (1/5) | -17.68213 ± 9.29651 (0/5) | -19.312 (0/5) | False | False |
| 0_to_3 | HA_128_NEAR_HA_NORMALIZED | -0.07526 ± 0.11975 (1/5) | -20.45744 ± 8.87540 (0/5) | -22.540 (0/5) | False | False |
| 0_to_3 | HA_128_NEAR_H_FIXED | 0.02811 ± 0.14728 (3/5) | -6.08146 ± 8.28086 (1/5) | -7.810 (1/5) | False | False |
| 0_to_3 | HA_128_NEAR_H_NORMALIZED | 0.02777 ± 0.13807 (3/5) | -8.32327 ± 8.66322 (1/5) | -10.409 (1/5) | False | False |
| 0_to_3 | HA_64_NEAR_HA_FIXED | -0.13248 ± 0.03437 (0/5) | -27.88381 ± 17.25277 (0/5) | -29.560 (0/5) | False | False |
| 0_to_3 | HA_64_NEAR_HA_NORMALIZED | -0.13609 ± 0.04817 (0/5) | -33.02668 ± 19.50211 (0/5) | -35.483 (0/5) | False | False |
| 0_to_3 | HA_64_NEAR_H_FIXED | -0.04448 ± 0.09949 (2/5) | -6.49419 ± 6.61546 (1/5) | -5.438 (1/5) | False | False |
| 0_to_3 | HA_64_NEAR_H_NORMALIZED | -0.06323 ± 0.09459 (1/5) | -13.58061 ± 10.49330 (1/5) | -13.952 (1/5) | False | False |
| 0_to_3 | H_128_NEAR_HA_FIXED | 0.07082 ± 0.19353 (4/5) | -34.74644 ± 44.89650 (0/5) | -24.169 (1/5) | False | False |
| 0_to_3 | H_128_NEAR_HA_NORMALIZED | 0.07858 ± 0.17277 (4/5) | -38.87423 ± 44.99859 (0/5) | -29.358 (1/5) | False | False |
| 0_to_3 | H_128_NEAR_H_FIXED | 0.12244 ± 0.12285 (4/5) | -4.45033 ± 8.92620 (1/5) | -4.619 (1/5) | False | False |
| 0_to_3 | H_128_NEAR_H_NORMALIZED | 0.13408 ± 0.13563 (4/5) | -6.42002 ± 7.56561 (1/5) | -8.700 (2/5) | False | False |
| 0_to_3 | H_64_NEAR_HA_FIXED | -0.09189 ± 0.19027 (1/5) | -66.38197 ± 64.84208 (0/5) | -74.427 (0/5) | False | False |
| 0_to_3 | H_64_NEAR_HA_NORMALIZED | -0.06438 ± 0.28428 (1/5) | -82.88677 ± 69.85155 (0/5) | -94.121 (0/5) | False | False |
| 0_to_3 | H_64_NEAR_H_FIXED | 0.09605 ± 0.29969 (2/5) | -2.38760 ± 16.19707 (2/5) | -0.660 (2/5) | False | False |
| 0_to_3 | H_64_NEAR_H_NORMALIZED | 0.10781 ± 0.28150 (4/5) | -12.77671 ± 12.08773 (0/5) | -17.341 (1/5) | False | False |

后一口径只有0→2的H_128_NEAR_H_FIXED通过一个任务（10.295%误差下降，4/5同向）；其余任务没有共同支持。前一口径所有配置均0/3。两种口径都不支持扩大局部路线。最终候选LOCAL按源端规则进入独立A6，其0/3结论另见A6报告，不按此处目标结果挑配置。

## 解释与边界

近邻共享输入特征，不保证共享标签条件规律；删除远源图同时减少监督约束。当前结果未证明距离完全无用，也未证明更多独立源标签无用。反而说明不能把旧支持分层相关性当成局部选择的因果收益。

## 审计与限制

独立逐条审计通过，聚合复核480项，最大差3.89e-16。普通模型副本重放6960次FULL/BASE参数步；null的系数与指标全部重放，但没有再次独立前向每一个null参数步。

保留主读出未收敛0条；教师内折未收敛0条，选中教师重拟合未收敛0条。不删除失败条目，不将工程通过等同研究通过。原始未收敛标记在各块profiles/教师元数据，summary中的覆盖率没有把有限未收敛结果偷偷丢弃。

[完整逐种子/快照指标](results.csv) · [门槛与配对汇总](results_summary.json) · [数值与权限审计](audit.json) · [独立聚合审计](summary_audit.json) · [冻结协议](protocol/method.md)

## Reproduction and export issues

Execution provenance: manifest.json; checkpoints/sample IDs are sealed in each worker blocks/*/ids.json. Recompute summary: `python optimization_feedback_report.py --root <stage>/A2b`; independently verify: `python optimization_summary_audit.py --root <stage>/A2b`. Numeric audit uses optimization_feedback_audit.py.

The audit script had a syntax error corrected before execution. The first summary export failed on missing control_valid in step logs; fixed by joining sealed gradient validity on token/profile/arm. No fitting, labels, coefficients or outcomes were changed. Two aggregation checks passed via direct invocation because pytest is not installed. Existing invalid or finite nonconverged records are retained.
