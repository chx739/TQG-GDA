# R10 全方向结构参照结果

Mutagenicity 边数划分上，源选模 WL 的普通 Accuracy 为 **71.49%**，高于 B0 **2.44 pp**，十二个方向中十个方向的五种子均值更高。节点数划分上的差为 **+0.12 pp**，PROTEINS 边数划分上的差为 **−2.18 pp**。后续 Mutagenicity 图级适配可把源选模 WL 作为独立的源监督参照，保留 B0、SourceCal 以及匹配的同架构对照。PROTEINS 同时保留大小参照与分类平衡性指标。数据为三个设置共 180 个留出开发块。[登记协议](structural-reference.md) · [覆盖核验](../results/attempt-003-structural/coverage.json)

## 五种子 Accuracy

每个迁移方向先在五个种子上取均值，再对十二个方向等权。各方向的均值和样本 SD，以及 BA、两类召回、AUROC、NLL、相对 B0 的纠正与损害计数，见完整表。[端点汇总](../results/attempt-003-structural/endpoint-summary.csv) · [逐块端点](../results/attempt-003-structural/endpoints.csv)

| 设置 | B0 | SourceCal | 固定 WL | 源选模 WL | 大小 |
|---|---:|---:|---:|---:|---:|
| Mutagenicity 边数 | 69.04% | 69.23% | 63.81% | 71.49% | 58.25% |
| Mutagenicity 节点数 | 70.87% | 71.09% | 64.08% | 70.99% | 58.56% |
| PROTEINS 边数 | 55.46% | 54.99% | 51.67% | 53.29% | 65.60% |

下表为同图同种子的 Accuracy 差及 95% 区间，单位为百分点。沿用 R11 按目标域及类分层的 2,000 次配对重抽样，同一目标域的全部方法、源方向及种子共用图权重。区间条件于这批已拟合模型。[配对均值、SD 与区间](../results/attempt-003-structural/paired-summary.csv) · [逐块纠正与损害](../results/attempt-003-structural/paired-blocks.csv)

| 比较 | Mutagenicity 边数 | Mutagenicity 节点数 | PROTEINS 边数 |
|---|---:|---:|---:|
| 源选模 WL − B0 | +2.44 [1.43, 3.38] | +0.12 [−0.74, 1.04] | −2.18 [−3.73, −0.60] |
| 源选模 WL − SourceCal | +2.25 [1.25, 3.19] | −0.10 [−0.97, 0.83] | −1.70 [−3.05, −0.31] |
| 源选模 WL − 大小 | +13.23 [11.49, 14.96] | +12.43 [10.78, 14.16] | −12.32 [−14.39, −10.34] |
| 固定 WL − B0 | −5.23 [−6.84, −3.65] | −6.79 [−8.33, −5.19] | −3.79 [−5.63, −1.91] |
| 源选模 WL − 固定 WL | +7.68 [6.23, 9.15] | +6.91 [5.49, 8.28] | +1.61 [0.38, 2.86] |

选模 WL 相对 B0，在节点数划分上五向提升、七向下降，在 PROTEINS 上三向提升、九向下降。相对大小参照，Mutagenicity 的两个设置分别有 12/12 和 11/12 个方向提升。PROTEINS 上大小参照的高 Accuracy 对应 51.56% BA、57.76% AUROC，60 块中有 38 块把整个 H 判为同一类。选模 WL 的 BA 为 56.36%，AUROC 为 65.84%。全部 60 源组中，WL 有 38 组选中 `C=100`，12 组选中 `C=10`，7 组选中 `C=1`，3 组选中 `C=0.1`。结果描述源图的监督分类能力。[全部 V 候选与选模](../results/attempt-003-structural/v-candidates.csv)

## 可核验的历史对照

原生 R10 的三个参照各覆盖 Mutagenicity 边数的两个方向，共 10 块。选模 WL 相对 GRADE，在 M0→M1 和 M0→M3 分别高 **6.06** [2.28, 9.97] 和 **10.06** [5.03, 14.73] pp。相对 SourceOnly 分别高 **5.73** 和 **10.53** pp，相对 SourceFusion 分别高 **3.65** 和 **1.89** pp。原生 R10 绑定 R7 attempt-020/P，结构参照绑定 attempt-016/C。每块在原档案确认原始 manifest 的 SHA256，再核对两方的 S/V/A/H ID、标签身份、R5 文件和源 checkpoint 来源，B0 与 SourceCal logits 的类间差及预测类别逐图一致。[配对方向表](../results/attempt-003-structural/paired-summary.csv) · [逐块 P/C 角色证明](../results/attempt-003-structural/roles.json)

R13 的同图同角色比较限于种子 3405，共 36 块，下表单位为 pp。每格对该设置的十二个方向等权。[R13 原报告](../../r13_目标图归一化与分类排序_20261121/report.md) · [同种子区间](../results/attempt-003-structural/paired-summary.csv)

| R13 臂 − 同种子源选模 WL | Mutagenicity 边数 | Mutagenicity 节点数 | PROTEINS 边数 |
|---|---:|---:|---:|
| A_node | −4.15 | −5.21 | +3.35 |
| A_graph | −4.06 | −5.56 | +3.08 |
| S_node | −1.14 | −0.55 | +4.76 |
| S_graph | −0.81 | −0.50 | +5.01 |

[R14 与 WL 同种子配对](structural-r14-comparison.md)覆盖三个设置共 36 块。

## 执行与封存

执行版本为 `e4247032dbf91611c02814695bdf8ff0efabb387`。60 组模型各复用到三个 H，全部 480 个源验证指标随 FIT 封存。H 图拓扑在 FIT 之后按原 ID 推断。OBS 固定全部预测与谱系后，评分独立读取 12 个目标留出标签组。核验重载全部封存成员，确认 360 个 B0/SourceCal 端点与 R11 一致，以及 12 组重抽样权重的 SHA256 与 R11 一致，复核期间使用封存产物。[实际配置](../results/attempt-003-structural/executed-config.json) · [核验记录](../results/attempt-003-structural/verification.json) · [H 读口](../results/attempt-003-structural/label-access.json) · [FIT](../results/attempt-003-structural/FIT.json) · [OBS](../results/attempt-003-structural/OBS.json) · [FINAL](../results/attempt-003-structural/FINAL.json)

Attempt-002 在线搜索报错后停在源拟合，报告的梯度无穷范数为 1.83×10⁻⁹。Attempt-003 为 `ABNORMAL` 的候选从相同初值重试，使用原有 10⁻⁷ 驻点容差作为提前收敛阈值，并在返回系数处复核原目标。两次恢复均发生在大小臂。18 项 R10 工程测试通过。R12–R14 档案索引补齐后复查目录规范，唯一失败项为历史 R5 缺少 `src/__init__.py`。[失败尝试](../results/attempt-002-structural/execution-failure.json) · [成功运行日志](../results/attempt-003-structural/run.log) · [工程测试](../results/attempt-003-structural/engineering-tests.log)

完整模型、180 个逐图预测文件、验证脚本与测试日志保存在运行档案。仓库镜像封存清单、候选值、角色审计和结果表。档案索引列出全部成员的 SHA256 和字节数。[档案索引](../results/structural-reference-archive.json)
