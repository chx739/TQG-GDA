# TQG v2 实现与使用

依据 [v2 设计稿](TQG_Graph_Level_GDA_Design_v2.md) 实现。旧版方法、配置和历史结果保留；设置 `method=tqg_v2` 才启用新方法。此处记录实现与工程验证，不代表已经证明目标分类收益。

## 公式与代码

| 文档 | 实现 | 处理 |
|---|---|---|
| §3.2 | `feedback_v2.contrast_basis` | 固定 Helmert 类别对比基，二分类也保留两个 logits |
| §3.4–3.5 | `ParameterSnapshot`、`target_space` | 全部可训练参数，包括分类头；目标 Jacobian 薄 SVD，保留全部数值非零方向 |
| §3.5 | `ParameterSnapshot.response` | 源图逐方向 JVP；目标响应复用本轮目标 Jacobian；不使用有限差分 |
| §3.7 | `signature_kernels` | 完整图表示与响应上的 RBF 核；两个平方带宽取源查询成对平方距离的中位数，统一用于 SS/ST/TT |
| §3.9–3.10 | `query_matrices`、`solve_calibration` | 图优先的类别对比块排列，平均误差＋最大方向误差＋权重正则 |
| §3.11 | `solve_calibration` | 分解标量源核，条件 Sylvester 求解；对偶投影上升及回溯；每个候选 Ω 都重新求自己的 W |
| §3.12 | `build_gradient_feedback` | 求出 W 后读取真实源残差；`d=W r_S`，不额外除以源查询图数 |
| §3.13 | `parameter_feedback_loss`、`TQGTrainer` | 参数坐标平方损失，不除以秩；源 CE 与反馈合并后一次无动量、无 weight decay 的 SGD 更新 |
| §3.14 | 查询模块与训练模块边界 | Jacobian、基、签名、核、权重、源残差全部停止梯度；只对源 CE 和当前参数坐标损失反传 |

所有图使用原图的全部节点与边。每步从查询池抽取多少张图由 `query_batch_size` 决定；大于池大小时使用全池，仍保留 fit/query 不重叠。源查询批次必须至少两张图。

数值秩阈值为 `max(J.shape) * eps(J.dtype) * sigma_max`，不提供固定 K 或主动截断秩的开关。基于当前批次的完整性不等于覆盖整个目标域。源和目标均使用 `eval()` 固定 dropout 和 BN 状态，全部可训练参数正常更新。

核矩阵与内部校准使用 FP64，GNN/Jacobian 支持 FP32 和 FP64。实现存储目标 Jacobian 和参数基，不形成 P×P 矩阵，也不显式展开类别块核。源查询响应采用逐方向 JVP，尚未做吞吐优化。

## 配置与对照

| 配置 | 含义 |
|---|---|
| `configs/v2/tqg_v2.json` | 文档主版本：表示＋响应、联合校准、SGD |
| `configs/v2/mean_only_krr.json` | 去掉最大方向误差项的普通核岭回归对照 |
| `configs/v2/representation_only.json` | 核只使用图表示，仍用目标梯度定义查询 |
| `configs/v2/source_only_sgd.json` | 相同源 fit 起点、相同源批次调度的继续源监督 SGD |
| `configs/v2/source_all_sgd.json` | 另用全部源训练标签预训练、继续训练的全源监督 SGD 基线 |
| `configs/v2/pseudo_labels_sgd.json` | 相同 SGD 的置信度伪标签基线 |
| `configs/v2/permuted_labels.json` | 查询标签在每批内打乱的负对照 |

默认 `lambda_w=0.01`；gap 绝对容差 1e-8、相对容差 1e-6，最多 200 次内部迭代。谱误差项与平均项的相对权重固定为文档值。学习率和正则是启动默认值，尚未验证最优，不能直接把旧版 Adam 的学习率视为等效 SGD 步长。

冻结源模型基线直接使用源预训练的 `best.pt`，无需适配配置。全源监督基线使用独立的 all_source 预训练及预处理统计，不能复用 fit-only 起点后声称全源预训练；它与 v2 的差异包含监督样本与预训练预算。

每步日志包括有效秩、核尺度、源目标核交互、校准 primal/dual/gap、是否收敛、源残差和系数范数，以及反馈梯度与 SGD 实际步长的等式误差。gap 是内部代理目标的求解误差，不是目标预测置信度。

达到内部迭代上限时保留有限近似解，明确记录 `converged=false` 和终止原因。非有限值、超出舍入误差的非 PSD 核或求解异常会将本轮反馈置零并记录原因。秩为零或逐图源残差全部为零时反馈为零。

## 运行入口

使用现有通用入口；无需旧研究目录的控制器或 TensorBoard 运行时。以下相对路径均以仓库根目录为工作目录。

```bash
python scripts/run_experiment.py all \
  --data-path /path/to/PROTEINS/processed/data.pt \
  --source-split data/splits/PROTEINS_official_cache_source.json \
  --target-index idx/idx_PROTEINS_1.npy \
  --target-policy transductive-unlabeled \
  --config configs/v2/tqg_v2.json \
  --output /path/outside/repo/v2_run
```

该示例使用 100 epoch 源预训练，而不是历史 PROTEINS 的 700 update 预算。要复用历史起点，使用 `adapt --source-checkpoint .../best.pt`，并把配置中的 `warmup` 字段改成该 checkpoint 的原始配置，包括 `update_budget` 和验证间隔；模型、源划分、预处理及 dtype 也必须匹配。只能使用源预训练 checkpoint 初始化 v2，不能把已适配模型冒充源起点。

断点续训使用 `adapt --resume-adapt .../step_N.pt`，同时传入相同的 `--source-checkpoint` 和新的输出目录。恢复参数、SGD、数据采样及打乱标签 RNG，保留 checkpoint 所在目录的前 N 步历史；不会重新运行预训练。

训练入口不评价目标标签。目标训练批次若携带标签会被拒绝；模型选择只看源验证 CE，末步与源验证选点分别保存。原版带运行目录依赖的批量实验控制器尚未改写为 v2，不能通过原控制器的旧 arm 名称启动 v2。

## 工程验证

2026-10-28完成[目标batch16/32/64敏感性实验](experiments/mutagenicity_target_batch_20261028/结果解读.md)：固定源fit16、query196和200步，新增30次适配。增大batch降低反馈范数波动，但没有获得跨任务稳定的分类增益；训练数值代码未改动。

2026-10-28完成了[对照设计稿的代码审查](validation/v2_review_20261028/README.md)：116项测试、5例独立稠密原始问题求解及运行快照/日志复核通过；没有修改训练算法。该报告同时说明KRR有效正则对照的局限。

执行：

```bash
python -m unittest discover -s tests -v
python scripts/smoke_v2.py --output /path/outside/repo/v2_smoke
```

测试覆盖多分类类别坐标、完整空间重构真实 CE 梯度、JVP 与显式 Jacobian 一致、块矩阵排列、KRR 闭式解、联合校准的独立解析例子、秩一退化、非收敛 gap、坐标旋转不变性、停止梯度、零残差/无支持/数值异常，以及 GIN/GINE、FP32、目标标签隔离、SGD 更新与精确恢复。

2026-10-26 另复用种子 3405 的既有源模型，对 PROTEINS 和 Mutagenicity 的 0→1 各执行两步 v2 与配对源 SGD 检查。分别使用全部 51/196 张独立 query，目标批次 16 张，均保留 16 个有效方向。四次联合校准均在 2–4 次内部迭代达到容差；反馈梯度等式误差为 0，SGD 步长等式误差小于 7e-16。两数据集的 v2 参数均与源 SGD 基线不同。

上述工程检查没有评价目标准确率，不能用于声称 v2 有效。随后完成了 [Mutagenicity 三方向五种子效果验证](experiments/mutagenicity_v2_20261026/run_report.md)：105次适配、225行开发评价，当前默认配置未获得跨任务稳定收益。超参数选择、等计算预算比较、与旧版及外部方法的充分对照仍待开展。早期工程验证摘要见 [validation/v2_20261026/](validation/v2_20261026/README.md)。
