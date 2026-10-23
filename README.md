# TQG-GDA 实验框架

根据用户提供的 TQG 设计稿，实现 graph-level、source-access、closed-set UDA。
沿用 SEGA-GDA 的 GIN/GINE 结构、原始域索引和 TU/PyG 缓存读取约定；训练核心独立实现。
推断只保留普通 GNN 与线性分类头，一张图一次前向。

这是可运行的实验起点。已在 PROTEINS 0→1 上完成四轮真实数据开发实验；当前默认 TQG 尚未优于合理内部基线，不能据此声称方法已经有效。
实验进度见 [有效性验证路线](docs/TQG_有效性验证路线.md)。[E2 报告](/home/hxchen/TQG_runs/20261023_proteins_followup_v2/run_report.md) 包含源方向、响应消融、打乱 query 标签和匹配更新步数基线；[E3 报告](/home/hxchen/TQG_runs/20261024_source_generalization_e3/run_report.md) 进一步检查 query 数量与 ridge 强度，并在源域选择封存后补做 20 次完整适配。
[E4 报告](/home/hxchen/TQG_runs/20261024_residual_baselines_e4/run_report.md) 比较平均残差与带常数项的响应回归，新增 20 次适配。它们是额外对照，默认 TQG 和原设计稿保持不变；相关数学与使用说明见 [E4 残差基线](docs/E4_残差基线说明.md)。

首个机制实验已单独设计：[E1：目标方向是否比随机方向更会纠错](docs/E1_首个验证实验.md)。
专用配置 `configs/diagnostics/e1_target_vs_random.json` 与 `scripts/run_diagnostic.py` 在固定源模型上做配对比较；
它只读取保留诊断子集的目标标签，最终目标标签另行留出，不属于常规无标签训练。

## 1. 训练逻辑

```text
固定源 train/validation
  └─ 将 source train 划分为 fit / query
       ├─ 仅 fit：源分类预训练，按独立 source validation CE 选择初始模型
       └─ 每轮适配：
            target 输出投影 → 编码器参数方向 → target JVP 规范化
            相同方向探测完整 source query / target 图，得到奇、偶响应
            两域共享列 RMS → 用 source query 的 Y-p 拟合岭回归系数
            组合 target 响应 → 固定参考 logits = 当前 logits + 反馈
            source fit CE + λ_fb × target 平方损失 → 一次 Adam 更新
```

- source query 参与训练，不是 validation；标准 TQG 不用它做 warmup 或源 CE。
- 输入连续属性的统计仅在 warmup 可访问的源池上拟合，验证/目标标签不参与。
- 探测只改变 encoder 的临时参数；分类头不探测，但最终训练正常更新。
- 用 `torch.func.functional_call` 评估快照与扰动，不修改真实模型或优化器。
- JVP 使用 `torch.autograd.functional.jvp`，兼容 PyG scatter；临时 AD 图不进入最终更新。
- 方向、响应、残差、系数、参考 logits 均停止梯度；默认 TQG 无有效方向时执行源分类更新。
- 适配使用 eval 前向固定 dropout/BN 状态，可学习参数仍训练；本骨干没有 BN/dropout。
- 每一步重建反馈，不在同一份参考输出下重复更新。
- 所有训练选模只用 source validation CE，初始模型也是候选；`selected_step=0` 是合法结果。

## 2. 环境与快速检查

已在 Python 3.12 / PyTorch 2.5.1 / PyG 2.6.1 的 CPU 环境验证。
先安装适合机器的 CPU/CUDA PyTorch，再安装其余依赖；本项目不需要 SEGA 的几何求解器或 SciPy。

```bash
cd /home/hxchen/TQG_GDA
python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python scripts/smoke_test.py --output /tmp/tqg-smoke
```

当前工作区已有可用解释器：`/home/hxchen/anaconda3/envs/bfn4jet/bin/python`。
也可在根目录 `pip install -e .`。脚本从任意工作目录调用均可，默认资产路径由脚本位置解析。
新输出目录必须不存在，避免覆盖旧实验。

## 3. 真实数据

图缓存不随包提供。默认使用原 SEGA 包的 PROTEINS 边数域 0→1：

- `idx/idx_PROTEINS_0.npy`：源域；保持原 250 train / 28 validation。
- 在这 250 张源训练图中，按类别固定划分 fit/query；比例与种子记录在 `data_roles.json`。
- `idx/idx_PROTEINS_1.npy`：整个无标签 transductive 目标池。
- `data/splits/PROTEINS_idx_0_source.json` 锁定原缓存的 SHA256。

支持原生 TU/PyG 三元组或四元组缓存：`(data_dict_or_Data, slices, sizes[, DataClass])`。
只读取已有缓存，不自动下载数据。属性型数据集应设置 `use_node_attributes=true`。
其他数据集需要相应源划分清单；现有 molhiv 索引不表示已经支持 OGB 原生加载格式。

如果只有重新处理的数据，必须明确创建新协议，不能覆盖原 manifest 的哈希来冒充原实验：

```bash
python scripts/prepare_source_split.py \
  --data-path /path/to/PROTEINS/processed/data.pt \
  --source-index /home/hxchen/TQG_GDA/idx/idx_PROTEINS_0.npy \
  --dataset PROTEINS --seed 431 --output /tmp/new_source_protocol.json
```

这会生成**新的**源 train/validation 划分；要与原划分比较，仍需原缓存或人工核验 ID 内容。

## 4. 从零运行

```bash
python scripts/run_experiment.py all \
  --data-path /path/to/PROTEINS/processed/data.pt \
  --target-policy transductive-unlabeled \
  --config /home/hxchen/TQG_GDA/configs/tqg_development.json \
  --device cpu --output /tmp/tqg-proteins-0-to-1
```

若使用新协议，额外传 `--source-split /tmp/new_source_protocol.json`。
其他任务可指定 `--dataset`、`--source-split`、`--source-domain`、`--target-domain`、`--index-family idx|node`，
或直接给 `--target-index`。

源预训练可复用到**相同 source fit/query、架构、预处理和 warmup 配置**的不同目标任务或消融：

```bash
python scripts/run_experiment.py source \
  --data-path /path/to/PROTEINS/processed/data.pt --output /tmp/tqg-source
python scripts/run_experiment.py adapt \
  --data-path /path/to/PROTEINS/processed/data.pt \
  --source-checkpoint /tmp/tqg-source/source/best.pt \
  --target-policy transductive-unlabeled --output /tmp/tqg-adapt
```

SEGA 原 source checkpoint 通常用过完整 source train，不满足 TQG 的 query 留出要求；
本框架拒绝把它自动当成标准 TQG 起点，应重新做 fit-only warmup。

## 5. 输出与恢复

```text
run/
├─ resolved_config.json
├─ source/       best.pt、last.pt、history.jsonl、data_roles.json、environment.json、result.json
└─ adaptation/   selected.pt、last.pt、step_N.pt、history.jsonl、data_roles.json、environment.json、result.json
```

日志包括源残差 RMS、残差拟合误差、有效方向数、响应列数、系数范数、目标反馈 RMS、
丢弃原因、源 CE、反馈损失和批次 ID。`explicit_forward_evaluations` 只统计显式前向，不包括 JVP 内部 AD 成本。
默认 float64 是为了降低有限差分抵消误差；float32 也经过合成验证，但应检查 rho 的敏感性。

在完整 step 边界恢复适配，使用新输出目录：

```bash
python scripts/run_experiment.py adapt \
  --data-path /path/to/PROTEINS/processed/data.pt \
  --source-checkpoint /tmp/tqg-source/source/best.pt \
  --resume-adapt /tmp/tqg-adapt/adaptation/step_50.pt \
  --target-policy transductive-unlabeled --output /tmp/tqg-resumed
```

配置、数据角色、源 checkpoint 和设备必须匹配；仅允许增加总适配 steps。
相同预定预算中从中间 step 恢复可逐位重现模型和选择状态。
若延长原本已结束的预算，原终点的额外验证可能成为选模候选，和一次性更长训练的选模时间表需分别记录。
源训练可用 `--resume-source .../source/last.pt` 在 epoch 边界恢复，仅允许延长 warmup epochs。
保存了 optimizer、每种数据角色的采样 RNG、探测 RNG 和全局 RNG。

## 6. 独立评估目标标签

训练代码不导入目标评估器。训练完成后显式运行：

```bash
python scripts/evaluate.py \
  --checkpoint /tmp/tqg-adapt/adaptation/selected.pt \
  --data-path /path/to/PROTEINS/processed/data.pt \
  --target-index /home/hxchen/TQG_GDA/idx/idx_PROTEINS_1.npy \
  --output /tmp/tqg-target-evaluation.json --predictions /tmp/tqg-predictions.csv
```

输出 accuracy、CE、混淆矩阵、各类召回率、balanced accuracy；二分类且两类都有样本时输出 AUROC。
缺类指标为 JSON `null`，不会写 NaN。目标测试指标不得用于训练选模或反复调参。

## 7. 已提供的消融

| 配置 | 用途 |
| --- | --- |
| tqg_development.json | 目标投影方向 + 奇/偶响应，设计稿默认主线 |
| source_only_same_fit.json | 相同 fit 数据、源 CE 训练的受控基线 |
| source_only_all_labels.json | warmup 与后续源 CE 均使用 fit+query 的全部源训练标签，较强基线 |
| random_probes.json | 随机参数方向；仍按目标输出响应规范化 |
| source_probes.json | 源图输出投影生成方向；仍按目标输出响应规范化 |
| odd_only.json | 只用奇响应 |
| linear_jvp.json | 用精确局部一阶 JVP 代替有限响应 |
| odd_only_8cols.json / linear_jvp_8cols.json | K=8，一阶/奇响应列数与 K=4 的 Odd+Even 匹配；计算成本仍不同 |
| permuted_query_113.json / 227 / 349 | 仅打乱求组合权重时的 query 标签，正常 fit CE 标签保持真实 |
| source_all_700updates.json | 全部源标签 warmup 严格 700 次更新、每 7 步验证，随后 200 步 CE；匹配更新次数，未匹配图次/墙钟时间 |
| mean_residual_q16.json / mean_residual_q32.json | source query 平均残差作为所有目标图的共同纠错，不做参数探测 |
| intercept_q16.json / intercept_q32.json | 在原响应回归中加入不惩罚的类常数项；query=16/32 |

```bash
python scripts/run_matrix.py --data-path /path/to/PROTEINS/processed/data.pt \
  --seeds 3405 3406 3407 --output /tmp/tqg-ablation-matrix
```

矩阵运行不会自动读取目标标签。不同响应模式的列数、时间成本不同；固定 K 不等于固定列数或固定算力。
source-only 同步数基线也不等于同时间预算基线。论文实验需要分别匹配并报告这些预算。
先跑小配置确认资源，再决定正式 seeds、任务矩阵和超参预算。

E2 的 `scripts/run_followup.py` 读取已冻结的实验方案，校验数据、执行代码和起点 checkpoint 指纹后运行。
作业只有在输入指纹及登记输出哈希都一致时才可复用；部分完成目录保留，不能靠一个完成标记跳过训练。
分类结果分别记录实际模型步数、该 checkpoint 的源验证 CE、历史最优源 CE；固定末步与源验证选模结果分开报告。
随机打乱种子是模型种子内的重复，汇总时先在同一模型内平均，不能当作独立模型种子。
固定模型诊断可通过 `diagnostics.heldout_response_features` 把同一方向和同一 RMS 缩放应用于另一批图，不重新拟合尺度。
`warmup.update_budget` 是独立的精确更新预算模式，暂不支持 epoch 边界的源训练恢复；普通 warmup 的恢复不受影响。

## 8. 代码与设计稿对应

| 设计稿 | 实现 |
| --- | --- |
| 3.2 中心化 logits | `tqg_gda/models.py` |
| 3.3 fit/query 与源初始化 | `tqg_gda/data.py`、`training.run_warmup` |
| 3.4–3.6 快照、方向、双向响应 | `feedback.Snapshot`、`feedback.build_feedback` |
| 3.7 共享 RMS | `feedback.shared_response_scale` |
| 3.8 岭回归 | `feedback.fit_residual` |
| 3.9–3.11 目标参考与单步更新 | `feedback.feedback_loss`、`training.TQGTrainer.step` |
| 3.13 单次 GNN 推断 | `GraphClassifier.forward`、独立 `scripts/evaluate.py` |

完整原始设计稿放在 `docs/TQG_Graph_Level_GDA_Design.md`。
测试覆盖反馈梯度恒等式、编码器联合探测、分类头训练、无副作用查询、两域共享尺度、
岭回归目标、无效方向回退、目标标签隔离和完整训练恢复。
