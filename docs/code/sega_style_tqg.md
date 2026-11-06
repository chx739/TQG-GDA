# TQG 分支：按 SEGA 范式重构 v1 / v2

更新日期：2026-11-06。开发分支为 `TQG`，其起点为 `eb39fd7c5433420bd51f14e6330ba5067d164456`。本轮不修改 `SEGA-ref`、`TSBA` 或默认分支；已有实验、冻结协议和本地汇报稿保持原状。

后续审查：[2026-11-07 独立代码 review 与合成检查](../validation/tqg_clean_review_20261107/README.md)。本次审查没有修改算法，尚未开展真实数据验证。

## 1. 采用什么范式，依据什么算法

参考为用户提供的 [SEGA-GDA 原样代码](../../references/sega-gda/README.md)，文件清单与 SHA 见[来源记录](../../references/sega-gda.provenance.json)。方法依据为 [TQG v1](../v1/method/design.md) 和 [TQG v2](../v2/method/design.md)。

| 环节 | 本轮实现 | 依据 |
|---|---|---|
| 代码组织 | config → data → encoder/classifier → pretrain → query/内部求解 → adapt → predict；pipeline 连接各阶段；run_task / run_all 为薄入口 | SEGA 组织方式 |
| 图编码器 | GIN 默认，可配置 GINE；每层 MLP 内有 BN，卷积后另有 BN；逐层 sum pooling 后拼接，再投影为图表示 | 新 SEGA 编码器；默认 3 层、hidden=64、embedding=16 |
| 分类器 | 图表示后接可训练线性头，按类别中心化 logits | 两份 TQG §3.2 |
| 初始化 | 只用固定 source fit 的普通 CE；Adam；独立 source validation 选择最低 CE epoch | 两份 TQG §3.3；SEGA 的源初始化/缓存阶段组织 |
| 源角色 | 保留既有 source train/validation 清单；train 按类别分成 fit/query；划分种子固定，与模型种子分开 | 两份 TQG §3.3；保留研究协议 |
| 适配状态 | 所有前向使用 eval，BN 运行统计和 dropout 固定；编码器、BN 仿射参数和线性头仍可训练 | 两份 TQG §3.4 |
| 参数执行 | 每轮新快照、新查询，源 CE + 一项反馈辅助损失；恰好一次更新 | v1 §3.11、v2 §3.13 |
| 推断 | 只保存 GNN、类别映射及输入预处理；一张完整图做一次前向 | v1 §3.13、v2 §3.16 |

SEGA 的 reference classifier、ref/fit/cal 三角色、目标伪标签筛选、冻结编码器后的几何适配、源风险约束和非线性接受/回退，属于 SEGA 的具体算法。把这些规则移植进 TQG 主版本会改变文档定义。本实现采用其模块和流水线范式，TQG 的更新以两份方法文档为准。v2 内部 Ω 求解仍按文档执行回溯；这与 SEGA 的网络/几何更新接受规则是不同层次。

新编码器与参考代码使用相同参数名称和层结构。在同权重、FP32 输入下，对 GIN/GINE 的训练态和 eval 态做逐元素一致性检查。唯一的通用数值适配是输入跟随模型参数 dtype，支持 FP64 探测；原 SEGA 会强制把输入转成 FP32。单批只有一个节点时无法训练 BN，预训练明确报错，不隐式替换层。

随机初始化前显式设定配置 dtype，构造结束后恢复调用方全局默认 dtype；包含 GINE 内部的边投影层。这样同一个配置和种子不会因为外部代码切换过默认精度而得到不同起点。

## 2. 精读入口

| 顺序 | 文件 | 负责什么 |
|---|---|---|
| 1 | [pipeline.py](../../tqg_gda/pipeline.py) | 查看完整任务如何串起来 |
| 2 | [config.py](../../tqg_gda/config.py)、[data.py](../../tqg_gda/data.py) | 配置检查、数据哈希、固定角色、目标无标签视图 |
| 3 | [encoder.py](../../tqg_gda/encoder.py)、[classifier.py](../../tqg_gda/classifier.py) | SEGA 图编码器 + TQG 线性分类头 |
| 4 | [pretrain.py](../../tqg_gda/pretrain.py) | source fit 预训练、source validation 选模、源缓存 |
| 5a | [v1/query.py](../../tqg_gda/v1/query.py) | v1 方向、有限响应、系数、输出反馈 |
| 5b | [v2/query.py](../../tqg_gda/v2/query.py)、[v2/calibration.py](../../tqg_gda/v2/calibration.py) | v2 完整作用空间、核、W 校准、参数反馈 |
| 6 | [adapt.py](../../tqg_gda/adapt.py) | 数据抽样和一次真实联合更新 |
| 7 | [predict.py](../../tqg_gda/predict.py)、[provenance.py](../../tqg_gda/provenance.py) | 独立推断和产物身份校验 |
| 8 | [专项测试](../../tests/test_sega_pipeline.py) | 可执行公式/状态/数据边界检查 |

旧 `feedback.py`、`feedback_v2.py`、`calibration_v2.py` 是导入兼容层，算法本体已迁至版本子目录。`models.py` / `training.py` 保留旧架构和旧 checkpoint 接口。已有阶段工作流仍通过旧接口运行，不会暗中替换它们的骨干。精确复现既有实验应使用对应归档中的代码快照。

## 3. v1 的逐节实现

| 文档 | 实现函数 | 必须保持的定义 |
|---|---|---|
| §3.2 | `GraphClassifier.forward` | 线性头；仅沿类别维中心化 |
| §3.3–3.4 | `load_task`、`pretrain_model`、`Adapter.step` | fit/query 不重叠；同一确定性快照；query 不参与预训练 |
| §3.5 | `build_feedback`、`Snapshot.jvp` | 目标随机零均值输出投影，编码器 VJP；按目标 JVP RMS 归一化；分类头探测为零 |
| §3.6–3.7 | `build_feedback`、`shared_response_scale` | 同方向 ±ρ；odd 除 2ρ、even 除 ρ²；两域共用每列 RMS，不跨图减均值 |
| §3.8 | `fit_residual` | `(ΨᵀΨ/bS + λI)a = Ψᵀ(Y-p)/bS`；求解方程，不显式求逆 |
| §3.9–3.10 | `feedback_loss` | 固定参考 `z_snapshot + ΨT a`；损失 `sum_classes(square)/2` 再按目标图求平均 |
| §3.11–3.14 | `Adapter.step` | 查询全部停止梯度；目标输出前向保留梯度；一次联合更新后丢弃反馈 |

主配置为 target 方向和 odd+even；默认 K=4、ρ=0.05、λa=0.01。旧 odd-only、linear、随机方向、源方向和额外截距等实验仍可从历史工作流使用，但新主版本不把它们作为默认算法。方向无效时丢弃该方向的两域响应；无有效方向或数值求解异常时只执行源 CE，并记录原因。

默认适配使用 SGD，便于验证实际参数步长。v1 文档允许 Adam，可通过 `adapt.optimizer` 显式切换；切换后不把实际步长声称为普通梯度下降增量。

## 4. v2 的逐节实现

| 文档 | 实现函数 | 必须保持的定义 |
|---|---|---|
| §3.2 | `contrast_basis` | 固定 Helmert 类别基，QᵀQ=I、QQᵀ=P |
| §3.4–3.5 | `ParameterSnapshot`、`target_space` | 全部可训练参数包含编码器和头；目标 Jacobian 薄 SVD；保留全部数值非零方向，没有 K 截断 |
| §3.5–3.7 | `response`、`signature_kernels` | 源 A 用 JVP；目标 A=JV；H+A 非线性核；源成对距离中位数决定共同尺度 |
| §3.8 | 无“自动判定成立”开关 | 四项迁移条件是研究假设，工程测试不能宣布成立；沿用既有诊断工作流 |
| §3.9–3.10 | `query_matrices` | graph-major 的 q 坐标块；B 含 1/bT；均值误差 + 最大方向误差 + 权重正则 |
| §3.11 | `solve_calibration` | Ω PSD、迹为 1；D=I+rΩ；仅分解标量源核，复用 Kronecker 结构；每个候选 Ω 重新求自身 W 后评估对偶目标 |
| §3.11 | gap/termination 日志 | gap 使用同一组 Ω/W；迭代或回溯上限保留有限近似解并明确报告未收敛 |
| §3.12 | `build_gradient_feedback` | W 求解不看源标签；随后读取 `rS=(Y-p)Q`；`d=W rS`，不再除 bS；估计梯度 `-Vd` |
| §3.13–3.14 | `parameter_feedback_loss`、`Adapter.step` | `0.5‖Vᵀ(θ-θsnapshot)-d‖²`，不除 rank；SGD 无动量、无 weight decay；只更新一次 |

默认 λW=0.01、绝对 gap 容差 1e-8、相对容差 1e-6、内部上限 200 次、回溯上限 40 次。前向/Jacobian 默认 FP32，可显式切换 FP64；核和校准使用 FP64。数值秩容差为 `max(J.shape) * machine_epsilon * 最大奇异值`。零秩反馈为零；明显非 PSD 或非有限校准走 source-only fallback 并记录。有限但未达到 gap 容差的结果会用于当前步，必须读取其 `converged=false`，不能当成精确求解。

## 5. 配置、运行和输出

示例配置：[Mutagenicity v1](../../configs/sega_style/mutagenicity_v1.json)、[Mutagenicity v2](../../configs/sega_style/mutagenicity_v2.json)。路径按仓库根目录解析。需要自行提供与已有源清单 SHA 一致的数据缓存；缓存不同必须先显式建立并审查新的源划分清单，不能修改旧清单冒充相同数据。

```bash
python -m workflows.tqg.run_task \
  --config configs/sega_style/mutagenicity_v2.json \
  --cache-path /path/to/Mutagenicity/processed/data.pt \
  --device cpu --out-dir /path/to/new-output

# v1 / fit-only continued training; same source cache and fit sampling sequence
python -m workflows.tqg.run_task \
  --config configs/sega_style/mutagenicity_v1.json \
  --cache-path /path/to/Mutagenicity/processed/data.pt --out-dir /path/to/new-output

python -m workflows.tqg.run_task \
  --config configs/sega_style/mutagenicity_v2.json --version source_only \
  --cache-path /path/to/Mutagenicity/processed/data.pt --out-dir /path/to/new-output
```

两版默认源预训练 100 epoch、batch=128、Adam lr=0.001；适配 200 step、fit=16、query=196、target=16、SGD lr=0.0003、反馈权重 0.1。模型种子 3405，源角色划分种子 431，fit fraction=0.8。Mutagenicity 示例沿用已有源 train=975、validation=109、fit=779、query=196 的清单和图规模域索引。以上是新流水线的起始配置，不是已证明最优的设置。

预训练每 epoch 对 fit 洗牌并完整遍历，保留最后不足 batch 的批次。适配每步独立无放回抽样，跨步允许重复；fit/query/target/probe 各自独立 RNG。请求数大于池大小时使用全部池，日志记录实际图 ID。v2 的实际 query 仍须至少两张。适配按固定步数结束，输出最后一步模型；不会再根据目标指标选模，也不会使用 source validation 挑适配步数。

`run_all` 接受完整配置对象的 JSON 列表，逐任务运行并保存成功/失败及 seed。它不按数组位置拼接不同 seed，不自动计算目标测试均值：

```bash
python -m workflows.tqg.run_all --manifest /path/to/tasks.json --output /path/to/new-summary.json
```

```text
<out_dir>/
  encoders/<source_key>/     best.pt、history.jsonl、config、data_roles、provenance、result
  tasks/<version>_<key>/     model.pt、history.jsonl、predictions.json、config、data_roles、provenance、result
```

缓存身份包含代码内容、数据/划分 SHA、源角色、模型、预训练配置、seed 和运行时；源缓存不含 target 或 TQG 版本，所以相同起点能复用。任务身份还包括完整适配配置和目标索引。已完成任务复用前核对每个产物 SHA；未完成目录或损坏缓存会明确报错，不因 JSON 同名就跳过。当前新流水线不支持中途恢复：保留失败目录，修复问题后使用新 out_dir。所有输出默认放在被 Git 忽略的 `runs/`。

当前验收环境是 CPU。使用 CUDA 确定性模式时，按 PyTorch 的要求在启动进程前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，例如在上述命令前加此前缀并指定 `--device cuda`。尚未完成 GPU 验收，不把 CPU 测试视为 CUDA 可用性或性能结论。

`predictions.json` 只有目标图 ID 和预测概率，无目标标签或准确率。本轮不读取 final 标签。后续若做研究对照，应明确 development/final 评价清单并单独实现评价流程。

## 6. 检查与证据边界

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_sega_pipeline.py' -v
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

专项检查覆盖：SEGA GIN/GINE 输出及状态一致性；新骨干上的 v1 有限响应和梯度等式；v2 多分类完整空间及梯度等式；停止梯度；BN/dropout 固定；单步 SGD；无反馈与同批次 source-only 一致；目标标签拒绝；源 query 不参与预训练；两版端到端训练、源缓存共用、独立预测；缓存配置/字节失配；数值失败回退。

已有 v1/v2 数学测试继续检查岭回归、共享尺度、坐标旋转、零秩、零支持、零残差、Kronecker 顺序、强凸校准闭式特例和 matched gap。详细验收及独立 review 结果记录在[本轮验收](../validation/sega_style_tqg_20261106.md)。

**旧实验仍使用旧骨干和当时的协议。本次工程改造没有生成新的 PROTEINS/Mutagenicity 成绩，不能把旧表格改名为新实现结果。**
