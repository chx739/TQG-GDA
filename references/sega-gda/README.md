# SEGA-GDA：受源任务约束的共享几何自演化图域适配（代码实现）

本代码完整实现 `SEGA_GDA_Revised_Design_CN` 设计文档第 3 节的算法，在 Mutagenicity 上按给定的 `idx` / `node` 四域划分运行全部 12 个有序迁移任务（M_s→M_t，s≠t）。

## 1. 环境

```bash
pip install torch torch_geometric cvxpy clarabel scikit-learn scipy numpy
```

测试环境：Python 3.13，torch 2.15，torch_geometric 2.8.1，cvxpy 1.9.3，Clarabel 0.11.1。GPU 可用时自动使用（`--device auto`）。

## 2. 数据

* `splits/` 中是你提供的 8 个划分文件（`{idx,node}_Mutagenicity_{0..3}.npy`），内容是 TUDataset 原始图序号（0–4336），四个域互不相交且覆盖全部 4337 张图。
* Mutagenicity 原图由 PyG `TUDataset` 读取。压缩包里已经带了 TU 原始 txt 文件（`data/Mutagenicity/raw/`，4337 图、131488 节点），首次运行时 PyG 直接处理生成 `processed/`，不需要联网。若删掉它，PyG 会从 TU 官网自动下载。
* 节点特征为 14 维原子类型 one-hot，边特征为 3 维键型 one-hot（`--encoder gine` 时使用）。

两种划分的域统计（由原始数据计算）：

| 划分 | 域 | 图数 | 正类比例 | 平均节点数 | 平均密度 |
|---|---|---|---|---|---|
| idx | M0 / M1 / M2 / M3 | 1084 / 1084 / 1084 / 1085 | 0.486 / 0.333 / 0.430 / 0.536 | 52.6 / 31.1 / 23.2 / 14.4 | 0.046 / 0.071 / 0.093 / 0.155 |
| node | M0 / M1 / M2 / M3 | 1084 / 1084 / 1084 / 1085 | 0.525 / 0.327 / 0.401 / 0.533 | 52.7 / 31.0 / 23.2 / 14.4 | 0.045 / 0.071 / 0.093 / 0.156 |

`node` 是按节点数四分位切分（各域节点数区间不重叠），`idx` 是按规模/密度切分但区间有重叠。两种划分都同时存在结构偏移和标签比例偏移。

## 3. 运行

单个任务：

```bash
python run_task.py --split_type idx --source 0 --target 1 --seed 0
```

全部任务（两种划分 × 12 个方向 × 多个种子），结束后打印并保存汇总表：

```bash
python run_all.py --split_types idx,node --seeds 0,1,2
python run_all.py --split_types node --seeds 0 --tasks 0-1,3-2      # 只跑部分任务
```

任何 `Config` 字段都可以用 `--字段名 值` 覆盖，或用 `--config my.json` 批量覆盖。源编码器只依赖源域和种子，会缓存在 `results/encoders/`，同一源域的 3 个目标任务共用同一个 φ0；已完成任务的 json 存在时 `run_all.py` 自动跳过（可断点续跑）。

数值检查：

```bash
python tests/test_sanity.py          # 或 python -m pytest -q tests/
```

检查内容：恒等几何下数值距离等于欧氏平方距离；路径能量的解析梯度与自动微分一致；无序点对对称、相同端点零距离、谱界 (1−ρ)D_B ≤ D ≤ (1+ρ)D_B；有效坐标满足 tr B=d、Σν_kE_k=0 且基正交归一；分数 Jacobian 与有限差分一致（unrolled 模式相对误差约 1e-10）；变形 Gram 与逐节点直接积分一致；子问题零步值、可行性与有效性判据；ρ=0、κ=1、K=17 等边界情形。

### 消融开关

| 开关 | 含义 |
|---|---|
| `--rho 0` | 去掉局部修正，只保留全局 Mahalanobis 度量 B（w/o local） |
| `--K 1` | 单区域，等价于只有全局部分 |
| `--use_reverse false` | 去掉目标参照识别源查询（w/o L_R）。由于 L_F(Θ0)=0，此时通常不会产生更新，这是设计的预期行为 |
| `--use_forward false` | 去掉源参照识别可信目标（w/o L_F） |
| `--use_risk_budget false` | 去掉逐类源 CE 预算（w/o budget） |
| `--target_filter all` | 不做稳健筛选，接受所有非并列 argmax 目标，权重为 1（w/o filtering） |
| `--jac_mode envelope` | 用固定路径的包络导数代替对有限松弛过程求导（更快） |
| `--encoder gine` | 编码器使用键型边特征 |

## 4. 输出

* `results/json/<task>.json`：目标准确率（源参照分类器初始几何 vs SEGA-GDA 最终几何）、源校准集准确率前后对比、可信目标数量、m_c、R_c、每轮候选的 pred/ared/预算/可行性/接受记录、停止原因、最终可行性指标。
* `results/models/<task>.pt`：最终推理需要的全部内容（编码器参数、源参照坐标与标签、标尺权重、τ、区域中心与带宽、最终 B 与 E_k、数值距离规格）。
* `results/results.csv` 和 `results/summary.md`：汇总表，行是方法，列是 M_s→M_t 任务和平均值（多种子时为 mean±std）。
* `results/logs/`：逐任务日志。

目标标签只用于报告准确率和可信目标伪标签准确率这一诊断量，不参与任何训练、筛选、检查点或超参数选择。

## 5. 设计文档到代码的对应

| 设计章节 | 内容 | 代码 |
|---|---|---|
| 3.1.2 | 按图 ID、按类分层划分 I_ref / I_fit / I_cal，以及参照库两个按类完整的子库 | `sega/data.py: split_source_roles` |
| 3.2 | N_g(z)、r_g、每类核平均 S_c、p(c\|z)；v_i = 1/(C n_{y_i}) | `sega/classifier.py` |
| 3.3 | 恒等几何下用同一参照分类器预训练编码器；每步重算查询、参照和分母；V_A>0 检查；按校准 CE 选检查点 | `sega/pretrain.py` |
| 3.4 | 三个参照头的稳健间隔 M_0、V_c、m_c、R_c（经验分位数取第 ⌈qn⌉ 个）、C_rel、J_+、w_j、C_T；\|C_T\|<2 时不适配 | `sega/target_refs.py` |
| 3.5.1 | μ_0（两域各半）、固定种子加权 k-means 中心、删除空/重复中心、σ_cov=c_h s_cov、α_k、ν_k | `sega/coverage.py` |
| 3.5.2, 3.8.1 | g_b(z)=B+Σα_kE_k，有效坐标 b∈R^P（P=Kd(d+1)/2−1），正交归一基，可行域检查 | `sega/geometry.py` |
| 3.6, 3.7 | 统一间隔缺口 ℓ_m、L_F、L_R、L_DA、逐类源 CE 风险与固定预算 | `sega/problem.py` |
| 3.8.2 | 分数 Jacobian（含归一化分母导数，不截断梯度） | `sega/problem.py: jacobian_pass` |
| 3.8.3–3.8.5 | 只线性化几何→分数映射，保留完整 max-hinge 与 CE 的凸近端子问题（cvxpy + Clarabel，含 LMI） | `sega/subproblem.py` |
| 3.8.4 | 变形 Gram 矩阵 G_r 与 M_r=G_r+ε_M I | `sega/gram.py` |
| 3.9 | pred / ared、四个接受条件、近端步长回缩、五个停止条件、返回最后接受状态 | `sega/adapt.py` |
| 3.10 | 分段直线路径、对称 Gauss–Legendre 积分、固定步数路径松弛、取最低能量有限迭代、字典序无序点对、相同端点零距离 | `sega/geodesic.py` |
| 3.11 D | 完整源库 + 最终 g_* 的统一预测，并列取固定类别顺序中的第一个 | `sega/predict.py` |

## 6. 实现中需要说明的数值约定

以下几点是把设计落成可运行代码时做出的具体选择，均在适配前固定、不使用目标标签。

1. **Jacobian 的求法。** 默认 `jac_mode=unrolled`，按 3.10.2 对实际采用的有限路径松弛过程求导（含内部节点对几何参数的依赖）。为了一次反传得到每个查询的一行 Jacobian，每个查询使用自己的一份 b 副本，因此对 Σ_i S_c(z_i) 反传即得到逐查询的行。`envelope` 模式只在最终路径上求偏导，速度更快，两者都经过同一个非线性验收，所以只影响候选质量，不影响接受规则的正确性。
2. **变形 Gram 的积分。** μ̄_r 包含数十万条路径上的积分节点，逐节点计算 P×P 矩阵不可行。实现利用 g(z) 只通过 α(z) 依赖位置这一结构：在 α 空间按分辨率 1/`gram_alpha_res` 分格，每格内的多项式矩 ψψ^T 精确累加，tr(g⁻¹·g⁻¹·) 对应的 s×s 矩阵在格内加权平均 α 处计算。b=0 时结果精确，一般情况下的误差由格宽控制（测试中与直接积分的相对误差低于 2e-3）。
3. **子问题中的 LMI 收紧。** 子问题内部把谱约束收紧 `sdp_margin`（默认 1e-6），使有限精度求解器的输出能够通过接受阶段对 b_r+Δ 的严格特征值检查。没有在求解后做投影。当前状态以同样的余量可行，所以 Δ=0 仍然是子问题的可行点。
4. **路径松弛步长。** η_path = c_path / (4 L_path m_+)，其中 m_+=(1+ρ)κ。它由数值规格和可行域常数唯一确定，对当前状态和所有候选完全相同。
5. **子问题的有效性。** 求解器状态为 optimal 或 optimal_inaccurate 时，再用 numpy 精确重算局部 CE 约束和子问题目标；只有局部可行、且目标不高于零步值的输出才视为有效候选。主求解器失败时退回 SCS；仍无有效候选时按回溯处理。
6. **前向距离的路径松弛使用解析梯度。** 不需要 Jacobian 的评估（当前状态、候选验收、最终预测）用能量对内部节点的解析梯度做松弛，迭代序列与自动微分完全相同（测试中差异约 1e-14），速度更快。unrolled Jacobian 仍用自动微分以保留对几何参数的依赖。
7. **默认超参数**（K=4、d=16、ρ=0.5、κ=3、τ=0.1、ε_src=0.05、q_m=0.5、q_R=0.95 等）是设计取值，没有调过。η_init=10 是依据 idx M0→M1 试跑中 ared/pred≈0.97 设定的（只看适配内部的预测/实际收益比，不看目标标签），回溯会在线性近似变差时自动缩步。P 随 K·d² 增长，d=16、K=4 时 P=543；ρ=0 时局部方向被删去，P=d(d+1)/2−1；κ=1 时全局方向被删去，P=(K−1)d(d+1)/2。
8. **其他。** 预训练时 BatchNorm 使用当前 batch（参照库 + 查询）的统计量，检查点选择和冻结坐标都用 eval 模式。数据划分按 TU 原始图序号进行，没有合并化学结构相同的不同记录。

## 7. 试跑结果与计算量

在 2 核 CPU、float64 下跑了 idx M0→M1（seed 0，预训练 40 轮，R_max=3，只用于验证流程）：

| 设置 | 每轮 Jacobian | 每次非线性评估 | 子问题求解 | L_DA | 目标准确率 |
|---|---|---|---|---|---|
| unrolled，η_init=1 | 约 120–150 s | 约 45 s* | 3–4 s | 0.248 → 0.188 | 77.49% → 77.95% |
| envelope，η_init=10 | 约 21 s | 约 25 s | 4–7 s | 0.248 → 0.146 | 77.49% → 78.60% |

\* 第一次试跑尚未使用解析梯度。

两次试跑每轮都是首个候选即被接受，ared/pred 在 0.97–1.0 之间，逐类源 CE 都在预算内，可行性检查通过；可信目标 316/1084 个，诊断用伪标签准确率 89.2%。L_F 初值精确为 0，随着反向任务推动空间变化开始出现小的正值，与设计 3.6.2 的描述一致。这只是 3 轮的流程验证，不代表完整设置（R_max=30、预训练 100 轮）下的结果。

主要开销来自路径距离：每轮需要 (|I_fit|+|J_+|)×|I_ref| + |I_fit^T|×|J_+| 条路径（这里约 60 万条），每条做 J_path 步松弛。按上面的 CPU 速度，完整的一个任务（30 轮）在 unrolled 模式下约 1–1.5 小时，envelope 模式下约 25 分钟。建议在 GPU 上跑：把 `pair_chunk_eval`、`pair_chunk_jac` 调大（例如 1e6 和 1e5），全部计算都是批量张量运算。CPU 上大规模扫描可以先用 `--jac_mode envelope`。

## 8. 目录

```
SEGA-GDA/
├── run_task.py            单任务入口
├── run_all.py             全部任务 + 汇总表
├── splits/                idx / node 四域划分
├── sega/
│   ├── config.py          全部超参数（与设计 3.12 节一一对应）
│   ├── data.py            数据读取、域划分、源数据角色
│   ├── encoder.py         GIN / GINE 图编码器
│   ├── classifier.py      统一参照分类器
│   ├── pretrain.py        源预训练与检查点选择
│   ├── target_refs.py     固定可信目标参照
│   ├── coverage.py        空间覆盖与区域权重
│   ├── geometry.py        度量场、有效坐标与可行域
│   ├── geodesic.py        数值测地距离
│   ├── problem.py         双向间隔任务、源风险、Jacobian
│   ├── gram.py            几何变形 Gram
│   ├── subproblem.py      凸近端子问题
│   ├── adapt.py           验收、回溯与停止
│   ├── predict.py         统一目标预测
│   └── pipeline.py        阶段 A–D 串联
└── tests/test_sanity.py   数值检查
```
