# SEGA-GDA 代码审查记录（2026-10-16，2026-10-19 增补效率改造）

对照 `docs/SEGA_GDA_Self_Evolving_Local_Geometry_Design_CN.md` 逐模块核对了实现。
审查环境没有 PyTorch/PyG，无法运行测试；所有修改都只做了字节码编译检查
（Taylor 二次型的数学用 numpy/scipy 独立核对过），合并前请在本机执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python scripts/smoke_test.py --output /tmp/sega-smoke
```

## 零、2026-10-19 效率改造（定义不变）

**时间去哪了。** 每个内容步和每个空间候选都要对整个关系记忆求 `2|T_s|+2|T_t|` 条
路径（NCI1 约 4000 对），每条路径 L×Q=12 个积分点、J_path=8 步展开。原实现在每个
积分点调用 `torch.matrix_exp(A)`：它的一阶导要对 `[[A,G],[0,A]]`（2d×2d）再做一次指数，
路径松弛的 `create_graph=True` 使外层反传还要穿过这个导数——即 4d×4d 的块矩阵指数。
d=32 时每个点的求导代价是前向的几十倍，这一项主导了全部训练时间；再叠加非重入
checkpoint 在前向阶段触发的重复重算（每次内部 `autograd.grad` 都重算一遍 chunk 前缀）、
`num_threads=1` 和 `chunk_size=4`。

**改法（四处，均有 `tests/test_s6_efficiency.py` 核对）：**

1. `metric.py::LocalMetricField.quadratic_form` + `taylor_degree`。能量只需要
   `Δᵀ g Δ`；对对称 A，`Δᵀ e^{A} Δ = ‖e^{A/2} Δ‖²`，而 `e^{A/2}Δ` 是矩阵-**向量**
   Taylor 级数。阶数 N 由谱界 c_g 推出：`2 e^{4h} h^{N+1}/(N+1)! ≤ 2^{-53}`（h=c_g/2）
   保证相对误差在双精度内（c_g=1 → N=15；numpy/scipy 核对 300 个随机可行 A，
   最大相对误差 9e-16）。B=0 时各项精确为零，与欧氏能量逐位相同，§3.5 的初始等价不变。
   `forward()` 仍返回 `matrix_exp` 的完整矩阵供测试/诊断。
2. `geodesic.py::_RecomputedChunk`。用显式的重入式 `autograd.Function` 替换
   `torch.utils.checkpoint(use_reentrant=False)`：前向只存 chunk 输入（无外层图），
   反向重算一次完整展开图再求导；度量基元 B 作为显式输入，重算使用保存的 B。
   内存仍由 chunk 限定，前向不再有 O(J²) 重算。外层只支持一阶导（设计只需要一阶）。
3. `real_data.py::RealTaskGraphs.graph` 缓存已 `separate` 并预处理的输入图（任务内不可变），
   每步不再对全部记忆图重复 `separate()` 与校验；编码仍每步重做（§3.6 要求）。
4. `evolution.py` 可选 `relation_triplets_per_step`（默认 `null` = 全记忆，RNG 消耗与
   日志逐位不变）。给正整数时内容步用 `batch_rng`、空间步用 `probe_rng` 派生的生成器各
   无放回抽 N 条三元组，空间步基线与全部候选共用同一抽样。这是**语义改动**（全均值 →
   无偏 minibatch 估计），只在确实需要时打开。

`configs/adaptation_development.json`：`chunk_size` 4→128，新增 `relation_triplets_per_step: null`。
旧 checkpoint 的 `config`/`solver` 与新配置不同，`--resume-adapt` 会按规则拒绝，属预期。
`num_threads` 仍为 1：设为物理核数可线性提速，但逐位结果依赖线程数，同一组实验内固定即可。

**预期效果。** 距离求值与求导的 FLOPs 下降一到两个数量级，自动求导图内存约降一个数量级
（允许更大的 chunk）。NCI1 单方向估计从 ~1 小时降到几分钟（单线程），多线程更快；
若再打开 `relation_triplets_per_step`（如 256），内容/空间步的成本与记忆大小脱钩。

## 一、设计 ↔ 实现核对（未发现偏差）

| 设计条目 | 实现位置 | 结论 |
|---|---|---|
| §3.3 固定覆盖、对称零迹谱界对数度量、恒等背景 | `metric.py` | 权重 `α_k = r_k/(1+Σr)`、`g=expm(ΣαB)`、特征值投影（单一根 ζ 的位移+截断）均正确 |
| §3.4 分段线性路径能量、固定 P0、当前可微尺度 | `geodesic.py` | 能量公式、对称高斯积分、路径松弛完整反传、尺度分母在空间步参与求导均正确；无序点对统一取向 |
| §3.5 线性头 → 几何头精确转换 | `initialization.py` | `q=τs²w/2, β=b+‖q‖²/(τs²)` 正确，softmax 等价成立 |
| §3.6 固定 ID 关系记忆、两域各自平均再平均、空侧不翻倍 | `relations.py` | 正确 |
| §3.7 半松弛 OT（列硬约束、行 KL）、log-space 缩放、停止梯度 | `transport.py` | 迭代式、KKT 驻点检验、导出检查正确；`∂F/∂C = Π*` 由 `fixed_plan_objective` 保证 |
| §3.8 目标 | `objectives.py` | 正确 |
| §3.9 内容/空间分块更新、冻结探针、信任区域 + Armijo 双判据、新空间重建对应 | `evolution.py` | 版本号校验、回退恢复、`finally` 中恢复标志位均正确 |
| §3.11 梯度归属表 | `evolution.py::_flags` | 正确（内容步 B 无梯度并有断言；空间步坐标/类别参照冻结） |
| 数据边界：目标标签在 `separate` 之前从输入视图剔除 | `real_data.py` | 正确 |

## 二、已修改的文件

### 1. `sega_gda/objectives.py::alignment_loss`（实现问题）

原实现把参考图×目标图的测地代价矩阵**完整求解两次**：先在 `no_grad` 下求一次给 OT，
再带梯度求一次用于反传，然后用 `10*eps` 的 `allclose` 断言两次结果"相同"，不同则抛
`RuntimeError('Correspondence and differentiated costs differ')`。

- 两次求解在数值上本来就是同一组运算，断言只是一个近似逐位相等的检查；
  任何核函数差异（例如不同设备/线程数下的归约顺序）都会让训练直接中止。
- 代价矩阵的每一项都是带 8 步展开的路径松弛，白白多算一倍。

现改为只求一次带梯度的代价，OT 读取其 `detach()` 视图。这与设计 §3.7.3 / §3.9.2
的要求一致（对应读取停止梯度的当前代价，固定 Π* 后通过有梯度代价反传），
语义不变；`tests/test_s3.py` 中对 `cost.requires_grad`、Π 不带梯度、
端点/度量有限差分的检查都不受影响。

### 2. `sega_gda/train_source.py`（鲁棒性）

`environment.json` 的指纹用的是相对路径 `'docs/...'`、`Path('sega_gda')`，
只有在仓库根目录下运行才不报错（`run_experiment.py` 恰好 `os.chdir(ROOT)`）。
改为基于包根目录解析；输出的键名不变（仍为 `sega_gda/xxx.py`）。

### 3. `sega_gda/geodesic.py::GeodesicSolver.__init__`（增加稳定性警告，不改数值）

内部路径松弛是固定步长的显式梯度下降。对局部常值度量 g，能量关于内部节点的
Hessian 为 `2L·(T⊗g)`，`T=tridiag(-1,2,-1)`，故 `λ_max < 8·L·e^{c_g}`，
稳定要求 `path_lr·8·L·e^{c_g} < 2`。默认 `L=4, c_g=1, path_lr=0.01` 满足
（上界 0.023）；但若加大 `segments` 或 `spectral_bound`（如 `tests/test_s4.py`
里 `spectral_bound=4`），内部迭代可能发散，表现为候选空间能量非有限、全部被记为
`INVALID_CANDIDATE`——空间更新会"安静地停滞"而不是报错。现在构造求解器时会发出
`RuntimeWarning` 提示。恒等背景会把实际场压到 `e^{c_g}` 以下，所以这是保守提示。

## 三、需要在你的环境中确认、本次未改动的问题

### A. GPU 上 `torch.use_deterministic_algorithms(True)` 与 PyG 聚合（很可能是 GPU 阻断性问题）

`train_source.seed_everything` 全局打开严格确定性模式。PyG 的 `GINConv/GINEConv`
聚合与 `global_mean_pool` 在普通 `edge_index` 下走 `scatter_add_`（2-D 张量）。
PyTorch 文档明确：`scatter_add_` 在输入维度 > 1 且为 CUDA 张量时，严格确定性模式下
**抛出** `RuntimeError: scatter_add_cuda_kernel does not have a deterministic implementation`。
CPU 不受影响。若 README 中的 CUDA 命令确实完整跑通过，请忽略；否则两种处理：

1. 保持严格确定性：用 `index_add_` 实现求和聚合（CUDA 下有确定性实现），
   例如子类化 `GINConv/GINEConv` 覆盖 `aggregate`，并自写 mean-pool
   （`index_add_` 求和后除以计数）。参数名不变，checkpoint 兼容。
2. 放弃 GPU 逐位可复现：`torch.use_deterministic_algorithms(True, warn_only=True)`，
   但此时 `smoke_test.py` 里的逐位恢复断言在 GPU 上不再成立。

### B. `checkpoint_chunks=true` 与求解器内部的 `torch.autograd.grad`（2026-10-19 已解决）

原实现用非重入 `torch.utils.checkpoint`，`_solve` 内每步的 `torch.autograd.grad`
都会在前向阶段触发 chunk 前缀重算（约 O(J²)）。现已改为 `geodesic._RecomputedChunk`
（见第零节第 2 条）：只在反向重算一次。该模式下 `diagnostics=True` 仍会直接抛错。

### C. 计算规模（设计如此，提醒预算）

- 每个内容步都要重新编码**全部**关系记忆图并求 `2·|T_s|+2·|T_t|` 条路径
  （PROTEINS 约 1000+ 对、NCI1 约 4000 对，每对 8 步展开的二阶图）；
- 空间步的每一次回缩候选都要重算同样规模的关系项（`no_grad`，8 次尝试上限）；
- `NormalizedGeodesicDistance` 每次调用都重算 `s_ψ²`（内容步内 B 固定时其实可复用，
  但设计明确不缓存，P0 很小，影响不大）。
- 若上述规模仍不可接受，`relation_triplets_per_step` 是唯一一个改变语义的开关（第零节第 4 条）。

### D. OT 收敛预算

`ot_tolerances` 是接近机器精度的开发阈值。半松弛缩放迭代在 sup 范数下是
`κ=ρ/(ρ+ε)` 的压缩映射，默认 `ρ=1, ε=0.3` 时 κ≈0.77，百余次迭代即可达到
1e-9，`max_iter=5000` 充足；若以后把 ρ 调大（κ→1）或把 ε 调得很小，
会更接近 `MAX_ITER`，而 `raise_on_failure=True` 会让整轮训练中止（设计要求：
未收敛的对应不得进入更新）。调参时留意。

### E. 其他小点

- `fit_fixed_coverage` 的覆盖中心在 CPU 上拟合后 `.to(source)`；`pilot` 中
  `us/ut` 都在 CPU，行为正确。
- `run_experiment.adapt` 的 `selected.pt` 里 `portable_selection.engine_state`
  会再嵌一份完整状态，文件体积约为两倍，无功能问题。
- FRANKENSTEIN 等只有节点属性、没有节点标签的数据集必须 `use_node_attributes=true`，
  否则 `SourceGraphs` 会因特征列为 0 而明确拒绝（这是有意的保护，不是 bug）。
