# SEGA-GDA 独立研究源码

共享图编码器在固定初始坐标上建立局部 RBF 对数度量场，以可微有限路径测地距离统一分类、半松弛 OT 与域内关系；内容学习和受信任区域约束的空间修订交替进行。方法定义见 `docs/SEGA_GDA_Self_Evolving_Local_Geometry_Design_CN.md`。本包包含完整实现与开发示例，不含训练结果、模型或原始图数据；示例参数不是正式论文最优配置。

## 目录

```text
SEGA_GDA/
├── sega_gda/        # 数据、编码器、初始化、度量/路径、OT/关系、演化与完整状态
├── scripts/        # 从零训练入口、监控、独立合成 smoke 与自生成测试夹具
├── configs/        # Source-only、真实适配开发示例及合成测试参数
├── tests/          # 数学、梯度、数据边界和恢复测试
├── idx/            # 原始 graph ID 域划分；不是图数据
├── data/splits/    # 原 PROTEINS source train/validation ID
├── docs/           # 原始方法设计
├── requirements.txt
├── SOURCE_MANIFEST.sha256
└── README.md
```

## 环境

已验证 Python 3.10、PyTorch 2.3.0+cu121、PyG 2.7.0、NumPy 2.2.6；SciPy 1.15.3 用于 OT 独立数值参考测试。无需导出整个服务器环境或安装额外 PyG 扩展。

```bash
conda create -n sega-gda python=3.10
conda activate sega-gda
# GPU 环境：先安装匹配驱动的已验证 CUDA wheel
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

CPU 环境将 torch 安装命令的索引换成 `https://download.pytorch.org/whl/cpu`。GPU 需要宿主 NVIDIA 驱动及可见设备；容器需正确映射 GPU。无需修改系统 CUDA。以下命令均从解压后的 SEGA_GDA 目录运行，输出路径请替换为自己的空目录。

## 数据准备

图数据不随包分发。准备与附带 ID 相符的可信原生 PROTEINS `processed/data.pt`，具体格式、版本哈希、PyG 准备方式与限制见 `data/README.md`。数据可在外部任意目录；入口要求 `--data-path`，索引默认本包 idx，也可 `--idx-dir /path/to/idx`。入口不访问旧项目源码或历史 runs。

默认保留原 source train 250 / validation 28，不重新划分；完整 idx_1 作为无标签 transductive 池，需要在命令显式声明该协议。源 validation 仅用于预定 CE 最小选模与诊断，不参与统计拟合、覆盖和关系。目标标签不参与任何训练/选择，本入口不提供目标评估。

## 从零训练

先训练仅自然源 CE 的 GIN/GINE 和线性头：

```bash
python scripts/run_experiment.py source --data-path /path/to/PROTEINS/processed/data.pt --output /tmp/sega-source --device cpu
```

然后使用**这次新训练**产生的 source checkpoint；自动重建几何头、固定 P0/覆盖及关系，不加载历史资产：

```bash
python scripts/run_experiment.py adapt --data-path /path/to/PROTEINS/processed/data.pt --target-policy transductive-unlabeled --source-checkpoint /tmp/sega-source/source/best.pt --output /tmp/sega-adapt --device cpu
```

或一次完成全部步骤：

```bash
python scripts/run_experiment.py all --data-path /path/to/PROTEINS/processed/data.pt --target-policy transductive-unlabeled --output /tmp/sega-all --device cpu
```

GPU 上先确认卡空闲，只暴露一张卡，将 device 改为 cuda:0；此序号是屏蔽后的逻辑序号：

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/run_experiment.py all --data-path /path/to/PROTEINS/processed/data.pt --target-policy transductive-unlabeled --output /tmp/sega-gpu --device cuda:0
```

源码包不自动启动多卡、正式任务矩阵或全量超参搜索。实际 d=32 完整路径反传成本较高，先使用小预算和分块配置确认资源。

## 配置与输出

`--source-config configs/source_development.json`：GIN/GINE 宽度/层数、图读出、自然 CE、epochs、batch、lr、seed、精确头转换参数；device/数据路径由命令覆盖。只用 source train 拟合预处理；source validation CE 最小选模，并列取最早 epoch。

`--adapt-config configs/adaptation_development.json`：内容窗口与总轮数、query/reference/target 数量、lambda/gamma/margin、epsilon/rho、OT 严格收敛阈值、P0/覆盖/关系 RNG、信任图对/组权重、步长回缩、delta_space 和 c_A。路径 L=4/Q=3/J_path=8、float64，支持点对 chunk 与等价重计算。分类、OT、关系共用当前归一化测地距离，空间尺度保留梯度；不得以替代距离或截断梯度节省资源。

## 计算效率

度量只以二次型 `Δᵀg(z)Δ` 进入路径能量；实现用 `Δᵀe^{A}Δ = ‖e^{A/2}Δ‖²`（A 对称）把它化成矩阵-向量 Taylor 级数（`LocalMetricField.quadratic_form`，阶数由谱界 c_g 推出，c_g=1 时 15 项即达双精度），不再对每个积分点求 `torch.matrix_exp` 及其 2d×2d / 4d×4d 块矩阵导数。`checkpoint_chunks=true` 时每个点对 chunk 只在反向重算一次（`geodesic._RecomputedChunk`），前向不再重复重算。这些都不改变距离定义，`tests/test_s6_efficiency.py` 逐项核对与 `matrix_exp` 定义的一致性。

运行时间主要由关系记忆的路径数决定（每个内容步和每个空间候选都求 `2|T_s|+2|T_t|` 条路径）。几个旋钮：

- `num_threads`：示例配置为 1。设为物理核数可线性提速；同一组实验内保持固定，因为逐位结果依赖线程数。
- `solver.chunk_size`：每个 chunk 在反向重算时持有完整展开图，d=32 时约 1–2 MB/点对；CPU 128–256、GPU 可用 512–1024。
- `relation_triplets_per_step`：`null`（默认）= 设计规定的全记忆均值；给正整数 N 则每个内容步/空间步从固定记忆中无放回均匀抽 N 条三元组（两域各 N），空间步的基线与全部回缩候选共用同一抽样。这是把全均值换成无偏 minibatch 估计的**语义改动**，默认关闭。

source 输出在指定根目录下 `source/`，有 best.pt、last.pt、initialization.pt、配置、源训练日志/结果。adapt 输出在 `adaptation/`，有 initial/round_N/last/selected.pt、固定关系/数据身份、rounds.jsonl、runtime_measurements.json 和结果。`inputs/task.json` 是本次输入协议。适配选模仍仅用源验证 CE，初始状态也是候选。已有输出目录会拒绝覆盖。

## 恢复训练

源训练按完整 epoch 边界恢复；新配置只允许延长 epochs（固定学习率），可迁移数据路径但缓存/划分字节与身份必须一致。创建一个 epochs 更大的 source 配置后：

```bash
python scripts/run_experiment.py source --data-path /path/to/PROTEINS/processed/data.pt --source-config /path/to/source_extended.json --resume-source /tmp/sega-source/source/last.pt --output /tmp/sega-source-resumed --device cpu
```

适配只支持完整演化轮次边界恢复，不能恢复内容步、路径或回缩中途。总轮数和训练配置保持一致，设备类型/逻辑编号保持一致；从最近已完成的 round_N.pt 或 last.pt 继续到原定总轮数（不是已完成总预算的 checkpoint）。源 checkpoint 必须是本次原起点，允许文件位置迁移并校验指纹：

```bash
python scripts/run_experiment.py adapt --data-path /path/to/PROTEINS/processed/data.pt --target-policy transductive-unlabeled --source-checkpoint /tmp/sega-source/source/best.pt --resume-adapt /tmp/sega-adapt/adaptation/round_1.pt --output /tmp/sega-adapt-resumed --device cpu
```

保留先前最佳完整状态和选择准则，不把恢复边界当作新的选模起点。失败时保留已完成内容/轮次的原始记录，不自动修改研究参数。

## 验证

不需要真实数据或预训练模型的核心测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

从零生成外部临时合成图缓存，实际运行 source→几何初始化→两轮完整适配，并检查源训练/适配断点恢复一致性：

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/smoke_test.py --output /tmp/sega-smoke
```

该测试所有图数据、checkpoint 和结果均写到包外；不会改变真实数据划分。目标标签故意设为无意义值以检查隔离，合成结果不能作为研究实验。发布时另有源码清单，运行前可用 `sha256sum -c SOURCE_MANIFEST.sha256` 检查文件完整性。
