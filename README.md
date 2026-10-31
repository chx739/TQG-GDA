# TQG-GDA

Target-Queried Gradient Learning for graph-level domain adaptation 的研究代码与实验归档。

本仓库依据 TQG 设计稿实现实验框架，沿用用户提供的 SEGA 图网络结构与域索引。**截至2026-10-26，在 PROTEINS 上尚未获得跨迁移任务的稳定收益。** 这里保留代码、负面结果及审计证据，供后续改进。

新增 [v2 方法实现](docs/TQG_V2_IMPLEMENTATION.md)：完整目标分类梯度空间、核查询联合校准、参数坐标反馈与无动量 SGD。Mutagenicity 三方向五种子的[v2完整验证](docs/experiments/mutagenicity_v2_20261026/run_report.md)已经完成，当前配置仍未获得跨任务稳定收益。v2 配置位于 `configs/v2/`，原配置继续使用原流程；原版与 v2 结果分开记录。

后续参考：[TQG v2详细实验方案与优先级总表（2026-10-31）](docs/research_plans/tqg_v2_20261031/README.md)。覆盖实现验收、梯度质量、单步作用、校准／签名诊断和条件改进；优先D0→D1→D2。**该文档是待执行方案，不代表新增实验结果。**

## 原版方法在做什么

1. 用有标签的源图预训练 GNN。
2. 用无标签目标图选择参数探测方向。
3. 沿相同方向扰动参数，测量源图和目标图的输出响应。
4. 用源标签拟合响应组合权重，预测目标图的输出纠错量。
5. 将源分类损失与目标纠错损失合并，进行一次 Adam 更新；下一步重新构建反馈。

方向、响应、源残差、组合权重和参考输出均停止梯度；普通 GNN 参数仍正常优化。推断阶段只需要普通 GNN 前向。

## PROTEINS实验

PROTEINS 共1,113张图，任务为图级二分类。四个域沿用 SEGA 代码包提供的边数域索引，**属于实验构造的图规模迁移，不是 PROTEINS 官方的四个域**。0域整体边数最多，3域最少；0→1、0→2、0→3分别以0域为源、其余域为目标。

- 源域278张：250张训练、28张独立源验证。
- 250张训练图按类别、划分种子431保留约80% fit：类别0为167/209张，类别1为32/41张，因此 fit=199、query=51。
- n51每步使用全部51张 query；n250每步使用199 fit＋51 query拟合响应组合权重。n250同时改变数量与样本角色，不是单纯的数据量对照。
- TQG 起点均由199张 fit 预训练得到，后续源分类主损失仍用 fit；全源监督基线另有250张源图预训练起点。
- 预训练预算700次更新，按源验证 CE 选起点；后续训练200次更新。目标标签不用于训练或选模。
- 每个设置使用5个模型种子3405–3409，报告开发集 balanced accuracy（%）的均值±样本标准差。各目标域开发集139张，final标签尚未评价。
- K=4、扰动半径0.05、ridge=0.01、反馈权重0.1、Adam lr=0.0003、float64；最新一轮未使用局部匹配、反馈投影或源 CV 选正则。

odd记录正反扰动之间的带方向输出变化；even记录正反扰动平均值相对未扰动输出的弯曲变化。odd＋even将两类响应拼成特征，K=4时分别为4列与8列。见[公式对应](docs/响应类型与原文公式对照.md)。

| 方法 | 0→1 | 0→2 | 0→3 |
|---|---:|---:|---:|
| fit-only 源模型不动 | 56.28 ± 0.94 | 64.47 ± 1.35 | 60.06 ± 1.00 |
| 继续源 fit 训练 | 56.49 ± 0.46 | 64.28 ± 0.34 | 61.05 ± 1.65 |
| 全源标签训练 | 56.49 ± 1.06 | 63.44 ± 1.72 | 59.51 ± 1.71 |
| 简单置信度伪标签 | 56.49 ± 0.46 | 64.28 ± 0.34 | 61.15 ± 1.96 |
| TQG n51，odd | 56.84 ± 0.48 | 63.94 ± 0.62 | 60.78 ± 0.77 |
| TQG n51，odd＋even | 57.08 ± 0.66 | 64.16 ± 1.10 | 60.40 ± 0.69 |
| TQG n250，odd | 56.49 ± 0.46 | 64.28 ± 0.34 | 60.96 ± 1.53 |
| TQG n250，odd＋even | 56.49 ± 0.46 | 64.28 ± 0.34 | 60.86 ± 1.67 |

固定200步时，n51 odd＋even在0→1比继续 fit 训练提高约0.59个百分点，但其他方向未超过该基线，按源验证选点也未保留这项优势。n250整体接近继续源监督训练，even未体现稳定收益。同分不代表未执行反馈，参数差异已有审计。开发集已被多轮观察，五种子结果属于探索性证据。

完整证据：[执行报告](docs/experiments/latest/run_report.md)、[逐种子结果](docs/experiments/latest/results.csv)、[汇总 JSON](docs/experiments/latest/results_summary.json)、[复现范围与限制](docs/experiments/latest/REPRODUCIBILITY.md)。

## Mutagenicity：SEGA配置下的完整对照

新增三个迁移方向、五个种子的完整实验，采用SEGA边数域索引和通用源训练配置。骨干为GINE（14维节点标签、3维边标签），32维、2层、mean readout、float64。源预训练100 epoch（fit 2,500次更新／全源3,100次更新），TQG适配200 step；没有使用SEGA几何/OT适配模块。

源域1,084张＝975训练＋109验证；训练图进一步分为779 fit＋196 query。目标开发集各542张，final标签未评价。下表为固定200步的balanced accuracy（%），均值±五种子样本标准差。

| 方法 | 0→1 | 0→2 | 0→3 |
|---|---:|---:|---:|
| fit-only 源模型不动 | 76.72 ± 1.38 | 67.34 ± 2.56 | 61.67 ± 3.36 |
| 继续源 fit 训练 | 76.91 ± 2.21 | 67.45 ± 2.88 | 62.05 ± 3.34 |
| 全源标签训练 | 77.20 ± 0.49 | 68.37 ± 1.65 | 63.11 ± 3.00 |
| 简单置信度伪标签 | 76.96 ± 2.18 | 67.70 ± 2.67 | 61.80 ± 3.29 |
| query196，odd | 76.79 ± 1.79 | 67.12 ± 2.92 | 62.62 ± 3.85 |
| query196，odd＋even | 76.85 ± 1.78 | 66.92 ± 2.86 | 62.48 ± 4.00 |
| 全源975，odd | 76.85 ± 2.07 | 67.41 ± 2.81 | 62.16 ± 3.23 |
| 全源975，odd＋even | 76.93 ± 2.10 | 67.46 ± 2.73 | 62.18 ± 3.34 |

query196＋odd在0→3较继续fit监督提高0.56个百分点，五个种子4胜1负，但其余方向没有提升；所有TQG设置的平均值均未超过各方向全源监督基线。even未体现稳定增量。源验证选点结果与配对比较见[完整报告](experiments/mutagenicity_sega_20261026/results/run_report.md)。这轮尚未获得跨任务稳定收益。

[实验归档](experiments/mutagenicity_sega_20261026/README.md)包含115份训练日志、755份checkpoint、225条评价索引、配置与来源记录，约260 MiB。更换数据集同时改变了GNN结构与预训练更新预算，不能把与PROTEINS的差异单独解释为数据量作用。

## Mutagenicity：v2完整验证

复用相同源预训练模型，运行0→1/2/3、五个种子，每步使用全部196张独立query图，适配200步；所有适配方法采用无动量SGD，lr0.0003、反馈权重0.1。105次适配及225行开发评价全部完成。指标为balanced accuracy（%），均值±五种子样本标准差。

| 方法 | 0→1 | 0→2 | 0→3 |
|---|---:|---:|---:|
| fit-only 源模型不动 | 76.72 ± 1.38 | 67.34 ± 2.56 | 61.67 ± 3.36 |
| 继续源fit（SGD） | 76.76 ± 1.50 | 67.37 ± 2.76 | 61.82 ± 3.48 |
| 全源监督（SGD） | 77.09 ± 0.67 | 68.43 ± 1.73 | 63.00 ± 2.94 |
| 置信度伪标签（SGD） | 76.72 ± 1.39 | 67.41 ± 2.68 | 61.95 ± 3.45 |
| 普通核回归（KRR） | 76.99 ± 1.53 | 67.23 ± 2.83 | 61.84 ± 3.57 |
| v2：仅表示核 | 76.84 ± 2.06 | 67.20 ± 3.05 | 62.21 ± 3.83 |
| v2：打乱query标签 | 77.28 ± 1.73 | 67.13 ± 2.76 | 61.95 ± 3.35 |
| v2：完整方法 | 77.02 ± 1.28 | 67.29 ± 2.94 | 61.81 ± 3.61 |

完整方法相对继续源fit为+0.264/−0.077/−0.0015个百分点，未满足至少两个方向获益的预设主条件。0→1提升主要来自一个种子；与KRR差别很小，打乱query标签也没有稳定变差。主版本3000个查询步骤均达到求解容差，参数更新等式与标签访问审计通过；这些执行检查不证明监督估计正确。final标签未读。

[完整报告与源验证选点结果](docs/experiments/mutagenicity_v2_20261026/run_report.md)、[逐种子结果](docs/experiments/mutagenicity_v2_20261026/results.csv)、[独立审计](docs/experiments/mutagenicity_v2_20261026/independent_audit.json)。[本轮实验归档](experiments/mutagenicity_v2_20261026/README.md)包含115份训练日志、650份checkpoint、225条评价索引及执行代码快照，约140 MiB。10次源预训练来自上一轮，105次适配为本轮新训练；归档操作没有重新训练模型。

## Mutagenicity v2：目标batch196追加对照，2026-10-28

新增目标batch196，0→1/2/3各五种子，共15次新适配。源fit每步16、query全部196、相同源模型起点、200步、SGD lr0.0003、反馈权重0.1及其余设置保持一致。16/32/64与源SGD使用已审计历史对照；指标为开发集balanced accuracy（%），均值±五种子样本标准差。

| 方法 | 0→1 | 0→2 | 0→3 |
|---|---:|---:|---:|
| 源fit模型不动 | 76.72 ± 1.38 | 67.34 ± 2.56 | 61.67 ± 3.36 |
| 继续源fit SGD | 76.76 ± 1.50 | 67.37 ± 2.76 | 61.82 ± 3.48 |
| TQG v2，目标batch16 | 77.02 ± 1.28 | 67.29 ± 2.94 | 61.81 ± 3.61 |
| TQG v2，目标batch32 | 77.13 ± 1.40 | 67.26 ± 2.94 | 61.89 ± 3.63 |
| TQG v2，目标batch64 | 77.08 ± 1.42 | 67.23 ± 2.89 | 61.96 ± 3.61 |
| TQG v2，目标batch196 | 77.08 ± 1.42 | 67.19 ± 2.88 | 61.92 ± 3.69 |

**196没有带来跨任务稳定的分类收益。** 相对16，配对均值差为+0.058/−0.108/+0.103个百分点，仅1/3任务满足预设支持条件；相对64，0→1持平，另两项分别−0.043/−0.040个百分点。反馈范数CV继续从64档的0.380降至0.261，但这只表示幅度波动更小。当前CPU实现每步计时约15.83秒，64档历史参考为4.55秒；耗时受机器负载影响，仅供参考。固定200步不等计算预算。

[结果解读](docs/experiments/mutagenicity_target_batch196_20261028/结果解读.md)、[完整报告](docs/experiments/mutagenicity_target_batch196_20261028/run_report.md)、[独立审计](docs/experiments/mutagenicity_target_batch196_20261028/independent_audit.json)。新归档包含80份日志、460份checkpoint、165条评价记录，约104.9 MiB原始来源文件；其中60次适配及5次源预训练复用，详见[归档说明](experiments/mutagenicity_target_batch196_20261028/README.md)。新增3000步的校准全部收敛，无数值回退，final标签未读。196是每步目标抽样数，不是目标域全量（目标池为1084/1084/1085图）。

## Mutagenicity v2：目标batch对照，2026-10-28

保持源fit每步16、query全部196、相同五个源模型、适配200步、SGD lr0.0003及反馈权重0.1，仅改变目标batch16/32/64。沿用0→1/2/3和种子3405–3409。batch16及源SGD复用已核对代码、数据和模型哈希的结果；batch32/64新增30次训练。指标为开发集balanced accuracy（%），均值±五种子样本标准差。

| 方法 | 0→1 | 0→2 | 0→3 |
|---|---:|---:|---:|
| fit-only 源模型不动 | 76.72 ± 1.38 | 67.34 ± 2.56 | 61.67 ± 3.36 |
| 继续源fit（SGD） | 76.76 ± 1.50 | 67.37 ± 2.76 | 61.82 ± 3.48 |
| TQG v2，目标batch16 | 77.02 ± 1.28 | 67.29 ± 2.94 | 61.81 ± 3.61 |
| TQG v2，目标batch32 | 77.13 ± 1.40 | 67.26 ± 2.94 | 61.89 ± 3.63 |
| TQG v2，目标batch64 | 77.08 ± 1.42 | 67.23 ± 2.89 | 61.96 ± 3.61 |

相对batch16，batch32在三个方向为+0.113/−0.032/+0.080个百分点，batch64为+0.058/−0.065/+0.143个百分点；两档均仅在0→3满足至少3/5种子获胜的正向收益条件，未达到至少两个任务的预设门槛。0→1的batch32增益来自一个种子，其他四个持平。没有获得跨任务稳定的分类收益。

反馈范数CV的跨任务均值从0.521降为0.448/0.380，目标图覆盖从约95%升至约99.8%/100%；但反馈相对源梯度的幅度也下降。当前CPU实现每步计时约1.29/2.42/4.55秒，batch16是历史参考，负载差异使耗时比较仅供参考。幅度波动减小不证明估计方向正确，当前证据不足以支持仅靠增大目标batch解决效果问题。

[结果解读与逐任务诊断](docs/experiments/mutagenicity_target_batch_20261028/结果解读.md)、[完整报告](docs/experiments/mutagenicity_target_batch_20261028/run_report.md)、[独立审计](docs/experiments/mutagenicity_target_batch_20261028/independent_audit.json)。[完整归档](experiments/mutagenicity_target_batch_20261028/README.md)包含65份训练日志、370份checkpoint和135条评价索引，约84.8 MiB；其中30次新适配、30次复用适配、5次共享源预训练。三档9000步查询均收敛且无数值回退，final标签未读。此次没有执行按epoch洗牌对照。

## PROTEINS训练日志与模型归档

该轮PROTEINS实验的原始运行文件已集中到 [experiments/proteins_full_query_20261026/](experiments/proteins_full_query_20261026/README.md)：

| 内容 | 数量 | 未压缩大小 |
|---|---:|---:|
| 逐步训练日志 `history.jsonl` | 115份 | 66.3 MiB |
| 原始模型 checkpoint | 755份 | 125.7 MiB |
| 配置、数据角色、环境与阶段记录等 | 配套归档 | 约2.9 MiB |

范围包括本轮新跑的85次适配、复用的20次基线适配，以及共享的10次源预训练；15个冻结基线引用这些源模型，不重复保存。每个文件与原运行目录逐字节校验，归档总大小约195 MiB。

- 按任务、种子、方法查找：[训练索引](experiments/proteins_full_query_20261026/job_index.csv)。
- 从表格指标找到实际模型：[评价索引](experiments/proteins_full_query_20261026/evaluation_index.csv)，覆盖225条末步、源验证选点及冻结模型评价记录。
- 浏览原始文件：[日志](experiments/proteins_full_query_20261026/logs/)、[checkpoint](experiments/proteins_full_query_20261026/checkpoints/)、[配置与来源](experiments/proteins_full_query_20261026/metadata/)。
- 下载后核验，不需要 PyTorch：

```bash
python scripts/verify_experiment_archive.py experiments/proteins_full_query_20261026
```

checkpoint目录包含源模型的 `best.pt` / `last.pt`，以及适配的 `step_0/10/50/100/200.pt`、`last.pt`、`selected.pt`。源验证选点可能选中step0。图数据缓存仍需自行提供；原始记录中的绝对路径保留为来源信息，索引提供仓库内对应路径。

## 安装与合成检查

推荐使用独立 Python 环境。已有验证环境为 Python 3.12、PyTorch 2.5.1、PyG 2.6.1；根据机器选择匹配的 CPU/CUDA PyTorch。

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python scripts/smoke_test.py --output /tmp/tqg-smoke-new
```

输出目录必须不存在。合成检查不会启动新的真实数据实验。

## 基础 TQG 命令行

提供与清单哈希一致的 TU/PyG 缓存后，可运行以下基础示例。`--target-domain`可设为1、2、3：

```bash
python scripts/run_experiment.py all \
  --data-path /path/to/PROTEINS/processed/data.pt \
  --source-split data/splits/PROTEINS_official_cache_source.json \
  --target-domain 1 \
  --target-policy transductive-unlabeled \
  --config configs/tqg_development.json \
  --seed 3405 --device cpu --output /tmp/tqg-basic-new
```

此配置是基础 TQG 的 query16示例，不能直接逐位重现最新全查询表格。源模型、适配模型和训练记录已归档；完整实验编排仍依赖原目录结构与数据缓存。[复现说明](docs/experiments/latest/REPRODUCIBILITY.md)列出使用已归档模型与重跑实验的区别。

如果自行重新处理的数据缓存哈希不同，应显式建立新的源划分清单，并说明与旧实验的区别：

```bash
python scripts/prepare_source_split.py \
  --data-path /path/to/PROTEINS/processed/data.pt \
  --source-index idx/idx_PROTEINS_0.npy \
  --dataset PROTEINS --seed 431 --output /tmp/new_source_protocol.json
```

不要替换旧清单中的哈希来冒充同一实验。

## 文件导航

- `tqg_gda/`：模型、参数探测、有限响应、残差回归、目标反馈、训练与数学模块。
- `scripts/`：基础训练、恢复、独立评价、诊断与消融入口。
- `scripts/archive_full_query.py`、`scripts/archive_v2.py`、`scripts/archive_target_batch.py`、`scripts/verify_experiment_archive.py`：归档既有实验、验证文件和指标对应关系；不训练模型。
- `configs/full_query/0_to_1.json`、`configs/full_query/0_to_2_3.json`：PROTEINS两阶段的原始实验设置。
- `configs/mutagenicity/sega_full_query.json`：Mutagenicity的SEGA源配置、TQG矩阵与协议边界。
- `configs/revision_v1/`：改进方案和合成测试使用的归档计划；状态字段是历史记录。
- `docs/TQG_Graph_Level_GDA_Design.md`：原始方法设计稿。
- `docs/TQG_Improvement_Experiment_Plan_v1.md`、`docs/失败原因与改进方案.md`：改进设计与失败分析，不代表其中所有实验均已完成。
- `docs/experiments/latest/`：最新结果、图表、核查与来源哈希。
- `experiments/`：实际训练日志、checkpoint、运行配置与文件索引，见[实验目录](experiments/README.md)。
- `docs/`：方法、改进设计和结果文档，见[文档目录](docs/README.md)。
- `docs/history/`：历史框架、pilot、revision说明与初始源清单，含原运行环境路径。

代码历史保留原实验提交：`8923cc9`对应0→2/0→3全查询实验，`3ea6b72`对应0→1扩展。这些原版结果作为历史记录保留；v2方法和本轮Mutagenicity实验另列配置、报告与归档。

后续完成代码修改或实验后，默认更新文档、核验产物并同步此GitHub仓库，项目约定见[AGENTS.md](AGENTS.md)。
