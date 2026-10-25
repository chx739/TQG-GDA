# TQG-GDA

Target-Queried Gradient Learning for graph-level domain adaptation 的研究代码与实验归档。

本仓库依据 TQG 设计稿实现实验框架，沿用用户提供的 SEGA 图网络结构与域索引。**截至2026-10-26，在 PROTEINS 上尚未获得跨迁移任务的稳定收益。** 这里保留代码、负面结果及审计证据，供后续改进。

## 方法在做什么

1. 用有标签的源图预训练 GNN。
2. 用无标签目标图选择参数探测方向。
3. 沿相同方向扰动参数，测量源图和目标图的输出响应。
4. 用源标签拟合响应组合权重，预测目标图的输出纠错量。
5. 将源分类损失与目标纠错损失合并，进行一次 Adam 更新；下一步重新构建反馈。

方向、响应、源残差、组合权重和参考输出均停止梯度；普通 GNN 参数仍正常优化。推断阶段只需要普通 GNN 前向。

## 最新实验

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

## 训练日志与模型归档

最新实验的原始运行文件已集中到 [experiments/proteins_full_query_20261026/](experiments/proteins_full_query_20261026/README.md)：

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
- `scripts/archive_full_query.py`、`scripts/verify_experiment_archive.py`：归档既有实验、验证文件和指标对应关系；不训练模型。
- `configs/full_query/0_to_1.json`、`configs/full_query/0_to_2_3.json`：最新两阶段的原始实验设置。
- `configs/revision_v1/`：改进方案和合成测试使用的归档计划；状态字段是历史记录。
- `docs/TQG_Graph_Level_GDA_Design.md`：原始方法设计稿。
- `docs/TQG_Improvement_Experiment_Plan_v1.md`、`docs/失败原因与改进方案.md`：改进设计与失败分析，不代表其中所有实验均已完成。
- `docs/experiments/latest/`：最新结果、图表、核查与来源哈希。
- `experiments/`：实际训练日志、checkpoint、运行配置与文件索引，见[实验目录](experiments/README.md)。
- `docs/`：方法、改进设计和结果文档，见[文档目录](docs/README.md)。
- `docs/history/`：历史框架、pilot、revision说明与初始源清单，含原运行环境路径。

代码历史保留原实验提交：`8923cc9`对应0→2/0→3全查询实验，`3ea6b72`对应0→1扩展。仓库整理补充文档、实验文件、来源索引及方案路径回退；数值训练实现与原实验结果未改动，也未启动新的真实数据训练。
