# 安装与运行

所有命令从仓库根目录执行。v1与v2使用同一包，但方法和优化器不同，先选版本再选配置。

**当前 SEGA式实现**使用 `python -m workflows.tqg.run_task --config configs/sega_style/mutagenicity_v2.json --cache-path /path/to/Mutagenicity/processed/data.pt`。见[配置、完整运行和产物说明](code/sega_style_tqg.md)。本页后续 `scripts/run_experiment.py` 命令属于旧骨干/历史实验接口，不能加载新流水线的 checkpoint。

阶段实验脚本已从根目录归入`workflows/`。基础CLI继续使用`python scripts/xxx.py`；阶段实验使用模块入口：

```bash
python -m workflows.v2.assumptions.assumptions_experiment --help
python -m workflows.v2.optimization.optimization_feedback --help
python -m workflows.v1.full_query_experiment --help
```

[按阶段查入口](../workflows/README.md) · [旧路径映射](workflow_migration.json)。旧归档的冻结命令在对应旧代码快照中执行，不直接套到当前主分支。

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

此配置是基础 TQG 的 query16示例，不能直接逐位重现最新全查询表格。源模型、适配模型和训练记录已归档；完整实验编排仍依赖原目录结构与数据缓存。[复现说明](v1/results/proteins_full_query_20261026/REPRODUCIBILITY.md)列出使用已归档模型与重跑实验的区别。

如果自行重新处理的数据缓存哈希不同，应显式建立新的源划分清单，并说明与旧实验的区别：

```bash
python scripts/prepare_source_split.py \
  --data-path /path/to/PROTEINS/processed/data.pt \
  --source-index idx/idx_PROTEINS_0.npy \
  --dataset PROTEINS --seed 431 --output /tmp/new_source_protocol.json
```

不要替换旧清单中的哈希来冒充同一实验。


## v2训练与诊断

训练配置见[v2实现说明](v2/implementation/method.md)；诊断入口见[代码导航](code/README.md)。原始运行依赖、数据缓存SHA和归档复现限制仍以具体报告为准。
