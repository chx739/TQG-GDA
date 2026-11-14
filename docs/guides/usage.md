# 数据协议、运行与推断

研究进展见[实验总览](../../experiments/README.md)：[模拟结果](../../experiments/synthetic/README.md)与[真实图验证计划](../../experiments/mutagenicity/README.md)分开记录；工程测试不代表真实任务有效。


## 显式输入

使用 `torch.save` 保存纯字典/张量，读取时 `weights_only=True`。顶层必须恰好包含：

```python
payload = {
    "schema_version": 1,
    "protocol": "transductive",  # 或 inductive，按预先确定的实验设置填写
    "source_train": [source_record, ...],
    "source_validation": [source_record, ...],  # 可为空
    "target": [target_record, ...],
}
source_record = {
    "id": "dataset/graph-123",   # 原始图身份；禁止通过改 ID 跨角色复用
    "x": node_features,         # 浮点 [N,F]，完整图，N>=1
    "edge_index": edge_index,   # int64 [2,E]，图内 0-based 下标
    "edge_attr": edge_features, # 可省略；浮点 [E,A]，GINE 必需
    "label": 0,                 # Python int，源训练覆盖 0..C-1
}
target_record = {
    "id": "dataset/graph-456",
    "x": node_features,
    "edge_index": edge_index,
    "edge_attr": edge_features,
}  # 禁止 label/y/其他类别字段
```

图 ID 必须唯一且跨角色互斥，两域特征接口一致。目标输入不得携带类别标签。数据准备阶段仅提取获准图的结构和属性；不要调用参考 SEGA 的全目标评价入口。

源训练预算、源验证角色和目标池身份由实验协议事先确定。目标池包含最终测试图的无标签结构时填写 transductive。程序检查显式数据契约，不能代替外部协议确认；不会自行访问历史 final 标签。

## 运行

```bash
python -m workflows.tsba.run_task \
  --input /absolute/path/task.pt \
  --config /absolute/path/config.json \
  --output /absolute/path/new-tsba-run
```

配置是 Config 字段的 JSON 子集，未知字段报错。默认 adaptation_dtype 为 float64。cg_tolerance 未指定或为 null 时，float64 使用 1e-7、float32 使用 1e-5；显式正容差原样保留。它同时控制直接求解和 CG 的真实相对残差验收，运行记录保存解析后的值。过严容差仍可能导致更新全部被拒绝，应检查 adaptation.history 中的原因与残差。Python 中对已解析配置调用 `dataclasses.replace` 更改精度时，若要重新使用该精度的默认容差，需同时传 `cg_tolerance=None`；否则保留原配置的显式数值。复用源模型时添加 `--source-checkpoint /absolute/path/previous-run/source.pt`；源数据、角色、架构或预训练配置不匹配会报错，目标池可以改变。

纯工程检查无需真实输入：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m workflows.tsba.run_task \
  --synthetic --output /tmp/tsba-generated-graph-check
```

synthetic 默认缩小网络和预算，覆盖值全部记录。它不声称完成真实任务复现，也不保证每个生成任务产生有效边界。

## 输出

- `model.pt`：冻结编码器、固定坐标、最终分类模块和来源元数据。
- `source.pt`：可跨目标复用的源模型，不包含源图或标签。
- `predictions.pt`：目标 ID、初始及最终概率，没有目标真值。
- `result.json`：完整配置、身份、训练记录、锚点与拒绝原因、求解残差、回溯结果及源风险。target_accuracy 固定 null。
- 初始化或训练错误写 failure.json 并失败退出；适配数值异常保存最后接受状态，结果标为 numerical_error，CLI 返回 2。

## 独立推断

```python
from tsba.predict import load_predictor
model = load_predictor("/absolute/path/new-tsba-run/model.pt", device="cpu")
probabilities = model.predict(unlabeled_pyg_graphs)
classes = probabilities.argmax(dim=1)
```

输入为具有兼容 x/edge_index/edge_attr 的完整 PyG 图列表。推断不需要原训练输入文件、源标签、源参照库、score 或目标池。

## 测试

```bash
python -m unittest discover -s tests/tsba -v
```

覆盖概率等价初始化、重复点带宽、投影有效性、位移符号、GGN/目标梯度、直接/CG 一致性、空反馈/异常/回溯回滚、BN 冻结、目标标签隔离、checkpoint 身份和合成图端到端独立推断。

## E1/E2 合成机制验证

按[冻结方案](../../experiments/validation-plan.md)运行40条固定表示轨迹：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.synthetic.e1_e2_20261109.run \
  --output /absolute/path/new-tsba-mechanism-run
```

输出目录必须不存在。入口不会读取真实图数据；使用解析源分类器、训练的score及两个解析score对照。保存输入生成参数、源坐标/初始头、score、逐轮分类器参数与候选、批次/锚点、独立几何观察、预测封存哈希、评价标签访问、风险结果和全文件清单。score和分类器checkpoint属于研究档案，不是包含GNN的生产predictor。所有40条轨迹完成或记录失败后统一评价，不按合成标签选轮次。

生产 `adapt_classifier(..., observer=callback)` 默认关闭观察器。回调只收到复制后的记录/张量，Torch RNG在回调前后隔离；回调异常向调用者传播，不能伪装成算法拒绝。不要在回调中通过外部闭包修改模型。锚点 `input_indices` 对应本轮目标批次位置，可与 `target_ids` 合成全池索引。

已完成的首轮结果见[E1/E2报告](../../experiments/synthetic/e1_e2_20261109/report.md)。复核与汇总使用独立入口，输出路径不能覆盖旧文件：

```bash
python -m experiments.common.synthetic.verify --run /absolute/path/finished-run --output /absolute/path/checkpoint-audit.json
python -m experiments.synthetic.e1_e2_20261109.summarize --run /absolute/path/finished-run --output /absolute/path/new-analysis
python -m experiments.synthetic.e1_e2_20261109.report --summary /absolute/path/new-analysis/summary.json --output /absolute/path/new-analysis/report.md
python -m experiments.common.synthetic.package --source /absolute/path/experiment-bundle --archive /absolute/other/path/bundle.tar.gz
```

`predictions.pt`在研究入口中保存logits；softmax得到概率。自动报告生成证据表格，正式报告另含人工核对后的解释与限制。归档工具逐项核验压缩内容并输出SHA256，不覆盖既有实验目录或档案。

## 补充源监督与零请求几何对照

[补充协议](../../experiments/synthetic/e2_controls_20261109/controls-plan.json)是查看首轮结果后登记的诊断。直接复用原run的输入、初始头和逐轮ID，运行20条新轨迹，不重训score或覆盖旧档案：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.synthetic.e2_controls_20261109.run_controls \
  --parent /absolute/path/original-run-001 \
  --output /absolute/path/new-control-run
```

`source_ce_ggn`没有边界约束；`zero_zeta_geometry`在自己的当前边界上保留位移损失、DᵀD和执行预算，强制ζ=0仍求解，空锚点跳过。与原TSBA完全相同的参考和矩阵只在首轮检查，后续各自演化。模型和批次记录保存到新目录，所有预测封存后才评价合成标签。

补充对照完成后，用下面的只读核验入口重载checkpoint并生成配对表格：

```bash
python -m experiments.synthetic.e2_controls_20261109.summarize_controls --run /absolute/path/finished-control-run --output /absolute/path/new-control-analysis
```

已发布的[补充对照报告](../../experiments/synthetic/e2_controls_20261109/report.md)包含20条新轨迹、与原40条轨迹的配对比较、源控制的局限以及对首轮解释的修正。

## 全源512样本候选与验收

按[全源批次协议](../../experiments/synthetic/e2_full_source_20261110/full-source-plan.json)运行三臂共30条轨迹：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.synthetic.e2_full_source_20261110.run_full_source \
  --original /absolute/path/original-mechanism-run \
  --controls /absolute/path/completed-control-run \
  --output /absolute/path/new-full-source-run
```

仅修改 `source_adapt_batch=512`；仍重放原随机数调用并核对目标ID。原有目标形式和预算不改，逐次回溯的源CE用独立的全源前向复算核验。运行器保留全源CE轨迹、原128批次的封存预测和30个新checkpoint，所有预测封存后再评价。


完成后，重载所有新checkpoint、复算每次候选的全源CE，并与旧128批次结果配对：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.synthetic.e2_full_source_20261110.summarize_full_source \
  --run /absolute/path/finished-full-source-run \
  --output /absolute/path/new-full-source-analysis
```

核验要求预测逐位一致、逐轮参数状态连续、源样本全覆盖及目标ID完全匹配。输出逐种子评价、配对差、全源CE曲线、审计清单与报告；原128批次的全源CE曲线从既有参数快照被动复算。

已发布的[全源批次报告](../../experiments/synthetic/e2_full_source_20261110/report.md)包含30条新轨迹、与原128批次的配对结果及完整归档索引。


## 既有全源checkpoint回放

按[回放协议](../../experiments/synthetic/e2_replay_20261110/replay-plan.json)评价源泛化、目标置信度及固定探针边界；只加载已保存的参数，参数梯度关闭，既有目录不改写：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.synthetic.e2_replay_20261110.run_replay \
  --parent /absolute/path/completed-full-source-run \
  --output /absolute/path/new-checkpoint-replay-run
```

930次状态评价包括三个臂各自的初始化与30个已接受状态，重复的初始化不是独立样本。完整记录与compact checkpoint向量留在新目录，失败原样保留。


回放结束后，重算目标logits、指标和边界根残差并汇总：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.synthetic.e2_replay_20261110.summarize_replay \
  --run /absolute/path/completed-checkpoint-replay-run \
  --output /absolute/path/new-replay-analysis
```

已发布[checkpoint回放报告](../../experiments/synthetic/e2_replay_20261110/report.md)。源条件CE和后验KL的MC标准误、逐轮目标NLL四组贡献、固定探针几何、每个种子的配对差均可从表格与完整档案复核。


## 解析源后验对照

[后验CE协议](../../experiments/synthetic/e2_source_posterior_20261110/source-posterior-plan.json)只替换研究入口的源损失，不改变生产TSBA。候选与验收同时替换，复用完整源批次、原目标ID及总体score：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.synthetic.e2_source_posterior_20261110.run_source_posterior \
  --parent /absolute/path/full-source-run \
  --replay /absolute/path/checkpoint-replay-run \
  --output /absolute/path/new-source-posterior-run
```

适配全部结束并封存后，自动按前次回放协议重新评价已保存的参数；独立源样本与边界探针必须逐位匹配。`source_ce`接受日志在本诊断中特指后验期望CE，硬标签CE只在封存后的诊断中单独报告。


核验候选方程、每次后验CE验收及checkpoint，并与旧硬标签结果配对：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.synthetic.e2_source_posterior_20261110.summarize_source_posterior \
  --run /absolute/path/completed-source-posterior-run \
  --output /absolute/path/new-source-posterior-analysis
```

[已发布报告](../../experiments/synthetic/e2_source_posterior_20261110/report.md)包含20条控制轨迹的浮点漂移、总体score净收益及完整档案索引。


## 静态score诊断

[五组单因素协议](../../experiments/synthetic/e2_static_score_20261110/static-score-plan.json)固定分类头、查询点和400次Adam更新，只改变目标样本量或带宽，不更新分类器：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.synthetic.e2_static_score_20261110.run_static_score \
  --parent /absolute/path/completed-original-e1-e2-run \
  --output /absolute/path/new-static-score-run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.synthetic.e2_static_score_20261110.summarize_static_score \
  --run /absolute/path/completed-static-score-run \
  --output /absolute/path/new-static-score-analysis
```

主结果使用最终400更新状态；补充保存按旧无标签噪声验证规则选中的快照。扩大样本池时σ仍取原512样本的基准；带宽实验的查询点不变。[已发布报告](../../experiments/synthetic/e2_static_score_20261110/report.md)分别记录估计误差和总体方向是否仍朝正确边界，避免把过度平滑后的准确估计当成成功适配。完整checkpoint、Adam状态、逐更新日志及查询记录见报告内本地存档索引。

### P0：延长固定score训练预算与简化oracle

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.synthetic.p0_20261110.run_p0 \
  --parent /absolute/path/completed-static-score-run \
  --original /absolute/path/completed-original-e1-e2-run \
  --output /absolute/path/new-p0-run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m experiments.synthetic.p0_20261110.summarize_p0 \
  --run /absolute/path/completed-p0-run \
  --output /absolute/path/new-p0-analysis
```

固定2维、8192样本和原带宽/查询；从原400步Adam状态连续训练，固定末尾400/1600/6400步比较。原父档案先后校验且不可覆盖；输出路径必须不存在。[P0报告](../../experiments/synthetic/p0_20261110/report.md)包含逐种子结果和完整本地archive索引。6400步静态检查5/5通过，但源约束下标量oracle收益仍小；本入口不运行学习score闭环。`p0_oracle.py`是固定斜率阈值诊断，不是生产分类头实现。

真实图R1详细设计：[Mutagenicity流程](../../experiments/mutagenicity/r1_20261111/mutagenicity-stage1-protocol.md)、[机器清单](../../experiments/mutagenicity/r1_20261111/stage1-plan.json)。诊断包含类别相关密度结构与源边界方向；480个小头拟合的执行驱动仍待实现/验收，当前未运行真实图训练。
