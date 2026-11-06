# 数据协议、运行与推断

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

配置是 Config 字段的 JSON 子集，未知字段报错。复用源模型时添加 `--source-checkpoint /absolute/path/previous-run/source.pt`；源数据、角色、架构或预训练配置不匹配会报错，目标池可以改变。

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
