# TU 图与标签权限

`smoke/` 含两套真实图的角色、来源摘要、逐图字节索引和独立的 train/val 标签。真实 H 类别不随仓库分发。每个 TU 图 ID 从 0 起，清单使用整数，侧车 JSON 使用与之对应的十进制字符串键。

新任务可参照 [Mutagenicity 清单](smoke/Mutagenicity.json)。四个非空角色为 train、val、A、H，图 ID 互斥。两个源侧车只含各自角色的原始整数类别。配置的 `data.manifest`、`data.raw_dir` 相对于配置文件，原始目录也可用 `--raw-dir` 显式指定。环境变量 `CELA_TU_ROOT` 的布局为 `<root>/<dataset>/raw`。

正式阶段按索引只打开所需图的字节范围并核验 SHA-256。已有真实清单附带索引；自有 TU 数据需先在清单声明 `tu_index` 的相对路径，并从原始图输入生成一次索引。TU 节点及邻接记录按图 ID 非降序分块。该导入步骤只需图输入。

```bash
python -m cela.data index --manifest task/manifest.json --raw-dir /path/to/raw --output task/tu_index.json
```

- 节点离散类型需在 `node_type_provenance` 声明外生来源。若文件中的值来自完整图度、位置编码或 WL，应填 null 以省略该列。连续节点列需在 `continuous_node_attributes` 中声明 `columns`、`exogenous: true` 和 `provenance`。本轮真实 smoke 使用 TU 原始离散节点类型。PROTEINS 的连续列未纳入本轮 smoke。
- 支持简单无向图与离散边类型。互反记录可归并成一条边，重复同向边、冲突标记、自环、连续边属性以及目标出现未纳入源词表的边标记会报错。未知节点类型有 OTHER 入口。
- 模型检查源 checkpoint 与适配文件的摘要，校验 A 身份以及 H 与源/A 的互斥关系。预测和独立评分产物均记录父产物摘要。
