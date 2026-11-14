# 数据与固定索引

源码不附带图数据或模型。自行准备可信的 PROTEINS TU/PyG 原生 `processed/data.pt`，可放在源码外任意目录，通过 `--data-path /path/to/PROTEINS/processed/data.pt` 指定；不会复制数据。

支持的序列化结构是 PyG TUDataset 的三元组 `(data_dict_or_Data, slices, sizes)` 或四元组 `(data_dict_or_Data, slices, sizes, DataClass)`。`sizes` 含 `num_node_attributes`、`num_node_labels`、`num_edge_attributes`、`num_edge_labels`；`slices` 是 InMemoryDataset 的图边界，edge_index 为各图局部索引。具体读取约定见 `sega_gda/data.py`。仅加载可信文件，torch.load 使用原生 pickle。

如果已有官方 TU 原始文件，可用 PyG `TUDataset(root=外部目录, name='PROTEINS')` 准备原生 processed 缓存；但重新处理、不同 PyG 版本/序列化可能改变缓存字节或图顺序。当前附带的固定 source 划分严格锁定原数据版本，要求缓存 SHA256 为 `19e75f23f52fb349238da0e46fac04268e570afbb446ec8d4ec87f30365ad207`。请迁移这一可信原生缓存；不能保证任意重新下载的文件字节相同。校验不匹配会明确拒绝，不会自动改哈希或重划分。重新准备的数据若版本不同，需另行核验内容和 graph ID 协议，不能静默冒充原任务。

`idx/*.npy` 是原缓存中从 0 开始的图 ID，不含图属性或标签。保留原有文件；`idx` 族是边数域，`node` 族是节点数域。默认真实任务是 PROTEINS idx_0→idx_1；idx_0 共 278 图，按附带 source manifest 的原 ID 固定为 250 train / 28 validation，完整 idx_1 的 278 图用于无标签 transductive adaptation。`--idx-dir` 可指向同一索引文件的外部副本，不能通过更换划分提升结果。

其他数据集的原索引也保留，但没有编造其 source train/validation 划分；运行其他任务必须显式提供已批准的 `--dataset`、`--source-split`、域索引和相符缓存。闭集类别语义必须由研究协议确认，不能根据目标标签拟合。目标标签在取图前从训练输入视图排除，不用于任何训练、覆盖、关系、选模或调参。
