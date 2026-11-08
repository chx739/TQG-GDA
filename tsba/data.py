"""Explicit graph roles; target records have no label field (§3.1).

The portable input is a weights_only torch archive of tensor dictionaries.
No TU-wide label reader, implicit split, or real-data evaluation is invoked.
"""
from dataclasses import dataclass
import torch
from torch_geometric.data import Data
from .utils import finite, identity, tensor_hash


@dataclass
class GraphPool:
    graphs: list
    ids: list
    labels: torch.Tensor | None = None

    def fingerprint(self):
        records = []
        for graph_id, graph in zip(self.ids, self.graphs):
            records.append(dict(id=graph_id, x=tensor_hash(graph.x), edges=tensor_hash(graph.edge_index),
                                edge_attr=None if graph.edge_attr is None else tensor_hash(graph.edge_attr)))
        return identity(dict(graphs=records, labels=None if self.labels is None else tensor_hash(self.labels)))


@dataclass
# §3.1–3.2：S={(G_i^S,Y_i^S)}、T={G_j^T}；源验证保留独立角色。
# 每个 graph 是完整图；目标池只有结构和属性，不能携带类别标签。
class TaskData:
    source_train: GraphPool
    source_validation: GraphPool
    target: GraphPool
    protocol: str

    # 检查协议声明、跨角色 ID 互斥、特征兼容与源类别范围。
    # 这些是数据契约检查，不替代外部确认源监督预算和 final 数据权限。
    def validate(self):
        if self.protocol not in ("transductive", "inductive"):
            raise ValueError("protocol must explicitly be transductive or inductive")
        if not self.source_train.graphs or not self.target.graphs:
            raise ValueError("source training and target pools must be nonempty")
        if self.target.labels is not None:
            raise ValueError("target labels are forbidden in adaptation input")
        seen, dims, edges = set(), set(), set()
        for role, pool in (("source_train", self.source_train), ("source_validation", self.source_validation),
                           ("target", self.target)):
            if len(pool.ids) != len(pool.graphs):
                raise ValueError(f"{role}: ID count mismatch")
            if role != "target" and (pool.labels is None or pool.labels.shape != (len(pool.graphs),)
                                     or pool.labels.dtype != torch.long):
                raise ValueError(f"{role}: integer labels required")
            for graph_id, graph in zip(pool.ids, pool.graphs):
                if not isinstance(graph_id, str) or not graph_id or graph_id in seen:
                    raise ValueError("graph IDs must be nonempty, unique and disjoint across roles")
                seen.add(graph_id)
                if graph.y is not None:
                    raise ValueError("graph objects must be label-free; source labels are separate")
                if graph.x is None or graph.x.ndim != 2 or min(graph.x.shape) < 1 or not graph.x.is_floating_point():
                    raise ValueError("x must be a nonempty floating [nodes, features] tensor")
                edge = graph.edge_index
                if edge is None or edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2:
                    raise ValueError("edge_index must be int64 [2, edges]")
                if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= len(graph.x)):
                    raise ValueError("edge endpoint outside graph")
                dims.add(graph.x.shape[1])
                finite(graph.x)
                if graph.edge_attr is not None:
                    if (graph.edge_attr.ndim != 2 or graph.edge_attr.shape[0] != edge.shape[1]
                            or not graph.edge_attr.is_floating_point() or graph.edge_attr.shape[1] < 1):
                        raise ValueError("edge_attr must be floating [edges, features]")
                    finite(graph.edge_attr)
                    edges.add(graph.edge_attr.shape[1])
                else:
                    edges.add(0)
        if len(dims) != 1 or len(edges) != 1:
            raise ValueError("incompatible feature interfaces across graphs")
        labels = self.source_train.labels
        classes = torch.unique(labels).cpu().tolist()
        if len(classes) < 2 or classes != list(range(len(classes))):
            raise ValueError("source training labels must cover consecutive classes 0..C-1, C>=2")
        validation = self.source_validation.labels
        if len(validation) and (int(validation.min()) < 0 or int(validation.max()) >= len(classes)):
            raise ValueError("source validation class absent in source training")
        return next(iter(dims)), next(iter(edges)), len(classes)

    def fingerprints(self):
        return {name: getattr(self, name).fingerprint() for name in ("source_train", "source_validation", "target")}


# §3.1：读取时白名单校验字段；目标记录不允许 label/y。
# 源标签单独放入 GraphPool.labels，编码器接收的图对象不带 y。
def from_payload(payload):
    required = {"schema_version", "protocol", "source_train", "source_validation", "target"}
    if set(payload) != required or payload["schema_version"] != 1:
        raise ValueError("invalid TSBA input schema")
    pools = []
    for role in ("source_train", "source_validation", "target"):
        graphs, ids, labels = [], [], []
        for record in payload[role]:
            needed = {"id", "x", "edge_index"} | ({"label"} if role != "target" else set())
            if not needed <= set(record) or set(record) - needed - {"edge_attr"}:
                raise ValueError(f"{role}: unsupported/missing fields (target labels are forbidden)")
            ids.append(record["id"])
            graphs.append(Data(x=record["x"], edge_index=record["edge_index"], edge_attr=record.get("edge_attr")))
            if role != "target":
                if type(record["label"]) is not int:
                    raise ValueError("source labels must be integer class IDs")
                labels.append(record["label"])
        pools.append(GraphPool(graphs, ids, None if role == "target" else torch.tensor(labels, dtype=torch.long)))
    data = TaskData(*pools, payload["protocol"])
    data.validate()
    return data


def load_task(path):
    return from_payload(torch.load(path, map_location="cpu", weights_only=True))
