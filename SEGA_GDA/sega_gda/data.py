"""Read existing TU/PyG caches and immutable source splits, without downloads.

Uses PyG's native separate() and the feature-column convention in TUDataset.
Only graph IDs explicitly present in the source manifest can be requested.
"""
from pathlib import Path
import hashlib
import json

import torch
from torch_geometric.data import Data
from torch_geometric.data.separate import separate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class SourceGraphs:
    def __init__(self, cache_path, split_path, use_node_attributes=False):
        self.manifest = json.loads(Path(split_path).read_text())
        self.cache_path = Path(cache_path)
        if sha256(cache_path) != self.manifest['dataset_cache_sha256']:
            raise ValueError('Dataset cache does not match the frozen source split')
        payload = torch.load(cache_path, map_location='cpu', weights_only=False)
        if not isinstance(payload, tuple) or len(payload) not in (3, 4):
            raise ValueError('Expected the existing TU/PyG processed cache format')
        values, self.slices, self.sizes = payload[:3]
        cls = payload[3] if len(payload) == 4 else Data
        self.data = cls.from_dict(values) if isinstance(values, dict) else values
        self.train_ids = list(self.manifest['source_train_indices'])
        self.val_ids = list(self.manifest['source_validation_indices'])
        self.allowed = set(self.train_ids + self.val_ids)
        if not self.train_ids or not self.val_ids:
            raise ValueError('Source train and validation must both be nonempty')
        if len(self.allowed) != len(self.train_ids) + len(self.val_ids):
            raise ValueError('Duplicate IDs or overlapping source train/validation')
        if min(self.allowed) < 0 or max(self.allowed) >= len(self.slices['x']) - 1:
            raise ValueError('Source graph index outside cached dataset')
        self.continuous_dim = int(self.sizes['num_node_attributes']) if use_node_attributes else 0
        if self.data.x is not None and not use_node_attributes:
            self.data.x = self.data.x[:, int(self.sizes['num_node_attributes']):]
        self.input_dim = int(self.data.x.shape[1]) if self.data.x is not None else 1
        if self.input_dim < 1:
            raise ValueError('No node features after the declared feature selection')
        self.edge_dim = int(self.data.edge_attr.shape[1]) if self.data.edge_attr is not None else 0
        # Class vocabulary and all learned preprocessing come only from source TRAIN.
        self.classes = sorted({int(self.graph(i).y.item()) for i in self.train_ids})
        if len(self.classes) < 2:
            raise ValueError('Source training must contain at least two classes')
        self.class_map = {label: i for i, label in enumerate(self.classes)}
        for i in self.val_ids:
            if int(self.graph(i).y.item()) not in self.class_map:
                raise ValueError('Validation contains a class absent from source train')

    def graph(self, index):
        index = int(index)
        if index not in self.allowed:
            raise PermissionError(f'Graph {index} is not in this source-only split')
        graph = separate(self.data.__class__, self.data, index, self.slices, decrement=False)
        graph = graph.clone()
        if graph.x is None:
            graph.x = torch.ones((graph.num_nodes, 1))
        graph.x = graph.x.to(torch.float64)
        if graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr.to(torch.float64)
        if (graph.x.ndim != 2 or graph.num_nodes < 1
                or graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2):
            raise ValueError(f'Invalid graph {index}')
        if not torch.isfinite(graph.x).all():
            raise ValueError(f'Nonfinite features in graph {index}')
        if graph.edge_index.numel() and (graph.edge_index.min() < 0
                                       or graph.edge_index.max() >= graph.num_nodes):
            raise ValueError(f'Invalid edge endpoints in graph {index}')
        if graph.edge_attr is not None and (graph.edge_attr.shape[0] != graph.edge_index.shape[1]
                                            or not torch.isfinite(graph.edge_attr).all()):
            raise ValueError(f'Invalid edge attributes in graph {index}')
        if graph.y is None or graph.y.numel() != 1 or not torch.isfinite(graph.y).all():
            raise ValueError(f'Expected one finite graph label for source graph {index}')
        if not torch.equal(graph.y, graph.y.long()):
            raise ValueError('Noninteger source class')
        graph.y = graph.y.reshape(1).long()
        graph.graph_id = torch.tensor([index], dtype=torch.long)
        return graph

    def fit_preprocessor(self):
        """Continuous columns only; categorical TU one-hot columns stay intact."""
        mean = torch.zeros(self.input_dim, dtype=torch.float64)
        scale = torch.ones_like(mean)
        if self.continuous_dim:
            values = torch.cat([self.graph(i).x[:, :self.continuous_dim] for i in self.train_ids])
            mean[:self.continuous_dim] = values.mean(0)
            std = values.std(0, unbiased=False)
            scale[:self.continuous_dim] = torch.where(std > 0, std, torch.ones_like(std))
        return {'mean': mean, 'scale': scale, 'fit_ids': self.train_ids,
                'continuous_dim': self.continuous_dim}

    def materialize(self, indices, preprocessing):
        graphs = []
        for index in indices:
            graph = self.graph(index)
            graph.x = (graph.x - preprocessing['mean']) / preprocessing['scale']
            graph.y = torch.tensor([self.class_map[int(graph.y.item())]])
            graphs.append(graph)
        return graphs
