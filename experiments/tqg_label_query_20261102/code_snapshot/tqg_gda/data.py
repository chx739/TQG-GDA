"""Native TU/PyG caches, fixed source roles, and label-free target input views.

Reader conventions originate in SEGA-GDA. Source train/validation IDs are
preserved; TQG partitions SOURCE TRAIN into fit/query before warmup.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data.separate import separate

from .utils import sha256, fingerprint


def load_cache(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, tuple) or len(payload) not in (3, 4):
        raise ValueError('Expected native TU/PyG (data, slices, sizes[, DataClass]) cache')
    values, slices, sizes = payload[:3]
    cls = payload[3] if len(payload) == 4 else Data
    raw = cls.from_dict(values) if isinstance(values, dict) else values
    if raw.x is None or 'x' not in slices:
        raise ValueError('This TU adapter requires existing node features')
    return raw, slices, sizes


def validate_ids(ids, count, name):
    if (not ids or any(isinstance(i, bool) or not isinstance(i, int) for i in ids)
            or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= count):
        raise ValueError(f'Invalid {name} graph IDs')


def source_partition(train_ids, labels, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError('fit_fraction must be between zero and one')
    generator = torch.Generator().manual_seed(seed)
    fit, query = [], []
    for label in sorted(set(labels.values())):
        pool = sorted(i for i in train_ids if labels[i] == label)
        if len(pool) < 2:
            raise ValueError('Every source class needs at least two graphs to reserve fit and query')
        shuffled = [pool[j] for j in torch.randperm(len(pool), generator=generator).tolist()]
        n = max(1, min(len(pool) - 1, math.floor(len(pool) * fraction)))
        fit.extend(shuffled[:n]); query.extend(shuffled[n:])
    return tuple(sorted(fit)), tuple(sorted(query))


class GraphDomainTask:
    def __init__(self, cache_path, source_split, *, target_index=None, target_policy=None,
                 fit_fraction=.8, split_seed=431, use_node_attributes=False,
                 preprocess_pool='fit'):
        self.cache_path = Path(cache_path).resolve()
        self.split_path = Path(source_split).resolve()
        self.manifest = json.loads(self.split_path.read_text())
        cache_hash = sha256(self.cache_path)
        if self.manifest.get('dataset_cache_sha256') != cache_hash:
            raise ValueError('Cache bytes differ from the source manifest; prepare and audit a new manifest explicitly')
        raw, self.slices, self.sizes = load_cache(self.cache_path)
        self._raw = raw
        count = len(self.slices['x']) - 1
        self.train_ids = tuple(self.manifest['source_train_indices'])
        self.validation_ids = tuple(self.manifest['source_validation_indices'])
        validate_ids(list(self.train_ids), count, 'source train')
        validate_ids(list(self.validation_ids), count, 'source validation')
        if set(self.train_ids) & set(self.validation_ids):
            raise ValueError('Source train and validation overlap')
        raw_labels = {i: self._source_label(i) for i in self.train_ids}
        self.classes = sorted(set(raw_labels.values()))
        if len(self.classes) < 2:
            raise ValueError('Need at least two source training classes')
        self.class_map = {label: j for j, label in enumerate(self.classes)}
        self.labels = {i: self.class_map[y] for i, y in raw_labels.items()}
        self.fit_ids, self.query_ids = source_partition(self.train_ids, self.labels, fit_fraction, split_seed)
        for i in self.validation_ids:
            label = self._source_label(i)
            if label not in self.class_map:
                raise ValueError('Source validation contains a class absent from source train')
            self.labels[i] = self.class_map[label]
        self.target_ids = ()
        target_hash = None
        if target_index is not None:
            if target_policy != 'transductive-unlabeled':
                raise ValueError('Declare target_policy=transductive-unlabeled before loading the target pool')
            index = np.load(target_index, allow_pickle=False)
            if index.ndim != 1 or index.dtype.kind not in 'iu':
                raise ValueError('Target index must be a one-dimensional integer array')
            self.target_ids = tuple(map(int, index.tolist()))
            validate_ids(list(self.target_ids), count, 'target')
            if set(self.target_ids) & (set(self.train_ids) | set(self.validation_ids)):
                raise ValueError('Target and source identities overlap')
            target_hash = sha256(target_index)
        self.continuous_dim = int(self.sizes['num_node_attributes']) if use_node_attributes else 0
        x = raw.x if use_node_attributes else raw.x[:, int(self.sizes['num_node_attributes']):]
        if x.ndim != 2 or x.shape[1] < 1:
            raise ValueError('No node features remain; enable use_node_attributes for attribute-only datasets')
        self.input_dim = x.shape[1]
        self.edge_dim = raw.edge_attr.shape[1] if raw.edge_attr is not None else 0
        # y is absent from this view BEFORE any target graph is separated.
        self._inputs = Data(x=x, edge_index=raw.edge_index)
        if raw.edge_attr is not None:
            self._inputs.edge_attr = raw.edge_attr
        self._input_slices = {k: v for k, v in self.slices.items() if k in self._inputs}
        if preprocess_pool not in ('fit', 'all_source'):
            raise ValueError('Invalid preprocessing source pool')
        fit_stats_ids = self.fit_ids if preprocess_pool == 'fit' else self.train_ids
        self.preprocessing = self._fit_preprocessing(fit_stats_ids)
        self._cache = {}
        self.identity = dict(dataset=self.manifest['dataset'], cache_sha256=cache_hash,
            source_split_sha256=sha256(self.split_path), target_index_sha256=target_hash,
            target_policy=target_policy if self.target_ids else None,
            source_fit_ids=list(self.fit_ids), source_query_ids=list(self.query_ids),
            source_validation_ids=list(self.validation_ids), target_ids=list(self.target_ids),
            source_labels={str(i): self.labels[i] for i in self.train_ids},
            class_labels=self.classes, use_node_attributes=use_node_attributes,
            preprocessing_pool=preprocess_pool, preprocessing_fit_ids=list(fit_stats_ids))
        self.identity['fingerprint'] = fingerprint(self.identity)

    def _source_label(self, index):
        if index not in set(self.train_ids) | set(self.validation_ids):
            raise PermissionError('Training data reader does not expose target labels')
        g = separate(self._raw.__class__, self._raw, index, self.slices, decrement=False)
        y = g.y
        if (y is None or y.numel() != 1 or not torch.isfinite(y).all()
                or not torch.equal(y, y.long())):
            raise ValueError('Expected one integer source label per graph')
        return int(y.item())

    def _input_graph(self, index):
        g = separate(Data, self._inputs, index, self._input_slices, decrement=False).clone()
        g.x = g.x.double()
        if g.edge_attr is not None:
            g.edge_attr = g.edge_attr.double()
        if (g.y is not None or g.x.ndim != 2 or g.num_nodes < 1 or not torch.isfinite(g.x).all()
                or g.edge_index.ndim != 2 or g.edge_index.shape[0] != 2):
            raise ValueError('Invalid label-free graph input')
        if g.edge_index.numel() and (g.edge_index.min() < 0 or g.edge_index.max() >= g.num_nodes):
            raise ValueError('Invalid graph-local edge endpoints')
        if g.edge_attr is not None and (g.edge_attr.shape[0] != g.num_edges or not torch.isfinite(g.edge_attr).all()):
            raise ValueError('Invalid edge attributes')
        return g

    def _fit_preprocessing(self, ids):
        mean = torch.zeros(self.input_dim, dtype=torch.float64)
        scale = torch.ones_like(mean)
        if self.continuous_dim:
            x = torch.cat([self._input_graph(i).x[:, :self.continuous_dim] for i in ids])
            mean[:self.continuous_dim] = x.mean(0)
            std = x.std(0, unbiased=False)
            scale[:self.continuous_dim] = torch.where(std > 0, std, torch.ones_like(std))
        return dict(mean=mean, scale=scale, fit_ids=list(ids))

    def ids(self, role):
        pools = dict(fit=self.fit_ids, query=self.query_ids, validation=self.validation_ids,
                     target=self.target_ids, all_source=self.train_ids)
        if role not in pools:
            raise ValueError('Unknown data role')
        return pools[role]

    def batch(self, ids, role, device='cpu', dtype=torch.float64):
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(self.ids(role)):
            raise PermissionError('Graph IDs do not belong to the requested data role')
        graphs = []
        for i in ids:
            if i not in self._cache:
                g = self._input_graph(i)
                g.x = (g.x - self.preprocessing['mean']) / self.preprocessing['scale']
                self._cache[i] = g
            g = self._cache[i].clone()
            if role != 'target':
                g.y = torch.tensor([self.labels[i]], dtype=torch.long)
            graphs.append(g)
        batch = Batch.from_data_list(graphs).to(device)
        batch.x = batch.x.to(dtype=dtype)
        if batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr.to(dtype=dtype)
        return batch

    def sample(self, role, count, generator, device='cpu', dtype=torch.float64):
        pool = self.ids(role)
        if not pool:
            raise ValueError(f'No graphs in role {role}')
        chosen = tuple(pool[i] for i in torch.randperm(len(pool), generator=generator)[:count].tolist())
        return self.batch(chosen, role, device, dtype), chosen
