"""Role-scoped TU inputs with separate source labels and frozen feature encoding."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
import torch

from .artifacts import output_file, read_json


@dataclass(frozen=True)
class Graph:
    id: str
    x: torch.Tensor
    node_type: torch.Tensor  # raw categorical ID, also retained for coarse source grouping
    edges: torch.Tensor      # local, distinct undirected pairs, smaller endpoint first
    edge_type: torch.Tensor  # source vocabulary index

    def __post_init__(self):
        x, nt, e, et = self.x, self.node_type, self.edges, self.edge_type
        if not isinstance(self.id, str) or not self.id:
            raise ValueError('graph ID must be a nonempty string')
        if x.ndim != 2 or min(x.shape) < 1 or not x.is_floating_point() or not x.isfinite().all():
            raise ValueError('expected finite floating x[n,d]')
        if nt.dtype != torch.long or nt.shape != (len(x),):
            raise ValueError('invalid node_type')
        if e.dtype != torch.long or e.ndim != 2 or e.shape[1] != 2:
            raise ValueError('expected int64 edges[m,2]')
        if et.dtype != torch.long or et.shape != (len(e),) or (et < 0).any():
            raise ValueError('invalid edge_type')
        if (e < 0).any() or (e >= len(x)).any() or (e[:, 0] >= e[:, 1]).any():
            raise ValueError('edges must be valid node pairs u < v')
        if len(e) != torch.unique(e, dim=0).shape[0]:
            raise ValueError('duplicate undirected edge')


@dataclass
class TaskData:
    graphs: dict[str, list[Graph]]
    labels: dict[str, torch.Tensor]
    preprocessing: dict


def _json(path):
    path = Path(path)
    if path.resolve().name.endswith('_graph_labels.txt'):
        raise ValueError('TU-wide class label files are outside graph-loading roles')
    return read_json(path)


def _raw_dir(config, name):
    if config.get('raw_dir'):
        return Path(config['raw_dir']).expanduser()
    if 'CELA_TU_ROOT' in os.environ:
        return Path(os.environ['CELA_TU_ROOT']).expanduser() / name / 'raw'
    # Existing project archives used to produce the checked-in smoke manifests.
    archive = {'Mutagenicity': 'Mutagenicity_official_20261026', 'PROTEINS': 'PROTEINS_official_20261023'}
    if name not in archive:
        raise ValueError('set raw_dir or CELA_TU_ROOT for this dataset')
    return Path.home() / 'research/GDA/TQG/data' / archive[name] / name / 'raw'


def _schema(manifest, raw, name):
    typed = manifest['typed_edges']
    if type(typed) is not bool or typed != (raw / f'{name}_edge_labels.txt').is_file():
        raise ValueError('typed_edges must agree with the TU edge label input')
    if (raw / f'{name}_edge_attributes.txt').is_file():
        raise ValueError('continuous edge attributes require a marked edge model')
    attr = manifest.get('continuous_node_attributes')
    if attr is not None:
        if (not isinstance(attr, dict) or attr.get('exogenous') is not True
                or not isinstance(attr.get('provenance'), str) or not attr['provenance'].strip()):
            raise ValueError('continuous node attributes need explicit exogenous provenance')
        cols = attr.get('columns')
        if (not isinstance(cols, list) or not cols or any(type(i) is not int or i < 0 for i in cols)
                or len(set(cols)) != len(cols)):
            raise ValueError('invalid or repeated continuous node column')
    node_file = (raw / f'{name}_node_labels.txt').is_file()
    if node_file and 'node_type_provenance' not in manifest:
        raise ValueError('declare exogenous node_type_provenance or set it to null to omit TU node labels')
    provenance = manifest.get('node_type_provenance')
    if provenance is not None and (not node_file or not isinstance(provenance, str) or not provenance.strip()):
        raise ValueError('node_type_provenance requires documented exogenous types and a TU node label file')
    return dict(typed_edges=typed, has_node_types=provenance is not None, node_type_provenance=provenance,
                continuous_node_attributes=attr)


def _fields(schema):
    names = ['graph_indicator', 'A']
    if schema['has_node_types']:
        names.append('node_labels')
    if schema['typed_edges']:
        names.append('edge_labels')
    if schema['continuous_node_attributes']:
        names.append('node_attributes')
    return names


def _table(src, field, schema):
    if field == 'node_attributes':
        cols = schema['continuous_node_attributes']['columns']
        return np.loadtxt(src, delimiter=',', dtype=np.float64, ndmin=2, usecols=cols)
    width = 2 if field == 'A' else 1
    size = src.getbuffer().nbytes if isinstance(src, BytesIO) else src.stat().st_size
    if size:
        # A strict integer converter also rejects fractional categorical IDs;
        # some NumPy versions accept and truncate them when only dtype is set.
        result = np.loadtxt(src, delimiter=',', dtype=np.int64, converters=int, ndmin=2)
        if result.shape[1] != width:
            raise ValueError(f'expected {width} integer columns for {field}')
    else:
        result = np.empty((0, width), dtype=np.int64)
    return result if field == 'A' else result[:, 0]


def _raw_graph(gid, nodes, edge_rows, node_types, edge_types, attributes):
    n = len(nodes)
    if n == 0 or (attributes is not None and (len(attributes) != n or not np.isfinite(attributes).all())):
        raise ValueError(f'graph {gid} has missing nodes or invalid continuous attributes')
    if node_types is None:
        node_types = np.zeros(n, dtype=np.int64)
    if edge_types is None:
        edge_types = np.zeros(len(edge_rows), dtype=np.int64)
    if len(edge_types) != len(edge_rows) or len(node_types) != n:
        raise ValueError(f'graph {gid} attribute row count mismatch')
    # TU raw A may contain each undirected pair once or both reciprocal records.
    # Repeated directions and conflicting reciprocal types do not describe a simple graph.
    offsets = {int(global_node): local for local, global_node in enumerate(nodes)}
    directed = set()
    pairs = {}
    for (a, b), tau in zip(edge_rows, edge_types, strict=True):
        if a not in offsets or b not in offsets or a == b:
            raise ValueError(f'graph {gid} has a cross-graph edge, invalid endpoint, or self-loop')
        u, v = offsets[a], offsets[b]
        pair = (min(u, v), max(u, v))
        if (u, v) in directed or (pair in pairs and pairs[pair] != tau):
            raise ValueError(f'graph {gid} has duplicate edges or conflicting reciprocal types')
        directed.add((u, v))
        pairs[pair] = int(tau)
    unique = sorted(pairs)
    return (node_types, np.asarray(unique, dtype=np.int64).reshape(-1, 2),
            np.asarray([pairs[p] for p in unique], dtype=np.int64), attributes)


def build_tu_index(config, output):
    """Prepare role-addressable byte ranges from nondecreasing TU graph blocks.

    This is a one-time import step. It reads raw graph inputs and writes a new
    index. It never opens class labels. Rows inside each graph can be in any
    order, including single or reciprocal records. Empty edge blocks are legal.
    """
    config = config.get('data', config)
    manifest = _json(config['manifest'])
    name = config['dataset']
    if manifest['dataset'] != name:
        raise ValueError('dataset identity mismatch')
    raw = _raw_dir(config, name)
    schema = _schema(manifest, raw, name)
    fields = _fields(schema)
    files = {f: raw / f'{name}_{f}.txt' for f in fields}
    indicator = _table(files['graph_indicator'], 'graph_indicator', schema)
    adj = _table(files['A'], 'A', schema)
    if not len(indicator) or (indicator < 1).any() or (adj < 1).any() or (adj > len(indicator)).any():
        raise ValueError('invalid TU node identity')
    owner = indicator[adj[:, 0]-1]
    if (owner != indicator[adj[:, 1]-1]).any():
        raise ValueError('TU adjacency crosses graph boundaries')
    if (np.diff(indicator) < 0).any() or (np.diff(owner) < 0).any():
        raise ValueError('TU index requires graph_indicator and A rows grouped by nondecreasing graph ID')
    ids = sorted(i for values in manifest['roles'].values() for i in values)
    if len(ids) != len(set(ids)) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError('TU index requires distinct zero-based role IDs')
    lower, upper = np.searchsorted(indicator, np.asarray(ids)+1), np.searchsorted(indicator, np.asarray(ids)+1, side='right')
    if (lower == upper).any():
        raise ValueError('role contains ID absent from TU graph_indicator')
    index = {str(i): dict(first_node=int(first)+1, spans={}) for i,first in zip(ids,lower)}
    for field, path in files.items():
        row_owner = owner if field in ('A', 'edge_labels') else indicator
        # Read bytes with original line endings. Blank/comment records are
        # unsupported because they would invalidate TU's row-aligned attributes.
        offsets = [0]
        with path.open('rb') as stream:
            for line in stream:
                offsets.append(offsets[-1] + len(line))
            if len(offsets) != len(row_owner)+1:
                raise ValueError(f'invalid TU row count for {field}; require exactly one line per record')
            for i in ids:
                left = offsets[np.searchsorted(row_owner, i+1)]
                right = offsets[np.searchsorted(row_owner, i+1, side='right')]
                stream.seek(left)
                digest = sha256(stream.read(right-left)).hexdigest()
                index[str(i)]['spans'][field] = [left, right-left, digest]
    with output_file(output) as tmp:
        tmp.write_text('{\n' + ',\n'.join('  ' + json.dumps(k) + ': ' + json.dumps(v, separators=(',', ':'))
                                       for k,v in index.items()) + '\n}\n', encoding='utf-8')
    return Path(output)


def _raw_graphs(raw, name, schema, ids, index):
    fields = _fields(schema)
    files = {f: raw / f'{name}_{f}.txt' for f in fields}
    # The fixed real smoke manifests carry hashes of graph-local byte ranges.
    # A train or A call can seek straight to its rows without reading H inputs.
    out = {}
    for i in ids:
        record = index[str(i)]
        arrays = {}
        for field, path in files.items():
            start, length, digest = record['spans'][field]
            if min(start, length) < 0:
                raise ValueError('invalid TU byte range')
            with path.open('rb') as stream:
                stream.seek(start)
                buf = stream.read(length)
            if len(buf) != length or sha256(buf).hexdigest() != digest:
                raise ValueError(f'graph {i} raw {field} bytes differ from manifest')
            arrays[field] = _table(BytesIO(buf), field, schema)
        indicator = arrays['graph_indicator']
        if not len(indicator) or not np.all(indicator == i+1):
            raise ValueError('TU index graph identity differs')
        nodes = record['first_node'] + np.arange(len(indicator))  # TU uses 1-based global nodes
        out[i] = _raw_graph(i, nodes, arrays['A'], arrays.get('node_labels'),
                            arrays.get('edge_labels'), arrays.get('node_attributes'))
    return out



def _fit(raw, ids, labels, schema, name):
    nodes = Counter(int(t) for i in ids for t in raw[i][0])
    edge_values = sorted({int(t) for i in ids for t in raw[i][2]}) if schema['typed_edges'] else [0]
    if not edge_values:
        raise ValueError('source training contains no observed edge type for a typed dataset')
    class_values = sorted(set(labels))
    if len(class_values) < 2:
        raise ValueError('source training must cover at least two classes')
    means, scales = [], []
    if schema['continuous_node_attributes']:
        attrs = np.concatenate([raw[i][3] for i in ids])
        mean, scale = attrs.mean(0), attrs.std(0)
        means, scales = mean.tolist(), np.where(scale > 0, scale, 1.).tolist()
    node_values = sorted(nodes) if schema['has_node_types'] else []
    return dict(version=1, dataset=name, fit_ids=ids, **schema, node_values=node_values,
                edge_values=edge_values, class_values=class_values, common_types=sorted(nodes, key=lambda t: (-nodes[t], t))[:4],
                continuous_mean=means, continuous_scale=scales, num_classes=len(class_values),
                num_edge_types=len(edge_values), input_dim=len(node_values)+1+len(means))


def _encode(gid, record, pre):
    nt, edges, et, attr = record
    values = pre['node_values']
    node_vocab = {t: i for i, t in enumerate(values)}
    x = np.zeros((len(nt), len(values) + 1), dtype=np.float32)
    x[np.arange(len(nt)), [node_vocab.get(t, len(values)) for t in nt]] = 1
    if attr is not None:
        normalized = (attr - np.asarray(pre['continuous_mean'])) / np.asarray(pre['continuous_scale'])
        x = np.concatenate((x, normalized.astype(np.float32)), axis=1)
    edge_vocab = {v: i for i, v in enumerate(pre['edge_values'])}
    try:
        edge_types = torch.tensor([edge_vocab[v] for v in et], dtype=torch.long)
    except KeyError as err:
        raise ValueError(f'graph {gid} edge type {err.args[0]} outside frozen source vocabulary') from err
    return Graph(str(gid), torch.from_numpy(x), torch.from_numpy(nt), torch.from_numpy(edges), edge_types)


def load_task(config, *, roles, preprocessing=None):
    """Read selected roles; fit source transforms only when train is requested.

    Manifest roles use zero-based TU graph indices; graph IDs and sidecar keys
    are their decimal strings. A and H have input graphs only, and checkpoint
    transforms are required when either role is requested.
    """
    config = config.get('data', config)
    roles = tuple(roles)
    if not roles or len(set(roles)) != len(roles) or set(roles) - {'train', 'val', 'A', 'H'}:
        raise ValueError('request distinct roles from train, val, A, H')
    if preprocessing is None and ('train' not in roles or set(roles) & {'A', 'H'}):
        raise ValueError('target or validation-only loading requires frozen source preprocessing')
    path = Path(config['manifest']).expanduser()
    manifest = _json(path)
    name = config['dataset']
    if manifest['dataset'] != name or not name or Path(name).name != name:
        raise ValueError('dataset identity mismatch')
    role_ids = manifest['roles']
    if set(role_ids) != {'train', 'val', 'A', 'H'} or set(manifest['source_labels']) != {'train','val'}:
        raise ValueError('expected four graph roles with two source-only label files')
    for values in role_ids.values():
        if not isinstance(values, list) or not values or any(type(i) is not int or i < 0 for i in values):
            raise ValueError('each role needs a nonempty list of zero-based TU IDs')
    all_ids = [i for values in role_ids.values() for i in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('graph IDs repeated or overlapping across roles')
    if 'tu_index' not in manifest:
        raise ValueError('stage loading requires tu_index; first prepare the raw TU inputs with python -m cela.data index')
    raw = _raw_dir(config, name)
    schema = _schema(manifest, raw, name)
    if preprocessing is not None and (preprocessing.get('dataset') != name or preprocessing.get('version') != 1
            or preprocessing.get('fit_ids') != role_ids['train']
            or any(preprocessing.get(k) != v for k, v in schema.items())):
        raise ValueError('frozen preprocessing does not match source identities or feature schema')
    labels = {}
    for role in roles:
        if role in ('train', 'val'):
            lookup = _json(path.parent / manifest['source_labels'][role])
            if set(lookup) != set(map(str, role_ids[role])) or any(type(y) is not int for y in lookup.values()):
                raise ValueError(f'{role} label sidecar must match exactly its graph IDs and contain raw integer labels')
            labels[role] = [lookup[str(i)] for i in role_ids[role]]
    index = _json(path.parent / manifest['tu_index'])
    records = _raw_graphs(raw, name, schema, [i for r in roles for i in role_ids[r]], index)
    if preprocessing is None:
        preprocessing = _fit(records, role_ids['train'], labels['train'], schema, name)
    class_map = {v: i for i, v in enumerate(preprocessing['class_values'])}
    for role, raw_labels in labels.items():
        if not set(raw_labels) <= class_map.keys():
            raise ValueError(f'{role} has a class absent from source training')
        labels[role] = torch.tensor([class_map[y] for y in raw_labels], dtype=torch.long)
    graphs = {r: [_encode(i, records[i], preprocessing) for i in role_ids[r]] for r in roles}
    return TaskData(graphs, labels, preprocessing)


def main():
    parser = argparse.ArgumentParser(description='Prepare graph-local reads for TU-format tasks')
    sub = parser.add_subparsers(dest='command', required=True)
    index = sub.add_parser('index', help='index input graph rows; class-label files are never opened')
    index.add_argument('--manifest', required=True, type=Path)
    index.add_argument('--raw-dir', required=True, type=Path)
    index.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest = _json(args.manifest)
    build_tu_index(dict(dataset=manifest['dataset'], manifest=args.manifest, raw_dir=args.raw_dir), args.output)


if __name__ == '__main__':
    main()
