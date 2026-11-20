"""Typed WL count fingerprints and normalized sparse graph kernel features."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import struct

import numpy as np
from scipy.sparse import csr_matrix

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import discrete_types, graph_counts


def hash_color(payload):
    return hashlib.sha256(payload).digest()


def fingerprints(graph, plan):
    widths = plan['node_types'], plan['edge_types']
    _, edges = graph_counts(graph, *widths)
    num_nodes = len(graph['x'])
    node_type = discrete_types(graph['x'], widths[0], num_nodes, 'node')
    e_type = discrete_types(graph['edge_attr'], widths[1], graph['edge_index'].shape[1], 'edge')
    neighbors = [[] for _ in node_type]
    for (source, target), label in zip(graph['edge_index'].T.tolist(), e_type):
        neighbors[source].append((label, target))
    colors = [hash_color(b'R9-WL0\0'+struct.pack('>I', value)) for value in node_type]
    counts = [Counter(colors)]
    for depth in (1, 2):
        new_colors = []
        for center, adjacent in enumerate(neighbors):
            keys = sorted((label, colors[target]) for label, target in adjacent)
            body = b''.join(struct.pack('>I', label)+color for label, color in keys)
            new_colors.append(hash_color(b'R9-WL\0'+struct.pack('>I', depth)+colors[center]
                                         +struct.pack('>I', len(keys))+body))
        colors = new_colors
        counts.append(Counter(colors))
    return counts, edges.total()


def primitive(row):
    divisor = math.gcd(*row.values())
    return {color: count//divisor for color, count in row.items()}


def normalize(counts):
    result = {}
    for depth, raw in enumerate(counts):
        row = primitive(raw)
        scale = math.sqrt(3*sum(count*count for count in row.values()))
        result.update({(depth, color): n/scale for color, n in row.items()})
    return result


def count_key(counts):
    value = [[[color.hex(), count] for color, count in sorted(primitive(row).items())] for row in counts]
    return json.dumps(value, ensure_ascii=True, separators=(',', ':')).encode('ascii')


def sparse_rows(rows, order, vocabulary):
    values, coordinates, starts = [], [], [0]
    for i in order:
        for key, val in rows[i].items():
            if key in vocabulary:
                coordinates.append(vocabulary[key])
                values.append(val)
        starts.append(len(values))
    matrix = csr_matrix((values, coordinates, starts), shape=(len(order), len(vocabulary)), dtype=np.float64)
    matrix.sort_indices()
    return matrix


def vocabulary(rows, ids):
    return {key: n for n, key in enumerate(sorted({key for i in ids for key in rows[i]}))}


def kernel(rows, query, candidate):
    # No query-query kernel is evaluated. Graphs retain all coordinates in
    # their per-layer denominators, including unseen colors.
    names = vocabulary(rows, candidate)
    left = sparse_rows(rows, query, names)
    right = sparse_rows(rows, candidate, names)
    return (left @ right.T).toarray()


def by_fingerprint(counts, ids):
    groups = defaultdict(list)
    for i in ids:
        groups[count_key(counts[i])].append(i)
    return {key: sorted(ids) for key, ids in groups.items()}


def make_records(graphs, plan):
    computed = {i: fingerprints(graph, plan) for i, graph in graphs.items()}
    counts = {i: pair[0] for i, pair in computed.items()}
    edges = {i: pair[1] for i, pair in computed.items()}
    rows = {i: normalize(vec) for i, vec in counts.items()}
    return counts, edges, rows
