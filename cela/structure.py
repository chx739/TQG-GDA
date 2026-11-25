"""Prefix-only topology and the ten fixed CELA structural tokens."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


NUM_TOKENS = 10


def block_bounds(m: int, blocks: int):
    """Yield half-open blocks, keeping the final step separate when possible."""
    if blocks < 1:
        raise ValueError("blocks must be positive")
    if m <= blocks:
        for i in range(m):
            yield i, i + 1
    elif blocks == 1:
        yield 0, m
    else:
        width, extra = divmod(m - 1, blocks - 1)
        start = 0
        for i in range(blocks - 1):
            stop = start + width + int(i < extra)
            yield start, stop
            start = stop
        yield start, m


def pair_ids(u, v, n):
    """Indices in np.triu_indices(n, 1), given u < v."""
    return u * (2 * n - u - 1) // 2 + v - u - 1


def unordered_categories(x, y, count):
    """Lexicographic index of the unordered category pair, including ties."""
    low, high = np.minimum(x, y), np.maximum(x, y)
    return low * (2 * count - low - 1) // 2 + high


@dataclass
class Prefix:
    u: np.ndarray
    v: np.ndarray
    degree: np.ndarray
    token: np.ndarray
    group: np.ndarray       # compact pair stratum, before crossing with edge type
    group_key: np.ndarray   # index of (coarse type pair, path category)
    group_count: np.ndarray
    token_count: np.ndarray # [pair stratum, token]


def prefix_layout(n, prefix_edges, u, v, coarse_type, num_coarse_types):
    """Build topology for the supplied remaining node pairs at block entry."""
    s, t = prefix_edges.T
    if len(s):
        row = np.concatenate((s, t))
        col = np.concatenate((t, s))
        adjacency = csr_matrix((np.ones(len(row), dtype=bool), (row, col)), shape=(n, n))
        degree = np.diff(adjacency.indptr).astype(np.int64)
        _, component = connected_components(adjacency, directed=False)
        shared = np.asarray((adjacency @ adjacency)[u, v]).reshape(-1)
        path = np.where(component[u] != component[v], 0, np.where(shared, 1, 2))
    else:
        degree = np.zeros(n, dtype=np.int64)
        path = np.zeros(len(u), dtype=np.int64)

    roles = np.minimum(degree, 3)
    token = unordered_categories(roles[u], roles[v], 4)
    type_pair = unordered_categories(coarse_type[u], coarse_type[v], num_coarse_types)
    group_key, group, group_count = np.unique(type_pair * 3 + path, return_inverse=True,
                                              return_counts=True)
    token_count = np.bincount(group * NUM_TOKENS + token,
                              minlength=len(group_key) * NUM_TOKENS)
    return Prefix(u, v, degree, token, group, group_key, group_count,
                  token_count.reshape(-1, NUM_TOKENS))
