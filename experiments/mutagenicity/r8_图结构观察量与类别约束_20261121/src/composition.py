"""Source-only graph composition vocabulary, normalization, and logistic scores."""
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch


def discrete_types(value, width, rows, name):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (rows, width)
            or not bool(((value == 0) | (value == 1)).all())
            or not bool((value.sum(dim=1) == 1).all())):
        raise ValueError(f'Unsupported {name} encoding: expected {width}-way one-hot types')
    return value.argmax(dim=1).tolist()


def graph_counts(graph, node_width, edge_width):
    """Count simple unordered pairs once, validating the sealed TSBA topology."""
    if type(graph) is not dict or set(graph) != {'x', 'edge_index', 'edge_attr'}:
        raise ValueError('Graph record must contain exactly x, edge_index and edge_attr')
    x, edges, edge_attr = graph['x'], graph['edge_index'], graph['edge_attr']
    if not isinstance(x, torch.Tensor) or not len(x):
        raise ValueError('A graph needs at least one node')
    nodes = discrete_types(x, node_width, len(x), 'node')
    if (not isinstance(edges, torch.Tensor) or edges.dtype != torch.int64
            or edges.ndim != 2 or edges.shape[0] != 2):
        raise ValueError('Invalid edge index')
    edge_types = discrete_types(edge_attr, edge_width, edges.shape[1], 'edge')
    directed = {}
    for (src, dst), kind in zip(edges.T.tolist(), edge_types):
        if not 0 <= src < len(nodes) or not 0 <= dst < len(nodes) or src == dst:
            raise ValueError('Invalid endpoint or self-loop in sealed TSBA graph')
        if (src, dst) in directed:
            raise ValueError('Duplicate directed edge in sealed TSBA graph')
        directed[src, dst] = kind
    pairs = Counter()
    for (src, dst), kind in directed.items():
        if directed.get((dst, src)) != kind:
            raise ValueError('Absent or inconsistent reverse edge type')
        if src < dst:
            lo, hi = sorted((nodes[src], nodes[dst]))
            pairs[lo, hi, kind] += 1
    return Counter(nodes), pairs


@dataclass
class Vocabulary:
    nodes: list
    edges: list

    @classmethod
    def fit(cls, source):
        return cls(sorted({key for nodes, _ in source for key in nodes}),
                   sorted({key for _, edges in source for key in edges}))

    def encode(self, records):
        def row(counter, vocabulary):
            total = counter.total()
            known = [counter[key] for key in vocabulary]
            values = np.array([*known, total - sum(known)], dtype=np.float64)
            return values / total if total else values

        return np.array([np.r_[row(nodes, self.nodes), row(edges, self.edges)] for nodes, edges in records])


def fit_logistic(x, y, offset, options, *, penalty=1.):
    """Minimize average CE + penalty*||weights||^2/(2*n), with a free intercept."""
    if set(y) != {0, 1} or x.ndim != 2 or x.shape[0] != len(y) or y.shape != offset.shape:
        raise ValueError('Invalid binary source logistic inputs')
    n = len(y)
    def objective(theta):
        w, intercept = theta[:-1], theta[-1]
        margin = x @ w + intercept + offset
        signed = (1 - 2*y) * margin
        residual = (1 - 2*y) * expit(signed)
        value = np.logaddexp(0., signed).mean() + penalty * (w @ w) / (2*n)
        return value, np.r_[x.T @ residual / n + penalty*w/n, residual.mean()]

    result = minimize(objective, np.zeros(x.shape[1]+1), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=options['max_iterations'], gtol=options['gradient_tolerance'],
                                   ftol=options['function_tolerance']))
    _, gradient = objective(result.x)
    error = float(np.max(np.abs(gradient)))
    if not result.success or not np.isfinite(error) or error > options['stationarity_tolerance']:
        raise ValueError(f'Source logistic fit failed: {result.message}, stationarity {error:g}')
    return dict(weights=result.x[:-1].tolist(), intercept=float(result.x[-1]), iterations=int(result.nit),
                training_objective=float(result.fun), gradient_inf=error)


def scores(x, model):
    return x @ np.asarray(model['weights']) + model['intercept']


def fit_composition(vocab, source, labels, config, solver):
    x = vocab.encode(source)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.
    model = fit_logistic((x-mean)/scale, labels, np.zeros(len(labels)), solver, penalty=1./config['C'])
    return dict(vocabulary=dict(nodes=vocab.nodes, edges=vocab.edges), mean=mean.tolist(), scale=scale.tolist(),
                model=model)


def composition_scores(vocabulary, counts, fitted):
    x = vocabulary.encode(counts)
    normalized = (x - fitted['mean']) / fitted['scale']
    return scores(normalized, fitted['model'])
