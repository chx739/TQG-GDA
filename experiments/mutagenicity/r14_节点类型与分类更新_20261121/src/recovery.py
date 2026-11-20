"""Masked-node observations, source type recovery, and per-graph BN gradients."""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import discrete_types
from . import activations


def positions(dataset, graph_id, n, maximum):
    if n < 1:
        raise ValueError('A graph must contain at least one node')
    return sorted(range(n), key=lambda i: (
        sha256(f'r14-node-v1|{dataset}|{graph_id}|{i}'.encode('utf-8')).digest(), i))[:maximum]


def masked_graph(graph, node):
    clone = graph.clone('x')
    clone.x[node].zero_()
    return clone


@dataclass(frozen=True)
class Masked:
    ids: np.ndarray
    owner: np.ndarray
    nodes: np.ndarray
    types: np.ndarray
    normalized: np.ndarray
    bn_output: np.ndarray

    @property
    def sizes(self):
        return np.bincount(self.owner, minlength=len(self.ids))

    @property
    def weights(self):
        return 1. / self.sizes[self.owner]

    def mean_by_graph(self, matrix):
        shape = (len(self.ids), *matrix.shape[1:])
        total = np.zeros(shape, dtype=np.float64)
        np.add.at(total, self.owner, matrix)
        return total / self.sizes.reshape((-1,) + (1,) * (matrix.ndim - 1))


@torch.inference_mode()
def masked_activations(model, graphs, ids, dataset, maximum, batch_size):
    if len(graphs) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Masked pool IDs and input graphs do not match')
    copies, owners, nodes, labels = [], [], [], []
    for index, (gid, g) in enumerate(zip(ids, graphs)):
        types = discrete_types(g.x, model.encoder.spec['in_dim'], len(g.x), 'node')
        for node in positions(dataset, gid, len(g.x), maximum):
            copies.append(masked_graph(g, node))
            owners.append(index)
            nodes.append(node)
            labels.append(types[node])
    bn = model.encoder.norms[-1]
    observed, pre_activation = [], []
    selected = None

    def collect(_m, inputs, output):
        pre_activation.append(activations.tensor(output[selected]))
        observed.append(activations.normalize(inputs[0][selected], bn))

    handle = bn.register_forward_hook(collect)
    try:
        for indices, batch in activations.batches(copies, batch_size):
            selected = batch.ptr[:-1] + torch.tensor([nodes[i] for i in indices])
            model.encoder(batch)
    finally:
        handle.remove()
    return Masked(np.asarray(ids), np.asarray(owners), np.asarray(nodes), np.asarray(labels),
                  np.concatenate(observed), np.concatenate(pre_activation))


def fit_type_head(masked, options):
    seen, inverse = np.unique(masked.types, return_inverse=True)
    nclasses, ngraphs = len(seen), len(masked.ids)
    weights = masked.weights
    x = np.maximum(masked.bn_output, 0.)
    width = x.shape[1]
    common = dict(seen=seen.tolist(), majority=int(seen[np.argmax(np.bincount(inverse, weights=weights))]),
                  fit_graphs=ngraphs, fit_masks=len(weights))
    if nclasses == 1:
        return common | dict(weights=[[0.] * width], intercept=[0.], fitted=False, stop='single_source_type')

    def unpack(params):
        return params[:nclasses*width].reshape(nclasses, width), np.r_[params[nclasses*width:], 0.]

    def objective(params):
        w, b = unpack(params)
        logits = x @ w.T + b
        norm = logsumexp(logits, axis=1)
        loss = (weights @ (norm - logits[np.arange(len(x)), inverse]) + .5*np.square(w).sum()) / ngraphs
        residual = np.exp(logits - norm[:, None])
        residual[np.arange(len(x)), inverse] -= 1.
        residual *= weights[:, None] / ngraphs
        gradient = np.r_[(residual.T @ x + w/ngraphs).ravel(), residual.sum(axis=0)[:-1]]
        return loss, gradient

    result = minimize(objective, np.zeros(nclasses*width + nclasses-1), jac=True, method='L-BFGS-B',
                      options=dict(ftol=options['ftol'], gtol=options['gtol'], maxiter=options['max_iterations']))
    w, b = unpack(result.x)
    error = float(np.max(np.abs(objective(result.x)[1])))
    if not result.success or not np.isfinite(error) or error > options['stationarity_tolerance']:
        raise ValueError(f'R14 source node recovery did not converge: {result.message}; residual={error}')
    return common | dict(weights=w.tolist(), intercept=b.tolist(), fitted=True, iterations=int(result.nit),
                         mean_penalized_loss=float(result.fun), gradient_inf=error)


def node_scores(masked, fitted):
    w = np.asarray(fitted['weights'])
    return np.maximum(masked.bn_output, 0.) @ w.T + fitted['intercept']


def recovery_diagnostic(masked, fitted):
    seen = np.asarray(fitted['seen'])
    correct = seen[node_scores(masked, fitted).argmax(1)] == masked.types
    majority = masked.types == fitted['majority']
    unknown = ~np.isin(masked.types, seen)
    return dict(recovery_accuracy=float(masked.mean_by_graph(correct).mean()),
                majority_accuracy=float(masked.mean_by_graph(majority).mean()),
                unknown_rate=float(np.mean(unknown)), unknown_masks=int(unknown.sum()),
                graphs=len(masked.ids), masks=len(masked.types))


def targets(masked, seen):
    seen = np.asarray(seen)
    one_hot = masked.types[:, None] == seen
    unknown = ~one_hot.any(axis=1)
    true = one_hot.astype(float)
    true[unknown] = 1. / len(seen)
    histogram = masked.mean_by_graph(true)[masked.owner]
    # Copy the identical soft target exactly for a graph with only one
    # observed type, including a type unseen while fitting the source head.
    start = np.r_[0, np.cumsum(masked.sizes)[:-1]]
    homogeneous = np.minimum.reduceat(masked.types, start) == np.maximum.reduceat(masked.types, start)
    histogram[homogeneous[masked.owner]] = true[homogeneous[masked.owner]]
    return true, histogram, unknown, homogeneous


def gradients(masked, fitted):
    true, hist, unknown, homogeneous = targets(masked, fitted['seen'])
    w = np.asarray(fitted['weights'])
    z = node_scores(masked, fitted)
    log_probs = z - logsumexp(z, axis=1, keepdims=True)
    prob = np.exp(log_probs)

    def per_graph(residual):
        # The source loss uses the native ReLU zero subgradient. H diagnostics
        # use the distinct right directional derivative at a zero.
        grad_beta = (residual @ w) * (masked.bn_output > 0.)
        return masked.mean_by_graph(np.c_[grad_beta * masked.normalized, grad_beta])

    grad = {'True': per_graph(prob - true), 'Hist': per_graph(prob - hist),
            'Binding': per_graph(hist - true)}
    np.testing.assert_allclose(grad['Binding'], grad['True'] - grad['Hist'], rtol=1e-12, atol=1e-12)
    if np.any(grad['Binding'][homogeneous] != 0.):
        raise AssertionError('A single-type graph produced nonzero Binding contribution')
    known = np.isin(masked.types, fitted['seen'])
    metrics = dict(graphs=len(masked.ids), masks=len(masked.types), distinct_true_types=len(set(masked.types)),
                   graphs_two_true_types=int((~homogeneous).sum()), known_masks=int(known.sum()),
                   unknown_masks=int(unknown.sum()), unknown_rate=float(unknown.mean()),
                   unknown_graph_weight=float(masked.mean_by_graph(unknown).mean()),
                   ce_true=float(masked.mean_by_graph(-np.sum(true * log_probs, axis=1)).mean()),
                   ce_hist=float(masked.mean_by_graph(-np.sum(hist * log_probs, axis=1)).mean()))
    return grad, metrics


def directions(gradients_by_graph, metrics, source_types, minimum_norm):
    output = {}
    for arm, by_graph in gradients_by_graph.items():
        g = by_graph.mean(axis=0)
        norm = float(np.linalg.norm(g))
        if not np.isfinite(norm):
            raise ValueError('Nonfinite R14 gradient norm')
        stop = ('single_source_type' if source_types < 2 else
                'single_pool_type' if metrics['distinct_true_types'] < 2 else
                'norm_below_floor' if norm <= minimum_norm else None)
        unit = np.zeros_like(g) if stop else -g/norm
        output[arm] = dict(norm=norm, gradient=g.tolist(), unit=unit.tolist(), fallback=stop)
    return output
