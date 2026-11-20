"""Source-fit topology and node-composition probes for R12."""
import numpy as np
import torch

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl


def node_frequency(graph, width):
    x = graph['x']
    if len(x) == 0:
        raise ValueError('A graph must contain nodes')
    types = composition.discrete_types(x, width, len(x), 'node')
    return np.bincount(types, minlength=width).astype(float) / len(types)


def topology_counts(graph):
    edge_index = graph['edge_index']
    untyped = dict(x=torch.ones((len(graph['x']), 1)), edge_index=edge_index,
                   edge_attr=torch.ones((edge_index.shape[1], 1)))
    return wl.fingerprints(untyped, {'node_types': 1, 'edge_types': 1})[0]


def quantile_bins(margins):
    margins = np.asarray(margins, dtype=np.float64)
    if margins.ndim != 1 or not len(margins) or not np.isfinite(margins).all():
        raise ValueError('B0 margins must be a finite nonempty vector')
    cuts = np.quantile(margins, [.25, .5, .75], method='linear')
    bins = np.searchsorted(cuts, margins, side='left') + 1
    return cuts, bins


def fit_linear(x_s, x_v, y_s, options):
    y_s = np.asarray(y_s, dtype=np.int64)
    fit = composition.fit_logistic(x_s, y_s, np.zeros(len(y_s)), options, penalty=1./options['C'])
    s_scores, v_scores = composition.scores(x_s, fit), composition.scores(x_v, fit)
    threshold = float(np.median(s_scores))
    return fit | {'threshold': threshold}, v_scores, (v_scores > threshold).astype(np.int8)


def fit(graphs, fit_ids, validation_ids, y_s, node_width, options):
    if set(graphs) != set(fit_ids) | set(validation_ids) or set(fit_ids) & set(validation_ids):
        raise PermissionError('Probe graph records do not match disjoint S/V roles')
    l_s = np.stack([node_frequency(graphs[i], node_width) for i in fit_ids])
    l_v = np.stack([node_frequency(graphs[i], node_width) for i in validation_ids])
    l_fit, l_score, l = fit_linear(l_s, l_v, y_s, options)
    rows = {i: wl.normalize(topology_counts(graphs[i])) for i in graphs}
    vocab = wl.vocabulary(rows, fit_ids)
    t_s = wl.sparse_rows(rows, fit_ids, vocab)
    t_v = wl.sparse_rows(rows, validation_ids, vocab)
    t_fit, t_score, t = fit_linear(t_s, t_v, y_s, options)
    return dict(L=l_fit, T=t_fit | {'vocabulary': [[depth, color.hex()] for depth, color in vocab]}), dict(
        l_score=l_score, t_score=t_score, l=l, t=t)
