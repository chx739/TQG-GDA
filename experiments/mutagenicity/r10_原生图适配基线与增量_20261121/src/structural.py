"""Fit the registered sparse WL and two-feature size models on S/V only."""
from itertools import chain

import numpy as np
import torch

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import fit_logistic, scores
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl


SETTINGS = {
    'mutagenicity_edge': (14, 3),
    'mutagenicity_node': (14, 3),
    'proteins_edge': (3, 1),
}
DOMAINS = [0, 1, 2, 3]
SEEDS = [3405, 3406, 3407, 3408, 3409]
STRENGTHS = [0.1, 1., 10., 100.]
METHODS = ('WL_fixed', 'WL_selected', 'Size')
IDENTITY = ('setting', 'source', 'target', 'seed')
GROUP = ('setting', 'source', 'seed')


def validate_plan(plan):
    if (plan['settings'] != list(SETTINGS) or plan['domains'] != DOMAINS
            or plan['seeds'] != SEEDS or plan['C'] != STRENGTHS):
        raise ValueError('The registered R10 structural grid or candidate set changed')


def size(graph):
    edges = {tuple(sorted((a, b))) for a, b in graph.edge_index.T.tolist() if a != b}
    return np.log1p(np.array([len(graph.x), len(edges)], dtype=np.float64))


def features(graph, setting):
    node_types, edge_types = SETTINGS[setting]
    attr = graph.edge_attr
    if setting == 'proteins_edge':
        if attr is not None:
            raise ValueError('Expected the original untyped PROTEINS topology')
        attr = torch.ones((graph.edge_index.shape[1], 1), dtype=graph.x.dtype)
    counts, _ = wl.fingerprints(
        {'x': graph.x, 'edge_index': graph.edge_index, 'edge_attr': attr},
        {'node_types': node_types, 'edge_types': edge_types})
    return wl.normalize(counts), size(graph)


def decisions(margins):
    return (margins > 0.).astype(np.int64)


def validation_metrics(c, margin, y):
    correct = int(np.count_nonzero(decisions(margin) == y))
    return dict(C=c, correct=correct, n=len(y), accuracy=correct / len(y),
                nll=float(np.logaddexp(0., (1 - 2*y) * margin).mean()))


def choose(candidates):
    return min(candidates, key=lambda row: (-row['correct'], row['nll'], row['C']))['C']


def fit_candidates(s, v, ys, yv, plan):
    fits, validation = {}, []
    options = plan['solver']
    for c in plan['C']:
        offset = np.zeros(len(ys))
        penalty = 1. / c
        try:
            fitted = fit_logistic(s, ys, offset, options, penalty=penalty)
        except ValueError as error:
            if not str(error).startswith('Source logistic fit failed: ABNORMAL:'):
                raise
            # Retry the same objective with the pre-registered stationarity limit
            # as the optimizer's success threshold before the line search stalls.
            retry = dict(options, gradient_tolerance=options['stationarity_tolerance'])
            fitted = fit_logistic(s, ys, offset, retry, penalty=penalty)
            weights = np.asarray(fitted['weights'])
            signed = (1 - 2*ys) * (scores(s, fitted) + offset)
            objective = float(np.logaddexp(0., signed).mean() + penalty * (weights @ weights) / (2*len(ys)))
            reported = fitted['training_objective']
            if (not np.isfinite(objective) or not np.isclose(reported, objective, rtol=1e-12, atol=1e-15)
                    or not 0. <= fitted['gradient_inf'] <= options['stationarity_tolerance']):
                raise ValueError(f'Uncertified line-search recovery for C={c}') from error
            fitted['training_objective'] = objective
            fitted['line_search_recovery'] = dict(original_error=str(error), reported_objective=reported,
                                                  gradient_tolerance=retry['gradient_tolerance'])
        fits[str(c)] = fitted
        validation.append(validation_metrics(c, scores(v, fitted), yv))
    return dict(candidates=fits, validation=validation, selected_C=choose(validation))


def fit_models(source, validation, source_y, validation_y, plan):
    s_wl, s_size = zip(*source)
    v_wl, v_size = zip(*validation)
    lexicon = wl.vocabulary(s_wl, range(len(s_wl)))
    source_matrix = wl.sparse_rows(s_wl, range(len(s_wl)), lexicon)
    validation_matrix = wl.sparse_rows(v_wl, range(len(v_wl)), lexicon)
    fitted_wl = fit_candidates(source_matrix, validation_matrix, source_y, validation_y, plan)
    s_size = np.asarray(s_size)
    mean, scale = s_size.mean(axis=0), s_size.std(axis=0)
    scale[np.ptp(s_size, axis=0) == 0.] = 1.
    fitted_size = fit_candidates((s_size - mean) / scale, (v_size - mean) / scale,
                                 source_y, validation_y, plan)
    return dict(wl=dict(fitted_wl, vocabulary=[[depth, color.hex()] for depth, color in lexicon]),
                size=dict(fitted_size, mean=mean.tolist(), scale=scale.tolist()))


def select_model(fitted):
    return fitted['candidates'][str(fitted['selected_C'])]


def predict(fitted, held):
    fitted_wl, fitted_size = fitted['wl'], fitted['size']
    ids = {(depth, bytes.fromhex(color)): index
           for index, (depth, color) in enumerate(fitted_wl['vocabulary'])}
    h_wl, h_size = zip(*held)
    individual = wl.sparse_rows(h_wl, range(len(h_wl)), ids)
    standardized = (np.asarray(h_size) - fitted_size['mean']) / fitted_size['scale']
    return dict(WL_fixed=scores(individual, fitted_wl['candidates']['1.0']),
                WL_selected=scores(individual, select_model(fitted_wl)),
                Size=scores(standardized, select_model(fitted_size)))


def check_roles(pools):
    """Validate the four full-domain blocks and the common S/V order."""
    if len(pools) != 3:
        raise PermissionError('Expected three registered target roles')
    first = pools[0]
    if any(set(item) != {'source', 'validation', 'adapt', 'holdout'}
                              or item['source'] != first['source']
                              or item['validation'] != first['validation'] for item in pools):
        raise PermissionError('Expected three target roles with the same ordered source S/V')
    distinct = [first['source'], first['validation'],
                *(item[role] for item in pools for role in ('adapt', 'holdout'))]
    all_ids = list(chain.from_iterable(distinct))
    if any(not ids for ids in distinct) or len(all_ids) != len(set(all_ids)):
        raise PermissionError('Archived S/V/A/H roles overlap or are empty')


def take_role(store, ids, roles, role, *, fitting):
    allowed = ('source', 'validation') if fitting else ('holdout',)
    if role not in allowed or ids != roles[role]:
        raise PermissionError('Topology request differs from the current graph role')
    return store.take(ids)
