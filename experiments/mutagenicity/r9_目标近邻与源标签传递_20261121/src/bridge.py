"""One-hop/two-hop weights and same-multiset source-identity controls."""
import numpy as np
from scipy.special import expit

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import fit_logistic, scores
from . import wl


class NoCandidates(ValueError):
    pass


def similarity_order(similarity, ids):
    return np.lexsort((ids, -similarity))


def row_weights(kernel, candidate_ids, k, allowed=None):
    ids = np.asarray(candidate_ids)
    if kernel.shape[1] != len(ids) or not np.isfinite(kernel).all() or np.any(kernel < 0):
        raise ValueError('Invalid nonnegative kernel matrix')
    result = np.zeros_like(kernel)
    fallbacks = 0
    for index, row in enumerate(kernel):
        valid = np.arange(len(ids)) if allowed is None else np.flatnonzero(allowed[index])
        if not len(valid):
            raise NoCandidates(f'No permitted source candidates for row {index}')
        positive = valid[row[valid] > 0]
        if not len(positive):
            result[index, valid] = 1./len(valid)
            fallbacks += 1
            continue
        chosen = positive[similarity_order(row[positive], ids[positive])[:k]]
        result[index, chosen] = row[chosen] / row[chosen].sum()
    return result, fallbacks


def reassign(main, direct_kernel, source_ids):
    result = np.zeros_like(main)
    ids = np.asarray(source_ids)
    for index, row in enumerate(main):
        ranked = similarity_order(direct_kernel[index], ids)
        result[index, ranked] = np.sort(row)[::-1]
    return result


def source_allowed(mids, source_ids, fingerprints):
    return np.asarray([[m != s and fingerprints[m] != fingerprints[s] for s in source_ids] for m in mids])


def probability_output(base, weights, labels, blend):
    base_logs = np.column_stack((-np.logaddexp(0., base), -np.logaddexp(0., -base)))
    if weights is None:
        return base_logs, expit(base)
    vote = weights @ labels
    if np.any(vote < -1e-12) or np.any(vote > 1.+1e-12):
        raise ValueError('Source vote is outside the simplex')
    vote = np.clip(vote, 0., 1.)
    with np.errstate(divide='ignore'):
        logs = np.logaddexp(np.log1p(-blend)+base_logs,
                            np.log(blend)+np.column_stack((np.log1p(-vote), np.log(vote))))
    return logs, (1.-blend)*expit(base)+blend*vote


def fit_source(rows, ids, query, labels, config):
    names = wl.vocabulary(rows, ids)
    x = wl.sparse_rows(rows, ids, names)
    if set(labels) != {0, 1}:
        raise ValueError('Pseudo-source logistic training requires both classes')
    fitted = fit_logistic(x, labels, np.zeros(len(ids)), config, penalty=1./config['C'])
    z = scores(wl.sparse_rows(rows, query, names), fitted)
    keys = [[depth, color.hex()] for depth, color in names]
    return z, dict(**fitted, vocabulary=keys, n_labels=len(labels))


def row_diagnostics(first, second):
    shared = ((first > 0) & (second > 0)).sum(axis=1)
    union = ((first > 0) | (second > 0)).sum(axis=1)
    return dict(overlap=shared, support_jaccard=shared/union,
                total_variation=np.abs(first-second).sum(axis=1)/2,
                top_weight_source_equal=first.argmax(axis=1)==second.argmax(axis=1))


def predict(rows, fingerprint_keys, s, h, a, mids, labels, base, plan):
    k, atol = plan['k'], plan['weight_atol']
    if not s or not h or not a or len(a) != len(mids) or not set(mids) <= set(s):
        raise NoCandidates('Pseudo-transfer role has an empty or unmatched middle pool')
    direct_k = wl.kernel(rows, h, s)
    direct, direct_fallback = row_weights(direct_k, s, k)
    h_a, h_a_fallback = row_weights(wl.kernel(rows, h, a), a, k)
    a_s, a_s_fallback = row_weights(wl.kernel(rows, a, s), s, k)
    target = h_a @ a_s
    h_m, h_m_fallback = row_weights(wl.kernel(rows, h, mids), mids, k)
    allowed = source_allowed(mids, s, fingerprint_keys)
    m_s, m_s_fallback = row_weights(wl.kernel(rows, mids, s), s, k, allowed=allowed)
    if np.any(m_s[~allowed]):
        raise ValueError('Forbidden source middle pairs received weight')
    controls = dict(Direct=direct, TargetBridge=target, MatchedWeights=reassign(target, direct_k, s),
                    SourceBridge=h_m @ m_s, PooledBridge=np.broadcast_to(a_s.mean(axis=0), target.shape))
    np.testing.assert_array_equal(np.sort(target, axis=1), np.sort(controls['MatchedWeights'], axis=1))
    arrays = dict(h_ids=np.asarray(h), s_ids=np.asarray(s), a_ids=np.asarray(a), middle_ids=np.asarray(mids),
                  h_to_a=h_a, a_to_s=a_s, h_to_source_middle=h_m, source_middle_to_s=m_s,
                  h_to_s_kernel=direct_k)
    logs, p = probability_output(base, None, labels, plan['blend_weight'])
    arrays.update(logp_B=logs, probability_B=p, action_B=(p > .5).astype(np.int64))
    normalization_error = {}
    for arm, weights in controls.items():
        if np.any(weights < 0):
            raise ValueError('A propagation arm returned negative weights')
        error = float(np.abs(weights.sum(axis=1)-1).max())
        if error > atol:
            raise ValueError(f'{arm} row weights do not sum to one')
        normalization_error[arm] = error
        logp, p = probability_output(base, weights, labels, plan['blend_weight'])
        arrays.update({f'weights_{arm}': weights, f'logp_{arm}': logp, f'probability_{arm}': p,
                       f'action_{arm}': (p > .5).astype(np.int64)})
        positive = np.where(weights > 0, weights, 1.)
        arrays[f'support_{arm}'] = (weights > 0).sum(axis=1)
        arrays[f'entropy_{arm}'] = -(weights*np.log(positive)).sum(axis=1)
        arrays[f'effective_neighbors_{arm}'] = 1./np.square(weights).sum(axis=1)
        if arm != 'TargetBridge':
            for key, value in row_diagnostics(target, weights).items():
                arrays[f'{key}_vs_{arm}'] = value
    for key in ('support', 'entropy', 'effective_neighbors'):
        np.testing.assert_allclose(arrays[f'{key}_TargetBridge'], arrays[f'{key}_MatchedWeights'], rtol=0, atol=atol)
    metadata = dict(fallback_rows=dict(Direct=direct_fallback, H_to_A=h_a_fallback, A_to_S=a_s_fallback,
                                       H_to_source_middle=h_m_fallback, source_middle_to_S=m_s_fallback),
                    row_normalization_max_error=normalization_error, exact_weight_multiset_match=True,
                    source_middle_excluded_pairs=int((~allowed).sum()))
    return arrays, metadata
