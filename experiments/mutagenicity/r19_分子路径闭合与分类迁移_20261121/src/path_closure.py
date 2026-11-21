"""Fit a fixed ridge correction based on path closures above a background rate."""
from fractions import Fraction
from hashlib import sha256

import numpy as np

from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src.feature_dropout_transfer import count_matched


CONTINUOUS = ('WL', 'Intercept', 'SourceBg', 'TargetBg', 'Raw', 'Mixture')
METHODS = (*CONTINUOUS, 'Count')


def reference_ids(key, roles):
    m = min(len(roles[name]) for name in ('C', 'D', 'U'))
    return {name: sorted(roles[name], key=lambda i: (
        sha256(f'ring-r19|{key}|{name}|{i}'.encode('utf-8')).digest(), i))[:m] for name in ('D', 'U')}


def moments(rows):
    """Exact sums of per-node closure and opportunity counts."""
    return tuple(tuple(sum((Fraction(row[j], row[0]) for row in rows), Fraction())
                       for j in group) for group in (range(6, 10), range(2, 6)))


def rate(closure, opportunity, fallback):
    return tuple(c / o if o else f for c, o, f in zip(closure, opportunity, fallback, strict=True))


def mixture(rows, labels, source, target, target_o, all_d_rate):
    groups = [[row for row, y in zip(rows, labels, strict=True) if y == k] for k in (0, 1)]
    if not all(groups):
        return source, None, None
    means = [tuple(tuple(v / len(group) for v in m) for m in moments(group)) for group in groups]

    def class_rate(prior):
        neg, pos = means
        c, o = ([a + prior * (b - a) for a, b in zip(n, p, strict=True)]
                for n, p in zip(neg, pos, strict=True))
        return rate(c, o, all_d_rate)

    anchor = class_rate(Fraction(len(groups[1]), len(rows)))
    best = None
    for i in range(1001):
        pi = Fraction(i, 1000)
        bg = tuple(min(1, max(0, s + h - a))
                   for s, h, a in zip(source, class_rate(pi), anchor, strict=True))
        error = sum(w * (p - q)**2 for w, p, q in zip(target_o, bg, target, strict=True))
        if best is None or error < best[2]:
            best = bg, pi, error
    return best


def per_node(rows):
    x = np.asarray(rows, dtype=np.float64)
    return x[:, 6:10] / x[:, :1], x[:, 2:6] / x[:, :1]


def excess(rows, rates):
    # Preserve exact zero columns before their RMS is computed.
    p = tuple(Fraction(v) for v in rates)
    return np.asarray([[(c - o*q) / row[0] for o, c, q in zip(row[2:6], row[6:10], p, strict=True)]
                       for row in rows], dtype=np.float64)


def design(x, rms):
    rms = np.asarray(rms, dtype=np.float64)
    scaled = np.divide(x, rms, out=np.zeros_like(x), where=rms != 0.)
    return scaled / np.sqrt(x.shape[1])


def ridge(x, residual):
    rms = np.sqrt(np.mean(x*x, axis=0))
    z = design(x, rms)
    w = np.linalg.solve(z.T @ z + np.eye(z.shape[1]), z.T @ residual)
    return dict(rms=rms.tolist(), coef=w.tolist())


def fit(key, rows, roles, y_d, y_c, f_c):
    refs = reference_ids(key, roles)
    d = [rows[i] for i in roles['D']]
    if (len(d) != len(y_d) or len(f_c) != len(y_c) or len(y_c) != len(roles['C'])
            or not np.isin(y_d, (0, 1)).all() or not np.isin(y_c, (0, 1)).all()):
        raise ValueError('D/C data and binary labels do not align')
    totals = {k: moments([rows[i] for i in v]) for k, v in refs.items()}
    totals['D_all'] = moments(d)
    all_d = rate(*totals['D_all'], (Fraction(),)*4)
    source = rate(*totals['D'], all_d)
    target = rate(*totals['U'], source)
    mix, pi, sqerr = mixture(d, y_d, source, target, totals['U'][1], all_d)
    c, o = per_node([rows[i] for i in roles['C']])
    residual = 2*y_c.astype(np.int64) - 1 - f_c
    source_origin = ['D' if a else ('D_all' if b else 'zero')
                     for a, b in zip(totals['D'][1], totals['D_all'][1], strict=True)]
    return dict(references=refs, m=len(refs['D']), intercept=float(residual.mean()),
                pS=list(map(str, source)), pU=list(map(str, target)), pMix=list(map(str, mix)),
                source_rate_origin=source_origin,
                target_rate_origin=['U' if a else 'pS' for a in totals['U'][1]],
                totals={name: [list(map(str, v)) for v in values] for name, values in totals.items()},
                mixture_prior=str(pi) if pi is not None else None,
                mixture_objective=str(sqerr) if sqerr is not None else None,
                background=ridge(excess([rows[i] for i in roles['C']], source), residual),
                raw=ridge(np.column_stack((c, o)), residual))


def actions(rows, ids, static, model):
    selected = [rows[i] for i in ids]
    c, o = per_node(selected)
    bg = model['background']

    def background(name):
        return static + design(excess(selected, model[name]), bg['rms']) @ bg['coef']

    raw = model['raw']
    scores = dict(WL=static, Intercept=static + model['intercept'], SourceBg=background('pS'),
                  TargetBg=background('pU'), Raw=static + design(np.column_stack((c, o)), raw['rms']) @ raw['coef'],
                  Mixture=background('pMix'))
    decisions = {method: z > 0 for method, z in scores.items()}
    decisions['Count'] = count_matched(ids, scores['SourceBg'], int(decisions['TargetBg'].sum())).astype(bool)
    return dict(ids=ids, has_paths=np.any(o > 0, axis=1), **decisions,
                **{f'margin_{m}': z for m, z in scores.items()})
