"""Known risk and matched product controls for deterministic source reweighting."""
from fractions import Fraction

import numpy as np

from . import identification as ident


def cell_counts(y, u, l, t):
    columns = [np.asarray(a) for a in (y, u, l, t)]
    if (len({a.shape for a in columns}) != 1 or columns[0].ndim != 1
            or any(not np.isin(a, allowed).all()
                   for a, allowed in zip(columns, ((0, 1), ident.BINS, (0, 1), (0, 1))))):
        raise ValueError('Expected aligned source Y, U bins 1–4, and binary L/T')
    y, u, l, t = (a.astype(np.int64, copy=False) for a in columns)
    return np.bincount((4 * y + u - 1) * 4 + 2 * l + t, minlength=32).reshape(2, 4, 2, 2)


def true_accuracy(positive_percent):
    positive = [Fraction(q, 400) for q in positive_percent]
    negative = [Fraction(1, 4) - p for p in positive]
    return [sum(negative[:j]) + sum(positive[j:]) for j in range(5)]


def tables(counts, positive_percent):
    n = counts.sum(axis=(2, 3))
    if np.any(n == 0):
        raise ValueError('Every (Y,U) stratum needs empirical support')
    q = np.asarray(positive_percent, dtype=np.float64) / 100
    class_bin = np.stack((1. - q, q)) / 4.
    conditional = counts / n[:, :, None, None]
    joint = conditional * class_bin[:, :, None, None]
    class_mass = class_bin.sum(axis=1)
    u, l, t = class_marginals(joint)
    product = ident.product_joint(class_mass, u, l, t)
    np.testing.assert_allclose(product.sum(axis=(2, 3)), class_bin, rtol=0, atol=1e-14)
    for actual, expected in zip(class_marginals(product), (u, l, t)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    return joint, product


def class_marginals(joint):
    p = joint.sum(axis=(1, 2, 3))
    return tuple(joint.sum(axis=axes) / p[:, None] for axes in ((2, 3), (1, 3), (1, 2)))


def conditional_stats(joint):
    axes = ident.VIEWS.astype(float)
    ordinal = axes.copy()
    ordinal[..., 0] = ident.GRID[0] + 1
    prior = joint.sum(axis=(1, 2, 3))
    conditional = joint / prior[:, None, None, None]
    means = [ident.moments(item)[0] for item in conditional]
    u, l, t = class_marginals(joint)
    contrasts = {name: float(np.max(np.abs(table[1] - table[0]))) for name, table in zip(('U', 'L', 'T'), (u, l, t))}
    order = 'positive' if means[1][1] - means[0][1] > ident.ROUND_TIE else (
        'negative' if means[0][1] - means[1][1] > ident.ROUND_TIE else 'zero')
    rows = []
    for cls, item in enumerate(conditional):
        row = {'class': cls, 'D_mean': float(means[cls][0]), 'L_mean': float(means[cls][1]), 'T_mean': float(means[cls][2])}
        for views, pairs in ((axes, ((0, 1, 'D_L'), (0, 2, 'D_T'), (1, 2, 'L_T'))),
                             (ordinal, ((0, 1, 'U_L'), (0, 2, 'U_T')))):
            _, cov, _ = ident.moments(item, views)
            for i, j, name in pairs:
                variance_product = cov[i, i] * cov[j, j]
                # When a view is constant, tiny positive roundoff in mu can
                # leave an artificial variance. Check support explicitly.
                varied = all(np.unique(views[..., v][item > 0]).size > 1 for v in (i, j))
                row[name] = float(np.clip(cov[i, j] / np.sqrt(variance_product), -1., 1.)) if (
                    varied and variance_product > 0) else None
        rows.append(row)
    return dict(prior=float(prior[1]), L_order=order, L_contrast=float(means[1][1] - means[0][1]),
                views_rank={k: (1 if d <= ident.ROUND_TIE else 2) for k, d in contrasts.items()},
                D_contrast=float(means[1][0] - means[0][0]), T_contrast=float(means[1][2] - means[0][2]),
                class_rows=rows)


def stratum_oracle(counts, observed, positive_percent, tolerance):
    """Audit reuse of P(L,T|U,Y) in the paired construction, per U bin."""
    g = counts / counts.sum(axis=(2, 3), keepdims=True)
    p_u = observed.sum(axis=(1, 2))
    h = observed / p_u[:, None, None]
    delta = (g[1] - g[0]).reshape(4, 4)
    residual = (h - g[0]).reshape(4, 4)
    recovered, rows = [], []
    for i, (d, r) in enumerate(zip(delta, residual)):
        square = float(d @ d)
        q = float(d @ r / square) if square > 0 else None
        expected = positive_percent[i] / 100.
        if q is not None and abs(q - expected) > tolerance:
            raise AssertionError('Empirical stratum construction did not recover its assigned label proportion')
        recovered.append(q)
        rows.append(dict(bin=i+1, identifiable=q is not None, squared_contrast=square,
                         recovered_positive_rate=q, assigned_positive_rate=expected))
    accuracy = None if None in recovered else ident.accuracies(p_u, p_u * recovered)
    return rows, None if accuracy is None else ident.choose(accuracy)
