"""Frozen source stacks and centered float64 ridge maps on one reference pool."""
from itertools import combinations_with_replacement
from hashlib import sha256

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist

from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural


def standardizer(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError('Expected finite V design')
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[np.ptp(matrix, axis=0) == 0.] = 1.
    return {'mean': mean.tolist(), 'scale': scale.tolist()}


def apply_standardizer(x, fitted):
    return (x - np.asarray(fitted['mean'])) / np.asarray(fitted['scale'])


def quadratic(x):
    return np.column_stack([x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(3), 2)])


def fit_stack(margins, sizes, labels, solver):
    first = standardizer(margins)
    x = apply_standardizer(margins, first)
    second = standardizer(quadratic(x))
    low = standardizer(np.column_stack((x[:, 1], sizes)))
    designs = (x, np.column_stack((x, apply_standardizer(quadratic(x), second))))
    fitted = [structural.fit_candidates(d, d, labels, labels, {'C': [1.0], 'solver': solver})
              for d in designs]
    return {'linear_scale': first, 'quadratic_scale': second, 'low_scale': low,
            'B': fitted[0]['candidates']['1.0'], 'B2': fitted[1]['candidates']['1.0']}


def design(margins, sizes, fitted):
    x = apply_standardizer(margins, fitted['linear_scale'])
    low = apply_standardizer(np.column_stack((x[:, 1], sizes)), fitted['low_scale'])
    extended = np.column_stack((x, apply_standardizer(quadratic(x), fitted['quadratic_scale'])))
    return x[:, 0], low, structural.scores(x, fitted['B']), structural.scores(extended, fitted['B2'])


def rbf(query, reference):
    return np.exp(-cdist(query, reference, metric='sqeuclidean') / 6.)


def vocabulary(rows):
    return wl.vocabulary(rows, range(len(rows)))


def vocabulary_digest(lexicon):
    return sha256(b''.join(bytes([depth]) + color for depth, color in lexicon)).hexdigest()


def sparse(rows, lexicon):
    return wl.sparse_rows(rows, range(len(rows)), lexicon)


def fit_ridge(kernel, target, tolerance):
    k, y = np.asarray(kernel, dtype=np.float64), np.asarray(target, dtype=np.float64)
    n = len(y)
    if k.shape != (n, n) or y.shape != (n,) or not n or not np.isfinite(k).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite or invalid R16 train kernel')
    np.testing.assert_allclose(k, k.T, rtol=0., atol=tolerance)
    mean = k.mean(axis=0)
    grand = float(mean.mean())
    centered = k - mean[None, :] - mean[:, None] + grand
    system = centered + np.eye(n)
    response = y - y.mean()
    alpha = solve(system, response, assume_a='pos', check_finite=True)
    error = float(np.max(np.abs(system @ alpha - response)))
    if not np.isfinite(error) or error > tolerance or abs(alpha.mean()) > tolerance:
        raise ValueError(f'Uncertified R16 ridge solve, residual={error}')
    midpoint = float(y.mean())
    return dict(alpha=alpha.tolist(), y_mean=midpoint, kernel_mean=mean.tolist(), grand_mean=grand,
                system_residual_max=error, alpha_mean=float(alpha.mean()), n=n)


def apply_ridge(query_kernel, model):
    k = np.asarray(query_kernel, dtype=np.float64)
    if k.ndim != 2 or k.shape[1] != model['n'] or not np.isfinite(k).all():
        raise ValueError('Nonfinite or invalid R16 query kernel')
    centered = k - k.mean(axis=1, keepdims=True) - np.asarray(model['kernel_mean']) + model['grand_mean']
    return model['y_mean'] + centered @ np.asarray(model['alpha'])


def fit_pool(x, g, rows, tolerance):
    k_low = rbf(x, x)
    lexicon = vocabulary(rows)
    ref = sparse(rows, lexicon)
    k_wl = (ref @ ref.T).toarray()
    low = fit_ridge(k_low, g, tolerance)
    residual = g - apply_ridge(k_low, low)
    if abs(residual.mean()) > tolerance:
        raise ValueError('Low ridge residual must have zero mean numerically')
    return dict(low=low, wl=fit_ridge(k_wl, residual, tolerance),
                low_twice=fit_ridge(k_low, residual, tolerance), residual_mean=float(residual.mean()),
                residual_rms=float(np.sqrt(np.mean(residual ** 2))),
                vocabulary_size=len(lexicon), vocabulary_sha256=vocabulary_digest(lexicon), x=x.tolist(),
                g=g.tolist(), residual=residual.tolist())


def predict_pool(pool, x, query_rows, reference_rows):
    lexicon = vocabulary(reference_rows)
    if vocabulary_digest(lexicon) != pool['vocabulary_sha256']:
        raise PermissionError('Reference WL vocabulary differs from FIT')
    low_kernel = rbf(x, pool['x'])
    # WL rows carry their original full per-graph denominators. A new H word
    # can never enter the reference dictionary, but it still affects the row.
    wl_kernel = (sparse(query_rows, lexicon) @ sparse(reference_rows, lexicon).T).toarray()
    return (apply_ridge(low_kernel, pool['low']), apply_ridge(wl_kernel, pool['wl']),
            apply_ridge(low_kernel, pool['low_twice']))


def compose(b, g, wg, own, v, other, blend):
    l, k, kk = own
    correction = blend * wg
    low = b + correction * (l - g)
    return dict(Low=low, Own=low + correction*k, LowTwice=low + correction*kk,
                VResidual=low + correction*v[1],
                OtherResidual1=low + correction*other[0][1],
                OtherResidual2=low + correction*other[1][1],
                VFull=b + correction*(v[0] - g + v[1]))
