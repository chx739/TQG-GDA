"""Identify a binary latent class using only a 4-by-2-by-2 probability table."""
from math import hypot

import numpy as np

BINS = np.arange(1, 5)
GRID = np.indices((4, 2, 2))
VIEWS = np.stack((GRID[0] >= 2, GRID[1] == 1, GRID[2] == 1), axis=-1)
# Treat errors at the scale of summing a few float64 probabilities as a tie.
ROUND_TIE = 8 * np.finfo(float).eps


def accuracies(bin_mass, positive):
    """Accuracy for j=0,...,4, from all-positive to all-negative."""
    tails = np.r_[np.cumsum((2 * positive - bin_mass)[::-1])[::-1], 0.]
    return 1. - positive.sum() + tails


def choose(accuracy):
    tied = np.flatnonzero(accuracy >= float(np.max(accuracy)) - ROUND_TIE)
    return int(min(tied, key=lambda j: (abs(j - 2), j)))


def moments(table, views=VIEWS):
    mean = np.einsum('ult,ultd->d', table, views)
    centered = views - mean
    cov = np.einsum('ult,ulti,ultj->ij', table, centered, centered)
    return mean, cov, float(np.sum(table * np.prod(centered, axis=-1)))


def product_joint(prior, cond_u, cond_l, cond_t):
    """Class-first joint distribution under conditional independence."""
    return np.einsum('y,yu,yl,yt->yult', prior, cond_u, cond_l, cond_t)


def recover(observed, *, covariance_tolerance, feasibility_tolerance):
    prob = np.asarray(observed, dtype=np.float64)
    if prob.shape != (4, 2, 2) or not np.isfinite(prob).all() or np.any(prob < 0):
        raise ValueError('The estimate requires an anonymous nonnegative 4×2×2 probability table')
    if abs(prob.sum() - 1.) > feasibility_tolerance:
        raise ValueError('The observations do not sum to probability one')
    prob = prob / prob.sum()
    mu, cov, third = moments(prob)
    dl, dt, lt = map(float, (cov[0, 1], cov[0, 2], cov[1, 2]))
    covariance = {'D_L': dl, 'D_T': dt, 'L_T': lt}
    ranks = [int(np.linalg.matrix_rank(np.moveaxis(prob, axis, 0).reshape(width, -1)))
             for axis, width in enumerate(prob.shape)]
    result = dict(j=2, status='fallback', reasons=[], prior=None, covariance=covariance,
                  third_moment=third, unfolding_rank=ranks, conditional_means=None, positive_mass=None,
                  estimated_accuracy=None, reconstructed_table_max_error=None, positive_L_contrast=None)

    small = [name for name, value in covariance.items() if abs(value) <= covariance_tolerance]
    if small:
        result['reasons'].extend(f'small_covariance_{name}' for name in small)
    product = dl * dt * lt
    if product <= 0:
        result['reasons'].append('nonpositive_covariance_product')
    if result['reasons']:
        return result

    ratio = np.sign(dt) * third / np.sqrt(product)
    prior = float((1. - ratio / hypot(ratio, 2.)) / 2.)
    result['prior'] = prior if np.isfinite(prior) else None
    if not 0 < prior < 1:
        result['reasons'].append('degenerate_or_nonfinite_prior')
        return result

    # The positive sign of d_L fixes the binary label convention.
    prior_var = prior * (1. - prior)
    d_l = np.sqrt(dl * lt / (dt * prior_var))
    d = np.array([dl / (prior_var * d_l), d_l, lt / (prior_var * d_l)])
    p_u = prob.sum(axis=(1, 2))
    cov_by_bin = prob[:, 1, :].sum(axis=1) - p_u * mu[1]
    positive = prior * p_u + cov_by_bin / d_l
    conditional = np.stack((mu - prior * d, mu + (1. - prior) * d))
    if not np.isfinite(np.r_[d, conditional.ravel(), positive]).all() or d_l <= 0:
        result['reasons'].append('nonfinite_or_invalid_contrast')
        return result
    result.update(positive_L_contrast=float(d_l), positive_mass=positive.tolist(),
                  conditional_means=conditional.tolist())
    tol = feasibility_tolerance
    for cls, row in enumerate(conditional):
        result['reasons'].extend(f'infeasible_{view}_class{cls}' for view, val in zip(('D', 'L', 'T'), row)
                                 if val < -tol or val > 1. + tol)
    result['reasons'].extend(f'infeasible_positive_mass_U{int(k)}' for k, m, size in zip(BINS, positive, p_u)
                             if m < -tol or m > size + tol)
    if result['reasons']:
        return result
    positive = np.clip(positive, 0., p_u)
    conditional = np.clip(conditional, 0., 1.)
    accuracy = accuracies(p_u, positive)
    l1, t1 = conditional[:, 1], conditional[:, 2]
    rebuilt = product_joint([1. - prior, prior],
                            [(p_u - positive) / (1. - prior), positive / prior],
                            np.column_stack((1. - l1, l1)),
                            np.column_stack((1. - t1, t1))).sum(axis=0)
    result.update(status='valid', j=choose(accuracy), positive_mass=positive.tolist(),
                  conditional_means=conditional.tolist(), estimated_accuracy=accuracy.tolist(),
                  reconstructed_table_max_error=float(np.max(np.abs(rebuilt - prob))))
    return result
