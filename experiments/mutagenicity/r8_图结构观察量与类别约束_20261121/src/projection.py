"""Projection of Bernoulli log-odds under the two registered moment bands."""
from dataclasses import dataclass
from itertools import chain, repeat
from math import fsum

import numpy as np
from scipy.optimize import brentq, linprog, minimize
from scipy.special import expit


@dataclass
class AffineRows:
    rows: np.ndarray
    rhs: np.ndarray
    offsets: np.ndarray
    slopes: np.ndarray

    def slack(self, q):
        return self.rhs - self.rows @ q / len(q)


@dataclass
class Bands(AffineRows):
    v: np.ndarray
    point_rows: AffineRows | None

    @classmethod
    def build(cls, v, mu, source_eps, target_eps):
        # Classes in order 0,1, with a separate positive and negative bound.
        signed = np.tile([1., -1.], 2)
        offsets = -signed*np.repeat(mu, 2) - np.repeat(source_eps, 2)
        row = offsets[:, None] + signed[:, None]*v
        rhs = np.full(4, target_eps, dtype=float)
        rhs[:2] -= row[:2].mean(axis=1)
        point_rows = None
        if target_eps == 0 and not np.any(source_eps):
            point_rows = _exact_coordinates(np.asarray(v), mu)
        return cls(row * np.array([-1., -1., 1., 1.])[:, None], rhs,
                   offsets * [-1., -1., 1., 1.], signed * [-1., -1., 1., 1.],
                   np.asarray(v), point_rows)


def _exact_coordinates(v, mu):
    # Adding the two class equations gives (mu1-mu0) E[q] = E[v-mu0].
    # Solve that identity before forming the LP, even for nearly equal means.
    difference = np.longdouble(mu[1]) - mu[0]
    # Accumulate v and -mu0 separately. Subtracting mu0 per graph can
    # round away a tiny v before positive and negative terms cancel.
    numerator = fsum(chain(v, repeat(-float(mu[0]), len(v))))
    mean_residual = np.longdouble(numerator) / len(v)
    if difference:
        center = float(np.mean(v, dtype=np.longdouble))
        ratio = mean_residual / difference
        # An out-of-range prior certifies an empty set. A bounded impossible
        # RHS also lets HiGHS report infeasibility for very small differences.
        if ratio < 0 or ratio > 1:
            prior = -1. if ratio < 0 else 2.
            moment = 0.
        else:
            prior = float(ratio)
            moment = (mu[1] - center) * prior
        mass_coefficient = 1.
    else:
        # Equal means identify no prior. A nonzero marginal discrepancy
        # makes the point equations inconsistent for every prior.
        center = float(mu[1])
        prior = float(np.sign(mean_residual))
        mass_coefficient = 0.
        moment = 0.
    centered = v - center
    scale = float(np.max(np.abs(centered))) or 1.
    mass = np.full(len(v), mass_coefficient)
    moment_row = centered / scale
    return AffineRows(np.stack([mass, -mass, moment_row, -moment_row]),
                      np.array([prior, -prior, moment/scale, -moment/scale]),
                      np.array([mass_coefficient, -mass_coefficient, -center/scale, center/scale]),
                      np.array([0., 0., 1/scale, -1/scale]))


def band_widths(source_counts, target_n, alpha):
    scale = np.log(8/alpha)/2
    return np.sqrt(scale/np.asarray(source_counts)), float(np.sqrt(scale/target_n))


def _lp(cost, bands, options, *, interior=False):
    matrix = bands.rows / bands.rows.shape[1]
    n = matrix.shape[1]
    if interior:
        from scipy.sparse import csr_matrix, eye, hstack, vstack
        # Maximize min_i(q_i,1-q_i) subject to the same four rows.
        ones = csr_matrix(np.ones((n, 1)))
        matrix = vstack([hstack([csr_matrix(matrix), csr_matrix((4, 1))]),
                         hstack([-eye(n), ones]), hstack([eye(n), ones])], format='csr')
        rhs = np.r_[bands.rhs, np.zeros(n), np.ones(n)]
    else:
        rhs = bands.rhs
    tol = options['lp_tolerance']
    return linprog(cost, A_ub=matrix, b_ub=rhs, bounds=(0., 1.), method='highs',
                   options=dict(primal_feasibility_tolerance=tol, dual_feasibility_tolerance=tol))


def prior_interval(bands, options):
    model = bands.point_rows or bands
    unit = np.ones(bands.rows.shape[1]) / bands.rows.shape[1]
    lower = _lp(unit, model, options)
    if lower.status == 2:
        return dict(status='infeasible', lp_message=lower.message)
    upper = _lp(-unit, model, options)
    if not lower.success or not upper.success:
        return dict(status='numerical_failure', lp_lower=lower.message, lp_upper=upper.message)
    lo, hi = float(lower.x.mean()), float(upper.x.mean())
    if (min(rows.slack(q).min() for rows in (model, bands) for q in (lower.x, upper.x))
            < -options['feasibility_tolerance']
            or lo > hi + options['feasibility_tolerance']):
        return dict(status='numerical_failure', lp_message='LP endpoints failed primal replay')
    return dict(status='feasible', lower=min(lo, hi), upper=max(lo, hi))


def _valid(intercept, slope, status, **metadata):
    return dict(status=status, valid=True, intercept=float(intercept), slope=float(slope), **metadata)


def prior_projection(logits, interval, options):
    if interval['status'] != 'feasible':
        return dict(valid=False, status=interval['status'])
    base = expit(logits).mean()
    target = np.clip(base, interval['lower'], interval['upper'])
    if interval['lower'] == interval['upper'] and target in (0., 1.):
        return dict(valid=False, status='boundary_only')
    if abs(base - target) <= options['lp_tolerance']:
        return _valid(0., 0., 'base_feasible', target_prior=float(base), prior_residual=0.)
    if not 0 < target < 1:
        return dict(valid=False, status='boundary_only')
    a = brentq(lambda shift: expit(logits+shift).mean()-target,
               -800-float(logits.max()), 800-float(logits.min()), xtol=1e-12)
    residual = abs(expit(logits+a).mean()-target)
    if residual > options['feasibility_tolerance']:
        return dict(valid=False, status='numerical_failure', prior_residual=float(residual))
    return _valid(a, 0., 'updated', target_prior=float(target), prior_residual=float(residual))


def joint_projection(logits, bands, interval, options, *, exact=False):
    if interval['status'] != 'feasible':
        return dict(valid=False, status=interval['status'])
    model = bands.point_rows or bands

    def primal_violation(q):
        mean = float(q.mean())
        return max(0., float(-model.slack(q).min()), float(-bands.slack(q).min()),
                   interval['lower'] - mean, mean - interval['upper'])

    if exact:
        interior = _lp(np.r_[np.zeros(len(logits)), -1.], model, options, interior=True)
        if not interior.success:
            return dict(valid=False, status='numerical_failure', lp_message=interior.message)
        if interior.x[-1] <= options['interior_tolerance']:
            return dict(valid=False, status='boundary_only_or_near_boundary',
                        lp_max_interior=float(interior.x[-1]))
    base_violation = primal_violation(expit(logits))
    if base_violation <= options['lp_tolerance']:
        return _valid(0., 0., 'base_feasible', base_violation=base_violation,
                      primal_violation=base_violation, dual_kkt_inf=base_violation)
    active_rows = [0, 2] if exact else [0, 1, 2, 3]
    rows = model.rows[active_rows]
    rhs = model.rhs[active_rows]
    scale = np.maximum(np.sqrt((rows**2).mean(axis=1)), 1e-12)
    normalized = rows / scale[:, None]
    baseline = np.logaddexp(0., logits)

    def objective(multiplier):
        projected_logits = logits - multiplier @ normalized
        value = (np.logaddexp(0., projected_logits) - baseline).mean() + multiplier @ (rhs/scale)
        gradient = rhs/scale - normalized @ expit(projected_logits)/len(logits)
        return value, gradient

    result = minimize(objective, np.zeros(len(rows)), jac=True, method='L-BFGS-B',
                      bounds=None if exact else [(0., None)]*len(rows),
                      options=dict(maxiter=options['dual_max_iterations'],
                                   gtol=options['dual_gradient_tolerance'],
                                   ftol=options['dual_function_tolerance']))
    dual = result.x/scale
    a = -dual @ model.offsets[active_rows]
    b = -dual @ model.slopes[active_rows]
    prediction = logits+a+b*bands.v
    q = expit(prediction)
    direct = logits - dual @ rows
    if not np.isfinite(prediction).all() or not np.isfinite(dual).all():
        return dict(valid=False, status='numerical_failure', optimizer_message=str(result.message))
    stationary = float(np.max(np.abs(prediction-direct)))
    slack = model.slack(q)
    primal = primal_violation(q)
    complementary = 0. if exact else float(np.max(np.abs(dual * slack)))
    dual_feasibility = 0. if exact else max(0., float(-dual.min()))
    kkt = max(primal, stationary, complementary, dual_feasibility)
    metadata = dict(base_violation=base_violation, primal_violation=primal,
                    affine_stationarity_inf=stationary, complementarity_inf=complementary,
                    dual_feasibility_violation=dual_feasibility, dual_kkt_inf=kkt,
                    multipliers=dual.tolist(), optimizer_iterations=int(result.nit),
                    optimizer_success=bool(result.success), optimizer_message=str(result.message))
    if primal > options['feasibility_tolerance'] or kkt > options['kkt_tolerance']:
        return dict(valid=False, status='numerical_failure', **metadata)
    return _valid(a, b, 'updated', **metadata)


def projected_logits(base, v, solution):
    if not solution['valid']:
        return None
    if solution['status'] == 'base_feasible':
        return base.copy()
    return base + solution['intercept'] + solution['slope']*v
