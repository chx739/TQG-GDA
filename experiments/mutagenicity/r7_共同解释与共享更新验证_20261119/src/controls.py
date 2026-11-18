"""R7 pointwise and restricted controls on the retained J posterior bank."""

import numpy as np
from scipy.linalg import helmert
from scipy.optimize import minimize

from tsba.shared_update.solver import (
    _array,
    _center_correction,
    _inputs,
    _risk_terms,
    _statistics,
    relative_risks,
    solve_master,
)


def _pointwise_inputs(z, u, qbank, mass, ids):
    z, u = _inputs(z, u)
    qbank = _array(qbank, "posterior bank", 3)
    if (not len(qbank) or qbank.shape[1:] != z.shape
            or ((qbank < 0) | (qbank > 1)).any()
            or not np.allclose(qbank.sum(axis=2), 1., rtol=0., atol=1e-10)):
        raise ValueError("Expected a nonempty [candidate, graph, class] probability bank")
    if mass and z.shape[1] != 2:
        raise ValueError("P_mass requires two classes")
    ids = np.arange(len(z)) if ids is None else np.asarray(ids)
    if ids.shape != (len(z),) or len(np.unique(ids)) != len(z):
        raise ValueError("Expected one unique graph ID per target row")
    return z, u, qbank, ids


def _separate(u, w, bank, mass, ids, feasibility_tol):
    delta = u @ _center_correction(w).T
    if not mass:
        best = np.argmin(np.einsum("knc,nc->kn", bank, delta), axis=0)
        posterior = bank[best, np.arange(len(u))]
        return posterior.T @ u / len(u)

    class_one = bank[:, :, 1]
    lower, upper = class_one.min(axis=0), class_one.max(axis=0)
    mass_lower, mass_upper = class_one.sum(axis=1).min(), class_one.sum(axis=1).max()
    margin = delta[:, 1] - delta[:, 0]
    probabilities = np.where(margin < 0, upper, lower)
    current = probabilities.sum()
    required = np.clip(current, mass_lower, mass_upper) - current
    if required:
        direction = np.sign(required)
        capacity = upper - probabilities if direction > 0 else probabilities - lower
        order = np.lexsort((ids, direction * margin))
        ordered_capacity = capacity[order]
        prefix = np.r_[0., np.cumsum(ordered_capacity[:-1])]
        adjustment = np.clip(abs(required) - prefix, 0., ordered_capacity)
        probabilities[order] += direction * adjustment

    average = probabilities.mean()
    if (np.max(lower - probabilities) > feasibility_tol
            or np.max(probabilities - upper) > feasibility_tol
            or not mass_lower / len(u) - feasibility_tol <= average <= mass_upper / len(u) + feasibility_tol):
        raise FloatingPointError("Pointwise mass separator returned an infeasible posterior")
    return np.stack((1 - probabilities, probabilities)) @ u / len(u)


def pointwise_risk(z, u, w, qbank, *, mass=False, ids=None):
    """Evaluate the exact P or P_mass worst relative NLL of one fixed update."""
    z, u, qbank, ids = _pointwise_inputs(z, u, qbank, mass, ids)
    w = _array(w, "correction", 2)
    if w.shape != (z.shape[1], u.shape[1]):
        raise ValueError("Correction has incompatible dimensions")
    statistic = _separate(u, w, qbank, mass, ids, 1e-10)
    return float(relative_risks(z, u, w, statistic[None])[0])


def solve_pointwise(z, u, qbank, cfg, max_rounds, *, mass=False, ids=None):
    """Solve and independently check the full product or mass-constrained risk."""
    z, u, qbank, ids = _pointwise_inputs(z, u, qbank, mass, ids)
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, (int, np.integer)) or max_rounds < 1:
        raise ValueError("Pointwise separation budget must be a positive integer")
    zero = np.zeros((z.shape[1], u.shape[1]))
    statistics = list(np.einsum("knc,nd->kcd", qbank, u) / len(u))
    w = zero.copy()
    info = dict(valid=False, status="unverified", stop_reason="search_budget_exhausted",
                found_worst_risk=None, exact_separation_violation=None, iterations=[])
    for _ in range(max_rounds):
        w, master = solve_master(
            z, u, statistics, regularization=cfg.regularization, feasibility_tol=cfg.feasibility_tol,
            optimality_tol=cfg.optimality_tol, master_maxiter=cfg.master_maxiter, initial=w,
        )
        entry = dict(master=master)
        info["iterations"].append(entry)
        if not master["valid"]:
            info.update(status="fallback", stop_reason="invalid_master", found_worst_risk=None,
                        exact_separation_violation=None)
            break
        statistic = _separate(u, w, qbank, mass, ids, cfg.feasibility_tol)
        risk = float(relative_risks(z, u, w, statistic[None])[0])
        violation = max(0., risk - master["epigraph"])
        objective = risk + cfg.regularization * float(np.sum(w * w)) / 2
        entry.update(found_worst_risk=risk, exact_separation_violation=violation, objective=objective)
        info.update(found_worst_risk=risk, exact_separation_violation=violation, objective=objective)
        if violation <= cfg.separation_tol and objective <= cfg.feasibility_tol:
            info.update(valid=True, status="completed", stop_reason="exact_separation_verified")
            return w, info
        if any(np.array_equal(statistic, s) for s in statistics):
            info["stop_reason"] = "duplicate_cut_without_verification"
            break
        statistics.append(statistic)
    return zero, info


def _restricted_kkt(theta, lower, risks, epigraph, gradient, statistics, cfg):
    slack = epigraph - risks
    active = np.flatnonzero(slack <= max(10 * cfg.feasibility_tol, 1e-8))
    if not len(active):
        active = np.array([int(np.argmax(risks))])
    target = gradient + cfg.regularization * theta
    columns = (target - statistics[active]).T
    scale = max(1., np.abs(target).max(), np.abs(statistics[active]).max())
    bound_slack = theta[0] - lower
    bound_active = bound_slack <= cfg.feasibility_tol

    def residual(weights):
        value = columns @ weights
        if bound_active:
            value[0] = min(value[0], 0.)
        return value

    weights = np.full(len(active), 1. / len(active))
    if len(active) > 1 and np.abs(residual(weights)).max() > cfg.optimality_tol * scale:
        constraint_scale = np.abs(columns).max()

        def objective(weights):
            value = residual(weights) / constraint_scale
            return .5 * float(value @ value), (columns / constraint_scale).T @ value

        weights = minimize(
            objective, weights, jac=True, method="SLSQP", bounds=[(0., 1.)] * len(active),
            constraints={"type": "eq", "fun": lambda p: p.sum() - 1.,
                         "jac": lambda p: np.ones_like(p)},
            options={"ftol": 1e-14, "maxiter": cfg.master_maxiter},
        ).x
    stationarity = float(np.abs(residual(weights)).max())
    simplex = max(abs(float(weights.sum() - 1.)), float(np.maximum(-weights, 0.).max()))
    complementarity = float(np.abs(weights * slack[active]).max())
    if bound_active:
        bound_multiplier = max(0., float(columns[0] @ weights))
        complementarity = max(complementarity, abs(bound_multiplier * bound_slack))
    valid = (stationarity <= cfg.optimality_tol * scale and simplex <= cfg.feasibility_tol
             and complementarity <= max(cfg.feasibility_tol, cfg.optimality_tol))
    return dict(valid=bool(valid), stationarity_residual=stationarity,
                stationarity_tolerance=float(cfg.optimality_tol * scale),
                simplex_residual=simplex, complementarity_residual=complementarity,
                active_indices=active.tolist(), active_weights=weights.tolist(), bound_active=bool(bound_active))


def solve_restricted(z, u, stats, v0, cfg, *, mode):
    """Solve the J epigraph with an intercept or nonnegative logit-scale restriction."""
    z, u = _inputs(z, u)
    classes = z.shape[1]
    stats = _statistics(stats, classes, u)
    zero = np.zeros((classes, u.shape[1]))
    if not np.all(u[:, -1] == 1.):
        raise ValueError("Restricted controls require a trailing unit intercept feature")
    if mode not in ("bias", "cal", "J_bias", "J_cal"):
        raise ValueError("Expected bias or cal restriction")
    biases = np.zeros((classes - 1, *zero.shape))
    biases[:, :, -1] = helmert(classes, full=False)
    basis, lower = biases, -np.inf
    if mode in ("cal", "J_cal"):
        v0 = _array(v0, "centered source readout", 2)
        if v0.shape != zero.shape:
            raise ValueError("Source readout has incompatible dimensions")
        v0 = v0 - v0.mean(axis=0)
        if not np.allclose(u @ v0.T, z - z.mean(axis=1, keepdims=True), atol=1e-10, rtol=1e-10):
            raise ValueError("Source readout does not reproduce the frozen class-centered logits")
        # The scale's intercept lies in the bias space. Strip it from the
        # scale direction so the parameter basis is orthonormal in W.
        direction = v0.copy()
        direction[:, -1] = 0.
        length = np.linalg.norm(direction)
        if length > 0.:
            basis = np.concatenate(((direction / length)[None], biases))
            lower = -length

    shape = zero.shape
    basis = basis.reshape(len(basis), -1)
    projected_stats = stats.reshape(len(stats), -1) @ basis.T

    def objective(vector):
        theta = vector[:-1]
        return (float(vector[-1] + cfg.regularization * (theta @ theta) / 2),
                np.r_[cfg.regularization * theta, 1.])

    def constraints(vector):
        w = (vector[:-1] @ basis).reshape(shape)
        return vector[-1] - _risk_terms(z, u, w, stats)[0]

    def jacobian(vector):
        w = (vector[:-1] @ basis).reshape(shape)
        gradient = basis @ _risk_terms(z, u, w, stats)[1].ravel()
        return np.column_stack((projected_stats - gradient, np.ones(len(stats))))

    result = minimize(
        objective, np.zeros(len(basis) + 1), jac=True, method="SLSQP",
        bounds=[(lower, None)] + [(None, None)] * len(basis),
        constraints={"type": "ineq", "fun": constraints, "jac": jacobian},
        options={"maxiter": cfg.master_maxiter,
                 "ftol": max(1e-14, min(1e-12, cfg.feasibility_tol * .001, cfg.optimality_tol ** 2))},
    )
    info = dict(valid=False, status="fallback", stop_reason="nonfinite_master_solution",
                found_worst_risk=None, solver_success=bool(result.success), solver_status=int(result.status),
                solver_message=str(result.message), iterations=int(result.nit))
    if not np.isfinite(result.x).all():
        return zero, info
    theta, epigraph = result.x[:-1], float(result.x[-1])
    w = (theta @ basis).reshape(shape)
    risks, gradient = _risk_terms(z, u, w, stats)
    kkt = _restricted_kkt(theta, lower, risks, epigraph, basis @ gradient.ravel(), projected_stats, cfg)
    worst = float(risks.max())
    penalty = cfg.regularization * float(np.sum(w * w)) / 2
    violation = max(0., worst - epigraph)
    valid = (violation <= cfg.feasibility_tol and theta[0] >= lower and kkt["valid"]
             and worst + penalty <= cfg.feasibility_tol)
    scale = 1. if not np.isfinite(lower) else 1. + theta[0] / -lower
    info.update(valid=bool(valid), status="completed" if valid else "fallback",
                stop_reason="solved" if valid else "invalid_master",
                found_worst_risk=worst, regularization_penalty=penalty, objective=worst + penalty,
                epigraph=epigraph, constraint_violation=violation, known_candidate_risks=risks.tolist(),
                source_logit_scale=float(scale), kkt=kkt)
    return (w if valid else zero), info


def check_controls():
    """Check exact separation and correction restrictions on small fixtures."""
    from scipy.optimize import linprog
    from tsba.shared_update.config import SharedUpdateConfig

    cfg = SharedUpdateConfig()
    u = np.array([[-2., 1.], [-1., 1.], [1., 1.], [2., 1.]])
    v0 = np.array([[-.8, -.3], [.8, .3]])
    z = u @ v0.T
    ids = [15, 8, 11, 1]
    q1 = np.array([[.1, .6, .3, .9], [.5, .3, .8, .3]])
    bank = np.stack((1. - q1, q1), axis=2)
    statistics = np.einsum("knc,nd->kcd", bank, u) / len(u)
    w = np.array([[-.2, -.7], [.2, .7]])
    delta = u @ w.T
    margin = delta[:, 1] - delta[:, 0]
    joint = relative_risks(z, u, w, statistics).max()
    pointwise = pointwise_risk(z, u, w, bank, ids=ids)
    mass = pointwise_risk(z, u, w, bank, mass=True, ids=ids)
    assert joint <= mass + 1e-13 and mass <= pointwise + 1e-13
    expected = linprog(margin, A_ub=np.vstack((np.ones(len(u)), -np.ones(len(u)))),
                       b_ub=[q1.sum(axis=1).max(), -q1.sum(axis=1).min()],
                       bounds=list(zip(q1.min(axis=0), q1.max(axis=0))))
    assert expected.success
    exact_q = np.stack((1. - expected.x, expected.x))
    np.testing.assert_allclose(mass, relative_risks(z, u, w, [(exact_q @ u) / len(u)])[0],
                               atol=1e-12, rtol=0.)
    reference, info = solve_master(z, u, statistics[:1], regularization=cfg.regularization)
    assert info["valid"]
    for use_mass in (False, True):
        point, info = solve_pointwise(z, u, bank[:1], cfg, 100, mass=use_mass, ids=ids)
        assert info["valid"], info
        np.testing.assert_allclose(point, reference, atol=1e-7, rtol=0.)
    bias, info = solve_restricted(z, u, statistics, v0, cfg, mode="bias")
    assert info["valid"], info
    np.testing.assert_array_equal(bias[:, :-1], 0.)
    cal, info = solve_restricted(z, u, statistics, v0, cfg, mode="cal")
    assert info["valid"] and info["source_logit_scale"] >= 0., info
    np.testing.assert_allclose(cal.sum(axis=0), 0., atol=1e-13)
    np.testing.assert_allclose(cal[:, :-1], (info["source_logit_scale"] - 1.) * v0[:, :-1],
                               atol=1e-13)
    np.testing.assert_allclose(info["regularization_penalty"], cfg.regularization * np.sum(cal ** 2) / 2)
