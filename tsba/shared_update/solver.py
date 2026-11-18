"""Exact finite-update risks and a finite-candidate convex master problem.

Only the outer problem is convex. A local explanation search does not certify
the worst risk over the complete explanation set, or the true target risk.
Matrices stay in [class, feature] form; flattening is an implementation detail.
SciPy is imported only when optimization is requested.
"""
from dataclasses import asdict, is_dataclass
from copy import deepcopy
import math

import numpy as np


def _array(value, name, ndim):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _inputs(source_logits, features):
    z = _array(source_logits, "source_logits", 2)
    u = _array(features, "features", 2)
    if not len(z) or z.shape[1] < 2 or u.shape[0] != z.shape[0] or not u.shape[1]:
        raise ValueError("Expected matching nonempty target rows and at least two classes")
    return z, u


def _lse(values):
    maximum = values.max(axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.exp(values - maximum).sum(axis=1))


def _partition(z, u, correction):
    shifted = z - z.max(axis=1, keepdims=True)
    logp0 = shifted - _lse(shifted)[:, None]
    updated = logp0 + u @ correction.T
    log_norm = _lse(updated)
    change = float(np.mean(log_norm - _lse(logp0)))
    probabilities = np.exp(updated - log_norm[:, None])
    gradient = probabilities.T @ u / len(u)
    if not math.isfinite(change) or not np.isfinite(gradient).all():
        raise FloatingPointError("Non-finite log-partition change or gradient")
    return change, gradient


def log_partition_change(source_logits, features, correction):
    """Return A(W), exactly for the specified finite affine logit update."""
    z, u = _inputs(source_logits, features)
    w = _array(correction, "correction", 2)
    if w.shape != (z.shape[1], u.shape[1]):
        raise ValueError("Correction has incompatible class/feature dimensions")
    return _partition(z, u, w)[0]


def relative_risks(source_logits, features, correction, statistics):
    """Return A(W)-<W,S_k> for all retained conditional-label statistics."""
    z, u = _inputs(source_logits, features)
    w = _array(correction, "correction", 2)
    stats = _array(statistics, "statistics", 3)
    if w.shape != (z.shape[1], u.shape[1]) or stats.shape[1:] != w.shape or not len(stats):
        raise ValueError("Correction/statistics have incompatible dimensions or no candidates")
    return _partition(z, u, w)[0] - np.einsum("kcd,cd->k", stats, w)


def _kkt(gradient, stats, risks, epigraph, regularization, w,
         feasibility_tol, optimality_tol, minimize):
    """Find simplex multipliers on near-active constraints, then check KKT."""
    slack = epigraph - risks
    active = np.flatnonzero(slack <= max(10 * feasibility_tol, 1e-8))
    if not len(active):
        active = np.array([int(np.argmax(risks))])
    columns = stats[active].reshape(len(active), -1).T
    target = (gradient + regularization * w).ravel()
    if len(active) == 1:
        weights = np.ones(1)
        multiplier_success = True
    else:
        def objective(weight):
            residual = columns @ weight - target
            return .5 * float(residual @ residual), columns.T @ residual
        fit = minimize(objective, np.full(len(active), 1 / len(active)), jac=True,
                       method="SLSQP", bounds=[(0., 1.)] * len(active),
                       constraints={"type": "eq", "fun": lambda v: v.sum() - 1,
                                    "jac": lambda v: np.ones_like(v)},
                       options={"ftol": 1e-14, "maxiter": 1000})
        weights = np.asarray(fit.x, dtype=np.float64)
        multiplier_success = bool(fit.success)
    stationarity = float(np.max(np.abs(target - columns @ weights)))
    complementarity = float(np.max(np.abs(weights * slack[active])))
    simplex_error = max(abs(float(weights.sum()) - 1), float(np.maximum(-weights, 0).max()))
    scale = max(1., float(np.max(np.abs(target))), float(np.max(np.abs(columns))))
    valid = (multiplier_success and math.isfinite(stationarity)
             and stationarity <= optimality_tol * scale
             and simplex_error <= feasibility_tol
             and complementarity <= max(feasibility_tol, optimality_tol))
    return dict(valid=bool(valid), stationarity_residual=stationarity,
                stationarity_tolerance=optimality_tol * scale,
                complementarity_residual=complementarity, simplex_residual=simplex_error,
                multiplier_solver_success=multiplier_success,
                active_indices=active.tolist(), active_weights=weights.tolist())


def solve_master(source_logits, features, statistics, *, regularization,
                 feasibility_tol=1e-8, optimality_tol=1e-6,
                 master_maxiter=1000, initial=None):
    """Solve the convex epigraph master; return a candidate and audit metadata.

    A candidate with info['valid'] == False MUST NOT be deployed. The enclosing
    solve_shared_update function replaces such a candidate with the zero update.
    """
    from scipy.optimize import minimize

    z, u = _inputs(source_logits, features)
    stats = _array(statistics, "statistics", 3)
    shape = (z.shape[1], u.shape[1])
    if not len(stats) or stats.shape[1:] != shape:
        raise ValueError("Expected at least one statistic with shape [classes, features]")
    if (not math.isfinite(regularization) or regularization <= 0
            or not 0 < feasibility_tol < 1 or not 0 < optimality_tol < 1
            or master_maxiter < 1):
        raise ValueError("Invalid regularization, tolerance or master iteration budget")
    w0 = np.zeros(shape) if initial is None else _array(initial, "initial", 2)
    if w0.shape != shape:
        raise ValueError("Initial correction has incompatible dimensions")
    risk0 = relative_risks(z, u, w0, stats)
    x0 = np.r_[w0.ravel(), risk0.max()]
    flattened = stats.reshape(len(stats), -1)

    def objective(vector):
        w = vector[:-1]
        return float(vector[-1] + regularization * (w @ w) / 2), np.r_[regularization * w, 1.]

    def constraints(vector):
        value, _ = _partition(z, u, vector[:-1].reshape(shape))
        return vector[-1] - value + flattened @ vector[:-1]

    def constraint_jacobian(vector):
        _, gradient = _partition(z, u, vector[:-1].reshape(shape))
        return np.column_stack((flattened - gradient.ravel(), np.ones(len(stats))))

    result = minimize(objective, x0, jac=True, method="SLSQP",
                      constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
                      options={"ftol": max(1e-14, min(1e-12, feasibility_tol * .001, optimality_tol ** 2)),
                               "maxiter": int(master_maxiter)})
    if not np.isfinite(result.x).all():
        return np.zeros(shape), dict(valid=False, reason="nonfinite_master_solution",
                                     solver_success=bool(result.success), solver_status=int(result.status),
                                     solver_message=str(result.message))
    w = result.x[:-1].reshape(shape).copy()
    value, gradient = _partition(z, u, w)
    risks = value - np.einsum("kcd,cd->k", stats, w)
    epigraph = float(result.x[-1])
    worst = float(risks.max())
    penalty = regularization * float(np.sum(w * w)) / 2
    violation = max(0., worst - epigraph)
    kkt = _kkt(gradient, stats, risks, epigraph, regularization, w,
               feasibility_tol, optimality_tol, minimize)
    checks = dict(solver_success=bool(result.success), feasible=violation <= feasibility_tol,
                  stationary=kkt["valid"], no_worse_than_zero=worst + penalty <= feasibility_tol)
    failures = [name for name, passed in checks.items() if not passed]
    return w, dict(valid=not failures, reason="solved" if not failures else "+".join(failures),
                   solver_status=int(result.status), solver_message=str(result.message),
                   iterations=int(result.nit), candidates=len(stats), **checks,
                   epigraph=epigraph, constraint_violation=violation,
                   known_candidate_risks=risks.tolist(), found_worst_risk=worst,
                   regularization_penalty=penalty, objective=worst + penalty, kkt=kkt)


def _json(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize explanation evidence of type {type(value).__name__}")


def solve_shared_update(source_logits, features, explanations, config):
    """Fit a fixed explanation set, solve masters, and search for violations.

    source_logits are fixed *target* logits produced by the source classifier.
    Search starts are Explanation objects. Target class labels are never inputs.
    Search-budget exhaustion retains only a finite-candidate guarantee; it is
    explicitly not a certificate for the complete explanation set.
    """
    z, u = _inputs(source_logits, features)
    zero = np.zeros((z.shape[1], u.shape[1]), dtype=np.float64)
    regularization = float(config.regularization)
    feasibility_tol = float(config.feasibility_tol)
    separation_tol = float(config.separation_tol)
    optimality_tol = float(getattr(config, "optimality_tol", 1e-6))
    rounds = int(config.search_rounds)
    if (regularization <= 0 or not math.isfinite(regularization) or rounds < 0
            or separation_tol < 0 or not math.isfinite(separation_tol)
            or not 0 < feasibility_tol < 1 or not 0 < optimality_tol < 1):
        raise ValueError("Invalid shared-update optimization configuration")
    candidates, history = [], []
    info = dict(status="fallback", stop_reason=None, iterations=history,
                full_set_certified=False, true_target_risk_certified=False,
                guarantee_scope="retained_feasible_explanations_only",
                reference_loss=None, threshold=None, regularization=regularization)
    numerical_errors = (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError)

    def finish(w, reason, fallback=False):
        if fallback:
            w = zero.copy()
        stats = np.stack([c.statistic for c in candidates]) if candidates else None
        risks = relative_risks(z, u, w, stats).tolist() if candidates else []
        penalty = regularization * float(np.sum(w * w)) / 2
        info.update(status="fallback" if fallback else "completed", stop_reason=reason,
                    used_zero_fallback=fallback, correction_norm=float(np.linalg.norm(w)),
                    candidates=[dict(statistic=_json(c.statistic), observation_loss=float(c.observation_loss),
                                     parameters=_json(c.parameters)) for c in candidates],
                    candidate_count=len(candidates), known_candidate_risks=risks,
                    found_worst_risk=max(risks) if risks else None,
                    final_regularized_objective=max(risks) + penalty if risks else None)
        return w, _json(info)

    try:
        reference = explanations.fit()
        reference_loss = float(explanations.reference_loss)
        threshold = float(explanations.threshold)
        if not math.isfinite(reference_loss) or not math.isfinite(threshold) or threshold < reference_loss:
            raise ValueError("Invalid fixed observation reference/threshold")
        info.update(reference_loss=reference_loss, threshold=threshold,
                    reference_fit_diagnostics=_json(getattr(explanations, "fit_report", None)))
    except numerical_errors as error:
        info["error"] = str(error)
        return finish(zero, "reference_fit_failed", True)

    def validate(candidate):
        if candidate is None:
            raise ValueError("No explanation returned")
        stat = _array(candidate.statistic, "explanation statistic", 2)
        loss = float(candidate.observation_loss)
        if stat.shape != zero.shape or not math.isfinite(loss) or loss > threshold + feasibility_tol:
            raise ValueError("Explanation is infeasible or has invalid dimensions")
        _json(candidate.parameters)

    try:
        validate(reference)
    except (TypeError, *numerical_errors) as error:
        info["error"] = str(error)
        return finish(zero, "invalid_reference", True)
    candidates.append(deepcopy(reference))
    w = zero.copy()
    stop_reason = "search_budget_exhausted"
    # An extra master solve after the last search is mandatory: the last found
    # violating statistic must constrain the returned correction as well.
    for iteration in range(rounds + 1):
        entry = dict(iteration=iteration, candidate_count=len(candidates))
        history.append(entry)
        try:
            w, master = solve_master(z, u, np.stack([c.statistic for c in candidates]),
                                     regularization=regularization, feasibility_tol=feasibility_tol,
                                     optimality_tol=optimality_tol, master_maxiter=config.master_maxiter,
                                     initial=w)
            entry["master"] = master
        except numerical_errors as error:
            entry["error"] = str(error)
            return finish(zero, "master_exception", True)
        if not master["valid"]:
            return finish(zero, "invalid_master", True)
        if iteration == rounds:
            break
        try:
            found = explanations.search(u @ w.T, deepcopy(candidates))
            entry["search_diagnostics"] = _json(getattr(explanations, "last_search", None))
            if (float(explanations.reference_loss) != reference_loss
                    or float(explanations.threshold) != threshold):
                raise ValueError("Explanation search changed the fixed observation set")
            if found is None:
                entry["search"] = "no_feasible_candidate_returned"
                stop_reason = "search_returned_no_candidate"
                break
            validate(found)
        except (TypeError, *numerical_errors) as error:
            entry["error"] = str(error)
            return finish(zero, "invalid_search_result", True)
        risk = float(relative_risks(z, u, w, np.asarray([found.statistic]))[0])
        violation = risk - master["found_worst_risk"]
        entry.update(search_observation_loss=float(found.observation_loss),
                     search_risk=risk, separation_violation=violation,
                     searched_statistic=_json(found.statistic))
        # Exact duplicate statistics introduce no new constraint. Distinct
        # feasible results are retained even if the current violation is small.
        duplicate = any(np.array_equal(found.statistic, c.statistic) for c in candidates)
        entry["duplicate_statistic"] = duplicate
        if not duplicate:
            candidates.append(deepcopy(found))
        if violation <= separation_tol or duplicate:
            diagnostics = entry["search_diagnostics"] or {}
            stop_reason = ("local_search_incomplete" if diagnostics.get("status") == "local_search_failed"
                           else "no_significant_violation_found")
            if not duplicate:
                try:
                    w, final_master = solve_master(z, u, np.stack([c.statistic for c in candidates]),
                                                   regularization=regularization, feasibility_tol=feasibility_tol,
                                                   optimality_tol=optimality_tol, master_maxiter=config.master_maxiter,
                                                   initial=w)
                    entry["final_master"] = final_master
                    if not final_master["valid"]:
                        return finish(zero, "invalid_final_master", True)
                except numerical_errors as error:
                    entry["error"] = str(error)
                    return finish(zero, "final_master_exception", True)
            break
    risks = relative_risks(z, u, w, np.stack([c.statistic for c in candidates]))
    objective = float(risks.max() + regularization * np.sum(w * w) / 2)
    if not math.isfinite(objective) or objective > feasibility_tol:
        return finish(zero, "final_zero_comparison_failed", True)
    return finish(w, stop_reason)
